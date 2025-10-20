# attacks/main.py
import argparse
import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from model import NodeFeatureEncoder, GraphSAGEClassifier, GCNClassifier, TransformerGraphClassifier, MLPBaseline

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_pth_graphs(pth_path):
    d = torch.load(pth_path, map_location='cpu')
    src_list, dst_list, x_n_list = d['src_list'], d['dst_list'], d['x_n_list']
    graphs = []
    n_graphs = len(x_n_list)
    for i in range(n_graphs):
        src = src_list[i]
        dst = dst_list[i]
        edge_index = torch.stack([src.long(), dst.long()], dim=0)
        #x = x_n_list[i].long()
        num_nodes = x_n_list[i].shape[0]
        x = torch.ones((num_nodes, 1), dtype=torch.float)

        num_nodes = x.size(0)
        if edge_index.numel() > 0:
            max_index = edge_index.max().item()
            min_index = edge_index.min().item()
            if max_index >= num_nodes or min_index < 0:
                print(f"Graph {i} skipped: edge_index out of bounds (max {max_index}, min {min_index}, nodes {num_nodes})")
                continue

        data = Data(x=x, edge_index=edge_index)
        graphs.append(data)
    return graphs

def compute_cardinalities(all_graphs):
    maxes = [0]*6
    for g in all_graphs:
        x = g.x
        for c in range(6):
            if x.size(1) <= c:
                raise ValueError("Expect 6 columns in node features")
            m = int(x[:, c].max().item()) if x.size(0) > 0 else 0
            if m > maxes[c]:
                maxes[c] = m
    cardinality = [m + 1 for m in maxes]
    print(cardinality)
    return cardinality

def build_dataset(real_pth, gen_pth):
    real_graphs = load_pth_graphs(real_pth)
    gen_graphs = load_pth_graphs(gen_pth)
    # Create label: real=1, gen=0
    for g in real_graphs:
        g.y = torch.tensor([1], dtype=torch.long)
    for g in gen_graphs:
        g.y = torch.tensor([0], dtype=torch.long)
    all_graphs = real_graphs + gen_graphs
    return all_graphs, len(real_graphs), len(gen_graphs)

def split(graphs, train_frac=0.7, val_frac=0.15, seed=42):
    n = len(graphs)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
    return [graphs[i] for i in train_idx], [graphs[i] for i in val_idx], [graphs[i] for i in test_idx]

def evaluate(model, loader, device):
    model.eval()
    ys, preds, probs = [], [], []
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y.view(-1).to(device))
            total_loss += loss.item() * batch.num_graphs
            p = torch.softmax(out, dim=1)[:,1].cpu().numpy()
            pred = out.argmax(dim=1).cpu().numpy()
            ys.extend(batch.y.view(-1).cpu().numpy().tolist())
            preds.extend(pred.tolist())
            probs.extend(p.tolist())
    if len(ys) == 0:
        return {"loss": None, "acc": None, "auc": None}
    acc = accuracy_score(ys, preds)
    try:
        auc = roc_auc_score(ys, probs)
    except Exception:
        auc = float('nan')
    avg_loss = total_loss / len(ys)
    return {"loss": avg_loss, "acc": acc, "auc": auc}

def train_loop(model, train_loader, val_loader, device, epochs=50, lr=1e-3, weight_decay=0.0, save_path=None):
    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best_val_auc = -1.0
    best_state = None
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        avg_train_loss = total_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d} | Train loss {avg_train_loss:.4f} | Val loss {val_metrics['loss']:.4f} | Val acc {val_metrics['acc']:.4f} | Val AUC {val_metrics['auc']:.4f}")
        if val_metrics['auc'] is not None and val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if save_path:
                torch.save(best_state, save_path)
    # load best
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

class FullModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, x, edge_index, batch):
        return self.base(x, edge_index, batch)

def main(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    all_graphs, n_real, n_gen = build_dataset(args.real, args.generated)
    print(f"Loaded {len(all_graphs)} graphs: {n_real} real, {n_gen} generated")

    train_graphs, val_graphs, test_graphs = split(all_graphs, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed)
    print(f"Split -> train: {len(train_graphs)}, val: {len(val_graphs)}, test: {len(test_graphs)}")

    # compute cardinalities for node embedding
   #cardinalities = compute_cardinalities(all_graphs)
    #print("Node categorical column cardinalities:", cardinalities)

    # Build Node feature encoder
    #node_emb_dim = args.emb_dim
    #encoder = NodeFeatureEncoder(cardinalities, emb_dim=node_emb_dim)
    # compute in_node_dim after embedding
    #in_node_dim = encoder.out_dim

    # Wrap dataset to apply encoder to Data.x before batching (we'll map)
    def encode_graphs(graphs):
        out = []
        for g in graphs:
            # encode x
            with torch.no_grad():
                # encoder expects LongTensor [N,6], but it's a module -> move to cpu to embed
                # We'll store raw x and embed on device in training loop by wrapping a small function:
                out.append(Data(x=g.x, edge_index=g.edge_index, y=g.y))
        return out

    train_graphs = encode_graphs(train_graphs)
    val_graphs = encode_graphs(val_graphs)
    test_graphs = encode_graphs(test_graphs)

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    if args.model.lower() == "graphsage":
        base_model = GraphSAGEClassifier(in_node_dim=1, hidden=128, num_layers=2, dropout=0.5).to(device)
        #base_model = GraphSAGEClassifier(in_node_dim=in_node_dim, hidden=args.hidden, num_layers=args.num_layers, dropout=args.dropout)
    elif args.model.lower() == "gcn" or args.model.lower() == "mpnn":
        base_model = GCNClassifier(in_node_dim=in_node_dim, hidden=args.hidden, num_layers=args.num_layers, dropout=args.dropout)
    elif args.model.lower() == "transformer":
        base_model = TransformerGraphClassifier(in_node_dim=in_node_dim, hidden=args.hidden, num_layers=args.num_layers, dropout=args.dropout)
    elif args.model.lower() == "mlp":
        base_model = MLPBaseline(in_node_dim=in_node_dim, hidden=args.hidden, dropout=args.dropout)
    else:
        raise ValueError("Unknown model: choose one of graphsage|gcn|transformer|mlp")

    model = FullModel(base_model).to(device)

    save_path = args.save_path if args.save_path else None
    model = train_loop(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, save_path=save_path)

    # final evaluation on test set
    test_metrics = evaluate(model, test_loader, device)
    print("Final Test -> loss: {:.4f}, acc: {:.4f}, auc: {:.4f}".format(
        test_metrics['loss'], test_metrics['acc'], test_metrics['auc']
    ))
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")
    

    # Better experiment
    if args.eval_unsupervised:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.neighbors import NearestNeighbors
        from torch_geometric.nn import global_mean_pool

        print("\n[Unsupervised Privacy Evaluation]")

        # Combine all graphs again (with label)
        full_loader = DataLoader(train_graphs + val_graphs + test_graphs, batch_size=args.batch_size, shuffle=False)

        all_embeds = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for batch in full_loader:
                batch = batch.to(device)
                #x = model.encoder(batch.x)
                out = model.base.get_graph_repr(batch.x, batch.edge_index, batch.batch)
                all_embeds.append(out.cpu())
                all_labels.extend(batch.y.cpu().tolist())

        all_embeds = torch.cat(all_embeds, dim=0).numpy()
        all_labels = np.array(all_labels)

        # -- Experiment 1: KMeans clustering and silhouette score
        print("→ Running KMeans clustering...")
        kmeans = KMeans(n_clusters=2, n_init="auto", random_state=args.seed)
        cluster_labels = kmeans.fit_predict(all_embeds)
        silhouette = silhouette_score(all_embeds, cluster_labels)
        print(f"[Clustering] Silhouette Score: {silhouette:.4f} (higher = more separable)")

        # -- Experiment 2: Nearest neighbor overlap
        print("→ Calculating nearest neighbor distances from real to generated...")

        real_embeds = all_embeds[all_labels == 1]
        gen_embeds = all_embeds[all_labels == 0]

        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(gen_embeds)
        dists, _ = nn.kneighbors(real_embeds)

        dists = dists.flatten()
        print(f"[NN Overlap] Real→Gen Nearest Neighbor Distances:")
        print(f"  Mean: {np.mean(dists):.4f}")
        print(f"  Median: {np.median(dists):.4f}")
        print(f"  5th percentile: {np.percentile(dists, 5):.4f}")
        print(f"  Min: {np.min(dists):.4f}")

        if np.min(dists) < 0.01:
            print("  ⚠️ Warning: Very close match found — possible memorization or low-diversity generation.")

    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    real_path = "/usr/scratch/vshitole6/TraceCraft/analysis/dataset/real.pth"
    generated_path = "/usr/scratch/vshitole6/TraceCraft/analysis/dataset/generated_tracecraft.pth"
    parser.add_argument("--real", type=str, default=real_path, help="Path to real.pth")
    parser.add_argument("--generated", type=str, default=generated_path, help="Path to generated.pth (fake)")
    parser.add_argument("--model", type=str, default="graphsage", help="graphsage|gcn|transformer|mlp")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--emb_dim", type=int, default=8, help="per-category embedding dim (each of 6 columns)")
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--eval_unsupervised", action="store_true", help="Run unsupervised clustering and NN overlap experiments after training")
    args = parser.parse_args()
    main(args)
