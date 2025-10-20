import torch
import torch.nn.functional as F
import numpy as np
import random
import argparse
import os
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling

from model import GraphAutoencoder


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pth_graphs(pth_path):
    data_dict = torch.load(pth_path, map_location='cuda')
    src_list, dst_list, x_n_list = data_dict['src_list'], data_dict['dst_list'], data_dict['x_n_list']

    graphs = []
    for i, x in enumerate(x_n_list):
        src, dst = src_list[i], dst_list[i]
        edge_index = torch.stack([src.long(), dst.long()], dim=0)
        num_nodes = x.shape[0]
        x = x.long()

        if edge_index.numel() > 0:
            max_index = edge_index.max().item()
            min_index = edge_index.min().item()
            if max_index >= num_nodes or min_index < 0:
                print(f"Graph {i} skipped: edge_index out of bounds (max {max_index}, min {min_index}, nodes {num_nodes})")
                continue

        graphs.append(Data(x=x, edge_index=edge_index))

    return graphs


def train_autoencoder(model, graphs, device='cuda', epochs=20, batch_size=32, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pos_edge_index = batch.edge_index
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=batch.num_nodes,
                num_neg_samples=pos_edge_index.size(1),
                method='sparse'
            )

            pos_out, _ = model(batch.x, pos_edge_index)
            neg_out, _ = model(batch.x, neg_edge_index)

            labels = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))]).to(device)
            preds = torch.cat([pos_out, neg_out], dim=0)

            loss = F.binary_cross_entropy_with_logits(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs

        avg_loss = total_loss / len(loader.dataset)
        print(f"[Epoch {epoch+1:02d}] Loss: {avg_loss:.4f}")

    return model


def extract_embeddings(model, graphs, device='cuda', batch_size=32):
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    model.eval()
    all_embeds = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            graph_embeds = model.get_graph_repr(batch.x, batch.edge_index, batch.batch)
            all_embeds.append(graph_embeds.cpu())

    return torch.cat(all_embeds, dim=0)


def plot_tsne(embeddings, labels, save_path=None):
    print("Running t-SNE projection...")
    tsne = TSNE(n_components=2, perplexity=10, learning_rate=200, init='random', random_state=42, metric='cosine')
    emb_2d = tsne.fit_transform(embeddings)

    colors = ['red' if l == 1 else 'blue' for l in labels]
    plt.figure(figsize=(8, 6))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, alpha=0.6, s=20, edgecolors='k', linewidths=0.3)
    plt.title("t-SNE Projection of Graph Embeddings\nRed=Real, Blue=Generated")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.grid(True)

    # Zoom out by expanding axis limits by 10%
    x_min, x_max = emb_2d[:, 0].min(), emb_2d[:, 0].max()
    y_min, y_max = emb_2d[:, 1].min(), emb_2d[:, 1].max()
    x_margin = (x_max - x_min) * 0.5
    y_margin = (y_max - y_min) * 0.5
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE plot to {save_path}")
    else:
        plt.show()



def main(args):
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Exiting because CPU is not allowed.")
    device = torch.device(args.device)

    real_graphs = load_pth_graphs(args.real)
    gen_graphs = load_pth_graphs(args.generated)

    if len(real_graphs) == 0 or len(gen_graphs) == 0:
        print("Error: One of the datasets is empty.")
        return

    print(f"Loaded {len(real_graphs)} real graphs, {len(gen_graphs)} generated graphs.")

    # Compute categorical_dims only from real graphs (training set)
    example_x = real_graphs[0].x
    n_cat_columns = example_x.shape[1]
    categorical_dims = [max([g.x[:, i].max().item() for g in real_graphs]) + 1 for i in range(n_cat_columns)]

    model = GraphAutoencoder(
        categorical_dims=categorical_dims,
        emb_dim=8,
        hidden_dim=64,
        latent_dim=32,
        encoder_type=args.model
    ).to(device)

    print("\n[Training Autoencoder on real graphs only]")
    model = train_autoencoder(model, real_graphs, device=device,
                        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    print("\n[Extracting embeddings for real and generated graphs]")
    real_embeds = extract_embeddings(model, real_graphs, device=device, batch_size=args.batch_size)
    gen_embeds = extract_embeddings(model, gen_graphs, device=device, batch_size=args.batch_size)

    embeddings = torch.cat([real_embeds, gen_embeds], dim=0)
    embeddings_np = embeddings.numpy()
    labels_np = np.array([1] * len(real_embeds) + [0] * len(gen_embeds))


    print("\n[Evaluation]")
    try:
        silhouette = silhouette_score(embeddings_np, labels_np)
        print(f"Silhouette Score: {silhouette:.4f} (lower = more mixed real/gen)")
    except Exception as e:
        print(f"Failed to compute silhouette score: {e}")

    plot_tsne(embeddings_np, labels_np, save_path="tsne_naive.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=str, default="/usr/scratch/vshitole6/TraceCraft/analysis/dataset/real.pth")
    parser.add_argument("--generated", type=str, default="/usr/scratch/vshitole6/TraceCraft/analysis/dataset/generated_naive.pth")
    parser.add_argument("--model", type=str, default="graphsage", help="graphsage|gcn")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
