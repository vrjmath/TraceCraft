import torch
from torch_geometric.loader import DataLoader
from torch import optim
import torch.nn.functional as F

from graph_dataset import TraceGraphDataset
from gnn_model import NodeEmbeddingModel, NodeAttributePredictor

def evaluate(encoder, decoder, loader, device):
    encoder.eval()
    decoder.eval()
    
    total_loss = 0
    total_nodes = 0
    correct_per_feature = [0] * 6
    total_per_feature = [0] * 6

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            h = encoder(batch.x, batch.edge_index)
            preds = decoder(h)
            
            loss = sum(F.cross_entropy(preds[i], batch.y[:, i]) for i in range(6))
            total_loss += loss.item() * batch.num_nodes
            total_nodes += batch.num_nodes
            
            for i in range(6):
                pred_labels = preds[i].argmax(dim=1)
                # Use batch.y here for accuracy too
                correct_per_feature[i] += (pred_labels == batch.y[:, i]).sum().item()
                total_per_feature[i] += batch.num_nodes

    avg_loss = total_loss / total_nodes
    accuracies = [correct / total for correct, total in zip(correct_per_feature, total_per_feature)]

    print(f"Test Loss: {avg_loss:.4f}")
    for i, acc in enumerate(accuracies):
        print(f"Feature {i+1} accuracy: {acc:.4f}")

    encoder.train()
    decoder.train()
    return avg_loss, accuracies

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = TraceGraphDataset("../data_files/traces_processed/train.pth")
    test_dataset = TraceGraphDataset("../data_files/traces_processed/test.pth")

    train_loader = DataLoader(train_dataset.graphs, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset.graphs, batch_size=1, shuffle=False)

    encoder = NodeEmbeddingModel().to(device)
    decoder = NodeAttributePredictor(in_dim=64).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

    for epoch in range(50):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)

            h = encoder(batch.x, batch.edge_index)
            preds = decoder(h)

            # Use batch.y for ground truth labels!
            loss = sum(F.cross_entropy(preds[i], batch.y[:, i]) for i in range(6))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

    print("Evaluating on test set...")
    evaluate(encoder, decoder, test_loader, device)

if __name__ == "__main__":
    main()
