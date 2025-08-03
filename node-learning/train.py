import torch
from torch_geometric.loader import DataLoader
from torch import optim
import torch.nn.functional as F

from graph_dataset import TraceGraphDataset
from gnn_model import NodeEmbeddingModel, NodeAttributePredictor

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = TraceGraphDataset("../data_files/traces_processed/train.pth")
    train_loader = DataLoader(train_dataset.graphs, batch_size=1, shuffle=True)

    encoder = NodeEmbeddingModel().to(device)
    decoder = NodeAttributePredictor(in_dim=64).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

    for epoch in range(50):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)

            h = encoder(batch.x, batch.edge_index)
            preds = decoder(h)

            # Use batch.y for true labels!
            loss = sum(F.cross_entropy(preds[i], batch.y[:, i]) for i in range(6))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

if __name__ == "__main__":
    main()
