# attacks/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, global_mean_pool, TransformerConv

class NodeFeatureEncoder(nn.Module):
    def __init__(self, cardinalities, emb_dim=8):
        super().__init__()
        assert len(cardinalities) == 6, "Expect 6 categorical node feature columns"
        self.embs = nn.ModuleList([nn.Embedding(c, emb_dim) for c in cardinalities])
        self.out_dim = emb_dim * len(cardinalities)

    def forward(self, x):
        parts = []
        for i, emb in enumerate(self.embs):
            parts.append(emb(x[:, i].long()))
        return torch.cat(parts, dim=-1)

class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_node_dim, hidden=64, num_layers=2, num_classes=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_node_dim, hidden))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.pool = global_mean_pool
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, num_classes)
        )
        self.act = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        g = self.pool(x, batch)
        return self.mlp(g)
    
    def get_graph_repr(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.dropout_layer(x)
        return global_mean_pool(x, batch)


class GCNClassifier(nn.Module):
    """Simple GCN-like MPNN baseline"""
    def __init__(self, in_node_dim, hidden=64, num_layers=2, num_classes=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_node_dim, hidden))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.pool = global_mean_pool
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, num_classes)
        )

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        g = self.pool(x, batch)
        return self.mlp(g)

class TransformerGraphClassifier(nn.Module):
    """Graph Transformer using TransformerConv"""
    def __init__(self, in_node_dim, hidden=64, heads=4, num_layers=2, num_classes=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(in_node_dim, hidden//heads, heads=heads))
        for _ in range(num_layers - 1):
            self.convs.append(TransformerConv(hidden, hidden//heads, heads=heads))
        self.pool = global_mean_pool
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, num_classes)
        )

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        g = self.pool(x, batch)
        return self.mlp(g)

class MLPBaseline(nn.Module):
    """
    Non-graph baseline: pool node embeddings (mean) and then an MLP.
    Assumes node features are already embedded (so in_node_dim is embedding dim).
    """
    def __init__(self, in_node_dim, hidden=128, num_classes=2, dropout=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_node_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, num_classes)
        )

    def forward(self, x, edge_index, batch):
        # ignore edge_index, perform mean pooling over nodes in each graph
        g = global_mean_pool(x, batch)
        return self.mlp(g)


# Add this to attacks/models.py

class GraphAutoencoder(nn.Module):
    def __init__(self, categorical_dims, emb_dim=8, hidden_dim=64, latent_dim=32, encoder_type='gcn'):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        self.feature_encoder = NodeFeatureEncoder(categorical_dims, emb_dim)
        in_node_dim = self.feature_encoder.out_dim  # Computed from emb_dim * num categorical columns

        if self.encoder_type == 'graphsage':
            self.encoder = nn.Sequential(
                SAGEConv(in_node_dim, hidden_dim),
                nn.ReLU(),
                SAGEConv(hidden_dim, latent_dim)
            )
        elif self.encoder_type == 'gcn':
            self.encoder = nn.Sequential(
                GCNConv(in_node_dim, hidden_dim),
                nn.ReLU(),
                GCNConv(hidden_dim, latent_dim)
            )
        else:
            raise ValueError("Unsupported encoder type: choose 'gcn' or 'graphsage'")

    def encode(self, x, edge_index):
        x = self.feature_encoder(x)
        for layer in self.encoder:
            if isinstance(layer, (SAGEConv, GCNConv)):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x

    def decode(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        logits = self.decode(z, edge_index)
        return logits, z

    def get_graph_repr(self, x, edge_index, batch):
        z = self.encode(x, edge_index)
        return global_mean_pool(z, batch)

