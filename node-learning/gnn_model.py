import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class NodeEncoder(nn.Module):
    def __init__(self, cat_dims=[2, 4, 24, 96, 15, 4], out_dim=32):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, min(16, num_classes))
            for num_classes in cat_dims
        ])
        emb_dim = sum(emb.embedding_dim for emb in self.embeddings)
        self.proj = nn.Linear(emb_dim, out_dim)

    def forward(self, x_cat):
        # Ensure input is long for embedding layers
        x_cat = x_cat.long()
        emb_list = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return self.proj(torch.cat(emb_list, dim=1))

class GNNEncoder(nn.Module):
    def __init__(self, node_emb_dim=32, hidden_dim=64, out_dim=64):
        super().__init__()
        self.conv1 = SAGEConv(node_emb_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class NodeEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_encoder = NodeEncoder()
        self.gnn = GNNEncoder()

    def forward(self, x_cat, edge_index):
        x = self.node_encoder(x_cat)
        return self.gnn(x, edge_index)

class NodeAttributePredictor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.out_dims = [2, 4, 24, 96, 15, 4]
        self.heads = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for out_dim in self.out_dims
        ])

    def forward(self, h):
        return [head(h) for head in self.heads]  # list of [num_nodes, out_dim]
