import torch
from torch_geometric.data import Data, InMemoryDataset

class TraceGraphDataset(InMemoryDataset):
    def __init__(self, path):
        super().__init__()
        raw_data = torch.load(path, map_location='cpu')

        self.graphs = []
        for x_n, src, dst in zip(raw_data['x_n_list'], raw_data['src_list'], raw_data['dst_list']):
            edge_index = torch.stack([src, dst], dim=0)  # shape [2, num_edges]
            num_nodes = x_n.size(0)

            # Dummy input features: all ones, shape [num_nodes, 1]
            x_dummy = torch.ones((num_nodes, 1), dtype=torch.float)

            # True node features as labels, shape [num_nodes, 6]
            y = x_n.long()

            data = Data(x=x_dummy, edge_index=edge_index, y=y)
            self.graphs.append(data)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]
