import torch
import networkx as nx
import pickle

# Load data
data = torch.load('train.pth', map_location='cpu')

src_list = data['src_list']
dst_list = data['dst_list']
x_n_list = data['x_n_list']

# Convert to networkx graphs
graph_list = []
for i in range(len(src_list)):
    G = nx.Graph()
    num_nodes = x_n_list[i].shape[0]
    G.add_nodes_from(range(num_nodes))  # add all nodes
    edges = zip(src_list[i].tolist(), dst_list[i].tolist())
    G.add_edges_from(edges)
    graph_list.append(G)

# Save as .pkl
with open('custom_dataset.pkl', 'wb') as f:
    pickle.dump(graph_list, f)

print(f"Saved {len(graph_list)} graphs to custom_dataset.pkl")
