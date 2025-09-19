import json
import torch
from torch_geometric.data import Data

with open("rank-0.json", "r") as f:
    raw = json.load(f)

nodes = raw["nodes"]  # <-- this is the list of ops

# Step 1: build node_id → index mapping
node_ids = [node["id"] for node in nodes]
id2idx = {nid: i for i, nid in enumerate(node_ids)}

# Step 2: build edges from ctrl_deps
edge_index = []
for node in nodes:
    src = node.get("ctrl_deps")
    if src is not None and src in id2idx:
        edge_index.append([id2idx[src], id2idx[node["id"]]])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Step 3: encode "name" as categorical feature
unique_names = list({node["name"] for node in nodes})
name2id = {name: i for i, name in enumerate(unique_names)}
x = torch.tensor([[name2id[node["name"]]] for node in nodes], dtype=torch.long)

# Step 4: build PyG Data
graph = Data(x=x, edge_index=edge_index)

print("x shape:", graph.x.shape)
print("edge_index shape:", graph.edge_index.shape)
print("Name vocab size:", len(unique_names))


import torch

# Save single graph
torch.save(graph, "graph.pth")

# Later load it back
loaded_graph = torch.load("graph.pth")
print(loaded_graph)



from collections import Counter

# Count frequencies of names
name_counts = Counter([node["name"] for node in nodes])

# Get top 10
top10 = name_counts.most_common(10)

print("Top 10 names and their frequencies:")
for name, freq in top10:
    print(f"{name}: {freq}")
