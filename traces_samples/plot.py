import torch
import matplotlib.pyplot as plt
import networkx as nx

# ----------------------------
# Load the data
# ----------------------------
data = torch.load('train.pth', map_location='cpu')
src_list = data['src_list']
dst_list = data['dst_list']

# ----------------------------
# Plot 3 graph structures side-by-side
# ----------------------------
plt.figure(figsize=(15, 5))

for i in range(3):
    G = nx.DiGraph()
    edges = list(zip(src_list[i].tolist(), dst_list[i].tolist()))
    G.add_edges_from(edges)
    
    plt.subplot(1, 3, i + 1)
    pos = nx.spring_layout(G, seed=42)  # or use nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=False, node_color='skyblue', edge_color='gray', node_size=50, arrows=True)
    plt.title(f'Graph {i + 1}')

plt.tight_layout()
plt.savefig("three_graphs.png", dpi=300)
plt.show()
