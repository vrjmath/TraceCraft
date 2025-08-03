import matplotlib.pyplot as plt
import networkx as nx
import torch

# Load the generated random DAGs
#data = torch.load('train.pth', map_location='cpu')
data = torch.load('random_dag_baseline.pth', map_location='cpu')
x_n_list = data['x_n_list']
src_list = data['src_list']
dst_list = data['dst_list']

# ----------------------------
# Plotting a few graphs
# ----------------------------
def plot_graph(ax, src, dst):
    G = nx.DiGraph()
    edges = list(zip(src.tolist(), dst.tolist()))
    G.add_edges_from(edges)
    nx.draw(
        G,
        ax=ax,
        with_labels=False,
        node_size=10,
        arrows=True,
        edge_color='gray',
        node_color='black',
        pos=nx.spring_layout(G, seed=42),
    )

# Select how many to show
num_graphs_to_show = 3
fig, axes = plt.subplots(1, num_graphs_to_show, figsize=(12, 4))

for i in range(num_graphs_to_show):
    src = src_list[i]
    dst = dst_list[i]
    plot_graph(axes[i], src, dst)
    axes[i].set_title(f'Graph {i+1}')

plt.tight_layout()
plt.savefig("check2.png", dpi=300)
print("📸 Saved visualization to 'check2.png'")
plt.show()
