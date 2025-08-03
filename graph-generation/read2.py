import torch
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter

# Load data
original_data = torch.load("/usr/scratch/vshitole6/TraceCraft/data_files/traces_processed/train.pth")
generated_data = torch.load("/usr/scratch/vshitole6/TraceCraft/traces_samples/train.pth")

def get_node_counts(x_n_list):
    return [len(x_n) for x_n in x_n_list]

def get_all_degrees(src_list, dst_list):
    degrees = []
    for src, dst in zip(src_list, dst_list):
        G = nx.DiGraph()
        G.add_edges_from(zip(src, dst))
        degrees.extend([d for _, d in G.degree()])
    return degrees

# Extract node counts
original_node_counts = get_node_counts(original_data['x_n_list'])
generated_node_counts = get_node_counts(generated_data['x_n_list'])

# Extract degrees
original_degrees = get_all_degrees(original_data['src_list'], original_data['dst_list'])
generated_degrees = get_all_degrees(generated_data['src_list'], generated_data['dst_list'])

# --- Plotting ---
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot node count distribution
axs[0].hist(original_node_counts, bins=30, alpha=0.6, label='Original', color='blue')
axs[0].hist(generated_node_counts, bins=30, alpha=0.6, label='Generated', color='orange')
axs[0].set_title("Node Count Distribution")
axs[0].set_xlabel("Number of Nodes")
axs[0].set_ylabel("Frequency")
axs[0].legend()

# Plot degree distribution
axs[1].hist(original_degrees, bins=30, alpha=0.6, label='Original', color='blue', density=True)
axs[1].hist(generated_degrees, bins=30, alpha=0.6, label='Generated', color='orange', density=True)
axs[1].set_title("Degree Distribution")
axs[1].set_xlabel("Node Degree")
axs[1].set_ylabel("Density")
axs[1].legend()

plt.tight_layout()
plt.show()

plt.savefig('node_and_degree_distribution.png', dpi=300)

import torch
import networkx as nx
import matplotlib.pyplot as plt

# Load data
original_data = torch.load("/usr/scratch/vshitole6/TraceCraft/data_files/traces_processed/train.pth")
generated_data = torch.load("/usr/scratch/vshitole6/TraceCraft/traces_samples/train.pth")

def build_nx_graph(src_list, dst_list, idx):
    G = nx.DiGraph()
    edges = list(zip(src_list[idx], dst_list[idx]))
    G.add_edges_from(edges)
    return G

num_to_plot = 3  # number of graphs from each set to plot

fig, axes = plt.subplots(2, num_to_plot, figsize=(4 * num_to_plot, 8))

for i in range(num_to_plot):
    # Plot original graph
    G_real = build_nx_graph(original_data['src_list'], original_data['dst_list'], i)
    ax = axes[0, i]
    nx.draw_networkx(G_real, ax=ax, node_size=50, arrowsize=10, with_labels=False)
    ax.set_title(f"Original Graph {i+1}")
    ax.axis('off')

    # Plot generated graph
    G_gen = build_nx_graph(generated_data['src_list'], generated_data['dst_list'], i)
    ax = axes[1, i]
    nx.draw_networkx(G_gen, ax=ax, node_size=50, arrowsize=10, with_labels=False)
    ax.set_title(f"Generated Graph {i+1}")
    ax.axis('off')

plt.tight_layout()
plt.show()

plt.savefig('gggggggggggggggggggggggggggggggg.png', dpi=300)


