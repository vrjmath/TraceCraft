import torch
import numpy as np
import random
import matplotlib.pyplot as plt

# Load real dataset
data = torch.load('train.pth', map_location='cpu')
x_n_list = data['x_n_list']
src_list = data['src_list']
dst_list = data['dst_list']

# Extract real graph stats
real_num_nodes = [x.shape[0] for x in x_n_list]
real_num_edges = [s.shape[0] for s in src_list]

# Calculate average edge density on real data
edge_densities = []
for n, e in zip(real_num_nodes, real_num_edges):
    max_edges = n * (n - 1) / 2 if n > 1 else 1
    edge_densities.append(e / max_edges)
avg_edge_density = np.mean(edge_densities)
print(f"Average edge density (real data): {avg_edge_density:.4f}")

def generate_random_dag(n_nodes, edge_prob):
    src, dst = [], []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if random.random() < edge_prob:
                src.append(i)
                dst.append(j)
    return torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)

# Generate random DAG baseline graphs
random_src_list = []
random_dst_list = []
random_num_nodes = []
random_num_edges = []

for n in real_num_nodes:
    src, dst = generate_random_dag(n, avg_edge_density)
    random_src_list.append(src)
    random_dst_list.append(dst)
    random_num_nodes.append(n)
    random_num_edges.append(len(src))

print(f"Generated {len(random_src_list)} random DAG graphs")

# Compute degree distributions helper
def get_degree_distribution(src_list, dst_list, num_nodes_list):
    in_degrees = []
    out_degrees = []
    for src, dst, n in zip(src_list, dst_list, num_nodes_list):
        in_deg = np.zeros(n, dtype=int)
        out_deg = np.zeros(n, dtype=int)
        for s, d in zip(src.tolist(), dst.tolist()):
            out_deg[s] += 1
            in_deg[d] += 1
        in_degrees.extend(in_deg.tolist())
        out_degrees.extend(out_deg.tolist())
    return in_degrees, out_degrees

# Get degree distributions for real and random
real_in_deg, real_out_deg = get_degree_distribution(src_list, dst_list, real_num_nodes)
random_in_deg, random_out_deg = get_degree_distribution(random_src_list, random_dst_list, random_num_nodes)

# Plot helper
def plot_histogram(real_data, random_data, xlabel, ylabel, title, filename, bins=50, log_scale=False):
    plt.figure(figsize=(8,5))
    plt.hist(real_data, bins=bins, alpha=0.5, label='Real', density=True)
    plt.hist(random_data, bins=bins, alpha=0.5, label='Random DAG Baseline', density=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if log_scale:
        plt.yscale('log')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Plot comparisons
plot_histogram(real_num_nodes, random_num_nodes,
               xlabel='Number of Nodes',
               ylabel='Density',
               title='Node Count Distribution',
               filename='node_count_distribution.png')

plot_histogram(real_num_edges, random_num_edges,
               xlabel='Number of Edges',
               ylabel='Density',
               title='Edge Count Distribution',
               filename='edge_count_distribution.png')

plot_histogram(real_in_deg, random_in_deg,
               xlabel='In-Degree',
               ylabel='Density',
               title='In-Degree Distribution',
               filename='in_degree_distribution.png',
               log_scale=True)

plot_histogram(real_out_deg, random_out_deg,
               xlabel='Out-Degree',
               ylabel='Density',
               title='Out-Degree Distribution',
               filename='out_degree_distribution.png',
               log_scale=True)

print("Plots saved: node_count_distribution.png, edge_count_distribution.png, in_degree_distribution.png, out_degree_distribution.png")
