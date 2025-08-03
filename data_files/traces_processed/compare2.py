import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import networkx as nx

# Load real and random datasets
real_data = torch.load('train.pth', map_location='cpu')
random_data = torch.load('random_dag_baseline.pth', map_location='cpu')

# Create output directory
os.makedirs('comparison_plots', exist_ok=True)

# Extract lists
def extract_stats(data):
    x_n_list = data['x_n_list']
    src_list = data['src_list']
    dst_list = data['dst_list']
    
    num_nodes = [x.shape[0] for x in x_n_list]
    num_edges = [s.shape[0] for s in src_list]
    features = torch.cat(x_n_list, dim=0)  # [total_nodes, 6]
    return num_nodes, num_edges, features, src_list, dst_list

real_nodes, real_edges, real_feats, real_src, real_dst = extract_stats(real_data)
rand_nodes, rand_edges, rand_feats, rand_src, rand_dst = extract_stats(random_data)

# Plotting KDE for continuous metrics
def plot_kde(data1, data2, label1, label2, xlabel, title, save_path):
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data1, label=label1, fill=True)
    sns.kdeplot(data2, label=label2, fill=True)
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 1. Node count distribution (KDE)
plot_kde(
    real_nodes, rand_nodes,
    'Real', 'Random DAG',
    'Number of nodes',
    'Node Count Distribution (KDE)',
    'comparison_plots/node_count_kde.png'
)

# 2. Edge count distribution (KDE)
plot_kde(
    real_edges, rand_edges,
    'Real', 'Random DAG',
    'Number of edges',
    'Edge Count Distribution (KDE)',
    'comparison_plots/edge_count_kde.png'
)

# 3. Attribute distributions (bar plots for categorical features)
num_features = real_feats.shape[1]

for i in range(num_features):
    r_vals = real_feats[:, i].tolist()
    g_vals = rand_feats[:, i].tolist()
    
    unique_vals = sorted(set(r_vals + g_vals))
    real_counts = [r_vals.count(v) for v in unique_vals]
    rand_counts = [g_vals.count(v) for v in unique_vals]

    x = np.arange(len(unique_vals))
    width = 0.4

    plt.figure(figsize=(6, 4))
    plt.bar(x - width/2, real_counts, width, label='Real', color='blue')
    plt.bar(x + width/2, rand_counts, width, label='Random DAG', color='orange')
    plt.xlabel(f'Feature {i} Value')
    plt.ylabel('Count')
    plt.title(f'Feature {i} Distribution')
    plt.xticks(ticks=x, labels=unique_vals)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'comparison_plots/feature_{i}_distribution.png')
    plt.close()

# Helper to build NetworkX graphs and compute structural metrics
def build_nx_graph(src_list, dst_list, idx):
    src = src_list[idx].tolist()
    dst = dst_list[idx].tolist()
    G = nx.DiGraph()
    G.add_nodes_from(range(len(set(src + dst))))
    edges = list(zip(src, dst))
    G.add_edges_from(edges)
    return G

def compute_graph_metrics(src_list, dst_list):
    in_degrees = []
    out_degrees = []
    clustering_coeffs = []
    diameters = []
    for i in range(len(src_list)):
        G = build_nx_graph(src_list, dst_list, i)
        # In-degree and out-degree for all nodes
        in_deg = [d for n, d in G.in_degree()]
        out_deg = [d for n, d in G.out_degree()]
        in_degrees.extend(in_deg)
        out_degrees.extend(out_deg)
        # Clustering coefficient (undirected)
        undirected = G.to_undirected()
        clust_coeffs = list(nx.clustering(undirected).values())
        clustering_coeffs.extend(clust_coeffs)
        # Diameter (only if weakly connected)
        if nx.is_weakly_connected(G):
            diam = nx.diameter(undirected)
            diameters.append(diam)
    return in_degrees, out_degrees, clustering_coeffs, diameters

print("Computing structural metrics for real graphs...")
real_in_deg, real_out_deg, real_clust, real_diam = compute_graph_metrics(real_src, real_dst)

print("Computing structural metrics for random graphs...")
rand_in_deg, rand_out_deg, rand_clust, rand_diam = compute_graph_metrics(rand_src, rand_dst)

# Plot in-degree distribution (KDE)
plot_kde(
    real_in_deg, rand_in_deg,
    'Real', 'Random DAG',
    'In-degree',
    'In-Degree Distribution (KDE)',
    'comparison_plots/in_degree_distribution_kde.png'
)

# Plot out-degree distribution (KDE)
plot_kde(
    real_out_deg, rand_out_deg,
    'Real', 'Random DAG',
    'Out-degree',
    'Out-Degree Distribution (KDE)',
    'comparison_plots/out_degree_distribution_kde.png'
)

# Plot clustering coefficient distribution (KDE)
plot_kde(
    real_clust, rand_clust,
    'Real', 'Random DAG',
    'Clustering Coefficient',
    'Clustering Coefficient Distribution (KDE)',
    'comparison_plots/clustering_coefficient_kde.png'
)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.kdeplot(real_diam, label='Real', bw_adjust=0.5, fill=True)
sns.kdeplot(rand_diam, label='Random DAG', bw_adjust=0.5, fill=True)
plt.xlabel('Diameter')
plt.ylabel('Density')
plt.title('Diameter Distribution (KDE)')
plt.legend()
plt.tight_layout()
plt.savefig('comparison_plots/diameter_distribution_kde.png')
plt.close()


print("✅ All comparison plots saved in 'comparison_plots/'")
