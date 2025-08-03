import torch
import numpy as np
import matplotlib.pyplot as plt
import os

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
    return num_nodes, num_edges, features

real_nodes, real_edges, real_feats = extract_stats(real_data)
rand_nodes, rand_edges, rand_feats = extract_stats(random_data)

# Plotting helper
def plot_histogram(data1, data2, label1, label2, xlabel, title, save_path, bins='auto'):
    plt.figure(figsize=(6, 4))
    plt.hist(data1, bins=bins, alpha=0.5, label=label1, color='blue', edgecolor='black')
    plt.hist(data2, bins=bins, alpha=0.5, label=label2, color='orange', edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 1. Node count distribution
plot_histogram(
    real_nodes, rand_nodes,
    'Real', 'Random DAG',
    'Number of nodes',
    'Node Count Distribution',
    'comparison_plots/node_count.png'
)

# 2. Edge count distribution
plot_histogram(
    real_edges, rand_edges,
    'Real', 'Random DAG',
    'Number of edges',
    'Edge Count Distribution',
    'comparison_plots/edge_count.png'
)

# 3. Attribute distributions
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

print("✅ All comparison plots saved in 'comparison_plots/'")
