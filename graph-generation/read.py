import torch
import os
import networkx as nx
from eval.stats import degree_stats, clustering_stats

# Utility to convert one graph from edge list + node features into NetworkX
def build_nx_graph(x_n, src, dst):
    G = nx.DiGraph()
    for i, feat in enumerate(x_n):
        G.add_node(i, feature=feat)
    for s, d in zip(src, dst):
        G.add_edge(s, d)
    return G

def load_graphs(path):
    data = torch.load(path)
    x_n_list = data['x_n_list']
    src_list = data['src_list']
    dst_list = data['dst_list']

    graphs = []
    for x_n, src, dst in zip(x_n_list, src_list, dst_list):
        G = build_nx_graph(x_n, src, dst)
        graphs.append(G)
    return graphs

def main():
    original_path = "/usr/scratch/vshitole6/TraceCraft/data_files/traces_processed/train.pth"
    generated_path = "/usr/scratch/vshitole6/TraceCraft/traces_samples/train.pth"

    print("Loading original graphs...")
    original_graphs = load_graphs(original_path)
    print(f"Loaded {len(original_graphs)} original graphs.")

    print("Loading generated graphs...")
    generated_graphs = load_graphs(generated_path)
    print(f"Loaded {len(generated_graphs)} generated graphs.")

    print("Calculating Degree Distribution MMD...")
    degree_mmd = degree_stats(original_graphs, generated_graphs, is_parallel=False)
    print(f"Degree MMD: {degree_mmd:.4f}")

    print("Calculating Clustering Coefficient MMD...")
    clustering_mmd = clustering_stats(original_graphs, generated_graphs, is_parallel=False)
    print(f"Clustering MMD: {clustering_mmd:.4f}")

if __name__ == "__main__":
    main()
