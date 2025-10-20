import networkx as nx
import torch
import numpy as np
from scipy.stats import wasserstein_distance


def build_nx_graph(src_list, dst_list, idx):
    src = src_list[idx].tolist()
    dst = dst_list[idx].tolist()
    G = nx.DiGraph()
    G.add_nodes_from(range(len(set(src + dst))))
    G.add_edges_from(zip(src, dst))
    return G

def compute_graph_metrics(src_list, dst_list, x_n_list=None):
    avg_in_deg, avg_out_deg, num_layers, num_nodes, num_edges = [], [], [], [], []
    num_dags, num_weakly_connected, num_graphs = 0, 0, len(src_list)  # initialize new metrics
    
    for i in range(len(src_list)):
        G = build_nx_graph(src_list, dst_list, i)
        
        in_degrees = [d for _, d in G.in_degree()]
        out_degrees = [d for _, d in G.out_degree()]
        
        avg_in_deg.append(np.mean(in_degrees))
        avg_out_deg.append(np.mean(out_degrees))
        
        num_nodes.append(G.number_of_nodes())
        num_edges.append(G.number_of_edges())
        
        if nx.is_directed_acyclic_graph(G):
            num_dags += 1
        
        if nx.is_weakly_connected(G):
            num_weakly_connected += 1
        
        if nx.is_directed_acyclic_graph(G):
            num_layers.append(len(nx.dag_longest_path(G)))
        else:
            num_layers.append(0)
    
    node_type_values = []
    if x_n_list is not None:
        for x_n in x_n_list:
            first_attr = x_n[:, 0].tolist()  # assuming x_n is a 2D tensor [num_nodes, 6]
            node_type_values.extend(first_attr)
    
    return {
        "avg_in_deg": avg_in_deg, 
        "avg_out_deg": avg_out_deg,  
        "num_layers": num_layers,    
        "num_nodes": num_nodes,      
        "num_edges": num_edges,
        "num_dags": num_dags,  
        "num_weakly_connected": num_weakly_connected,  
        "num_graphs": num_graphs,
        "node_types": node_type_values
    }

def compare_distributions(real_metrics, generated_metrics):
    metrics = ["num_nodes", "num_edges", "num_layers", "avg_in_deg", "avg_out_deg"]

    kl_divergences, mmds, wassersteins = {}, {}, {}

    for metric in metrics:
        real_data = real_metrics[metric]
        generated_data = generated_metrics[metric]
        
        kl_div = kl_divergence(real_data, generated_data)
        kl_divergences[metric] = f"{kl_div:.3g}"
        
        mmd_value = compute_mmd(real_data, generated_data)
        mmds[metric] = f"{mmd_value:.3g}"
        
        wasserstein_value = wasserstein_1d(real_data, generated_data)
        wassersteins[metric] = f"{wasserstein_value:.3g}"

    return {
        "kl_divergences": kl_divergences,
        "mmds": mmds,
        "wassersteins": wassersteins
    }   

def kl_divergence(p, q, eps=1e-10):
    p = np.array(p) + eps
    q = np.array(q) + eps
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))

def rbf_kernel_np(X, Y, gamma=1.0):
    X = np.asarray(X).reshape(-1,1)
    Y = np.asarray(Y).reshape(-1,1)
    dist2 = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    return np.exp(-gamma * dist2)

def compute_mmd(x, y, gamma=1.0, max_samples=1000):
    x = np.array(x)
    y = np.array(y)
    if len(x) > max_samples:
        x = np.random.choice(x, max_samples, replace=False)
    if len(y) > max_samples:
        y = np.random.choice(y, max_samples, replace=False)

    XX = rbf_kernel_np(x, x, gamma=gamma)
    YY = rbf_kernel_np(y, y, gamma=gamma)
    XY = rbf_kernel_np(x, y, gamma=gamma)

    m = len(x)
    n = len(y)

    return (XX.sum() - np.trace(XX)) / (m * (m - 1)) + \
           (YY.sum() - np.trace(YY)) / (n * (n - 1)) - \
           2 * XY.mean()

def wasserstein_1d(u, v, normalize=True):
    u = np.asarray(u)
    v = np.asarray(v)

    if normalize:
        combined = np.concatenate([u, v])
        min_val = combined.min()
        max_val = combined.max()
        range_val = max_val - min_val

        if range_val > 0:
            u = (u - min_val) / range_val
            v = (v - min_val) / range_val
        else:
            u = np.zeros_like(u)
            v = np.zeros_like(v)

    return wasserstein_distance(u, v)

def compute_percentiles(data):
    return [
        round(np.min(data), 3),
        round(np.percentile(data, 25), 3),
        round(np.median(data), 3),
        round(np.percentile(data, 75), 3),
        round(np.max(data), 3)
    ]