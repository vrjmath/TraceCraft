# metrics.py
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm, trange

# -----------------------------
# Load datasets
# -----------------------------
real_data = torch.load('train.pth', map_location='cpu')
rand_data = torch.load('randomB.pth', map_location='cpu')

# -----------------------------
# Extract statistics
# -----------------------------
def extract_stats(data):
    x_n_list = data['x_n_list']
    src_list = data['src_list']
    dst_list = data['dst_list']
    num_nodes = [x.shape[0] for x in x_n_list]
    return num_nodes, src_list, dst_list

real_nodes, real_src, real_dst = extract_stats(real_data)
rand_nodes, rand_src, rand_dst = extract_stats(rand_data)

# -----------------------------
# Graph-level metrics
# -----------------------------
def build_nx_graph(src_list, dst_list, idx):
    src = src_list[idx].tolist()
    dst = dst_list[idx].tolist()
    G = nx.DiGraph()
    G.add_nodes_from(range(len(set(src + dst))))
    G.add_edges_from(zip(src, dst))
    return G

def compute_graph_metrics(src_list, dst_list):
    degrees, num_layers = [], []
    for i in tqdm(range(len(src_list)), desc="Processing graphs"):
        G = build_nx_graph(src_list, dst_list, i)
        # Combine in- and out-degree
        degs = [d for _, d in G.in_degree()] + [d for _, d in G.out_degree()]
        degrees.extend(degs)
        # Compute # layers = length of longest path
        if nx.is_directed_acyclic_graph(G):
            num_layers.append(len(nx.dag_longest_path(G)))
        else:
            num_layers.append(0)
    return degrees, num_layers

real_deg, real_layers = compute_graph_metrics(real_src, real_dst)
rand_deg, rand_layers = compute_graph_metrics(rand_src, rand_dst)

# -----------------------------
# Distance metrics
# -----------------------------
def kl_divergence(p, q, eps=1e-10):
    p = np.array(p) + eps
    q = np.array(q) + eps
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))

def rbf_kernel_np(X, Y, gamma=1.0):
    X = np.asarray(X).reshape(-1,1)
    Y = np.asarray(Y).reshape(-1,1)
    dist2 = (X**2).sum(axis=1)[:,None] + (Y**2).sum(axis=1)[None,:] - 2*X.dot(Y.T)
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
    return XX.mean() + YY.mean() - 2*XY.mean()


def wasserstein_1d(u, v):
    u_sorted = np.sort(u)
    v_sorted = np.sort(v)
    n = min(len(u_sorted), len(v_sorted))
    return np.mean(np.abs(u_sorted[:n] - v_sorted[:n]))

def compare_metric(name, real_vals, rand_vals, bins=50, normalize=True):
    real_array = np.array(real_vals)
    rand_array = np.array(rand_vals)

    if normalize:
        max_val = max(real_array.max(), rand_array.max())
        if max_val > 0:
            real_array = real_array / max_val
            rand_array = rand_array / max_val

    w1 = wasserstein_1d(real_array, rand_array)

    hist_real, bin_edges = np.histogram(real_array, bins=bins, density=True)
    hist_rand, _ = np.histogram(rand_array, bins=bin_edges, density=True)
    kl = kl_divergence(hist_real, hist_rand)
    mmd = compute_mmd(real_array, rand_array)

    print(f"\n=== {name} ===")
    print(f"Wasserstein-1 (normalized): {w1:.4f}")
    print(f"KL Divergence: {kl:.4f}")
    print(f"MMD: {mmd:.4f}")


# -----------------------------
# Run comparisons
# -----------------------------
metric_names = ["# Layers", "# Nodes", "Degree"]
real_vals_list = [real_layers, real_nodes, real_deg]
rand_vals_list = [rand_layers, rand_nodes, rand_deg]

for i in trange(len(metric_names), desc="Comparing metrics"):
    compare_metric(metric_names[i], real_vals_list[i], rand_vals_list[i])

print("\n✅ Metrics computed successfully")
