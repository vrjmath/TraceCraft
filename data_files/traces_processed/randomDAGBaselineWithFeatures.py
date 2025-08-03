import torch
import numpy as np
import random

# ----------------------------
# Load your real dataset
# ----------------------------
data = torch.load('train.pth', map_location='cpu')
x_n_list = data['x_n_list']
src_list = data['src_list']
dst_list = data['dst_list']

# ----------------------------
# Compute graph sizes and edge densities on real data
# ----------------------------
num_nodes = [x.shape[0] for x in x_n_list]
num_edges = [s.shape[0] for s in src_list]

edge_densities = []
for n, e in zip(num_nodes, num_edges):
    max_edges = n * (n - 1) / 2 if n > 1 else 1
    edge_densities.append(e / max_edges)
avg_edge_density = np.mean(edge_densities)
print(f"✅ Average edge density (real data): {avg_edge_density:.4f}")

# ----------------------------
# Build empirical distribution for each node feature
# ----------------------------
all_features = torch.cat(x_n_list, dim=0)  # [total_nodes, num_features]
feature_value_lists = []

for i in range(all_features.shape[1]):
    unique_vals, counts = torch.unique(all_features[:, i], return_counts=True)
    probs = counts.float() / counts.sum()
    feature_value_lists.append((unique_vals.tolist(), probs.tolist()))

def sample_node_features(num_nodes, feature_value_lists):
    """
    Sample node features for a graph with `num_nodes` nodes.
    """
    features = []
    for unique_vals, probs in feature_value_lists:
        sampled = np.random.choice(unique_vals, size=num_nodes, p=probs)
        features.append(sampled)
    return torch.tensor(np.stack(features, axis=1), dtype=torch.long)

# ----------------------------
# DAG Generator
# ----------------------------
def generate_weakly_connected_random_dag(n_nodes, edge_prob):
    """
    Generate a weakly connected DAG with n_nodes.
    Uses a chain for weak connectivity, then adds extra edges with probability edge_prob.
    """
    if n_nodes == 1:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    src = list(range(n_nodes - 1))
    dst = list(range(1, n_nodes))

    # Number of possible edges excluding the chain edges
    total_possible_edges = n_nodes * (n_nodes - 1) // 2
    chain_edges = n_nodes - 1
    remaining_edges = total_possible_edges - chain_edges

    # Adjusted edge probability for remaining edges to match expected total edges = original e
    if remaining_edges > 0:
        adjusted_edge_prob = edge_prob
    else:
        adjusted_edge_prob = 0.0

    for i in range(n_nodes):
        for j in range(i + 2, n_nodes):
            if random.random() < adjusted_edge_prob:
                src.append(i)
                dst.append(j)

    return torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)


# ----------------------------
# Generate random DAG baseline dataset with node features
# ----------------------------
random_dags_src = []
random_dags_dst = []
random_dags_x_n = []

for n, e in zip(num_nodes, num_edges):
    total_possible_edges = n * (n - 1) // 2
    chain_edges = n - 1
    remaining_edges = total_possible_edges - chain_edges

    # Avoid division by zero if remaining_edges == 0
    if remaining_edges > 0:
        edge_prob = max((e - chain_edges) / remaining_edges, 0.0)
    else:
        edge_prob = 0.0

    src_edges, dst_edges = generate_weakly_connected_random_dag(n, edge_prob)
    random_dags_src.append(src_edges)
    random_dags_dst.append(dst_edges)

    sampled_features = sample_node_features(n, feature_value_lists)
    random_dags_x_n.append(sampled_features)

print(f"\n📦 Generated Random DAG Baseline")
print(f"Number of graphs: {len(random_dags_src)}")
print(f"Sample graph: {random_dags_x_n[0].shape[0]} nodes, {random_dags_src[0].shape[0]} edges")
print(f"Sample node features (first 5 nodes):\n{random_dags_x_n[0][:5]}")

# ----------------------------
# Save dataset
# ----------------------------
save_dict = {
    'x_n_list': random_dags_x_n,
    'src_list': random_dags_src,
    'dst_list': random_dags_dst
}

torch.save(save_dict, 'random_dag_baseline.pth')
print("💾 Saved random DAG baseline to 'random_dag_baseline.pth'")
