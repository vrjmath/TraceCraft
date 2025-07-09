import torch
import numpy as np

# Load the data
data = torch.load('train.pth', map_location='cpu')

# Get lists
x_n_list = data['x_n_list']
src_list = data['src_list']
dst_list = data['dst_list']

# ----------------------------
# Graph-level stats
# ----------------------------
num_nodes = [x.shape[0] for x in x_n_list]
num_edges = [s.shape[0] for s in src_list]

def describe(arr):
    arr = np.array(arr)
    return {
        'min': int(np.min(arr)),
        '25%': int(np.percentile(arr, 25)),
        '50% (median)': int(np.median(arr)),
        '75%': int(np.percentile(arr, 75)),
        'max': int(np.max(arr)),
    }

print("📊 Graph-level Statistics:")
print("Number of nodes per graph:", describe(num_nodes))
print("Number of edges per graph:", describe(num_edges))

# ----------------------------
# Node feature stats (per attribute)
# ----------------------------
all_features = torch.cat(x_n_list, dim=0)  # shape [total_nodes, 6]
feature_stats = {}

print("\n📈 Node Feature Statistics (per attribute):")
for i in range(all_features.shape[1]):
    values = all_features[:, i].numpy()
    feature_stats[f'Feature {i + 1}'] = {
        'min': float(np.min(values)),
        '25%': float(np.percentile(values, 25)),
        '50% (median)': float(np.median(values)),
        '75%': float(np.percentile(values, 75)),
        'max': float(np.max(values)),
    }

# Pretty print
for feat, stats in feature_stats.items():
    print(f"\n{feat}:")
    for k, v in stats.items():
        print(f" - {k}: {v}")
