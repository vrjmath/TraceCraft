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
# Node feature unique values (per attribute)
# ----------------------------
all_features = torch.cat(x_n_list, dim=0)  # shape [total_nodes, num_features]

print("\n📈 Unique Values of Each Node Feature (per attribute):")
for i in range(all_features.shape[1]):
    values = all_features[:, i]
    unique_vals = torch.unique(values)
    print(f"\nFeature {i + 1}: {sorted(unique_vals.tolist())}")
