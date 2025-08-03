import torch
import numpy as np
import random

# Load your real dataset
data = torch.load('random_dag_baseline.pth', map_location='cpu')
x_n_list = data['x_n_list']
src_list = data['src_list']
dst_list = data['dst_list']

# Compute number of nodes and edges per graph in real data
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

print("📊 Real Graph Statistics:")
print("Number of nodes per graph:", describe(num_nodes))
print("Number of edges per graph:", describe(num_edges))

import numpy as np

# Assuming node features are categorical or discrete integers
num_features = x_n_list[0].shape[1]

# Collect all values per feature across all graphs
all_features = [np.concatenate([x[:, i].numpy() for x in x_n_list]) for i in range(num_features)]

for i, feature_values in enumerate(all_features):
    unique_vals = np.unique(feature_values)
    print(f"Node attribute {i}: possible values = {unique_vals}")




