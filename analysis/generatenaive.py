import torch
import random

real_path = "/usr/scratch/vshitole6/TraceCraft/analysis/dataset/real.pth"
save_path = "/usr/scratch/vshitole6/TraceCraft/analysis/synthetic_generated_with_attrs.pth"

real = torch.load(real_path)
x_n_list = real['x_n_list']
src_list = real['src_list']
dst_list = real['dst_list']

real_stats = []
for x, src, dst in zip(x_n_list, src_list, dst_list):
    num_nodes = x.shape[0]
    num_edges = len(src)
    max_edges = num_nodes * (num_nodes - 1) // 2
    if num_edges <= max_edges:
        real_stats.append((num_nodes, num_edges))

valid_attributes = []
for x_n in x_n_list:
    mask = (
        ((x_n[:, 0] == 0) & (x_n[:, 1] == 3) & (x_n[:, 2] != 23) & (x_n[:, 3] != 95) & (x_n[:, 4] == 14) & (x_n[:, 5] == 3)) |
        ((x_n[:, 0] != 0) & (x_n[:, 1] != 3) & (x_n[:, 2] == 23) & (x_n[:, 3] == 95) & (x_n[:, 4] != 14) & (x_n[:, 5] != 3))
    )
    valid_attributes.append(x_n[mask])

valid_attributes = torch.cat(valid_attributes, dim=0)
num_valid = valid_attributes.size(0)

def generate_dag(n, m):
    possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    selected_edges = random.sample(possible_edges, m)
    src = torch.tensor([u for u, v in selected_edges], dtype=torch.long)
    dst = torch.tensor([v for u, v in selected_edges], dtype=torch.long)
    return src, dst

def assign_attributes(num_nodes):
    sampled_idx = torch.randint(0, num_valid, (num_nodes,))
    sampled_attrs = valid_attributes[sampled_idx]
    return sampled_attrs

synthetic_src_list = []
synthetic_dst_list = []
synthetic_x_n_list = []

for i in range(400):
    num_nodes, num_edges = random.choice(real_stats)
    src, dst = generate_dag(num_nodes, num_edges)
    attrs = assign_attributes(num_nodes)
    
    synthetic_src_list.append(src)
    synthetic_dst_list.append(dst)
    synthetic_x_n_list.append(attrs)

synthetic_data = {
    'src_list': synthetic_src_list,
    'dst_list': synthetic_dst_list,
    'x_n_list': synthetic_x_n_list,
}

torch.save(synthetic_data, save_path)
print(f"Saved 400 synthetic DAGs with attributes to:\n{save_path}")
