import torch

real = torch.load("/usr/scratch/vshitole6/TraceCraft/analysis/real.pth")
generated_naive = torch.load("/usr/scratch/vshitole6/TraceCraft/analysis/generated.pth")
generated_proteus = torch.load("/usr/scratch/vshitole6/TraceCraft/analysis/proteus_generated.pth")

valid_attributes = []

for x_n in real['x_n_list']:
    mask = (
        ((x_n[:, 0] == 0) & (x_n[:, 1] == 3) & (x_n[:, 2] != 23) & (x_n[:, 3] != 95) &
         (x_n[:, 4] == 14) & (x_n[:, 5] == 3)) |
        ((x_n[:, 0] != 0) & (x_n[:, 1] != 3) & (x_n[:, 2] == 23) & (x_n[:, 3] == 95) &
         (x_n[:, 4] != 14) & (x_n[:, 5] != 3))
    )
    valid_attributes.append(x_n[mask])

valid_attributes = torch.cat(valid_attributes, dim=0)
num_valid = valid_attributes.size(0)

def assign_attributes(generated):
    new_x_n_list = []
    for graph in generated['src_list']:
        num_nodes = graph.max().item() + 1
        sampled_idx = torch.randint(0, num_valid, (num_nodes,))
        sampled_attrs = valid_attributes[sampled_idx]
        new_x_n_list.append(sampled_attrs)
    return new_x_n_list

generated_naive['x_n_list'] = assign_attributes(generated_naive)
generated_proteus['x_n_list'] = assign_attributes(generated_proteus)

torch.save(generated_naive, "/usr/scratch/vshitole6/TraceCraft/analysis/generated_with_attrs.pth")
torch.save(generated_proteus, "/usr/scratch/vshitole6/TraceCraft/analysis/proteus_generated_with_attrs.pth")

