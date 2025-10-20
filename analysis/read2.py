import torch

G = torch.load("/usr/scratch/vshitole6/TraceCraft/traces_old2/train.pth")

print(len(G['src_list']))
"""
G_expanded = {}
for key, value in G.items():
    G_expanded[key] = value * 4 

print({k: len(v) for k, v in G_expanded.items()})

torch.save(G_expanded, "/usr/scratch/vshitole6/TraceCraft/analysis/generated_tracecraft_old.pth")
"""