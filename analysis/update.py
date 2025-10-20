import torch

real_path = "/usr/scratch/vshitole6/TraceCraft/analysis/dataset/real.pth"
real = torch.load(real_path)

metrics_list = real["metrics_list"]

generated_paths = [
    "/usr/scratch/vshitole6/TraceCraft/analysis/dataset/generated_tracecraft.pth",
    "/usr/scratch/vshitole6/TraceCraft/analysis/dataset/generated_naive.pth",
    "/usr/scratch/vshitole6/TraceCraft/analysis/dataset/generated_proteus.pth"
]

for path in generated_paths:
    data = torch.load(path)
    data["metrics_list"] = metrics_list
    torch.save(data, path)
    print(f"Updated and saved: {path}")
