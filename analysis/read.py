import torch
from pathlib import Path

current_dir = Path(__file__).parent

train_path = current_dir.parent / 'data_files' / 'traces_processed' / 'train.pth'
test_path = current_dir.parent / 'data_files' / 'traces_processed' / 'test.pth'
val_path = current_dir.parent / 'data_files' / 'traces_processed' / 'val.pth'

train_dict = torch.load(str(train_path), map_location="cpu")
test_dict = torch.load(str(test_path), map_location="cpu")
val_dict = torch.load(str(val_path), map_location="cpu")

data_dict = {}

for key in train_dict:
    if key in test_dict and key in val_dict:
        data_dict[key] = train_dict[key] + test_dict[key] + val_dict[key]
    else:
        data_dict[key] = train_dict.get(key, []) + test_dict.get(key, []) + val_dict.get(key, [])



print("Keys in dictionary:", list(data_dict.keys()))

num_graphs = len(data_dict['src_list'])
print("Number of graphs:", num_graphs)

total_nodes = 0
total_edges = 0

for i in range(num_graphs):
    src = data_dict['src_list'][i]
    dst = data_dict['dst_list'][i]
    nodes = data_dict['x_n_list'][i].size(0) 
    edges = src.size(0) 

    total_nodes += nodes
    total_edges += edges

avg_nodes = total_nodes / num_graphs
avg_edges = total_edges / num_graphs

print(f"Average number of nodes per graph: {avg_nodes:.2f}")
print(f"Average number of edges per graph: {avg_edges:.2f}")


from pprint import pprint

print("Example from metric_list:")
pprint(data_dict['metrics_list'][0])

