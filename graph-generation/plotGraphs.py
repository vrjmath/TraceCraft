import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import torch

def load_generated_graphs(file_path):
    with open(file_path, 'rb') as f:
        graphs = pickle.load(f)
    return graphs

def load_original_traces_graphs(pth_file_path):
    data = torch.load(pth_file_path, map_location='cpu')
    x_n_list = data['x_n_list']
    src_list = data['src_list']
    dst_list = data['dst_list']

    graphs = []
    for x_n, src, dst in zip(x_n_list, src_list, dst_list):
        G = nx.Graph()
        G.add_nodes_from(range(x_n.shape[0]))
        edge_list = list(zip(src.tolist(), dst.tolist()))
        G.add_edges_from(edge_list)
        for i in range(x_n.shape[0]):
            G.nodes[i]['feat'] = x_n[i].numpy()
        graphs.append(G)
    return graphs

def draw_graphs(graphs, save_dir, prefix, num_graphs=5):
    os.makedirs(save_dir, exist_ok=True)
    for i, g in enumerate(graphs[:num_graphs]):
        plt.figure(figsize=(6,6))
        nx.draw_networkx(g, node_size=50, with_labels=False)
        plt.title(f'{prefix} {i} - Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}')
        plt.axis('off')
        plt.tight_layout()
        filename = os.path.join(save_dir, f'{prefix}_{i}.png')
        plt.savefig(filename)
        plt.close()
    print(f"Saved {num_graphs} {prefix} images to '{save_dir}'")

if __name__ == '__main__':
    save_folder = 'graph_images_combined'

    generated_file = "graphs/GraphRNN_RNN_traces_4_128_pred_3000_1.dat"  # example generated graphs (predicted)
    original_pth = "../data_files/traces_processed/train.pth"  # original graphs (train set)

    # Load graphs
    generated_graphs = load_generated_graphs(generated_file)
    original_graphs = load_original_traces_graphs(original_pth)

    # Draw and save images
    draw_graphs(generated_graphs, save_folder, prefix='generated')
    draw_graphs(original_graphs, save_folder, prefix='original')
