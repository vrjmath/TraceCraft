import pickle
import networkx as nx
import numpy as np
import torch

def induce_orientation(G):
    G_prime = nx.DiGraph()
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component).copy()
        nodes = list(subgraph.nodes)
        if len(nodes) <= 1:
            continue 

        root = nodes[0]
        ord_ = {node: None for node in subgraph.nodes}
        bfs_order = list(nx.bfs_edges(subgraph, root))
        for i, (u, v) in enumerate(bfs_order):
            ord_[u] = i
            ord_[v] = i

        E_prime = []
        for u, v in subgraph.edges:
            if ord_[u] is not None and ord_[v] is not None:
                if ord_[u] < ord_[v]:
                    E_prime.append((u, v))
                else:
                    E_prime.append((v, u))

        G_prime.add_edges_from(E_prime)
    
    return G_prime


def load_graphs_from_dat(file_path):
    with open(file_path, 'rb') as f:
        graph_list = pickle.load(f)
    return graph_list

graph_file = '/usr/scratch/vshitole6/TraceCraft/graph-generation/graphs/GraphRNN_RNN_traces_4_128_pred_100_1.dat'
graphs = load_graphs_from_dat(graph_file)

x_n_list = []
src_list = []
dst_list = []

for g in graphs[:400]: 
    directed_g = induce_orientation(g)
    
    src_nodes = torch.tensor([u for u, v in directed_g.edges()])
    dst_nodes = torch.tensor([v for u, v in directed_g.edges()])
    
    src_list.append(src_nodes)
    dst_list.append(dst_nodes)
    
    n_nodes = directed_g.number_of_nodes()
    node_features = torch.zeros(n_nodes, 1) 
    x_n_list.append(node_features)

data = {
    'x_n_list': x_n_list,
    'src_list': src_list,
    'dst_list': dst_list
}

output_file = 'proteus_generated.pth'
torch.save(data, output_file)

print(f"Data saved to {output_file}")
