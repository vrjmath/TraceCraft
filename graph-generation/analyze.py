import pickle
import numpy as np

def analyze_graphs(graphs):
    node_counts = [g.number_of_nodes() for g in graphs]
    edge_counts = [g.number_of_edges() for g in graphs]

    stats = {}
    for name, counts in [('nodes', node_counts), ('edges', edge_counts)]:
        stats[name] = {
            'median': np.median(counts),
            '25%': np.percentile(counts, 25),
            '75%': np.percentile(counts, 75),
            'min': np.min(counts),
            'max': np.max(counts),
        }
    return stats

if __name__ == '__main__':
    filename = "graphs/GraphRNN_RNN_traces_4_128_pred_3000_1.dat"

    with open(filename, 'rb') as f:
        graphs = pickle.load(f)

    stats = analyze_graphs(graphs)
    print(f"Statistics for graphs in {filename}:")
    print("Node counts:", stats['nodes'])
    print("Edge counts:", stats['edges'])
