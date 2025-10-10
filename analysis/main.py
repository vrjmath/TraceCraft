import torch
from util import compute_graph_metrics, compute_percentiles, compare_distributions

def generate_report(real, generated, output_file):
    real_src, real_dst = real['src_list'], real['dst_list']
    generated_src, generated_dst = generated['src_list'], generated['dst_list']
    
    real_metrics = compute_graph_metrics(real_src, real_dst)
    generated_metrics = compute_graph_metrics(generated_src, generated_dst)

    report_data = {
        'Real Data': {
            '# Graphs': real_metrics["num_graphs"],
            '# DAGs': real_metrics["num_dags"],
            '# Weakly Connected': real_metrics["num_weakly_connected"],
            'Nodes': compute_percentiles(real_metrics["num_nodes"]),
            'Edges': compute_percentiles(real_metrics["num_edges"]),
            '# Layers': compute_percentiles(real_metrics["num_layers"]),
            'Avg In-degree': compute_percentiles(real_metrics["avg_in_deg"]),
            'Avg Out-degree': compute_percentiles(real_metrics["avg_out_deg"])
        },
        'Generated Data': {
            '# Graphs': generated_metrics["num_graphs"],
            '# DAGs': generated_metrics["num_dags"],
            '# Weakly Connected': generated_metrics["num_weakly_connected"],
            'Nodes': compute_percentiles(generated_metrics["num_nodes"]),
            'Edges': compute_percentiles(generated_metrics["num_edges"]),
            '# Layers': compute_percentiles(generated_metrics["num_layers"]),
            'Avg In-degree': compute_percentiles(generated_metrics["avg_in_deg"]),
            'Avg Out-degree': compute_percentiles(generated_metrics["avg_out_deg"])
        }
    }

    dist_results = compare_distributions(real_metrics, generated_metrics)

    with open(output_file, 'w') as f:
        f.write("=== Graph Metrics Report ===\n")
        
        f.write("\n=== KL Divergence Results ===\n")
        for metric, value in dist_results["kl_divergences"].items():
            f.write(f"{metric}: {value}\n")
        
        f.write("\n=== MMD Results ===\n")
        for metric, value in dist_results["mmds"].items():
            f.write(f"{metric}: {value}\n")
        
        f.write("\n=== Wasserstein Distance Results ===\n")
        for metric, value in dist_results["wassersteins"].items():
            f.write(f"{metric}: {value}\n")
        
        for dataset, metrics in report_data.items():
            f.write(f"\n=== {dataset} ===\n")
            for metric, percentiles in metrics.items():
                f.write(f"{metric}: {percentiles}\n")
                    
    print(f"Report has been saved to {output_file}")


real = torch.load("/usr/scratch/vshitole6/TraceCraft/analysis/real.pth")
generated = torch.load("/usr/scratch/vshitole6/TraceCraft/analysis/generated.pth")
generate_report(real, generated, "graph_report.txt")
