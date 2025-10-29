import torch
from util import compute_graph_metrics, compute_percentiles, compare_distributions
from pathlib import Path

def generate_report(real, generated, output_file):
    real_metrics = compute_graph_metrics(real['src_list'], real['dst_list'], real['x_n_list'])
    generated_metrics = compute_graph_metrics(generated['src_list'], generated['dst_list'], generated['x_n_list'])

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
        f.write("=== Dataset Keys: ")
        f.write(", ".join(generated.keys()) + "\n")
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


base_dir = Path(__file__).resolve().parent

dataset_dir = base_dir / "dataset"
report_dir = base_dir / "report"

real = torch.load(dataset_dir / "real.pth")
generated_layerdag = torch.load(dataset_dir / "generated_tracecraft.pth")
generated_naive = torch.load(dataset_dir / "generated_naive.pth")
generated_proteus = torch.load(dataset_dir / "generated_proteus.pth")
generated_layerdag_old = torch.load(dataset_dir / "generated_tracecraft_old.pth")
generated_layerdag_old2 = torch.load(dataset_dir / "generated_tracecraft_old2.pth")
#generate_report(real, generated_naive, report_dir / "naive_baseline_report.txt")
#generate_report(real, generated_layerdag, report_dir / "tracecraft_report.txt")
#generate_report(real, generated_proteus, report_dir / "proteus_report.txt")
#generate_report(real, generated_layerdag_old, report_dir / "tracecraft_old_report.txt")
#generate_report(real, generated_layerdag_old2, report_dir / "tracecraft_old_report2.txt")
generate_report(torch.load('/usr/scratch/vshitole6/TraceCraft/mixed/train.pth'), torch.load('/usr/scratch/vshitole6/TraceCraft/mixed/train.pth'), report_dir / "REPORTOFF.txt")
