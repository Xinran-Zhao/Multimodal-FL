"""
Analysis and Visualization for Multimodal FL Experiments.
Generates convergence plots, comparison bar charts, and per-disease breakdowns.
"""

import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENTS = {
    "E1":  {"name": "Uni-IID",      "partition": "iid",    "mm_clients": []},
    "E2":  {"name": "Uni-nonIID",   "partition": "noniid", "mm_clients": []},
    "E3":  {"name": "1MM-IID",      "partition": "iid",    "mm_clients": [0]},
    "E4":  {"name": "1MM-nonIID",   "partition": "noniid", "mm_clients": [0]},
    "E5":  {"name": "2MM-IID",      "partition": "iid",    "mm_clients": [0, 1]},
    "E6":  {"name": "2MM-nonIID",   "partition": "noniid", "mm_clients": [0, 1]},
    "E7":  {"name": "3MM-IID",      "partition": "iid",    "mm_clients": [0, 1, 2]},
    "E8":  {"name": "3MM-nonIID",   "partition": "noniid", "mm_clients": [0, 1, 2]},
    "E9":  {"name": "4MM-IID",      "partition": "iid",    "mm_clients": [0, 1, 2, 3]},
    "E10": {"name": "4MM-nonIID",   "partition": "noniid", "mm_clients": [0, 1, 2, 3]},
}


def load_results(results_dir):
    """Load all experiment results."""
    results = {}
    for exp_id, exp_cfg in EXPERIMENTS.items():
        exp_name = f"{exp_id}_{exp_cfg['name']}"
        history_path = os.path.join(results_dir, exp_name, "history.json")
        config_path = os.path.join(results_dir, exp_name, "config.json")

        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
            config = {}
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
            results[exp_id] = {"history": history, "config": config}

    return results


def plot_convergence(results, output_dir):
    """Plot AUROC convergence curves for all experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Colors for different number of MM clients
    colors = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868", 3: "#C44E52", 4: "#8172B2"}
    linestyles = {"iid": "-", "noniid": "--"}

    for ax, partition_type, title in [
        (axes[0], "iid", "IID Partition"),
        (axes[1], "noniid", "Non-IID Partition (Dirichlet α=0.1)")
    ]:
        for exp_id, exp_cfg in EXPERIMENTS.items():
            if exp_cfg["partition"] != partition_type:
                continue
            if exp_id not in results:
                continue

            history = results[exp_id]["history"]
            num_mm = len(exp_cfg["mm_clients"])
            aurocs = history.get("test_auroc", [])

            if aurocs:
                ax.plot(
                    range(1, len(aurocs) + 1),
                    aurocs,
                    label=f"{exp_cfg['name']} ({num_mm}MM)",
                    color=colors[num_mm],
                    linewidth=2,
                )

        ax.set_xlabel("Communication Round", fontsize=12)
        ax.set_ylabel("Test AUROC (macro)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 0.9)

    plt.tight_layout()
    path = os.path.join(output_dir, "convergence_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_final_comparison(results, output_dir):
    """Bar chart comparing final test AUROC across all experiments."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Group by IID vs non-IID
    iid_exps = [eid for eid, cfg in EXPERIMENTS.items() if cfg["partition"] == "iid"]
    noniid_exps = [eid for eid, cfg in EXPERIMENTS.items() if cfg["partition"] == "noniid"]

    x = np.arange(5)  # 5 modality configs: 0MM, 1MM, 2MM, 3MM, 4MM
    width = 0.35

    iid_aurocs = []
    noniid_aurocs = []
    labels_x = ["0 MM\n(All Unimodal)", "1 MM", "2 MM", "3 MM", "4 MM\n(All Multimodal)"]

    for exp_id in iid_exps:
        if exp_id in results:
            iid_aurocs.append(results[exp_id]["history"].get("final_test_auroc", 0))
        else:
            iid_aurocs.append(0)

    for exp_id in noniid_exps:
        if exp_id in results:
            noniid_aurocs.append(results[exp_id]["history"].get("final_test_auroc", 0))
        else:
            noniid_aurocs.append(0)

    bars1 = ax.bar(x - width/2, iid_aurocs, width, label="IID", color="#4C72B0",
                   edgecolor="white")
    bars2 = ax.bar(x + width/2, noniid_aurocs, width, label="Non-IID", color="#DD5544",
                   edgecolor="white")

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                        f"{h:.3f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Number of Multimodal Clients (out of 4)", fontsize=12)
    ax.set_ylabel("Test AUROC (macro)", fontsize=12)
    ax.set_title("Final Test AUROC: Unimodal vs Partial Multimodal FL",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.4, 0.9)

    plt.tight_layout()
    path = os.path.join(output_dir, "final_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_per_disease(results, output_dir, label_names=None):
    """Per-disease AUROC comparison between key experiments."""
    if label_names is None:
        # Try to load from first available result
        for exp_id, data in results.items():
            if "label_names" in data.get("config", {}):
                label_names = data["config"]["label_names"]
                break
        if label_names is None:
            label_names = [f"Label_{i}" for i in range(8)]

    # Compare E1 vs E6 (Uni-IID vs 2MM-nonIID) and E2 vs E6
    key_comparisons = [
        ("E1", "E2",  "Unimodal: IID vs Non-IID"),
        ("E1", "E5",  "IID: 0 MM vs 2 MM"),
        ("E2", "E6",  "Non-IID: 0 MM vs 2 MM"),
        ("E2", "E10", "Non-IID: 0 MM vs 4 MM"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    for idx, (exp_a, exp_b, title) in enumerate(key_comparisons):
        ax = axes[idx // 2, idx % 2]

        if exp_a not in results or exp_b not in results:
            ax.set_title(f"{title}\n(Data not available)")
            continue

        hist_a = results[exp_a]["history"]
        hist_b = results[exp_b]["history"]

        # Get per-label AUROC from final evaluation
        per_label_a = hist_a.get("final_per_label_test_auroc", {})
        per_label_b = hist_b.get("final_per_label_test_auroc", {})

        x = np.arange(len(label_names))
        width = 0.35

        aurocs_a = [per_label_a.get(str(i), 0) for i in range(len(label_names))]
        aurocs_b = [per_label_b.get(str(i), 0) for i in range(len(label_names))]

        name_a = EXPERIMENTS[exp_a]["name"]
        name_b = EXPERIMENTS[exp_b]["name"]

        ax.bar(x - width/2, aurocs_a, width, label=f"{exp_a}: {name_a}", color="#4C72B0")
        ax.bar(x + width/2, aurocs_b, width, label=f"{exp_b}: {name_b}", color="#DD5544")

        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("_", "\n") for n in label_names],
                           fontsize=8, rotation=0)
        ax.set_ylabel("AUROC")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0.3, 1.0)

    plt.suptitle("Per-Disease AUROC Comparisons", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "per_disease_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def generate_summary_table(results, output_dir):
    """Generate a text summary table."""
    lines = []
    lines.append("=" * 80)
    lines.append("MULTIMODAL FEDERATED LEARNING - RESULTS SUMMARY")
    lines.append("=" * 80)
    lines.append(
        f"{'Exp':<8}{'Name':<18}{'Partition':<10}{'#MM':<6}"
        f"{'Val AUROC':<14}{'Test AUROC':<14}{'Best Round'}"
    )
    lines.append("-" * 80)

    for exp_id in ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10"]:
        if exp_id not in results:
            continue
        cfg = EXPERIMENTS[exp_id]
        hist = results[exp_id]["history"]
        lines.append(
            f"{exp_id:<8}{cfg['name']:<18}{cfg['partition']:<10}"
            f"{len(cfg['mm_clients']):<6}"
            f"{hist.get('best_val_auroc', 0):<14.4f}"
            f"{hist.get('final_test_auroc', 0):<14.4f}"
            f"{hist.get('best_round', 0)}"
        )

    lines.append("=" * 80)

    table_str = "\n".join(lines)
    print(table_str)

    path = os.path.join(output_dir, "results_summary.txt")
    with open(path, "w") as f:
        f.write(table_str)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze FL experiment results")
    parser.add_argument(
        "--results_dir", type=str,
        default="/data/amciilab/xinran/indiana_cxr/results",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/data/amciilab/xinran/indiana_cxr/results/figures",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        return

    print(f"Found results for: {list(results.keys())}")

    generate_summary_table(results, args.output_dir)
    plot_convergence(results, args.output_dir)
    plot_final_comparison(results, args.output_dir)
    plot_per_disease(results, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
