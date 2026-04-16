"""
Visualize data partitions for all 10 FL experiments.
Generates grouped bar charts showing per-client label distribution
and modality assignment for each experiment.
"""

import os
import json
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {"id": "E1",  "name": "Uni-IID",       "partition": "iid",    "mm_clients": []},
    {"id": "E2",  "name": "Uni-nonIID",     "partition": "noniid", "mm_clients": []},
    {"id": "E3",  "name": "1MM-IID",        "partition": "iid",    "mm_clients": [0]},
    {"id": "E4",  "name": "1MM-nonIID",     "partition": "noniid", "mm_clients": [0]},
    {"id": "E5",  "name": "2MM-IID",        "partition": "iid",    "mm_clients": [0, 1]},
    {"id": "E6",  "name": "2MM-nonIID",     "partition": "noniid", "mm_clients": [0, 1]},
    {"id": "E7",  "name": "3MM-IID",        "partition": "iid",    "mm_clients": [0, 1, 2]},
    {"id": "E8",  "name": "3MM-nonIID",     "partition": "noniid", "mm_clients": [0, 1, 2]},
    {"id": "E9",  "name": "4MM-IID",        "partition": "iid",    "mm_clients": [0, 1, 2, 3]},
    {"id": "E10", "name": "4MM-nonIID",     "partition": "noniid", "mm_clients": [0, 1, 2, 3]},
]


def load_data(prepared_csv, partition_dir):
    """Load prepared data and both partition files."""
    # Load training samples
    samples = []
    with open(prepared_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "train":
                samples.append(row)

    # Load partitions
    with open(os.path.join(partition_dir, "partition_iid.json"), "r") as f:
        iid_partition = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(partition_dir, "partition_noniid.json"), "r") as f:
        noniid_partition = {int(k): v for k, v in json.load(f).items()}

    # Load metadata
    with open(os.path.join(partition_dir, "partition_meta.json"), "r") as f:
        meta = json.load(f)

    return samples, iid_partition, noniid_partition, meta


def compute_client_label_counts(partition, samples, label_names):
    """Compute label counts per client. Returns (num_clients, num_labels) array."""
    num_clients = len(partition)
    num_labels = len(label_names)
    counts = np.zeros((num_clients, num_labels), dtype=int)

    for c in range(num_clients):
        for idx in partition[c]:
            for j, label in enumerate(label_names):
                counts[c, j] += int(samples[idx][label])

    return counts


def plot_single_experiment(ax, exp, counts, label_names, num_clients):
    """Plot grouped bar chart for one experiment on a given axes."""
    num_labels = len(label_names)
    x = np.arange(num_labels)
    width = 0.18  # bar width

    # Color scheme: image-only clients in blue shades, multimodal in orange/red shades
    img_colors = ["#4C72B0", "#5B8BD6", "#7BAAF2", "#A0C4FF"]
    mm_colors = ["#DD5544", "#E87764", "#F09988", "#F5BBB0"]

    mm_clients = exp["mm_clients"]

    for c in range(num_clients):
        offset = (c - num_clients / 2 + 0.5) * width
        is_mm = c in mm_clients
        color = mm_colors[c] if is_mm else img_colors[c]
        modality = "img+txt" if is_mm else "img"
        label = f"Client {c} ({modality})"

        ax.bar(x + offset, counts[c], width, label=label, color=color,
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("_", "\n") for name in label_names],
                       fontsize=7, rotation=0)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title(f"{exp['id']}: {exp['name']}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")

    # Add total sample count per client as text
    for c in range(num_clients):
        total = sum(counts[c])
        is_mm = c in mm_clients
        modality = "MM" if is_mm else "Img"


def plot_all_experiments(samples, iid_partition, noniid_partition, label_names,
                         output_dir, num_clients=4):
    """Generate a comprehensive figure with all 10 experiments."""

    # --- Figure 1: All 10 experiments in a 5x2 grid ---
    fig, axes = plt.subplots(5, 2, figsize=(20, 28))
    fig.suptitle("Data Distribution Across Clients for All 10 FL Experiments\n"
                 "(Blue = Image-Only, Red = Image+Text Multimodal)",
                 fontsize=16, fontweight="bold", y=0.98)

    for i, exp in enumerate(EXPERIMENTS):
        row, col = i // 2, i % 2
        ax = axes[row, col]

        partition = iid_partition if exp["partition"] == "iid" else noniid_partition
        counts = compute_client_label_counts(partition, samples, label_names)
        plot_single_experiment(ax, exp, counts, label_names, num_clients)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path1 = os.path.join(output_dir, "all_experiments_distribution.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path1}")

    # --- Figure 2: IID vs non-IID side-by-side comparison ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(20, 12))
    fig2.suptitle("IID vs Non-IID Partition Comparison\n"
                  "(Same data, different allocation strategies)",
                  fontsize=14, fontweight="bold")

    # Top row: sample count per client
    for col, (partition, title) in enumerate([
        (iid_partition, "IID Partition"),
        (noniid_partition, "Non-IID (Dirichlet α=0.1)")
    ]):
        ax = axes2[0, col]
        client_sizes = [len(partition[c]) for c in range(num_clients)]
        colors = ["#4C72B0", "#5B8BD6", "#7BAAF2", "#A0C4FF"]
        bars = ax.bar(range(num_clients), client_sizes, color=colors, edgecolor="white")
        ax.set_xlabel("Client ID")
        ax.set_ylabel("Number of Samples")
        ax.set_title(f"{title} — Sample Count per Client", fontweight="bold")
        ax.set_xticks(range(num_clients))
        for bar, size in zip(bars, client_sizes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(size), ha='center', fontsize=10, fontweight='bold')

    # Bottom row: label proportion heatmaps
    for col, (partition, title) in enumerate([
        (iid_partition, "IID — Label Proportion per Client"),
        (noniid_partition, "Non-IID — Label Proportion per Client")
    ]):
        ax = axes2[1, col]
        counts = compute_client_label_counts(partition, samples, label_names)

        # Normalize to proportions (per label across clients)
        col_sums = counts.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1  # avoid division by zero
        proportions = counts / col_sums

        im = ax.imshow(proportions, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(label_names)))
        ax.set_xticklabels([name.replace("_", "\n") for name in label_names],
                           fontsize=8)
        ax.set_yticks(range(num_clients))
        ax.set_yticklabels([f"Client {c}" for c in range(num_clients)])
        ax.set_title(title, fontweight="bold", fontsize=10)

        # Add text annotations
        for i in range(num_clients):
            for j in range(len(label_names)):
                text = f"{proportions[i, j]:.2f}\n({counts[i, j]})"
                color = "white" if proportions[i, j] > 0.6 else "black"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=7, color=color)

        plt.colorbar(im, ax=ax, shrink=0.8, label="Proportion")

    plt.tight_layout()
    path2 = os.path.join(output_dir, "iid_vs_noniid_comparison.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {path2}")

    # --- Figure 3: Per-experiment detail with sample counts as table ---
    # Text summary table
    summary_path = os.path.join(output_dir, "partition_summary.txt")
    with open(summary_path, "w") as f:
        for exp in EXPERIMENTS:
            partition = iid_partition if exp["partition"] == "iid" else noniid_partition
            counts = compute_client_label_counts(partition, samples, label_names)
            mm_clients = exp["mm_clients"]

            f.write(f"\n{'='*100}\n")
            f.write(f"Experiment {exp['id']}: {exp['name']}\n")
            f.write(f"Partition: {exp['partition'].upper()}, "
                    f"Multimodal clients: {mm_clients if mm_clients else 'None (all unimodal)'}\n")
            f.write(f"{'='*100}\n")

            # Header
            header = f"{'Client':<20}{'Modality':<12}{'Total':<8}"
            for name in label_names:
                header += f"{name:<18}"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            # Per client
            for c in range(num_clients):
                modality = "img+txt" if c in mm_clients else "img"
                total = len(partition[c])
                line = f"Client {c:<13}{modality:<12}{total:<8}"
                for j in range(len(label_names)):
                    line += f"{counts[c, j]:<18}"
                f.write(line + "\n")

            # Total
            f.write("-" * len(header) + "\n")
            total_line = f"{'TOTAL':<20}{'':<12}{sum(len(partition[c]) for c in range(num_clients)):<8}"
            for j in range(len(label_names)):
                total_line += f"{counts[:, j].sum():<18}"
            f.write(total_line + "\n")

    print(f"Saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize FL partition distributions")
    parser.add_argument(
        "--prepared_csv",
        type=str,
        default="/data/amciilab/xinran/indiana_cxr/prepared/prepared_data.csv",
    )
    parser.add_argument(
        "--partition_dir",
        type=str,
        default="/data/amciilab/xinran/indiana_cxr/prepared/partitions",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/amciilab/xinran/indiana_cxr/prepared/partitions/figures",
    )
    args = parser.parse_args()

    from data_prep import LABEL_NAMES

    os.makedirs(args.output_dir, exist_ok=True)

    samples, iid_partition, noniid_partition, meta = load_data(
        args.prepared_csv, args.partition_dir
    )
    print(f"Loaded {len(samples)} training samples")
    print(f"Labels: {LABEL_NAMES}")

    plot_all_experiments(
        samples, iid_partition, noniid_partition, LABEL_NAMES,
        args.output_dir, num_clients=meta["num_clients"],
    )

    print(f"\nAll visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
