"""
Federated Learning Data Partitioning
IID and non-IID (Dirichlet) partitioning of the Indiana CXR dataset
across N clients for FL experiments.
"""

import os
import csv
import json
import argparse
import random
from collections import defaultdict

import numpy as np


def load_prepared_data(prepared_csv: str, split: str = "train"):
    """Load prepared_data.csv and return list of dicts for the given split."""
    samples = []
    with open(prepared_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == split:
                samples.append(row)
    return samples


def get_label_matrix(samples, label_names):
    """Return (N, K) binary numpy array of labels."""
    N = len(samples)
    K = len(label_names)
    matrix = np.zeros((N, K), dtype=np.int32)
    for i, s in enumerate(samples):
        for j, label in enumerate(label_names):
            matrix[i, j] = int(s[label])
    return matrix


# ---------------------------------------------------------------------------
# IID Partitioning
# ---------------------------------------------------------------------------

def partition_iid(samples, num_clients: int, seed: int = 42):
    """
    Randomly and uniformly split samples across clients.
    Returns: dict {client_id: [sample_indices]}
    """
    rng = np.random.RandomState(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    # Split into num_clients roughly equal chunks
    chunks = np.array_split(indices, num_clients)
    partition = {i: chunk.tolist() for i, chunk in enumerate(chunks)}
    return partition


# ---------------------------------------------------------------------------
# Non-IID Partitioning (Dirichlet)
# ---------------------------------------------------------------------------

def partition_dirichlet(samples, label_names, num_clients: int,
                        alpha: float = 0.1, seed: int = 42):
    """
    Non-IID partitioning using Dirichlet distribution with EQUAL client sizes.

    Each client gets the same number of samples (N // num_clients), but the
    disease label distribution differs across clients. This isolates the
    effect of label heterogeneity from sample size imbalance.

    Strategy:
    1. Assign each sample a "dominant label" (its rarest positive label).
    2. For each label group, sample Dirichlet to get per-client preferences.
    3. Use a scoring mechanism to assign samples to clients while enforcing
       an equal-size budget per client.

    Args:
        samples: list of sample dicts
        label_names: list of label column names
        num_clients: number of FL clients
        alpha: Dirichlet concentration (smaller = more heterogeneous)
        seed: random seed

    Returns: dict {client_id: [sample_indices]}
    """
    rng = np.random.RandomState(seed)
    N = len(samples)
    K = len(label_names)

    # Equal budget per client
    budget = N // num_clients
    remainder = N - budget * num_clients

    # Build label matrix
    label_matrix = get_label_matrix(samples, label_names)

    # Compute label frequencies for rarity ranking
    label_freq = label_matrix.sum(axis=0).astype(float)  # (K,)

    # Assign each sample a "dominant label" = its rarest positive label
    dominant_label = np.full(N, -1, dtype=np.int32)
    for i in range(N):
        positive_labels = np.where(label_matrix[i] == 1)[0]
        if len(positive_labels) > 0:
            rarest_idx = positive_labels[np.argmin(label_freq[positive_labels])]
            dominant_label[i] = rarest_idx
        else:
            dominant_label[i] = rng.randint(0, K)

    # Sample Dirichlet proportions per label group
    # dirichlet_probs[k, c] = preference of client c for label k
    dirichlet_probs = np.zeros((K, num_clients))
    for k in range(K):
        dirichlet_probs[k] = rng.dirichlet([alpha] * num_clients)

    # Score each sample for each client based on its dominant label
    # Then greedily assign samples to their highest-scoring client
    # while respecting the per-client budget
    client_budgets = np.array([budget] * num_clients)
    # Distribute remainder to first few clients
    for i in range(remainder):
        client_budgets[i] += 1

    partition = {i: [] for i in range(num_clients)}

    # Process label groups one at a time, shuffled
    label_order = list(range(K))
    rng.shuffle(label_order)

    # Collect all (sample_idx, client_preference_scores) pairs
    all_assignments = []
    for k in label_order:
        group_indices = np.where(dominant_label == k)[0]
        if len(group_indices) == 0:
            continue
        rng.shuffle(group_indices)
        for idx in group_indices:
            # Score = Dirichlet weight for this sample's dominant label
            scores = dirichlet_probs[k].copy()
            all_assignments.append((int(idx), scores))

    # Shuffle all assignments to avoid ordering bias
    rng.shuffle(all_assignments)

    # Greedy assignment: each sample goes to its highest-scoring client
    # that still has budget remaining
    assigned = set()
    for idx, scores in all_assignments:
        # Sort clients by descending score
        client_order = np.argsort(-scores)
        for c in client_order:
            if client_budgets[c] > 0:
                partition[c].append(idx)
                client_budgets[c] -= 1
                assigned.add(idx)
                break

    # Safety: assign any remaining unassigned samples
    unassigned = [i for i in range(N) if i not in assigned]
    for idx in unassigned:
        for c in range(num_clients):
            if client_budgets[c] > 0:
                partition[c].append(idx)
                client_budgets[c] -= 1
                break

    # Shuffle each client's indices
    for c in partition:
        rng.shuffle(partition[c])

    return partition


# ---------------------------------------------------------------------------
# Save / Load partitions
# ---------------------------------------------------------------------------

def save_partition(partition, output_path):
    """Save partition as JSON."""
    # Convert numpy types to Python ints
    clean = {int(k): [int(i) for i in v] for k, v in partition.items()}
    with open(output_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"Partition saved to {output_path}")


def load_partition(partition_path):
    """Load partition from JSON."""
    with open(partition_path, "r") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


# ---------------------------------------------------------------------------
# Partition statistics
# ---------------------------------------------------------------------------

def print_partition_stats(partition, samples, label_names):
    """Print label distribution per client."""
    label_matrix = get_label_matrix(samples, label_names)

    print(f"\n{'Client':<10}", end="")
    print(f"{'Total':<8}", end="")
    for name in label_names:
        short = name[:12]
        print(f"{short:<15}", end="")
    print()
    print("-" * (10 + 8 + 15 * len(label_names)))

    for client_id in sorted(partition.keys()):
        indices = partition[client_id]
        client_labels = label_matrix[indices]
        counts = client_labels.sum(axis=0)

        print(f"Client {client_id:<3}", end="")
        print(f"{len(indices):<8}", end="")
        for count in counts:
            print(f"{count:<15}", end="")
        print()

    # Total row
    print("-" * (10 + 8 + 15 * len(label_names)))
    total = label_matrix.sum(axis=0)
    print(f"{'Total':<10}{len(samples):<8}", end="")
    for t in total:
        print(f"{t:<15}", end="")
    print()


# ---------------------------------------------------------------------------
# Generate all partitions for the 10 experiments
# ---------------------------------------------------------------------------

def generate_all_partitions(prepared_csv: str, output_dir: str,
                            num_clients: int = 4, alpha: float = 0.1,
                            seed: int = 42):
    """
    Generate IID and non-IID partitions and save them.

    Experiments:
      E1/E3/E5/E7/E9  use IID partition  (same partition, different modality configs)
      E2/E4/E6/E8/E10  use non-IID partition (same partition, different modality configs)
    """
    from data_prep import LABEL_NAMES

    samples = load_prepared_data(prepared_csv, split="train")
    print(f"Loaded {len(samples)} training samples")

    os.makedirs(output_dir, exist_ok=True)

    # IID partition
    iid_partition = partition_iid(samples, num_clients, seed)
    save_partition(iid_partition, os.path.join(output_dir, "partition_iid.json"))
    print("\n=== IID Partition ===")
    print_partition_stats(iid_partition, samples, LABEL_NAMES)

    # Non-IID (Dirichlet) partition
    noniid_partition = partition_dirichlet(
        samples, LABEL_NAMES, num_clients, alpha, seed
    )
    save_partition(noniid_partition, os.path.join(output_dir, "partition_noniid.json"))
    print(f"\n=== Non-IID Partition (Dirichlet α={alpha}) ===")
    print_partition_stats(noniid_partition, samples, LABEL_NAMES)

    # Save metadata
    meta = {
        "num_clients": num_clients,
        "dirichlet_alpha": alpha,
        "seed": seed,
        "num_train_samples": len(samples),
        "label_names": LABEL_NAMES,
    }
    with open(os.path.join(output_dir, "partition_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return iid_partition, noniid_partition, samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FL data partitions")
    parser.add_argument(
        "--prepared_csv",
        type=str,
        default="/data/amciilab/xinran/indiana_cxr/prepared/prepared_data.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/amciilab/xinran/indiana_cxr/prepared/partitions",
    )
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_all_partitions(
        args.prepared_csv, args.output_dir,
        args.num_clients, args.alpha, args.seed,
    )
