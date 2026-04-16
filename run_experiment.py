"""
Experiment Runner for Multimodal Federated Learning
Orchestrates all 10 experiments (E1-E10) comparing unimodal vs partial
multimodal FL under IID and non-IID settings.
"""

import os
import json
import argparse
import logging
import time

import torch

from data_prep import LABEL_NAMES
from dataset import IndianaDataset
from partition import load_partition
from federated import FLClient, FedAvgServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

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


def run_single_experiment(
    exp_id: str,
    prepared_csv: str,
    images_dir: str,
    partition_dir: str,
    output_dir: str,
    num_rounds: int = 50,
    local_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    embed_dim: int = 256,
    patience: int = 10,
    device: str = "cuda",
    seed: int = 42,
):
    """Run a single experiment and save results."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    exp_config = EXPERIMENTS[exp_id]
    exp_name = f"{exp_id}_{exp_config['name']}"
    checkpoint_dir = os.path.join(output_dir, exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"Experiment {exp_name}")
    logger.info(f"  Partition: {exp_config['partition']}")
    logger.info(f"  Multimodal clients: {exp_config['mm_clients']}")
    logger.info(f"{'='*60}")

    # Load partition
    partition_file = os.path.join(
        partition_dir, f"partition_{exp_config['partition']}.json"
    )
    partition = load_partition(partition_file)
    num_clients = len(partition)

    # Create client datasets and FL clients
    clients = []
    for c_id in range(num_clients):
        is_mm = c_id in exp_config["mm_clients"]
        modality = "multimodal" if is_mm else "image"

        client_dataset = IndianaDataset(
            prepared_csv=prepared_csv,
            images_dir=images_dir,
            indices=partition[c_id],
            modality=modality,
            max_text_len=128,
            image_size=224,
            augment=True,  # training augmentation for clients
            label_names=LABEL_NAMES,
        )

        client = FLClient(
            client_id=c_id,
            dataset=client_dataset,
            modality=modality,
            num_labels=len(LABEL_NAMES),
            embed_dim=embed_dim,
            lr=lr,
            batch_size=batch_size,
            local_epochs=local_epochs,
            device=device,
        )
        clients.append(client)

        logger.info(
            f"  Client {c_id}: {modality}, {len(client_dataset)} samples"
        )

    # Validation and test datasets (always image-only for fair comparison)
    val_dataset = IndianaDataset(
        prepared_csv=prepared_csv,
        images_dir=images_dir,
        split="val",
        modality="image",
        image_size=224,
        augment=False,
        label_names=LABEL_NAMES,
    )
    test_dataset = IndianaDataset(
        prepared_csv=prepared_csv,
        images_dir=images_dir,
        split="test",
        modality="image",
        image_size=224,
        augment=False,
        label_names=LABEL_NAMES,
    )

    logger.info(f"  Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    # Create server and run
    server = FedAvgServer(
        clients=clients,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        num_labels=len(LABEL_NAMES),
        embed_dim=embed_dim,
        device=device,
        batch_size=64,
    )

    start_time = time.time()
    history = server.run(
        num_rounds=num_rounds,
        patience=patience,
        checkpoint_dir=checkpoint_dir,
        experiment_name=exp_name,
    )
    elapsed = time.time() - start_time

    # Save experiment config alongside results
    config = {
        "exp_id": exp_id,
        "exp_name": exp_name,
        "partition": exp_config["partition"],
        "mm_clients": exp_config["mm_clients"],
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "embed_dim": embed_dim,
        "patience": patience,
        "seed": seed,
        "total_time_seconds": elapsed,
        "best_val_auroc": history.get("best_val_auroc", 0),
        "best_round": history.get("best_round", 0),
        "final_test_auroc": history.get("final_test_auroc", 0),
        "label_names": LABEL_NAMES,
    }
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Experiment {exp_name} done in {elapsed:.1f}s")
    logger.info(
        f"  Best Val AUROC={history.get('best_val_auroc', 0):.4f} "
        f"(round {history.get('best_round', 0)})"
    )
    logger.info(f"  Final Test AUROC={history.get('final_test_auroc', 0):.4f}")

    return history


def main():
    parser = argparse.ArgumentParser(
        description="Run multimodal FL experiments"
    )
    parser.add_argument(
        "--experiments", type=str, nargs="+",
        default=list(EXPERIMENTS.keys()),
        help="Which experiments to run (e.g., E1 E2 E3). Default: all",
    )
    parser.add_argument(
        "--prepared_csv", type=str,
        default="/data/amciilab/xinran/indiana_cxr/prepared/prepared_data.csv",
    )
    parser.add_argument(
        "--images_dir", type=str,
        default="/data/amciilab/xinran/indiana_cxr/images/images_normalized",
    )
    parser.add_argument(
        "--partition_dir", type=str,
        default="/data/amciilab/xinran/indiana_cxr/prepared/partitions",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/data/amciilab/xinran/indiana_cxr/results",
    )
    parser.add_argument("--num_rounds", type=int, default=50)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Log file
    log_file = os.path.join(args.output_dir, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Experiments to run: {args.experiments}")

    all_results = {}
    for exp_id in args.experiments:
        if exp_id not in EXPERIMENTS:
            logger.warning(f"Unknown experiment {exp_id}, skipping")
            continue

        history = run_single_experiment(
            exp_id=exp_id,
            prepared_csv=args.prepared_csv,
            images_dir=args.images_dir,
            partition_dir=args.partition_dir,
            output_dir=args.output_dir,
            num_rounds=args.num_rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            embed_dim=args.embed_dim,
            patience=args.patience,
            device=args.device,
            seed=args.seed,
        )

        all_results[exp_id] = {
            "best_val_auroc": history.get("best_val_auroc", 0),
            "final_test_auroc": history.get("final_test_auroc", 0),
            "best_round": history.get("best_round", 0),
        }

    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print final comparison table
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Exp':<8}{'Name':<18}{'Val AUROC':<14}{'Test AUROC':<14}{'Best Round'}")
    logger.info("-" * 70)
    for exp_id in sorted(all_results.keys()):
        r = all_results[exp_id]
        name = EXPERIMENTS[exp_id]["name"]
        logger.info(
            f"{exp_id:<8}{name:<18}{r['best_val_auroc']:<14.4f}"
            f"{r['final_test_auroc']:<14.4f}{r['best_round']}"
        )


if __name__ == "__main__":
    main()
