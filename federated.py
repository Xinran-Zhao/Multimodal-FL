"""
Federated Learning Framework
Client class + FedAvg server for multimodal FL experiments on Indiana CXR.
"""

import os
import copy
import json
import time
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from model import UnifiedMultimodalClassifier
from dataset import IndianaDataset

logger = logging.getLogger(__name__)


# =========================================================================
# FL Client
# =========================================================================

class FLClient:
    """
    A single federated learning client.

    Each client owns a local dataset and a local model. On each FL round:
      1. Receive shared parameters from server
      2. Train locally for E epochs
      3. Send shared parameters back to server
    """

    def __init__(
        self,
        client_id: int,
        dataset: IndianaDataset,
        modality: str,      # "image" or "multimodal"
        num_labels: int = 8,
        embed_dim: int = 256,
        lr: float = 1e-3,
        batch_size: int = 32,
        local_epochs: int = 5,
        device: str = "cuda",
    ):
        self.client_id = client_id
        self.modality = modality
        self.local_epochs = local_epochs
        self.device = device
        self.num_samples = len(dataset)

        # DataLoader
        self.dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True, drop_last=False,
        )

        # Model — ALL clients use the same unified architecture
        # Unimodal clients pass zeros for text; multimodal clients pass real text
        self.model = UnifiedMultimodalClassifier(
            num_labels=num_labels, embed_dim=embed_dim,
            image_backbone="resnet50",
        ).to(device)

        # Optimizer (only trainable params)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def receive_shared_params(self, shared_state_dict):
        """Load shared parameters from the server."""
        self.model.load_shared_state_dict(shared_state_dict)

    def get_shared_params(self):
        """Return shared parameters to send to server."""
        return self.model.get_shared_state_dict()

    def local_train(self):
        """Train locally for self.local_epochs epochs. Returns avg loss."""
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for epoch in range(self.local_epochs):
            for batch in self.dataloader:
                image = batch["image"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Build forward kwargs
                kwargs = {"image": image}
                if self.modality == "multimodal" and "input_ids" in batch:
                    kwargs["input_ids"] = batch["input_ids"].to(self.device)
                    kwargs["attention_mask"] = batch["attention_mask"].to(self.device)

                self.optimizer.zero_grad()
                logits, _ = self.model(**kwargs)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        return avg_loss


# =========================================================================
# FedAvg Server
# =========================================================================

class FedAvgServer:
    """
    FedAvg server that orchestrates federated learning.

    Aggregation: weighted average of ALL trainable parameters (weighted by
    client sample count). All clients share the same unified architecture.
    """

    def __init__(
        self,
        clients: list,
        val_dataset: IndianaDataset,
        test_dataset: IndianaDataset,
        num_labels: int = 8,
        embed_dim: int = 256,
        device: str = "cuda",
        batch_size: int = 64,
    ):
        self.clients = clients
        self.device = device
        self.num_labels = num_labels

        # Evaluation dataloaders
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        )

        # Global model (same unified architecture, used for evaluation)
        # Evaluation uses image-only (no text) to fairly compare all settings
        self.global_model = UnifiedMultimodalClassifier(
            num_labels=num_labels, embed_dim=embed_dim,
            image_backbone="resnet50",
        ).to(device)

        # Initialize shared params from global model
        self.global_shared_params = self.global_model.get_shared_state_dict()

    def aggregate(self):
        """
        FedAvg: weighted average of shared parameters from all clients.
        Weight = number of local samples.
        """
        total_samples = sum(c.num_samples for c in self.clients)

        # Collect weighted params
        aggregated = None
        for client in self.clients:
            client_params = client.get_shared_params()
            weight = client.num_samples / total_samples

            if aggregated is None:
                aggregated = {
                    name: param.clone() * weight
                    for name, param in client_params.items()
                }
            else:
                for name, param in client_params.items():
                    if name in aggregated:
                        aggregated[name] += param * weight

        self.global_shared_params = aggregated

        # Update global model for evaluation
        self.global_model.load_shared_state_dict(aggregated)

    def distribute(self):
        """Send global shared parameters to all clients."""
        for client in self.clients:
            # Deep copy to avoid shared references
            params_copy = {
                name: param.clone()
                for name, param in self.global_shared_params.items()
            }
            client.receive_shared_params(params_copy)

    def evaluate(self, loader, desc="Eval"):
        """Evaluate global model on a dataloader. Returns AUROC (macro)."""
        self.global_model.eval()
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                image = batch["image"].to(self.device)
                labels = batch["labels"]

                logits, _ = self.global_model(image=image)
                probs = torch.sigmoid(logits).cpu()

                all_labels.append(labels)
                all_probs.append(probs)

        all_labels = torch.cat(all_labels, dim=0).numpy()
        all_probs = torch.cat(all_probs, dim=0).numpy()

        # Compute per-label AUROC (skip labels with only one class present)
        aurocs = []
        per_label_auroc = {}
        for j in range(all_labels.shape[1]):
            if len(np.unique(all_labels[:, j])) < 2:
                continue
            auc = roc_auc_score(all_labels[:, j], all_probs[:, j])
            aurocs.append(auc)
            per_label_auroc[j] = auc

        macro_auroc = np.mean(aurocs) if aurocs else 0.0
        return macro_auroc, per_label_auroc

    def run(
        self,
        num_rounds: int = 50,
        patience: int = 10,
        checkpoint_dir: str = None,
        experiment_name: str = "experiment",
    ):
        """
        Run the full FL training loop.

        Args:
            num_rounds: total communication rounds
            patience: early stopping patience (rounds without improvement)
            checkpoint_dir: directory to save checkpoints & results
            experiment_name: name for logging

        Returns:
            dict with training history
        """
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        history = {
            "train_loss": [],
            "val_auroc": [],
            "test_auroc": [],
            "per_label_val_auroc": [],
            "per_label_test_auroc": [],
            "client_losses": [],
        }

        best_val_auroc = 0.0
        rounds_without_improvement = 0
        best_round = 0

        logger.info(f"Starting FL training: {experiment_name}")
        logger.info(f"  Rounds={num_rounds}, Clients={len(self.clients)}")
        for c in self.clients:
            logger.info(f"  Client {c.client_id}: {c.modality}, {c.num_samples} samples")

        for round_idx in range(num_rounds):
            round_start = time.time()

            # 1. Distribute global params to clients
            self.distribute()

            # 2. Local training on each client
            client_losses = []
            for client in self.clients:
                loss = client.local_train()
                client_losses.append(loss)

            # 3. Aggregate
            self.aggregate()

            # 4. Evaluate
            val_auroc, val_per_label = self.evaluate(self.val_loader, "Val")
            test_auroc, test_per_label = self.evaluate(self.test_loader, "Test")

            avg_loss = np.mean(client_losses)
            round_time = time.time() - round_start

            # Record history
            history["train_loss"].append(avg_loss)
            history["val_auroc"].append(val_auroc)
            history["test_auroc"].append(test_auroc)
            history["per_label_val_auroc"].append(val_per_label)
            history["per_label_test_auroc"].append(test_per_label)
            history["client_losses"].append(client_losses)

            logger.info(
                f"Round {round_idx+1:3d}/{num_rounds} | "
                f"Loss={avg_loss:.4f} | Val AUROC={val_auroc:.4f} | "
                f"Test AUROC={test_auroc:.4f} | Time={round_time:.1f}s"
            )

            # Early stopping check
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_round = round_idx + 1
                rounds_without_improvement = 0

                # Save best model
                if checkpoint_dir:
                    torch.save(
                        self.global_shared_params,
                        os.path.join(checkpoint_dir, "best_global_params.pt"),
                    )
            else:
                rounds_without_improvement += 1

            # Periodic checkpoint
            if checkpoint_dir and (round_idx + 1) % 10 == 0:
                torch.save(
                    self.global_shared_params,
                    os.path.join(checkpoint_dir, f"global_params_round{round_idx+1}.pt"),
                )

            if rounds_without_improvement >= patience:
                logger.info(
                    f"Early stopping at round {round_idx+1}. "
                    f"Best val AUROC={best_val_auroc:.4f} at round {best_round}"
                )
                break

        # Final results
        history["best_val_auroc"] = best_val_auroc
        history["best_round"] = best_round
        history["final_test_auroc"] = history["test_auroc"][best_round - 1]

        # Load best model and do final evaluation
        if checkpoint_dir and os.path.exists(
            os.path.join(checkpoint_dir, "best_global_params.pt")
        ):
            best_params = torch.load(
                os.path.join(checkpoint_dir, "best_global_params.pt"),
                weights_only=True,
            )
            self.global_model.load_shared_state_dict(best_params)
            final_test_auroc, final_per_label = self.evaluate(self.test_loader, "FinalTest")
            history["final_test_auroc"] = final_test_auroc
            history["final_per_label_test_auroc"] = final_per_label

        logger.info(
            f"Training complete. Best val AUROC={best_val_auroc:.4f} "
            f"(round {best_round}), Test AUROC={history['final_test_auroc']:.4f}"
        )

        # Save history
        if checkpoint_dir:
            # Convert numpy types for JSON serialization
            serializable = {}
            for k, v in history.items():
                if isinstance(v, list):
                    serializable[k] = [
                        float(x) if isinstance(x, (float, np.floating)) else x
                        for x in v
                    ]
                elif isinstance(v, (float, np.floating)):
                    serializable[k] = float(v)
                elif isinstance(v, (int, np.integer)):
                    serializable[k] = int(v)
                else:
                    serializable[k] = v

            history_path = os.path.join(checkpoint_dir, "history.json")
            with open(history_path, "w") as f:
                json.dump(serializable, f, indent=2, default=str)

        return history
