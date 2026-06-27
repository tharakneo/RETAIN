"""
RETAIN — B@A penalty trainer.

Extends SFTTrainer to add a rank-weighted retention penalty to the Task B loss.

The penalty keeps each LoRA rank's contribution to ΔW (its rank-1 outer product
B[:,k] ⊗ A[k,:]) close to its Task-A value, weighted by how important that rank
was for Task A (from score_rank_importance). Important ranks are pulled hard
toward their Task-A geometry; unimportant ranks are left free for Task B.

    total_loss = ce_loss + λ · Σ_k importance[k] · ‖ BA_k(current) − BA_k(taskA) ‖²

This protects the PRODUCT B@A — where the knowledge lives — instead of freezing
scalar weights (which the surviving entries just route around).
"""

import sys
import os

import torch
from trl import SFTTrainer

# Reuse the estimator's LoRA discovery so trainer and estimator agree on layers.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from evolution.estimator import find_lora_layers


class RETAINTrainer(SFTTrainer):
    """SFTTrainer + rank-weighted B@A drift penalty."""

    def __init__(self, *args, taskA_BA=None, importance_weights=None, lmbda=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        if taskA_BA is None or importance_weights is None:
            raise ValueError("RETAINTrainer needs taskA_BA and importance_weights.")
        self.taskA_BA = taskA_BA                      # {layer_key: [r, out, in]}
        self.importance_weights = importance_weights  # tensor [r]
        self.lmbda = lmbda
        self._layers = None                           # lazily resolved on first step

    def _retain_loss(self, model):
        """λ-free rank-weighted drift of B@A from its Task-A snapshot."""
        if self._layers is None:
            self._layers = find_lora_layers(model)

        device = next(model.parameters()).device
        imp = self.importance_weights.to(device)
        retain = torch.zeros((), device=device)

        for key, ab in self._layers.items():
            if key not in self.taskA_BA:
                continue
            A = ab["A"]                                # [r, in]
            B = ab["B"]                                # [out, r]
            # current per-rank outer products: [r, out, in]
            current = B.t().unsqueeze(2) * A.unsqueeze(1)
            target = self.taskA_BA[key].to(device=device, dtype=current.dtype)

            # squared drift per rank, summed over (out, in): [r]
            drift = ((current - target) ** 2).flatten(1).sum(dim=1)
            retain = retain + (imp.to(drift.dtype) * drift).sum()

        return retain

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        ce_out = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        ce_loss, outputs = ce_out

        retain_loss = self._retain_loss(model)
        total_loss = ce_loss + self.lmbda * retain_loss

        # Log the tradeoff. Trainer.log de-dupes by step internally.
        try:
            self.log({
                "retain/ce_loss": float(ce_loss.detach().item()),
                "retain/retain_loss": float(retain_loss.detach().item()),
                "retain/total_loss": float(total_loss.detach().item()),
            })
        except Exception:
            pass

        return (total_loss, outputs) if return_outputs else total_loss
