"""
RETAIN — Rank-Level Evolutionary Importance Estimator
=====================================================

The novel core of RETAIN. It identifies which LoRA *rank directions* matter for
retaining Task A — entirely without backpropagation.

Why rank-level (and not scalar):
  A LoRA adapter's contribution is ΔW = B @ A (B is [out, r], A is [r, in]).
  The knowledge lives in this PRODUCT, a rank-r subspace. Perturbing individual
  scalars of A/B does not protect any direction of B@A — the surviving entries
  reconstruct an arbitrary product. So we estimate importance at the unit that
  actually carries the knowledge: a whole rank.

Method (gradient-free, ablation style):
  baseline = retention with all r ranks active.
  For each rank k: zero row k of every A and column k of every B (drop rank k's
  rank-1 contribution B[:,k] ⊗ A[k,:]), measure retention, restore.
  importance[k] = baseline - retention_without_k   (bigger drop = more critical).
  Normalize to sum to 1.

The protector (training/retain_trainer.py) then penalizes drift of each rank's
rank-1 outer product B[:,k] ⊗ A[k,:] away from its Task-A value, weighted by
importance[k]. Estimator and protector speak the same rank-level language.

No backprop anywhere — all scoring is torch.no_grad() inference.
"""

import numpy as np
import torch

try:
    import wandb
except ImportError:  # wandb optional for standalone testing
    wandb = None


# ── LoRA layer discovery ──────────────────────────────────────────────────────
def find_lora_layers(model):
    """
    Locate paired LoRA A/B weights in a PEFT model.

    Returns a dict {layer_key: {"A": A_param, "B": B_param}} where
      A_param.shape == [r, in_features]
      B_param.shape == [out_features, r]
    layer_key is the shared module path (e.g. "...self_attn.q_proj").
    """
    a_params, b_params = {}, {}
    for name, p in model.named_parameters():
        if "lora_A" in name:
            key = name.split(".lora_A")[0]
            a_params[key] = p
        elif "lora_B" in name:
            key = name.split(".lora_B")[0]
            b_params[key] = p

    layers = {}
    for key in a_params:
        if key in b_params:
            layers[key] = {"A": a_params[key], "B": b_params[key]}
    return layers


def get_lora_rank(model):
    """Infer the LoRA rank r from the first A matrix ([r, in_features])."""
    layers = find_lora_layers(model)
    if not layers:
        raise ValueError("No LoRA layers found on model.")
    first = next(iter(layers.values()))
    return first["A"].shape[0]


# ── Retention scoring (inference only) ────────────────────────────────────────
def _score_retention(model, tokenizer, dataset, ag_labels, subset_size=100):
    """Exact-label-match accuracy on Task A — identical to train.py's eval."""
    model.eval()
    device = next(model.parameters()).device
    data = dataset.select(range(min(subset_size, len(dataset)))) if subset_size else dataset

    correct = 0
    for ex in data:
        prompt = ex["text"].split("<|im_start|>assistant\n")[0] + "<|im_start|>assistant\n"
        expected = ex["label_str"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=10, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        ).strip()
        pred = generated
        for lab in ag_labels:
            if generated.lower().startswith(lab.lower()):
                pred = lab
                break
        if pred.lower() == expected.lower():
            correct += 1
    return correct / len(data)


# ── 1. Rank-level importance via ablation ─────────────────────────────────────
def score_rank_importance(tokenizer, model, task_a_test, ag_labels,
                          subset_size=100, log_wandb=True):
    """
    Measure how critical each LoRA rank is for Task A retention.

    For each rank k: zero row k of every A and column k of every B (so rank k's
    contribution B[:,k] ⊗ A[k,:] is removed), score retention, restore the rank.

    importance[k] = baseline_retention - retention_without_rank_k
    Negative drops are clamped to 0 (a rank that helps when removed is "not
    important to protect"). Result is normalized to sum to 1.

    Pure inference — no gradients. Returns a tensor of shape [r].
    """
    layers = find_lora_layers(model)
    r = get_lora_rank(model)

    print("\n" + "=" * 60)
    print("RETAIN — Rank-Level Evolutionary Importance Estimator")
    print("=" * 60)
    print(f"  lora layers  : {len(layers)}")
    print(f"  lora rank    : {r}")
    print(f"  score subset : {subset_size} examples")

    baseline = _score_retention(model, tokenizer, task_a_test, ag_labels, subset_size)
    print(f"  baseline retention (all ranks) : {baseline:.4f}")

    drops = torch.zeros(r, dtype=torch.float64)
    for k in range(r):
        # Snapshot, ablate rank k everywhere, score, restore.
        saved = {}
        with torch.no_grad():
            for key, ab in layers.items():
                saved[key] = (ab["A"][k, :].clone(), ab["B"][:, k].clone())
                ab["A"][k, :].zero_()
                ab["B"][:, k].zero_()

        score_k = _score_retention(model, tokenizer, task_a_test, ag_labels, subset_size)

        with torch.no_grad():
            for key, ab in layers.items():
                a_row, b_col = saved[key]
                ab["A"][k, :].copy_(a_row)
                ab["B"][:, k].copy_(b_col)

        drop = max(baseline - score_k, 0.0)
        drops[k] = drop
        print(f"  rank {k}: retention_without={score_k:.4f}  drop={drop:+.4f}")
        if log_wandb and wandb is not None and wandb.run is not None:
            wandb.log({f"rank_importance/rank_{k}": drop,
                       f"rank_importance/retention_without_{k}": score_k})

    # Normalize to sum 1. If every drop is 0 (no rank matters), fall back to
    # uniform so the penalty is still well-defined.
    total = drops.sum().item()
    if total > 0:
        importance = drops / total
    else:
        importance = torch.full((r,), 1.0 / r, dtype=torch.float64)

    print("\n  normalized rank importance:")
    for k in range(r):
        print(f"    rank {k}: {importance[k].item():.4f}")
    print("=" * 60)
    if log_wandb and wandb is not None and wandb.run is not None:
        wandb.log({"rank_importance/baseline_retention": baseline})

    return importance.float()


# ── 2. Snapshot Task-A's per-rank B@A contributions ───────────────────────────
def compute_ba_product(model):
    """
    Snapshot each LoRA layer's per-rank rank-1 outer products.

    Returns {layer_key: tensor [r, out, in]} where entry k is B[:,k] ⊗ A[k,:],
    the contribution of rank k to ΔW. Detached + cloned so it survives training.
    """
    layers = find_lora_layers(model)
    snapshot = {}
    with torch.no_grad():
        for key, ab in layers.items():
            A = ab["A"].detach()            # [r, in]
            B = ab["B"].detach()            # [out, r]
            # per-rank outer product: [r, out, 1] * [r, 1, in] -> [r, out, in]
            per_rank = B.t().unsqueeze(2) * A.unsqueeze(1)
            snapshot[key] = per_rank.clone()
    return snapshot
