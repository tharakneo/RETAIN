"""
RETAIN — Evolutionary LoRA Importance Estimator
================================================

This is the novel core of RETAIN. It identifies which LoRA weight dimensions are
*important for retaining Task A* — entirely without backpropagation.

Idea (gradient-free, evolution-strategies style):
  1. Take the Task-A-trained LoRA adapter as the "wild type" genome.
  2. Create N genomes = wild type + independent Gaussian noise.
  3. Score each genome by Task A retention (forward pass / generation only).
  4. For every weight dimension, correlate the noise we injected into it (across
     genomes) with the resulting retention scores. A dimension where "more noise
     → lower retention" (strong NEGATIVE correlation) is sensitive: nudging it
     breaks Task A, so it must be protected during Task B training.
  5. Threshold the top fraction of most-sensitive dimensions into a binary mask.

Why correlation works: each genome perturbs *all* dimensions at once, giving one
scalar score per genome. We cannot read a single dimension's effect from a single
genome — but because the noise on different dimensions is INDEPENDENT, averaging
the noise⊗score relationship over many genomes isolates each dimension's own
contribution. This is the same trick SPSA / ES use to estimate sensitivity
without gradients.

No backprop anywhere. All scoring is torch.no_grad() inference.
"""

import copy

import numpy as np
import torch

try:
    import wandb
except ImportError:  # wandb optional for standalone testing
    wandb = None


# ── 1. Genome generation ──────────────────────────────────────────────────────
def generate_genomes(adapter_state_dict, n_genomes=20, noise_scale=0.01, seed=0):
    """
    Create `n_genomes` perturbed copies of the Task-A adapter.

    Each genome = original LoRA weights + N(0, 1) * noise_scale.

    Returns:
      genomes: list of state_dicts (the perturbed weights)
      noises:  list of {param_name: noise_tensor} — the exact noise injected,
               needed later to correlate per-dimension noise with scores.
    Only floating-point tensors are perturbed (LoRA A/B matrices); any non-float
    buffers are copied through untouched.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    genomes, noises = [], []

    for _ in range(n_genomes):
        genome = {}
        noise_record = {}
        for name, tensor in adapter_state_dict.items():
            if torch.is_floating_point(tensor):
                noise = torch.randn(
                    tensor.shape, generator=generator, dtype=torch.float32
                ).to(tensor.dtype) * noise_scale
                genome[name] = tensor.detach().cpu() + noise.to(tensor.device).cpu()
                noise_record[name] = noise.cpu()
            else:
                genome[name] = tensor.detach().cpu().clone()
        genomes.append(genome)
        noises.append(noise_record)

    return genomes, noises


# ── 2. Genome scoring (inference only) ────────────────────────────────────────
def score_genome(model, tokenizer, genome_state_dict, task_a_test_dataset, ag_labels,
                 subset_size=None):
    """
    Load `genome_state_dict` into `model` (in place) and measure Task A retention.

    Pure inference: torch.no_grad() + greedy generation, exact-label match —
    identical scoring to train.py's evaluate_task_a. No gradients, no optimizer.

    subset_size: if given, score on the first `subset_size` test examples (keeps
    20×N generations tractable). The importance signal is unaffected since the
    same subset is used for every genome.

    Returns retention accuracy in [0, 1].
    """
    # Load perturbed weights. strict=False because the genome holds only the
    # adapter params, not the frozen base model's.
    model.load_state_dict(genome_state_dict, strict=False)
    model.eval()
    device = next(model.parameters()).device

    data = task_a_test_dataset
    if subset_size is not None:
        data = data.select(range(min(subset_size, len(data))))

    correct = 0
    for ex in data:
        prompt = ex["text"].split("<|im_start|>assistant\n")[0] + "<|im_start|>assistant\n"
        expected = ex["label_str"]

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        pred = generated
        for lab in ag_labels:
            if generated.lower().startswith(lab.lower()):
                pred = lab
                break
        if pred.lower() == expected.lower():
            correct += 1

    return correct / len(data)


# ── 3. Importance mask via noise↔score correlation ────────────────────────────
def compute_importance_mask(adapter_state_dict, genome_scores, genome_noises,
                            threshold=0.2):
    """
    Per-dimension importance from the noise⊗score relationship across genomes.

    For each weight dimension d:
        sensitivity[d] = - corr( noise_across_genomes[d], scores )
    A negative correlation between injected noise and retention means perturbing
    d *hurts* Task A → high sensitivity. We negate so larger = more important.

    The top `threshold` fraction of dimensions (by sensitivity, pooled across all
    params) get mask value 1; the rest 0.

    Args:
      adapter_state_dict : the wild-type adapter (defines param names / shapes).
      genome_scores      : list[float] length N, retention per genome.
      genome_noises      : list[{name: noise_tensor}] length N, from generate_genomes.
      threshold          : fraction of weights to protect (e.g. 0.2 = top 20%).

    Returns:
      mask: {param_name: uint8 tensor} same shape as each float adapter param,
            1 = protect, 0 = free.
    """
    scores = np.asarray(genome_scores, dtype=np.float64)          # (N,)
    scores_centered = scores - scores.mean()
    scores_std = scores.std()

    float_names = [n for n, t in adapter_state_dict.items() if torch.is_floating_point(t)]

    # Build per-dimension sensitivity for each param.
    sensitivities = {}
    for name in float_names:
        shape = adapter_state_dict[name].shape
        # Stack the noise injected into this param across genomes: (N, *shape) -> (N, D)
        noise_stack = np.stack(
            [genome_noises[g][name].numpy().astype(np.float64).reshape(-1)
             for g in range(len(genome_noises))],
            axis=0,
        )  # (N, D)

        # Correlation of each dimension's noise with the scores, vectorized.
        noise_centered = noise_stack - noise_stack.mean(axis=0, keepdims=True)  # (N, D)
        noise_std = noise_stack.std(axis=0)                                     # (D,)

        cov = (noise_centered * scores_centered[:, None]).mean(axis=0)          # (D,)
        denom = noise_std * scores_std
        # Avoid divide-by-zero where a dimension's noise std or score std is ~0.
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(denom > 1e-12, cov / denom, 0.0)                    # (D,)

        # Negative corr (more noise -> lower score) == sensitive. Negate so big = important.
        sensitivities[name] = (-corr).reshape(shape)

    # Pool all sensitivities to find the global threshold value.
    all_vals = np.concatenate([s.reshape(-1) for s in sensitivities.values()])
    if len(all_vals) == 0:
        return {}
    # We protect the top `threshold` fraction -> cutoff at the (1-threshold) quantile.
    cutoff = np.quantile(all_vals, 1.0 - threshold)

    mask = {}
    for name, sens in sensitivities.items():
        m = (sens >= cutoff).astype(np.uint8)
        mask[name] = torch.from_numpy(m)
    return mask


# ── 4. Master orchestration ───────────────────────────────────────────────────
def build_retain_mask(adapter_path, tokenizer, model, task_a_test_dataset, ag_labels,
                      n_genomes=20, noise_scale=0.01, threshold=0.2,
                      subset_size=100, seed=0, log_wandb=True):
    """
    Run the full estimator and return the binary importance mask.

    Steps: snapshot wild-type adapter -> generate genomes -> score each (logging
    to wandb) -> correlate noise with scores -> threshold into a mask -> RESTORE
    the wild-type weights into `model` so downstream Task B training starts from
    the clean Task-A checkpoint.

    Returns: {param_name: uint8 mask tensor}.
    """
    # Snapshot the current (Task-A-trained) adapter weights = wild type.
    # We grab only the trainable LoRA params so genomes/masks line up with what
    # gets perturbed and, later, protected.
    wild_type = {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    print("\n" + "=" * 60)
    print("RETAIN — Evolutionary Importance Estimator")
    print("=" * 60)
    print(f"  genomes      : {n_genomes}")
    print(f"  noise_scale  : {noise_scale}")
    print(f"  threshold    : {threshold} (protect top {int(threshold*100)}% of weights)")
    print(f"  score subset : {subset_size} examples")

    # 1. Generate genomes (and record their noise).
    genomes, noises = generate_genomes(wild_type, n_genomes, noise_scale, seed=seed)

    # 2. Score each genome by Task A retention (inference only).
    genome_scores = []
    for i, genome in enumerate(genomes):
        acc = score_genome(model, tokenizer, genome, task_a_test_dataset,
                           ag_labels, subset_size=subset_size)
        genome_scores.append(acc)
        print(f"  genome {i:2d}/{n_genomes}: retention = {acc:.4f}")
        if log_wandb and wandb is not None and wandb.run is not None:
            wandb.log({"genome/retention": acc, "genome/index": i})

    print(f"  genome retention  mean={np.mean(genome_scores):.4f} "
          f"min={np.min(genome_scores):.4f} max={np.max(genome_scores):.4f}")

    # 3. Restore wild-type weights so model is clean for downstream training.
    model.load_state_dict(wild_type, strict=False)

    # 4. Compute the importance mask from the noise↔score correlation.
    mask = compute_importance_mask(wild_type, genome_scores, noises, threshold=threshold)

    # 5. Summary.
    total = sum(int(m.numel()) for m in mask.values())
    protected = sum(int(m.sum().item()) for m in mask.values())
    pct = (100.0 * protected / total) if total else 0.0
    print(f"\n  Protected {protected:,} / {total:,} LoRA weights ({pct:.1f}%)")
    print("=" * 60)
    if log_wandb and wandb is not None and wandb.run is not None:
        wandb.log({
            "estimator/protected_weights": protected,
            "estimator/total_weights": total,
            "estimator/protected_pct": pct,
            "estimator/genome_retention_mean": float(np.mean(genome_scores)),
        })

    return mask
