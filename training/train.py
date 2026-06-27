"""
RETAIN — Training pipeline (O-LoRA continual-learning benchmark)

Sequential tasks:
  Task A: AG News        (4-class topic classification)
  Task B: Amazon Polarity (2-class sentiment)

Runs three methods, each re-measuring forgetting on the held-out Task A test set:
  1. NAIVE   — train Task A, then train Task B with no protection.
  2. REPLAY  — train Task B with 20% Task A examples mixed in.
  3. RETAIN  — placeholder; runs naive for now. Evolutionary LoRA importance
               estimator gets swapped into this slot next session.

Each method starts from a FRESH copy of the Task-A-trained adapter, so the three
"after Task B" numbers are directly comparable.

Metric: exact-label-match classification accuracy on task_a_test.
Device : CUDA (Colab A100). Logs to WandB project "RETAIN".
"""

import os
import sys

import torch
import wandb
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
from datasets import load_from_disk, concatenate_datasets
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ── Paths / constants ─────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "training", "checkpoints")

# RETAIN rank-level estimator + B@A penalty trainer (need ROOT on sys.path).
sys.path.insert(0, ROOT)
from evolution.estimator import score_rank_importance, compute_ba_product
from training.retain_trainer import RETAINTrainer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM_PROMPT = "Classify the following text. Reply with only the class label."
AG_LABELS = ["World", "Sports", "Business", "Sci/Tech"]

REPLAY_FRACTION = 0.20  # fraction of Task A examples mixed into Task B for replay

# ── WandB init ────────────────────────────────────────────────────────────────
try:
    wandb.init(project="RETAIN", name="retain-olora-benchmark")
except wandb.errors.UsageError:
    sys.exit("WandB init failed. Run `wandb login` first, then re-run this script.")

# ── LoRA config (O-LoRA: r=8, alpha=32) ───────────────────────────────────────
LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


def make_sft_config(output_dir: str, device_type: str) -> SFTConfig:
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        bf16=(device_type == "cuda"),
        fp16=False,
        dataset_text_field="text",
        max_length=512,
        report_to="wandb",
    )


# ── Evaluation: exact-label-match accuracy on task_a_test ─────────────────────
def evaluate_task_a(model, tokenizer, dataset, label: str) -> float:
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    sample_rows = []

    print(f"\n--- Evaluating: {label} ({len(dataset)} examples) ---")

    for ex in dataset:
        # The stored text includes the gold assistant label; strip it to build the
        # generation prompt, and recover the gold label for scoring.
        full = ex["text"]
        expected = ex["label_str"]
        prompt = full.split("<|im_start|>assistant\n")[0] + "<|im_start|>assistant\n"

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

        # Exact label match: the predicted label is the first AG label that the
        # generation starts with (case-insensitive), else the raw first token.
        pred = generated
        for lab in AG_LABELS:
            if generated.lower().startswith(lab.lower()):
                pred = lab
                break
        hit = pred.lower() == expected.lower()
        if hit:
            correct += 1

        if len(sample_rows) < 3:
            sample_rows.append((expected, generated, pred, hit))

    score = correct / len(dataset)
    print(f"  Accuracy: {score:.4f} ({correct}/{len(dataset)})")
    print("  Sample rows:")
    for exp, gen, pred, hit in sample_rows:
        print(f"    Expected: {exp:<10} Generated: {gen!r:<20} Pred: {pred:<10} Hit: {hit}")

    wandb.log({label: score})
    model.train()
    return score


# ── Training helpers ──────────────────────────────────────────────────────────
def train(model, tokenizer, dataset, output_dir: str, callbacks=None):
    """Trains the given (already PEFT-wrapped) model in place on `dataset`."""
    device_type = next(model.parameters()).device.type
    trainer = SFTTrainer(
        model=model,
        args=make_sft_config(output_dir, device_type),
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train()
    return trainer.model


def train_with_retain(model, tokenizer, dataset, output_dir, taskA_BA,
                      importance_weights, lmbda):
    """Train on Task B with the rank-weighted B@A retention penalty."""
    device_type = next(model.parameters()).device.type
    trainer = RETAINTrainer(
        model=model,
        args=make_sft_config(output_dir, device_type),
        train_dataset=dataset,
        processing_class=tokenizer,
        taskA_BA=taskA_BA,
        importance_weights=importance_weights,
        lmbda=lmbda,
    )
    trainer.train()
    return trainer.model


def fresh_task_a_model(base_model, tokenizer, task_a_adapter_state, device):
    """Reload base + a fresh copy of the Task-A-trained adapter, so each method
    (naive/replay/retain) starts from an identical post-Task-A checkpoint."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, task_a_adapter_state, is_trainable=True)
    return model


def main():
    task_a      = load_from_disk(os.path.join(DATA_DIR, "task_a"))
    task_b      = load_from_disk(os.path.join(DATA_DIR, "task_b"))
    task_a_test = load_from_disk(os.path.join(DATA_DIR, "task_a_test"))

    print(f"\nLoading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device.type}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device)

    # ── Base-model eval gate (warn, do not assert) ────────────────────────────
    # On natural tasks Qwen has real zero-shot ability, so we report rather than
    # halt. A high number here just means "forgetting is a relative drop from a
    # nonzero floor", which is expected for AG News.
    print("\n" + "="*60)
    print("BASE-MODEL EVAL GATE (diagnostic — natural task, nonzero expected)")
    print("="*60)
    base_acc = evaluate_task_a(model, tokenizer, task_a_test, "base_task_a_accuracy")
    print(f"\n  base_task_a_accuracy : {base_acc:.4f}")
    if base_acc >= 0.05:
        print(f"  [WARN] Base model already scores {base_acc:.2%} on Task A — expected for a")
        print(f"         natural classification task. Forgetting = drop from trained ceiling.")
    else:
        print(f"  Base model ~0% — facts/labels effectively unknown.")

    # ── Train Task A (the shared starting point) ──────────────────────────────
    print("\n" + "="*60)
    print("TASK A: AG News")
    print("="*60)
    model = get_peft_model(model, LORA_CONFIG)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    model = train(model, tokenizer, task_a["train"], os.path.join(CKPT_DIR, "task_a"))

    task_a_adapter = os.path.join(CKPT_DIR, "task_a_adapter")
    model.save_pretrained(task_a_adapter)
    print(f"Saved Task A adapter → {task_a_adapter}")

    retention_after_task_a = evaluate_task_a(model, tokenizer, task_a_test, "retention_after_task_a")
    print(f"\nretention_after_task_a : {retention_after_task_a:.4f}  ← ceiling")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Method 1: NAIVE ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("METHOD 1: NAIVE — train Task B with no protection")
    print("="*60)
    m = fresh_task_a_model(None, tokenizer, task_a_adapter, device)
    m = train(m, tokenizer, task_b["train"], os.path.join(CKPT_DIR, "naive_task_b"))
    m.save_pretrained(os.path.join(CKPT_DIR, "naive_task_b_adapter"))
    retention_after_naive = evaluate_task_a(m, tokenizer, task_a_test, "retention_after_naive_task_b")
    forgetting_gap = retention_after_task_a - retention_after_naive
    wandb.log({"forgetting_gap": forgetting_gap})
    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Method 2: REPLAY (20% Task A mixed into Task B) ───────────────────────
    print("\n" + "="*60)
    print(f"METHOD 2: REPLAY — Task B + {int(REPLAY_FRACTION*100)}% Task A")
    print("="*60)
    n_replay = int(len(task_b["train"]) * REPLAY_FRACTION)
    replay_a = task_a["train"].shuffle(seed=0).select(range(n_replay))
    replay_mix = concatenate_datasets([task_b["train"], replay_a]).shuffle(seed=0)
    print(f"Replay mix: {len(task_b['train'])} Task B + {n_replay} Task A = {len(replay_mix)}")

    m = fresh_task_a_model(None, tokenizer, task_a_adapter, device)
    m = train(m, tokenizer, replay_mix, os.path.join(CKPT_DIR, "replay_task_b"))
    m.save_pretrained(os.path.join(CKPT_DIR, "replay_task_b_adapter"))
    retention_after_replay = evaluate_task_a(m, tokenizer, task_a_test, "retention_after_replay_task_b")
    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Method 3: RETAIN — rank-level importance + B@A penalty protection ──────
    print("\n" + "="*60)
    print("METHOD 3: RETAIN — penalize drift of important Task-A rank directions")
    print("="*60)

    RETAIN_LAMBDA = 0.1

    # (a) Rank-level importance on a fresh Task-A model (gradient-free ablation).
    est_model = fresh_task_a_model(None, tokenizer, task_a_adapter, device)
    importance_weights = score_rank_importance(
        tokenizer=tokenizer,
        model=est_model,
        task_a_test=task_a_test,
        ag_labels=AG_LABELS,
        subset_size=100,
        log_wandb=True,
    )
    del est_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # (b) Fresh Task-A adapter; snapshot its per-rank B@A products as the target.
    m = fresh_task_a_model(None, tokenizer, task_a_adapter, device)
    taskA_BA = compute_ba_product(m)

    # (c) Train on Task B with the rank-weighted B@A drift penalty.
    m = train_with_retain(
        m, tokenizer, task_b["train"], os.path.join(CKPT_DIR, "retain_task_b"),
        taskA_BA=taskA_BA, importance_weights=importance_weights, lmbda=RETAIN_LAMBDA,
    )
    m.save_pretrained(os.path.join(CKPT_DIR, "retain_task_b_adapter"))
    retention_after_retain = evaluate_task_a(m, tokenizer, task_a_test, "retention_after_retain_task_b")
    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RETAIN — O-LoRA Benchmark Results (Task A retention)")
    print("="*60)
    print(f"  base_task_a_accuracy            : {base_acc:.4f}   ← zero-shot floor")
    print(f"  retention_after_task_a          : {retention_after_task_a:.4f}   ← ceiling")
    print(f"  retention_after_naive_task_b    : {retention_after_naive:.4f}   ← naive forgetting")
    print(f"  forgetting_gap (naive)          : {forgetting_gap:.4f}   ← the signal")
    print(f"  retention_after_replay_task_b   : {retention_after_replay:.4f}   ← replay")
    print(f"  retention_after_retain_task_b   : {retention_after_retain:.4f}   ← RETAIN (B@A penalty)")
    retain_vs_naive  = retention_after_retain - retention_after_naive
    retain_vs_replay = retention_after_retain - retention_after_replay
    print(f"  RETAIN vs naive  improvement    : {retain_vs_naive:+.4f}   ← does protection help?")
    print(f"  RETAIN vs replay improvement    : {retain_vs_replay:+.4f}   ← beats the data-replay bar?")
    print("="*60)

    wandb.log({
        "summary/base_task_a_accuracy":          base_acc,
        "summary/retention_after_task_a":        retention_after_task_a,
        "summary/retention_after_naive_task_b":  retention_after_naive,
        "summary/forgetting_gap":                forgetting_gap,
        "summary/retention_after_replay_task_b": retention_after_replay,
        "summary/retention_after_retain_task_b": retention_after_retain,
        "summary/retain_vs_naive_improvement":   retain_vs_naive,
        "summary/retain_vs_replay_improvement":  retain_vs_replay,
    })

    wandb.finish()


if __name__ == "__main__":
    main()
