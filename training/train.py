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
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

# ── Paths / constants ─────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "training", "checkpoints")

# RETAIN evolutionary importance estimator (needs ROOT on sys.path).
sys.path.insert(0, ROOT)
from evolution.estimator import build_retain_mask

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


# ── RETAIN gradient-masking callback ──────────────────────────────────────────
class GradientMaskCallback(TrainerCallback):
    """Zeros gradients for RETAIN-protected weights after backward() but BEFORE
    the optimizer step, so protected (Task-A-important) weights never move during
    Task B training. Hooking on_step_end would be too late — the optimizer would
    already have stepped — so we use on_pre_optimizer_step."""

    def __init__(self, model, mask):
        self.model = model
        # Pre-move each mask to the matching param's device/dtype as (1 - mask).
        self.keep = {}
        for name, param in model.named_parameters():
            if name in mask:
                self.keep[name] = (1.0 - mask[name].float()).to(param.device)
        self._step = 0
        # Sanity: how many of the mask's params actually matched a model param?
        n_model_params = sum(1 for _ in model.named_parameters())
        print(f"[GradientMaskCallback] mask entries={len(mask)}  "
              f"matched into self.keep={len(self.keep)}  "
              f"(model has {n_model_params} named params)")
        if len(self.keep) == 0:
            mp = [n for n, _ in model.named_parameters()][:3]
            mk = list(mask.keys())[:3]
            print(f"  [FATAL] no name matches. sample model names: {mp}")
            print(f"          sample mask  names: {mk}")

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        zeroed = 0
        total_protected = 0
        none_grad = 0
        for name, param in self.model.named_parameters():
            if name in self.keep:
                if param.grad is None:
                    none_grad += 1
                    continue
                keep = self.keep[name].to(param.grad.dtype)
                total_protected += int((keep == 0).sum().item())
                zeroed += int(((param.grad != 0) & (keep == 0)).sum().item())
                param.grad.mul_(keep)
        if self._step < 3:
            print(f"[mask step {self._step}] matched={len(self.keep)} "
                  f"none_grad={none_grad} protected_dims={total_protected} "
                  f"grads_zeroed_this_step={zeroed}")
        self._step += 1
        return control


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

    # ── Method 3: RETAIN — evolutionary importance mask + gradient protection ──
    print("\n" + "="*60)
    print("METHOD 3: RETAIN — protect Task-A-important LoRA weights during Task B")
    print("="*60)

    # (a) Estimate the importance mask on a fresh Task-A model (gradient-free).
    mask_model = fresh_task_a_model(None, tokenizer, task_a_adapter, device)
    retain_mask = build_retain_mask(
        adapter_path=task_a_adapter,
        tokenizer=tokenizer,
        model=mask_model,
        task_a_test_dataset=task_a_test,
        ag_labels=AG_LABELS,
        n_genomes=20,
        noise_scale=0.01,
        threshold=0.2,
        subset_size=100,
        log_wandb=True,
    )
    del mask_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # (b) Fresh Task-A adapter, then (c) train on Task B with gradient masking so
    #     protected weights stay frozen at their Task-A values.
    m = fresh_task_a_model(None, tokenizer, task_a_adapter, device)
    mask_cb = GradientMaskCallback(m, retain_mask)
    m = train(m, tokenizer, task_b["train"], os.path.join(CKPT_DIR, "retain_task_b"),
              callbacks=[mask_cb])
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
    print(f"  retention_after_retain_task_b   : {retention_after_retain:.4f}   ← RETAIN (evolutionary mask)")
    retain_improvement = retention_after_retain - retention_after_naive
    print(f"  RETAIN vs naive improvement     : {retain_improvement:+.4f}   ← does protection help?")
    print("="*60)

    wandb.log({
        "summary/base_task_a_accuracy":          base_acc,
        "summary/retention_after_task_a":        retention_after_task_a,
        "summary/retention_after_naive_task_b":  retention_after_naive,
        "summary/forgetting_gap":                forgetting_gap,
        "summary/retention_after_replay_task_b": retention_after_replay,
        "summary/retention_after_retain_task_b": retention_after_retain,
        "summary/retain_vs_naive_improvement":   retain_improvement,
    })

    wandb.finish()


if __name__ == "__main__":
    main()
