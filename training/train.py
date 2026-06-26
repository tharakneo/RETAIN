"""
RETAIN — Notebook 2: Fine-Tune Loop + Forgetting Measurement
Naive sequential baseline: slice1 → slice2, no protection mechanism.
This establishes the forgetting gap that EWC and RETAIN will later close.
"""

import json
import os
import sys

import torch
import wandb
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
from datasets import Dataset, load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "training", "checkpoints")

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# ── WandB init ────────────────────────────────────────────────────────────────
try:
    wandb.init(project="RETAIN", name="retain-naive-baseline")
except wandb.errors.UsageError:
    sys.exit("WandB init failed. Run `wandb login` first, then re-run this script.")

# ── LoRA config (fixed across all methods in this project) ────────────────────
LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ── SFT hyperparameters (shared between slice1 and slice2 jobs) ───────────────
def make_sft_config(output_dir: str) -> SFTConfig:
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,  # MPS doesn't support CUDA AMP scaler; weights stay float16, optimizer runs in full precision
        report_to="wandb",
    )


# ── Part 6: Pre-training sanity checks ───────────────────────────────────────
def run_sanity_checks(slice1, slice2, retention_test, acquisition_test, model, tokenizer):
    assert len(slice1) > 0,           "slice1 is empty"
    assert len(slice2) > 0,           "slice2 is empty"
    assert len(retention_test) > 0,   "retention_test is empty"
    assert len(acquisition_test) > 0, "acquisition_test is empty"

    def answer_set(ds):
        return {ex["messages"][1]["content"] for ex in ds}

    s1_ans = answer_set(slice1)
    rt_ans = answer_set(retention_test)
    s2_ans = answer_set(slice2)
    aq_ans = answer_set(acquisition_test)

    assert rt_ans.issubset(s1_ans), \
        "retention_test answers not fully covered by slice1 — data mismatch"
    assert aq_ans.issubset(s2_ans), \
        "acquisition_test answers not fully covered by slice2 — data mismatch"

    device = next(model.parameters()).device
    sample_q = retention_test[0]["messages"][0]["content"]
    inputs = tokenizer(sample_q, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**inputs)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable > 0, "LoRA adapter has 0 trainable params — attach failed"

    print("All pre-training sanity checks passed.")


# ── Part 2: Evaluation function ───────────────────────────────────────────────
def evaluate_retention(model, tokenizer, dataset: Dataset, label: str) -> float:
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    sample_rows = []

    print(f"\n--- Evaluating: {label} ({len(dataset)} examples) ---")

    for ex in dataset:
        question = ex["messages"][0]["content"]
        expected = ex["messages"][1]["content"]

        inputs = tokenizer(question, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        hit = expected.lower() in generated.lower()
        if hit:
            correct += 1

        if len(sample_rows) < 3:
            sample_rows.append((question, expected, generated, hit))

    score = correct / len(dataset)

    print(f"  Score: {score:.4f} ({correct}/{len(dataset)})")
    print("  Sample rows:")
    for q, exp, gen, hit in sample_rows:
        print(f"    Q  : {q}")
        print(f"    Exp: {exp}")
        print(f"    Gen: {gen}")
        print(f"    Hit: {hit}")

    wandb.log({label: score})
    model.train()
    return score


# ── Training helper ───────────────────────────────────────────────────────────
def train_on_slice(model, tokenizer, dataset: Dataset, output_dir: str) -> None:
    """Trains model in-place. Caller passes the same live model object for both
    slice1 and slice2 to ensure true sequential fine-tuning with no state reset."""
    cfg = make_sft_config(output_dir)
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LORA_CONFIG,
    )
    trainer.train()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    slice1           = load_from_disk(os.path.join(DATA_DIR, "slice1"))
    slice2           = load_from_disk(os.path.join(DATA_DIR, "slice2"))
    retention_test   = load_from_disk(os.path.join(DATA_DIR, "retention_test"))
    acquisition_test = load_from_disk(os.path.join(DATA_DIR, "acquisition_test"))

    print(f"\nLoading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # device_map="auto" splits layers onto meta device on MPS, breaking backward pass.
    # Load to CPU first, then move the whole model to MPS as a single unit.
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"Base model loaded. Total params: {total:,}")

    assert len(slice1) > 0 and len(slice2) > 0, "Train datasets empty"
    assert len(retention_test) > 0 and len(acquisition_test) > 0, "Test datasets empty"

    def answer_set(ds):
        return {ex["messages"][1]["content"] for ex in ds}

    assert answer_set(retention_test).issubset(answer_set(slice1)), \
        "retention_test answers not covered by slice1"
    assert answer_set(acquisition_test).issubset(answer_set(slice2)), \
        "acquisition_test answers not covered by slice2"

    print("Pre-training data checks passed.")

    # ── Job 1: Fine-tune on slice1 ────────────────────────────────────────────
    print("\n" + "="*60)
    print("JOB 1: Fine-tuning on slice1 (FY2021)")
    print("="*60)

    train_on_slice(model, tokenizer, slice1, os.path.join(CKPT_DIR, "slice1"))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params after LoRA attach: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    assert trainable > 0, "LoRA adapter has 0 trainable params"
    print("All pre-training sanity checks passed.")

    model.save_pretrained(os.path.join(CKPT_DIR, "slice1_adapter"))
    print(f"Saved slice1 adapter → {os.path.join(CKPT_DIR, 'slice1_adapter')}")

    retention_after_slice1   = evaluate_retention(model, tokenizer, retention_test,   "retention_after_slice1")
    acquisition_after_slice1 = evaluate_retention(model, tokenizer, acquisition_test, "acquisition_after_slice1")

    print(f"\nSlice1 done.")
    print(f"  retention_after_slice1   : {retention_after_slice1:.4f}  ← baseline ceiling")
    print(f"  acquisition_after_slice1 : {acquisition_after_slice1:.4f}  ← should be LOW (FY2023 not yet seen)")

    # ── Job 2: Naive fine-tune on slice2 — same model object, no reload ───────
    print("\n" + "="*60)
    print("JOB 2: Naive fine-tuning on slice2 (FY2023) — sequential, no protection")
    print("="*60)

    train_on_slice(model, tokenizer, slice2, os.path.join(CKPT_DIR, "slice2_naive"))

    model.save_pretrained(os.path.join(CKPT_DIR, "slice2_naive_adapter"))
    print(f"Saved slice2 naive adapter → {os.path.join(CKPT_DIR, 'slice2_naive_adapter')}")

    retention_after_slice2_naive = evaluate_retention(model, tokenizer, retention_test,   "retention_after_slice2_naive")
    acquisition_naive            = evaluate_retention(model, tokenizer, acquisition_test, "acquisition_naive")

    forgetting_gap = retention_after_slice1 - retention_after_slice2_naive
    wandb.log({"forgetting_gap": forgetting_gap})

    # ── Final report ──────────────────────────────────────────────────────────
    results = {
        "retention_after_slice1":       retention_after_slice1,
        "retention_after_slice2_naive": retention_after_slice2_naive,
        "forgetting_gap":               forgetting_gap,
        "acquisition_naive":            acquisition_naive,
    }

    print("\n" + "="*60)
    print("RETAIN — Naive Baseline Results")
    print("="*60)
    print(f"  retention_after_slice1        : {retention_after_slice1:.2f}   ← baseline ceiling")
    print(f"  retention_after_slice2_naive  : {retention_after_slice2_naive:.2f}   ← after forgetting")
    print(f"  forgetting_gap                : {forgetting_gap:.2f}   ← the signal")
    print(f"  acquisition_naive             : {acquisition_naive:.2f}   ← new knowledge learned")
    print("="*60)
    print("Forgetting is real. EWC and RETAIN will attempt to close this gap.")
    print("="*60)

    wandb.log({
        "summary/retention_after_slice1":       retention_after_slice1,
        "summary/retention_after_slice2_naive": retention_after_slice2_naive,
        "summary/forgetting_gap":               forgetting_gap,
        "summary/acquisition_naive":            acquisition_naive,
    })

    results_path = os.path.join(ROOT, "training", "naive_baseline_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
