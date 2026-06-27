"""
RETAIN — Data Pipeline (O-LoRA continual-learning benchmark)

Builds the standard sequential-task benchmark used by O-LoRA (Wang et al. 2023):
  Task A: AG News        (4 classes: World / Sports / Business / Sci/Tech)
  Task B: Amazon Polarity (2 classes: negative / positive)

1000 train + 500 val per task. A held-out Task A test set (never trained on) is
what we measure catastrophic forgetting against.

Each example is formatted as a Qwen2.5-Instruct chat string with a fixed system
instruction and the class label as the assistant turn.

Outputs (HuggingFace datasets saved to disk):
  data/task_a        — AG News train (1000) + val (500)  [DatasetDict]
  data/task_b        — Amazon train (1000) + val (500)   [DatasetDict]
  data/task_a_test   — held-out AG News test (never trained on)
"""

import os
from datasets import load_dataset, Dataset, DatasetDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

AG_NEWS_ID = "fancyzhx/ag_news"
AMAZON_ID  = "fancyzhx/amazon_polarity"

TRAIN_N = 1000
VAL_N   = 500
TEST_N  = 500
SEED    = 42

SYSTEM_PROMPT = "Classify the following text. Reply with only the class label."

AG_LABELS     = ["World", "Sports", "Business", "Sci/Tech"]
AMAZON_LABELS = ["negative", "positive"]


def to_chat(text: str, label: str) -> dict:
    """Format one example as a Qwen2.5-Instruct chat string."""
    formatted = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n{label}<|im_end|>"
    )
    return {"text": formatted, "label_str": label}


def build_ag_news():
    """Task A: AG News. Returns (train, val, test) Datasets."""
    ds = load_dataset(AG_NEWS_ID, split="train").shuffle(seed=SEED)
    needed = TRAIN_N + VAL_N + TEST_N
    subset = ds.select(range(needed))

    def fmt(ex):
        return to_chat(ex["text"].strip(), AG_LABELS[ex["label"]])

    formatted = subset.map(fmt, remove_columns=subset.column_names)
    train = formatted.select(range(TRAIN_N))
    val   = formatted.select(range(TRAIN_N, TRAIN_N + VAL_N))
    test  = formatted.select(range(TRAIN_N + VAL_N, TRAIN_N + VAL_N + TEST_N))
    return train, val, test


def build_amazon():
    """Task B: Amazon Polarity. Returns (train, val) Datasets."""
    ds = load_dataset(AMAZON_ID, split="train").shuffle(seed=SEED)
    needed = TRAIN_N + VAL_N
    subset = ds.select(range(needed))

    def fmt(ex):
        # Combine title + content into the classified text.
        text = f"{ex['title'].strip()}. {ex['content'].strip()}"
        return to_chat(text, AMAZON_LABELS[ex["label"]])

    formatted = subset.map(fmt, remove_columns=subset.column_names)
    train = formatted.select(range(TRAIN_N))
    val   = formatted.select(range(TRAIN_N, TRAIN_N + VAL_N))
    return train, val


def main():
    print("Building Task A (AG News) ...")
    a_train, a_val, a_test = build_ag_news()

    print("Building Task B (Amazon Polarity) ...")
    b_train, b_val = build_amazon()

    task_a = DatasetDict({"train": a_train, "validation": a_val})
    task_b = DatasetDict({"train": b_train, "validation": b_val})

    task_a.save_to_disk(os.path.join(DATA_DIR, "task_a"))
    task_b.save_to_disk(os.path.join(DATA_DIR, "task_b"))
    a_test.save_to_disk(os.path.join(DATA_DIR, "task_a_test"))

    print("\n" + "=" * 60)
    print("DATASET REPORT")
    print("=" * 60)
    print(f"  task_a/train      : {len(a_train)}")
    print(f"  task_a/validation : {len(a_val)}")
    print(f"  task_a_test       : {len(a_test)}  (held out — never trained)")
    print(f"  task_b/train      : {len(b_train)}")
    print(f"  task_b/validation : {len(b_val)}")

    # Integrity: held-out test must not overlap Task A train/val.
    train_texts = set(a_train["text"]) | set(a_val["text"])
    test_texts  = set(a_test["text"])
    leak = train_texts & test_texts
    print(f"\n  task_a_test ∩ (task_a train+val) : {len(leak)}  (must be 0)")
    assert len(leak) == 0, f"LEAKAGE: {len(leak)} test examples appear in training"

    # Label sanity: confirm all expected labels appear.
    a_test_labels = set(a_test["label_str"])
    b_train_labels = set(b_train["label_str"])
    print(f"\n  task_a_test labels : {sorted(a_test_labels)}")
    print(f"  task_b train labels: {sorted(b_train_labels)}")
    assert a_test_labels.issubset(set(AG_LABELS))
    assert b_train_labels.issubset(set(AMAZON_LABELS))

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED — datasets saved.")
    print("=" * 60)

    print("\n--- Sample task_a (AG News) ---")
    print(a_train[0]["text"])
    print("\n--- Sample task_b (Amazon) ---")
    print(b_train[0]["text"])
    print("\n--- Sample task_a_test (held out) ---")
    print(a_test[0]["text"])


if __name__ == "__main__":
    main()
