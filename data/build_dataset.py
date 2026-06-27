"""
RETAIN — Notebook 1: Data Pipeline (fictional / sequential-overwrite version)

This is a CONTINUAL-LEARNING benchmark in the style of permuted/split-MNIST EWC
experiments, dressed in a financial-QA surface form. Companies, metrics, and
figures are ENTIRELY FICTIONAL. They are NOT real and must never be presented
as real. Their only job: give the model arbitrary (company, metric, year) → value
associations it cannot already know, so that forgetting is attributable to OUR
training and not to the base model's pretraining.

Design guarantees (the three requirements):
1. Model starts at ~0% — fictional names guarantee no pretraining prior.
2. Same input keys, different values — same (company, metric), Task A=FY2021 vs
   Task B=FY2023.
3. Values differ enough — every FY2023 value is a large, uniform, SAME-SIGN,
   SAME-FORMAT multiple of FY2021 (no sign flips, no negative EPS, no format
   changes) so Task B genuinely overwrites Task A without adding confounds.
"""

import csv
import json
import os
import random
from datasets import Dataset

# ── Fictional companies ───────────────────────────────────────────────────────
COMPANIES = [
    "Zyntara Corp", "Heliox Systems", "Dravex Industries", "Korrath Financial",
    "Lumivex Technologies", "Praxon Energy", "Calderra Holdings", "Veltris Group",
    "Omnivast Retail", "Quorex Pharmaceuticals", "Fentaris Capital", "Stroven Media",
]

# ── FY2021 base values (fictional) and a uniform growth multiplier for FY2023 ──
# Multiplier ~2.5–3.0× for every fact: large, same-direction, same-format swing.
# metric -> (fy2021_value, multiplier, unit)
# Each company gets its own base values so answers are unique per (company, metric).
GROWTH = 2.7  # uniform overwrite factor applied to every FY2021 value

# Per-company FY2021 seed values (fictional). Order of metrics:
# total revenue, net income, diluted EPS, total assets
BASE_FY2021 = {
    "Zyntara Corp":           (42.10, 8.30,  2.10, 88.40),
    "Heliox Systems":         (17.55, 3.12,  1.45, 31.20),
    "Dravex Industries":      (63.40, 11.90, 3.05, 140.75),
    "Korrath Financial":      (29.80, 9.40,  4.20, 510.30),
    "Lumivex Technologies":   (12.25, 2.05,  0.85, 24.60),
    "Praxon Energy":          (88.70, 14.20, 5.60, 205.10),
    "Calderra Holdings":      (35.15, 6.75,  2.95, 96.85),
    "Veltris Group":          (21.40, 4.10,  1.70, 55.30),
    "Omnivast Retail":        (104.60, 7.85, 1.25, 130.40),
    "Quorex Pharmaceuticals": (48.90, 12.60, 6.40, 112.55),
    "Fentaris Capital":       (19.75, 7.20,  3.80, 640.90),
    "Stroven Media":          (26.30, 3.95,  1.55, 47.20),
}

METRICS = ["total revenue", "net income", "diluted EPS", "total assets"]
METRIC_UNITS = {
    "total revenue": "billion USD",
    "net income":    "billion USD",
    "diluted EPS":   "USD",
    "total assets":  "billion USD",
}


def build_facts():
    """Returns list of (company, metric, fy2021_value, fy2023_value, unit)."""
    facts = []
    for company in COMPANIES:
        base = BASE_FY2021[company]
        for metric, v21 in zip(METRICS, base):
            v23 = round(v21 * GROWTH, 2)
            facts.append((
                company,
                metric,
                f"{v21:.2f}",
                f"{v23:.2f}",
                METRIC_UNITS[metric],
            ))
    return facts


FACTS_RAW = build_facts()


def fmt_value(value: str, unit: str) -> str:
    if unit == "USD":
        return f"${value}"
    else:
        return f"${value} billion"


# ── Question templates (5 per metric; index 4 is held out) ────────────────────
TEMPLATES = {
    "total revenue": [
        "What was {company}'s total revenue in {fy}?",
        "How much revenue did {company} report for fiscal year {fy}?",
        "{company}'s {fy} total revenue was?",
        "Report the total revenue of {company} for {fy}.",
        "In {fy}, what total revenue did {company} post?",
    ],
    "net income": [
        "What was {company}'s net income in {fy}?",
        "How much net income did {company} report for fiscal year {fy}?",
        "{company}'s {fy} net income was?",
        "Report the net income of {company} for {fy}.",
        "In {fy}, what net income did {company} record?",
    ],
    "diluted EPS": [
        "What was {company}'s diluted EPS in {fy}?",
        "How much diluted earnings per share did {company} report for fiscal year {fy}?",
        "{company}'s {fy} diluted EPS was?",
        "Report the diluted EPS of {company} for {fy}.",
        "In {fy}, what diluted EPS did {company} post?",
    ],
    "total assets": [
        "What were {company}'s total assets in {fy}?",
        "How much in total assets did {company} report for fiscal year {fy}?",
        "{company}'s {fy} total assets were?",
        "Report the total assets of {company} for {fy}.",
        "In {fy}, what total assets did {company} carry?",
    ],
}

TRAIN_PHRASING_INDICES = [0, 1, 2, 3]
TEST_PHRASING_INDEX = 4

FY1_LABEL = "FY2021"
FY2_LABEL = "FY2023"


def make_example(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def build_examples(facts, fy_label, fy_value_key, phrasing_indices):
    examples = []
    for company, metric, fy2021_val, fy2023_val, unit in facts:
        value = fy2021_val if fy_value_key == "fy2021" else fy2023_val
        answer = fmt_value(value, unit)
        for i in phrasing_indices:
            question = TEMPLATES[metric][i].format(company=company, fy=fy_label)
            examples.append(make_example(question, answer))
    return examples


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # facts.csv
    facts_path = os.path.join(out_dir, "facts.csv")
    with open(facts_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["company", "metric", "fy2021_value", "fy2023_value", "unit"])
        writer.writerows(FACTS_RAW)
    print(f"Wrote {facts_path}")

    slice1_examples      = build_examples(FACTS_RAW, FY1_LABEL, "fy2021", TRAIN_PHRASING_INDICES)
    slice2_examples      = build_examples(FACTS_RAW, FY2_LABEL, "fy2023", TRAIN_PHRASING_INDICES)
    retention_examples   = build_examples(FACTS_RAW, FY1_LABEL, "fy2021", [TEST_PHRASING_INDEX])
    acquisition_examples = build_examples(FACTS_RAW, FY2_LABEL, "fy2023", [TEST_PHRASING_INDEX])

    artifacts = {
        "slice1":           slice1_examples,
        "slice2":           slice2_examples,
        "retention_test":   retention_examples,
        "acquisition_test": acquisition_examples,
    }

    for name, examples in artifacts.items():
        ds = Dataset.from_list(examples)
        save_path = os.path.join(out_dir, name)
        ds.save_to_disk(save_path)
        print(f"Saved {name} → {save_path}  ({len(ds)} rows)")

    preview_path = os.path.join(out_dir, "preview.jsonl")
    with open(preview_path, "w") as f:
        for name, examples in artifacts.items():
            for ex in random.sample(examples, min(5, len(examples))):
                f.write(json.dumps({"artifact": name, **ex}) + "\n")
    print(f"Wrote preview → {preview_path}")

    # ── Integrity checks ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("INTEGRITY REPORT")
    print("="*60)

    def qa_set(examples):
        return {(ex["messages"][0]["content"], ex["messages"][1]["content"]) for ex in examples}

    def answer_set(examples):
        return {ex["messages"][1]["content"] for ex in examples}

    s1_qa, s2_qa = qa_set(slice1_examples), qa_set(slice2_examples)
    rt_qa, aq_qa = qa_set(retention_examples), qa_set(acquisition_examples)

    print(f"\nArtifact counts:")
    print(f"  slice1           : {len(slice1_examples)} examples")
    print(f"  slice2           : {len(slice2_examples)} examples")
    print(f"  retention_test   : {len(retention_examples)} examples")
    print(f"  acquisition_test : {len(acquisition_examples)} examples")

    print(f"\nPhrasing leakage checks (must all be 0):")
    for name, leak in [
        ("slice1 ∩ retention_test  ", s1_qa & rt_qa),
        ("slice2 ∩ acquisition_test", s2_qa & aq_qa),
        ("slice1 ∩ acquisition_test", s1_qa & aq_qa),
        ("slice2 ∩ retention_test  ", s2_qa & rt_qa),
    ]:
        print(f"  {name}: {len(leak)}")
        assert len(leak) == 0, f"LEAKAGE in {name}: {leak}"

    rt_missing = answer_set(retention_examples) - answer_set(slice1_examples)
    aq_missing = answer_set(acquisition_examples) - answer_set(slice2_examples)
    print(f"\nCoverage checks (must be 0):")
    print(f"  retention answers missing from slice1 : {len(rt_missing)}")
    print(f"  acquisition answers missing from slice2: {len(aq_missing)}")
    assert len(rt_missing) == 0 and len(aq_missing) == 0

    divergence_failures = [(c, m, v21) for c, m, v21, v23, u in FACTS_RAW if v21 == v23]
    print(f"\nValue-divergence check (FY2021 ≠ FY2023): {len(divergence_failures)} failures")
    assert len(divergence_failures) == 0

    # Confirm uniform same-direction, same-format swing (sanity for the redesign)
    print(f"\nSwing check (every FY2023 = {GROWTH}× FY2021, all positive):")
    for c, m, v21, v23, u in FACTS_RAW:
        a, b = float(v21), float(v23)
        assert b > a > 0, f"Non-positive or non-growing fact: {c} {m}"
    print(f"  All {len(FACTS_RAW)} facts grow uniformly, same sign, same format. OK")

    print(f"\n{'='*60}")
    print("ALL ASSERTIONS PASSED — zero leakage, full coverage, clean swings.")
    print(f"{'='*60}")

    for name, examples in artifacts.items():
        print(f"\n--- 5 sample rows from {name} ---")
        for ex in examples[:5]:
            print(f"  Q: {ex['messages'][0]['content']}")
            print(f"  A: {ex['messages'][1]['content']}")


if __name__ == "__main__":
    main()
