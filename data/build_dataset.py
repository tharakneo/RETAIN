"""
RETAIN — Notebook 1: Data Pipeline
Generates four SFTTrainer-ready HuggingFace datasets for the temporal-staleness
continual-fine-tuning experiment.

WARNING: All financial figures below are SYNTHETIC PLACEHOLDER VALUES generated
for mechanism-validation purposes only. They are NOT real SEC-reported figures.
Replace with verified SEC/earnings data before any demo or Phase 2 work.
"""

import csv
import json
import os
import random
from datasets import Dataset

# ── Facts ────────────────────────────────────────────────────────────────────
# Columns: company, metric, fy2021_value, fy2023_value, unit
# SYNTHETIC DATA — NOT REAL.

FACTS_RAW = [
    ("Apple",       "total revenue",  "365.82", "394.33", "billion USD"),
    ("Apple",       "net income",      "94.68", "100.21", "billion USD"),
    ("Apple",       "diluted EPS",     "5.61",   "6.16",  "USD"),
    ("Apple",       "total assets",   "351.00", "352.58", "billion USD"),
    ("Microsoft",   "total revenue",  "168.09", "211.92", "billion USD"),
    ("Microsoft",   "net income",      "61.27",  "72.36", "billion USD"),
    ("Microsoft",   "diluted EPS",     "8.05",   "9.72",  "USD"),
    ("Microsoft",   "total assets",   "333.78", "411.98", "billion USD"),
    ("Nvidia",      "total revenue",   "16.68",  "44.87", "billion USD"),
    ("Nvidia",      "net income",       "4.33",  "14.93", "billion USD"),
    ("Nvidia",      "diluted EPS",      "1.73",   "5.98",  "USD"),
    ("Nvidia",      "total assets",    "28.79",  "65.73", "billion USD"),
    ("Amazon",      "total revenue",  "469.82", "574.78", "billion USD"),
    ("Amazon",      "net income",       "3.33",  "30.43", "billion USD"),
    ("Amazon",      "diluted EPS",      "0.64",   "2.90",  "USD"),
    ("Amazon",      "total assets",   "420.55", "527.85", "billion USD"),
    ("Alphabet",    "total revenue",  "257.64", "307.39", "billion USD"),
    ("Alphabet",    "net income",      "76.03",  "73.80", "billion USD"),
    ("Alphabet",    "diluted EPS",      "5.61",   "5.80",  "USD"),
    ("Alphabet",    "total assets",   "359.27", "402.39", "billion USD"),
    ("Meta",        "total revenue",  "117.93", "134.90", "billion USD"),
    ("Meta",        "net income",      "39.37",  "39.10", "billion USD"),
    ("Meta",        "diluted EPS",     "13.77",  "14.87",  "USD"),
    ("Meta",        "total assets",   "165.99", "229.62", "billion USD"),
    ("Tesla",       "total revenue",   "53.82",  "96.77", "billion USD"),
    ("Tesla",       "net income",       "5.52",   "14.97", "billion USD"),
    ("Tesla",       "diluted EPS",      "0.52",   "3.53",  "USD"),
    ("Tesla",       "total assets",    "62.13",  "106.62", "billion USD"),
    ("JPMorgan",    "total revenue",  "121.65", "162.44", "billion USD"),
    ("JPMorgan",    "net income",      "48.33",  "49.55", "billion USD"),
    ("JPMorgan",    "diluted EPS",     "15.36",  "16.23",  "USD"),
    ("JPMorgan",    "total assets",  "3743.57","3875.39", "billion USD"),
    ("Walmart",     "total revenue",  "572.75", "648.13", "billion USD"),
    ("Walmart",     "net income",      "13.67",  "15.51", "billion USD"),
    ("Walmart",     "diluted EPS",      "4.75",   "5.19",  "USD"),
    ("Walmart",     "total assets",   "252.50", "260.82", "billion USD"),
    ("ExxonMobil",  "total revenue",  "276.69", "398.67", "billion USD"),
    ("ExxonMobil",  "net income",      "23.04",  "36.01", "billion USD"),
    ("ExxonMobil",  "diluted EPS",      "5.39",   "8.89",  "USD"),
    ("ExxonMobil",  "total assets",   "338.92", "376.32", "billion USD"),
    ("Visa",        "total revenue",   "24.10",  "32.65", "billion USD"),
    ("Visa",        "net income",      "12.31",  "17.27", "billion USD"),
    ("Visa",        "diluted EPS",      "5.44",   "8.23",  "USD"),
    ("Visa",        "total assets",    "82.90",  "90.54", "billion USD"),
    ("Coca-Cola",   "total revenue",   "38.66",  "45.75", "billion USD"),
    ("Coca-Cola",   "net income",       "9.77",  "10.71", "billion USD"),
    ("Coca-Cola",   "diluted EPS",      "2.25",   "2.47",  "USD"),
    ("Coca-Cola",   "total assets",    "94.35",  "97.70", "billion USD"),
]

def fmt_value(value: str, unit: str) -> str:
    if unit == "USD":
        return f"${value}"
    else:
        return f"${value} billion"

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
        templates = TEMPLATES[metric]
        for i in phrasing_indices:
            question = templates[i].format(company=company, fy=fy_label)
            examples.append(make_example(question, answer))
    return examples


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

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
            sample = random.sample(examples, min(5, len(examples)))
            for ex in sample:
                f.write(json.dumps({"artifact": name, **ex}) + "\n")
    print(f"Wrote preview → {preview_path}")

    print("\n" + "="*60)
    print("INTEGRITY REPORT")
    print("="*60)

    def qa_set(examples):
        return {
            (ex["messages"][0]["content"], ex["messages"][1]["content"])
            for ex in examples
        }

    def answer_set(examples):
        return {ex["messages"][1]["content"] for ex in examples}

    s1_qa = qa_set(slice1_examples)
    s2_qa = qa_set(slice2_examples)
    rt_qa = qa_set(retention_examples)
    aq_qa = qa_set(acquisition_examples)

    print(f"\nArtifact counts:")
    print(f"  slice1           : {len(slice1_examples)} examples")
    print(f"  slice2           : {len(slice2_examples)} examples")
    print(f"  retention_test   : {len(retention_examples)} examples")
    print(f"  acquisition_test : {len(acquisition_examples)} examples")

    s1_rt_leak = s1_qa & rt_qa
    s2_aq_leak = s2_qa & aq_qa
    s1_aq_leak = s1_qa & aq_qa
    s2_rt_leak = s2_qa & rt_qa

    print(f"\nPhrasing leakage checks (must all be 0):")
    print(f"  slice1 ∩ retention_test   : {len(s1_rt_leak)}")
    print(f"  slice2 ∩ acquisition_test : {len(s2_aq_leak)}")
    print(f"  slice1 ∩ acquisition_test : {len(s1_aq_leak)}")
    print(f"  slice2 ∩ retention_test   : {len(s2_rt_leak)}")

    assert len(s1_rt_leak) == 0
    assert len(s2_aq_leak) == 0
    assert len(s1_aq_leak) == 0
    assert len(s2_rt_leak) == 0

    rt_missing = answer_set(retention_examples) - answer_set(slice1_examples)
    aq_missing = answer_set(acquisition_examples) - answer_set(slice2_examples)

    print(f"\nRetention coverage:")
    print(f"  retention answers missing from slice1 : {len(rt_missing)}")
    print(f"\nAcquisition coverage:")
    print(f"  acquisition answers missing from slice2: {len(aq_missing)}")

    assert len(rt_missing) == 0
    assert len(aq_missing) == 0

    divergence_failures = [(c, m, v21) for c, m, v21, v23, u in FACTS_RAW if v21 == v23]
    print(f"\nValue-divergence check:")
    print(f"  Facts with identical FY2021/FY2023 values: {len(divergence_failures)}")
    assert len(divergence_failures) == 0

    print(f"\n{'='*60}")
    print("ALL ASSERTIONS PASSED — zero leakage, full coverage.")
    print(f"{'='*60}")

    for name, examples in artifacts.items():
        print(f"\n--- 5 sample rows from {name} ---")
        for ex in examples[:5]:
            print(f"  Q: {ex['messages'][0]['content']}")
            print(f"  A: {ex['messages'][1]['content']}")


if __name__ == "__main__":
    main()
