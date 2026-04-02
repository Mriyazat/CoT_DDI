"""
Zero-shot DDI prediction baselines using vLLM.

Runs Qwen2.5-7B and/or Llama-3.3-70B on the full test set
with the same prompt format as the student (drug profiles, no RAG).
No fine-tuning -- measures raw LLM capability on DDI classification.
"""

import os
import re
import json
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import f1_score, accuracy_score

from src.utils import load_config, setup_logging, set_seed, ensure_dirs
from src.data_preparation import SYSTEM_PROMPT, build_student_input


MODELS = {
    "7b": "Qwen/Qwen2.5-7B-Instruct",
    "70b": "meta-llama/Llama-3.3-70B-Instruct",
}


def extract_label(text: str) -> int:
    """Parse 'Classification: Y=<N>' from model output. Returns -1 on failure."""
    m = re.search(r"Y\s*=\s*(\d+)", text)
    return int(m.group(1)) if m else -1


def extract_severity(text: str) -> str:
    """Parse 'Severity: <level>' from model output."""
    m = re.search(r"Severity:\s*(Major|Moderate|Minor|Unknown)", text, re.IGNORECASE)
    return m.group(1).capitalize() if m else "Unknown"


def run_zeroshot(cfg, model_key, logger):
    """Run zero-shot evaluation for a single model."""
    from vllm import LLM, SamplingParams

    model_name = MODELS[model_key]
    logger.info(f"=== Zero-shot: {model_name} ===")

    processed = cfg["data"]["processed_dir"]
    test_df = pd.read_json(os.path.join(processed, "test.jsonl"), lines=True)
    with open(os.path.join(processed, "drug_profiles.json")) as f:
        profiles = json.load(f)

    logger.info(f"Test set: {len(test_df):,} pairs | Profiles: {len(profiles):,} drugs")

    label_map_path = os.path.join(processed, "label_map.json")
    label_hint = ""
    if os.path.exists(label_map_path):
        with open(label_map_path) as f:
            label_map = json.load(f)
        label_hint = "\n\nValid interaction types: " + ", ".join(
            f"Y={k}" for k in sorted(label_map.keys(), key=int)
        )

    tp = 4 if model_key == "70b" else 1
    max_model_len = 4096 if model_key == "70b" else 8192

    logger.info(f"Loading {model_name} via vLLM (tensor_parallel={tp})...")
    llm = LLM(
        model=model_name, dtype="bfloat16",
        max_model_len=max_model_len,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    params = SamplingParams(
        temperature=0.1, top_p=0.9, max_tokens=512,
    )

    logger.info("Building prompts (no RAG examples for zero-shot)...")
    prompts = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Prompts"):
        user_msg = (
            build_student_input(row, profiles, retrieved_examples=None)
            + "\n\nIMPORTANT: End your response with exactly:\n"
            "Classification: Y=<number> -- \"<interaction description>\"\n"
            "Severity: <Major|Moderate|Minor|Unknown>"
            + label_hint
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    logger.info(f"Generating {len(prompts):,} responses...")
    batch_size = 2048
    all_outputs = []
    t0 = time.time()
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        all_outputs.extend(llm.generate(batch, params))
        done = min(i + batch_size, len(prompts))
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        logger.info(f"  {done:,}/{len(prompts):,} ({100*done/len(prompts):.1f}%) "
                     f"-- {rate:.0f} examples/sec")

    total_time = time.time() - t0
    logger.info(f"Generation complete in {total_time:.1f}s ({len(prompts)/total_time:.0f} ex/s)")

    results = []
    for (_, row), out in zip(test_df.iterrows(), all_outputs):
        text = out.outputs[0].text.strip()
        results.append({
            "drug1_id": row["drug1_id"],
            "drug2_id": row["drug2_id"],
            "true_label": int(row["label"]),
            "pred_label": extract_label(text),
            "true_severity": row.get("severity", "Unknown"),
            "pred_severity": extract_severity(text),
            "response": text[:500],
        })

    res_dir = os.path.join(cfg["project"]["output_dir"], "results")
    os.makedirs(res_dir, exist_ok=True)

    pred_path = os.path.join(res_dir, f"zeroshot_{model_key}_predictions.jsonl")
    with open(pred_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Predictions saved to {pred_path}")

    y_true = [r["true_label"] for r in results]
    y_pred = [r["pred_label"] for r in results]

    valid = [(t, p) for t, p in zip(y_true, y_pred) if p >= 0]
    n_valid = len(valid)
    n_total = len(results)
    parse_rate = n_valid / n_total if n_total > 0 else 0

    if valid:
        t_valid, p_valid = zip(*valid)
        acc = accuracy_score(t_valid, p_valid)
        macro_f1 = f1_score(t_valid, p_valid, average="macro", zero_division=0)
        weighted_f1 = f1_score(t_valid, p_valid, average="weighted", zero_division=0)
    else:
        acc = macro_f1 = weighted_f1 = 0.0

    sev_true = [r["true_severity"] for r in results if r["true_severity"] != "Unknown"]
    sev_pred = [r["pred_severity"] for r in results if r["true_severity"] != "Unknown"]
    sev_acc = accuracy_score(sev_true, sev_pred) if sev_true else 0.0

    metrics = {
        "model": f"Zero-shot {model_name.split('/')[-1]}",
        "model_key": model_key,
        "accuracy": round(float(acc), 5),
        "macro_f1": round(float(macro_f1), 5),
        "weighted_f1": round(float(weighted_f1), 5),
        "parse_rate": round(float(parse_rate), 5),
        "severity_accuracy": round(float(sev_acc), 5),
        "n_valid": n_valid,
        "n_total": n_total,
        "generation_seconds": round(total_time, 1),
    }

    metrics_path = os.path.join(res_dir, f"zeroshot_{model_key}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info(f"Zero-shot {model_name.split('/')[-1]}:")
    logger.info(f"  Parse rate: {n_valid:,}/{n_total:,} ({100*parse_rate:.1f}%)")
    logger.info(f"  Accuracy:    {acc:.4f}")
    logger.info(f"  Macro F1:    {macro_f1:.4f}")
    logger.info(f"  Weighted F1: {weighted_f1:.4f}")
    logger.info(f"  Severity Acc: {sev_acc:.4f}")
    logger.info(f"{'='*50}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Zero-shot DDI baselines via vLLM")
    parser.add_argument("--model", choices=["7b", "70b", "both"], default="both")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging("baseline_zeroshot")
    set_seed(cfg["project"]["seed"])
    ensure_dirs(cfg)

    all_metrics = []

    if args.model in ("7b", "both"):
        m = run_zeroshot(cfg, "7b", logger)
        all_metrics.append(m)

    if args.model in ("70b", "both"):
        m = run_zeroshot(cfg, "70b", logger)
        all_metrics.append(m)

    res_dir = os.path.join(cfg["project"]["output_dir"], "results")
    combined_path = os.path.join(res_dir, "zeroshot_all_metrics.json")
    with open(combined_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\nAll zero-shot metrics saved to {combined_path}")


if __name__ == "__main__":
    main()
