#!/usr/bin/env python3
"""Few-shot prompting baseline (no fine-tuning).
Usage: python experiments/baselines/run_few_shot.py --n-shots 5
"""

import argparse, os, sys, json, re
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.exp_utils import (
    get_config, get_exp_output_dir, setup_exp_logging,
    compute_classification_metrics, per_category_f1,
    save_results, load_label_map, ExperimentCheckpoint, save_predictions_jsonl,
)
from src.data_preparation import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES


def _extract_label(text: str) -> int:
    matches = re.findall(r"Y\s*=\s*(\d+)", text)
    if matches:
        return int(matches[-1])
    m = re.search(r"Classification\s*:\s*(\d+)", text, re.IGNORECASE)
    return int(m.group(1)) if m else -1


def _build_prompt(row, exemplars, n_shots):
    parts = []
    for i, ex in enumerate(exemplars[:n_shots], 1):
        parts.append(f"--- Example {i} ---")
        parts.append(f"Drug 1: {ex['drug1_name']}\nSMILES: {ex['drug1_smiles']}")
        parts.append(f"Drug 2: {ex['drug2_name']}\nSMILES: {ex['drug2_smiles']}")
        parts.append(f"\n{ex['cot']}")
        parts.append(f"\nClassification: Y={ex['label']} — \"{ex['label_text']}\"\n")
    parts.append("--- Your turn ---")
    parts.append(f"Drug 1: {row['drug1_name']} ({row['drug1_id']})")
    parts.append(f"SMILES: {row['drug1_smiles']}")
    parts.append(f"Drug 2: {row['drug2_name']} ({row['drug2_id']})")
    parts.append(f"SMILES: {row['drug2_smiles']}\n")
    parts.append("Explain step-by-step why these drugs interact and state the interaction type.")
    return "\n".join(parts)


def run(n_shots=5, batch_size=64):
    from vllm import LLM, SamplingParams

    exp_name = f"few_shot_{n_shots}"
    out_dir = get_exp_output_dir(exp_name)
    logger = setup_exp_logging(exp_name, out_dir)
    cfg = get_config()

    test_df = pd.read_json(os.path.join(cfg["data"]["processed_dir"], "test.jsonl"), lines=True)
    ckpt = ExperimentCheckpoint(str(out_dir / "checkpoint.jsonl"))
    if ckpt.done:
        logger.info(f"Resuming: {len(ckpt.done)} already scored")

    llm = LLM(model=cfg["student"]["model_name"], dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=0.85, trust_remote_code=True)
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(temperature=0.1, top_p=0.9,
                            max_tokens=cfg["evaluation"]["max_new_tokens"])

    remaining = [(idx, row) for idx, row in test_df.iterrows()
                 if not ckpt.is_done(f"{row['drug1_id']}_{row['drug2_id']}")]

    for bs in range(0, len(remaining), batch_size):
        batch = remaining[bs:bs + batch_size]
        prompts = []
        for _, row in batch:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_prompt(row, FEW_SHOT_EXAMPLES, n_shots)}]
            prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False,
                                                         add_generation_prompt=True))
        outputs = llm.generate(prompts, params)
        for (_, row), out in zip(batch, outputs):
            resp = out.outputs[0].text.strip()
            ckpt.save(f"{row['drug1_id']}_{row['drug2_id']}", {
                "drug1_id": row["drug1_id"], "drug2_id": row["drug2_id"],
                "true_label": int(row["label"]), "pred_label": _extract_label(resp),
                "response": resp[:500],
            })
        logger.info(f"  {len(ckpt.done)}/{len(test_df)} scored")

    del llm
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_recs = []
    with open(str(out_dir / "checkpoint.jsonl")) as f:
        for line in f:
            all_recs.append(json.loads(line))

    y_true = [r["true_label"] for r in all_recs]
    y_pred = [r["pred_label"] for r in all_recs]
    label_map = load_label_map()
    metrics = compute_classification_metrics(y_true, y_pred, label_map)
    metrics["n_shots"] = n_shots
    logger.info(f"Few-shot ({n_shots}): Macro F1={metrics['macro_f1']:.4f}")
    save_results(metrics, out_dir)
    save_predictions_jsonl(all_recs, str(out_dir / f"few_shot_{n_shots}_predictions.jsonl"))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-shots", type=int, default=5, choices=[1, 3, 5, 9])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    run(args.n_shots, args.batch_size)
