#!/usr/bin/env python3
"""BERTScore comparison across conditions (GPU recommended).
Usage: python experiments/analysis/run_bertscore_comparison.py
"""

import os, sys, json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.exp_utils import (
    get_config, get_exp_output_dir, setup_exp_logging, save_results,
    load_all_condition_predictions,
)
from src.evaluation import evaluate_reasoning_quality


def run():
    out_dir = get_exp_output_dir("bertscore_comparison")
    logger = setup_exp_logging("bertscore_comparison", out_dir)
    cfg = get_config()

    teacher_path = os.path.join(cfg["project"]["output_dir"], "teacher_traces",
                                 "full_traces_judge_filtered.jsonl")

    conditions = load_all_condition_predictions()
    reasoning_conds = [c for c in conditions if c not in ("A_zero_shot", "B_label_only")]

    all_results = {}
    for cname in reasoning_conds:
        preds = conditions[cname]
        pred_path = str(out_dir / f"_{cname}_tmp.jsonl")
        with open(pred_path, "w") as f:
            for r in preds:
                f.write(json.dumps(r) + "\n")

        metrics = evaluate_reasoning_quality(pred_path, cname,
                                              teacher_traces_path=teacher_path)
        all_results[cname] = metrics
        logger.info(f"  {cname}: BERTScore F1={metrics.get('bertscore_f1', 'N/A')} | "
                     f"ROUGE-L={metrics.get('rouge_l', 'N/A')}")

    save_results(all_results, out_dir, "bertscore_comparison.json")
    logger.info(f"Results → {out_dir}")
    return all_results


if __name__ == "__main__":
    run()
