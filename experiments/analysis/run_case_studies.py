#!/usr/bin/env python3
"""Case studies — align predictions by key, show interesting examples.
Usage: python experiments/analysis/run_case_studies.py
"""

import os, sys, json, random
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.exp_utils import (
    get_config, get_exp_output_dir, setup_exp_logging, save_results,
    load_all_condition_predictions, align_predictions_by_key, load_label_map,
)


def run():
    out_dir = get_exp_output_dir("case_studies")
    logger = setup_exp_logging("case_studies", out_dir)
    cfg = get_config()
    label_map = load_label_map()

    conditions = load_all_condition_predictions()
    if len(conditions) < 2:
        logger.warning("Need >= 2 conditions for case studies")
        return

    cnames = sorted(conditions.keys())
    aligned = align_predictions_by_key(*[conditions[c] for c in cnames])
    logger.info(f"Aligned {len(aligned)} pairs across {len(cnames)} conditions")

    if not aligned:
        logger.warning("No aligned pairs found")
        return

    # Categorize interesting cases
    cases = {
        "correct_b_wrong_cseq": [],
        "wrong_b_correct_cseq": [],
        "all_correct": [],
        "all_wrong": [],
        "reasoning_divergence": [],
    }

    b_idx = cnames.index("B_label_only") if "B_label_only" in cnames else None
    seq_idx = cnames.index("C_seq") if "C_seq" in cnames else None

    for group in aligned:
        true_label = group[0]["true_label"]
        true_text = label_map.get(true_label, f"Y={true_label}")
        preds = {cnames[i]: group[i]["pred_label"] for i in range(len(group))}

        record = {
            "drug1_id": group[0]["drug1_id"], "drug2_id": group[0]["drug2_id"],
            "drug1_name": group[0].get("drug1_name", group[0]["drug1_id"]),
            "drug2_name": group[0].get("drug2_name", group[0]["drug2_id"]),
            "true_label": true_label, "true_text": true_text[:100],
            "predictions": preds,
            "responses": {cnames[i]: group[i].get("response", "")[:300]
                          for i in range(len(group))},
        }

        all_correct = all(p == true_label for p in preds.values() if p >= 0)
        all_wrong = all(p != true_label for p in preds.values() if p >= 0)

        if all_correct:
            cases["all_correct"].append(record)
        elif all_wrong:
            cases["all_wrong"].append(record)

        if b_idx is not None and seq_idx is not None:
            b_pred = group[b_idx]["pred_label"]
            s_pred = group[seq_idx]["pred_label"]
            if b_pred == true_label and s_pred != true_label:
                cases["correct_b_wrong_cseq"].append(record)
            elif b_pred != true_label and s_pred == true_label:
                cases["wrong_b_correct_cseq"].append(record)

    rng = random.Random(cfg["project"]["seed"])
    sampled = {}
    n_per = cfg["evaluation"].get("n_manual_review", 50)
    for cat, records in cases.items():
        if len(records) > n_per:
            sampled[cat] = rng.sample(records, n_per)
        else:
            sampled[cat] = records
        logger.info(f"  {cat}: {len(records)} total, {len(sampled[cat])} sampled")

    save_results(sampled, out_dir, "case_studies.json")
    logger.info(f"Results → {out_dir}")
    return sampled


if __name__ == "__main__":
    run()
