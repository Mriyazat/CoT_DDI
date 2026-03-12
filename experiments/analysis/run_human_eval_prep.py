#!/usr/bin/env python3
"""Prepare human evaluation spreadsheet (CPU only).
Usage: python experiments/analysis/run_human_eval_prep.py
"""

import os, sys, json, random, csv
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.exp_utils import (
    get_config, get_exp_output_dir, setup_exp_logging,
    load_all_condition_predictions, align_predictions_by_key, load_label_map,
)


def run():
    out_dir = get_exp_output_dir("human_eval")
    logger = setup_exp_logging("human_eval", out_dir)
    cfg = get_config()
    label_map = load_label_map()
    n_eval = cfg["evaluation"].get("n_manual_review", 50)

    conditions = load_all_condition_predictions()
    reasoning_conds = [c for c in sorted(conditions.keys())
                       if c not in ("A_zero_shot", "B_label_only")]

    if not reasoning_conds:
        logger.warning("No reasoning conditions found")
        return

    cnames = reasoning_conds
    aligned = align_predictions_by_key(*[conditions[c] for c in cnames])

    if not aligned:
        logger.warning("No aligned pairs")
        return

    eval_rows = []
    for group in aligned:
        true_label = group[0]["true_label"]
        d1 = group[0].get("drug1_name", group[0]["drug1_id"])
        d2 = group[0].get("drug2_name", group[0]["drug2_id"])
        true_text = label_map.get(true_label, f"Y={true_label}")
        row = {"Drug_1": d1, "Drug_2": d2,
               "True_Label": true_label, "True_Text": true_text[:100]}
        for i, c in enumerate(cnames):
            row[f"{c}_response"] = group[i].get("response", "")[:500]
            row[f"{c}_pred"] = group[i]["pred_label"]
        eval_rows.append(row)

    rng = random.Random(cfg["project"]["seed"])
    if len(eval_rows) > n_eval:
        eval_rows = rng.sample(eval_rows, n_eval)
    logger.info(f"Prepared {len(eval_rows)} evaluation pairs")

    csv_path = str(out_dir / "human_eval_sheet.csv")
    if eval_rows:
        fieldnames = list(eval_rows[0].keys())
        for c in cnames:
            for suffix in ["_reasoning_score", "_factual_score", "_notes"]:
                fieldnames.append(f"{c}{suffix}")

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in eval_rows:
                writer.writerow(row)

    json_path = str(out_dir / "human_eval_data.json")
    with open(json_path, "w") as f:
        json.dump(eval_rows, f, indent=2, default=str)

    logger.info(f"CSV → {csv_path}")
    logger.info(f"JSON → {json_path}")
    return eval_rows


if __name__ == "__main__":
    run()
