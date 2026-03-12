#!/usr/bin/env python3
"""ECE and calibration analysis for all conditions (CPU only).
Usage: python experiments/analysis/run_ece_calibration.py
"""

import os, sys, json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.exp_utils import (
    get_config, get_exp_output_dir, setup_exp_logging,
    save_results, load_all_condition_predictions,
)
from src.evaluation import compute_ece


def run():
    out_dir = get_exp_output_dir("ece_calibration")
    logger = setup_exp_logging("ece_calibration", out_dir)
    cfg = get_config()
    fig_dir = str(out_dir / "figures")

    conditions = load_all_condition_predictions()
    all_results = {}

    for cname, preds in conditions.items():
        pred_path = str(out_dir / f"_{cname}_tmp.jsonl")
        with open(pred_path, "w") as f:
            for r in preds:
                f.write(json.dumps(r) + "\n")

        ece_result = compute_ece(pred_path, cfg, cname, fig_dir=fig_dir)
        all_results[cname] = ece_result
        ece_val = ece_result.get("ece", ece_result.get("ece_proxy", -1))
        logger.info(f"  {cname}: ECE = {ece_val:.4f}")

    save_results(all_results, out_dir, "ece_results.json")
    logger.info(f"Results → {out_dir}")
    return all_results


if __name__ == "__main__":
    run()
