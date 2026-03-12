#!/usr/bin/env python3
"""Per-category F1 breakdown across conditions (CPU only).
Usage: python experiments/analysis/run_category_analysis.py
"""

import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.exp_utils import (
    get_config, get_exp_output_dir, setup_exp_logging, save_results,
    load_all_condition_predictions, load_label_map, per_category_f1,
)
from src.utils import categorize_interaction, LABEL_CATEGORY_GROUPS


def run():
    out_dir = get_exp_output_dir("category_analysis")
    logger = setup_exp_logging("category_analysis", out_dir)
    cfg = get_config()

    conditions = load_all_condition_predictions()
    label_map = load_label_map()

    all_cat_results = {}
    for cname, preds in conditions.items():
        valid = [(r["true_label"], r["pred_label"]) for r in preds if r["pred_label"] >= 0]
        if not valid:
            continue
        vt, vp = zip(*valid)
        per_label = per_category_f1(list(vt), list(vp), label_map)

        # Aggregate by category group
        cat_agg = {cat: {"f1s": [], "n": 0} for cat in LABEL_CATEGORY_GROUPS}
        for lbl, info in per_label.items():
            cat = categorize_interaction(info.get("label_text", ""))
            cat_agg[cat]["f1s"].append(info["f1"])
            cat_agg[cat]["n"] += 1

        cat_means = {}
        for cat, vals in cat_agg.items():
            if vals["f1s"]:
                cat_means[cat] = {"mean_f1": float(np.mean(vals["f1s"])),
                                  "n_labels": vals["n"]}
        all_cat_results[cname] = cat_means
        logger.info(f"  {cname}: {len(per_label)} labels across {len(cat_means)} categories")

    save_results(all_cat_results, out_dir, "category_analysis.json")

    # Bar chart
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    cnames = list(all_cat_results.keys())
    cats = list(LABEL_CATEGORY_GROUPS.keys())

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(cats))
    w = 0.8 / max(len(cnames), 1)
    for i, cname in enumerate(cnames):
        vals = [all_cat_results[cname].get(c, {}).get("mean_f1", 0) for c in cats]
        ax.bar(x + i * w, vals, w, label=cname, alpha=0.8)

    ax.set_xticks(x + w * len(cnames) / 2)
    ax.set_xticklabels([LABEL_CATEGORY_GROUPS[c] for c in cats], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean F1")
    ax.set_title("Per-Category F1 Across Conditions")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(str(fig_dir / "category_comparison.png"), dpi=200)
    plt.close()

    logger.info(f"Results → {out_dir}")
    return all_cat_results


if __name__ == "__main__":
    run()
