#!/usr/bin/env python3
"""Statistical tests: McNemar's, bootstrap CI, Cohen's kappa (CPU only).
Usage: python experiments/analysis/run_statistical_tests.py
"""

import os, sys, json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.exp_utils import (
    get_config, get_exp_output_dir, setup_exp_logging, save_results,
    load_all_condition_predictions, align_predictions_by_key,
    bootstrap_ci, macro_f1_score, load_label_map,
)


def _mcnemar_test(preds_a, preds_b, y_true):
    """McNemar's test for paired predictions — fixed length alignment."""
    n01 = sum(1 for pa, pb, yt in zip(preds_a, preds_b, y_true)
              if pa == yt and pb != yt)
    n10 = sum(1 for pa, pb, yt in zip(preds_a, preds_b, y_true)
              if pa != yt and pb == yt)

    if n01 + n10 < 25:
        from math import factorial
        n = n01 + n10
        p = 0.0
        for k in range(n01, n + 1):
            p += factorial(n) / (factorial(k) * factorial(n - k)) * (0.5 ** n)
        return {
            "test": "exact_mcnemar", "n01": n01, "n10": n10,
            "p_value": min(2 * p, 1.0),
            "significant_05": bool(min(2 * p, 1.0) < 0.05),
            "significant_01": bool(min(2 * p, 1.0) < 0.01),
            "significant_001": bool(min(2 * p, 1.0) < 0.001),
        }

    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if (n01 + n10) > 0 else 0
    from scipy.stats import chi2 as chi2_dist
    p = 1 - chi2_dist.cdf(chi2, df=1)
    return {
        "test": "mcnemar_cc", "n01": n01, "n10": n10,
        "chi2": float(chi2), "p_value": float(p),
        "significant_05": bool(p < 0.05),
        "significant_01": bool(p < 0.01),
        "significant_001": bool(p < 0.001),
    }


def _cohens_kappa(preds_a, preds_b):
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(preds_a, preds_b))


def run():
    out_dir = get_exp_output_dir("statistical_tests")
    logger = setup_exp_logging("statistical_tests", out_dir)
    cfg = get_config()

    conditions = load_all_condition_predictions()
    cond_names = sorted(conditions.keys())
    logger.info(f"Found {len(cond_names)} conditions: {cond_names}")

    if len(cond_names) < 2:
        logger.warning("Need at least 2 conditions for statistical tests")
        return

    results = {"bootstrap_ci": {}, "mcnemar": {}, "cohens_kappa": {}}

    # Bootstrap CI for each condition
    for name in cond_names:
        recs = conditions[name]
        valid = [(r["true_label"], r["pred_label"]) for r in recs if r["pred_label"] >= 0]
        if not valid:
            continue
        vt, vp = zip(*valid)
        ci = bootstrap_ci(list(vt), list(vp), macro_f1_score, n_bootstrap=2000)
        results["bootstrap_ci"][name] = ci
        logger.info(f"  {name}: Macro F1 = {ci['mean']:.4f} "
                     f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    # Pairwise McNemar's and Cohen's kappa — key-aligned
    for i, name_a in enumerate(cond_names):
        for name_b in cond_names[i+1:]:
            aligned = align_predictions_by_key(conditions[name_a], conditions[name_b])
            if not aligned:
                continue

            y_true = [a["true_label"] for a, _ in aligned]
            preds_a = [a["pred_label"] for a, _ in aligned]
            preds_b = [b["pred_label"] for _, b in aligned]

            # Filter to records where both have valid predictions
            triples = [(yt, pa, pb) for yt, pa, pb in zip(y_true, preds_a, preds_b)
                       if pa >= 0 and pb >= 0]
            if not triples:
                continue
            yt_f, pa_f, pb_f = zip(*triples)

            pair = f"{name_a}_vs_{name_b}"
            results["mcnemar"][pair] = _mcnemar_test(list(pa_f), list(pb_f), list(yt_f))
            results["cohens_kappa"][pair] = {"kappa": _cohens_kappa(list(pa_f), list(pb_f))}

            mc = results["mcnemar"][pair]
            logger.info(f"  {pair}: McNemar p={mc['p_value']:.4e} "
                         f"{'***' if mc['significant_001'] else '**' if mc['significant_01'] else '*' if mc['significant_05'] else 'ns'}")

    save_results(results, out_dir, "statistical_test_results.json")
    logger.info(f"Results → {out_dir}")
    return results


if __name__ == "__main__":
    run()
