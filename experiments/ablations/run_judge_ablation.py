#!/usr/bin/env python3
"""Judge filtering ablation — analyzes existing judge scores (CPU only).
Usage: python experiments/ablations/run_judge_ablation.py
"""

import os, sys, json, itertools
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.exp_utils import get_config, get_exp_output_dir, setup_exp_logging, save_results


def _load_judge_scores(cfg):
    traces_dir = os.path.join(cfg["project"]["output_dir"], "teacher_traces")
    all_scores = {}
    for jcfg in cfg["judge"]["models"]:
        short = jcfg["model_name"].split("/")[-1]
        path = os.path.join(traces_dir, f"full_judge_scores_{short}.jsonl")
        if not os.path.exists(path):
            continue
        scores = {}
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                key = rec.get("idx", f"{rec.get('drug1_id', '')}_{rec.get('drug2_id', '')}")
                scores[key] = rec
        all_scores[short] = scores
    return all_scores


def _apply_consensus(pair_scores, strategy, min_score=3):
    overalls = [s.get("overall", 0) for s in pair_scores if s.get("overall", 0) > 0]
    if not overalls:
        return False
    if strategy == "all_pass":
        return all(o >= min_score for o in overalls)
    elif strategy == "majority":
        return sum(1 for o in overalls if o >= min_score) > len(overalls) / 2
    elif strategy == "any_pass":
        return any(o >= min_score for o in overalls)
    elif strategy == "average":
        return np.mean(overalls) >= min_score
    return False


def run():
    out_dir = get_exp_output_dir("judge_ablation")
    logger = setup_exp_logging("judge_ablation", out_dir)
    cfg = get_config()

    all_scores = _load_judge_scores(cfg)
    judge_names = list(all_scores.keys())
    logger.info(f"Loaded {len(judge_names)} judges: {judge_names}")

    if not judge_names:
        logger.error("No judge score files found.")
        return

    all_pairs = set()
    for scores in all_scores.values():
        all_pairs |= set(scores.keys())
    total = len(all_pairs)
    results = {}

    strategy = cfg["judge"]["consensus_strategy"]
    min_score = cfg["judge"]["min_overall_score"]

    for n_j in range(1, len(judge_names) + 1):
        for combo in itertools.combinations(judge_names, n_j):
            n_pass = sum(1 for pk in all_pairs
                         if _apply_consensus([all_scores[j][pk] for j in combo
                                              if pk in all_scores[j]], strategy, min_score))
            results[f"n{n_j}_{'_'.join(sorted(combo))}"] = {
                "n_judges": n_j, "judges": list(combo),
                "n_pass": n_pass, "pass_rate": round(100 * n_pass / total, 2),
            }
            logger.info(f"  {n_j} judges [{'+'.join(combo)}]: {n_pass:,}/{total:,} pass")

    for strat in ["all_pass", "majority", "any_pass", "average"]:
        n_pass = sum(1 for pk in all_pairs
                     if _apply_consensus([all_scores[j][pk] for j in judge_names
                                          if pk in all_scores[j]], strat, min_score))
        results[f"strategy_{strat}"] = {"strategy": strat, "n_pass": n_pass,
                                         "pass_rate": round(100 * n_pass / total, 2)}
        logger.info(f"  {strat}: {n_pass:,}/{total:,}")

    for thresh in [2, 3, 4]:
        n_pass = sum(1 for pk in all_pairs
                     if _apply_consensus([all_scores[j][pk] for j in judge_names
                                          if pk in all_scores[j]], strategy, thresh))
        results[f"threshold_{thresh}"] = {"min_score": thresh, "n_pass": n_pass,
                                           "pass_rate": round(100 * n_pass / total, 2)}

    save_results(results, out_dir, "judge_ablation_results.json")
    logger.info(f"Results saved to {out_dir}")
    return results


if __name__ == "__main__":
    run()
