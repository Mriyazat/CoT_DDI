"""
Shared utilities for all ACL experiments.
Data loading, metrics, checkpointing, bootstrap CI, result management.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import load_config


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_config() -> dict:
    return load_config()


def get_exp_output_dir(experiment_name: str) -> Path:
    d = get_project_root() / "outputs" / "experiments" / experiment_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup_exp_logging(name: str, output_dir: Path = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if output_dir:
        fh = logging.FileHandler(output_dir / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def load_ddi_split(split: str = "test") -> pd.DataFrame:
    cfg = get_config()
    path = os.path.join(cfg["data"]["processed_dir"], f"{split}.jsonl")
    return pd.read_json(path, lines=True)


def load_label_map() -> dict:
    cfg = get_config()
    path = os.path.join(cfg["data"]["processed_dir"], "label_map.json")
    with open(path) as f:
        return {int(k): v for k, v in json.load(f).items()}


def load_predictions(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_all_condition_predictions() -> dict[str, list[dict]]:
    cfg = get_config()
    res_dir = Path(cfg["project"]["output_dir"]) / "results"
    conditions = {}
    names = {
        "A_zero_shot": "zeroshot_predictions.jsonl",
        "B_label_only": "B_label_only_predictions.jsonl",
        "C_naive": "C_naive_predictions.jsonl",
        "C_seq": "C_seq_predictions.jsonl",
        "C_mix": "C_mix_predictions.jsonl",
        "D_real": "D_real_predictions.jsonl",
    }
    for cond, fname in names.items():
        p = res_dir / fname
        if p.exists():
            conditions[cond] = load_predictions(str(p))
    return conditions


def align_predictions_by_key(*pred_lists) -> list[tuple]:
    """Align multiple prediction lists by (drug1_id, drug2_id) key.
    Returns list of tuples of matched records."""
    if not pred_lists:
        return []
    key_maps = []
    for preds in pred_lists:
        km = {}
        for r in preds:
            key = (r["drug1_id"], r["drug2_id"])
            km[key] = r
        key_maps.append(km)

    common = set(key_maps[0].keys())
    for km in key_maps[1:]:
        common &= set(km.keys())

    return [tuple(km[k] for km in key_maps) for k in sorted(common)]


def compute_classification_metrics(y_true, y_pred, label_map=None) -> dict:
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p >= 0]
    if not valid:
        return {"macro_f1": 0, "micro_f1": 0, "accuracy": 0,
                "valid_pct": 0, "n_total": len(y_true)}

    vt, vp = zip(*valid)
    return {
        "macro_f1": float(f1_score(vt, vp, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(vt, vp, average="micro", zero_division=0)),
        "accuracy": float(accuracy_score(vt, vp)),
        "valid_pct": 100 * len(valid) / len(y_true),
        "n_total": len(y_true),
        "n_valid": len(valid),
    }


def per_category_f1(y_true, y_pred, label_map=None) -> dict:
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p >= 0]
    if not valid:
        return {}
    vt, vp = zip(*valid)
    labels = sorted(set(vt) | set(vp))
    scores = f1_score(vt, vp, labels=labels, average=None, zero_division=0)
    result = {}
    for lbl, sc in zip(labels, scores):
        name = label_map.get(lbl, str(lbl)) if label_map else str(lbl)
        result[lbl] = {"f1": float(sc), "label_text": name[:80]}
    return result


class ExperimentCheckpoint:
    def __init__(self, path: str):
        self.path = path
        self.done = set()
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    self.done.add(rec.get("_ckpt_key", ""))

    def is_done(self, key: str) -> bool:
        return key in self.done

    def save(self, key: str, data: dict):
        data["_ckpt_key"] = key
        with open(self.path, "a") as f:
            f.write(json.dumps(data) + "\n")
        self.done.add(key)


def save_results(results: dict, output_dir: Path, filename: str = "results.json"):
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return path


def save_predictions_jsonl(records: list[dict], path: str):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=1000,
                 ci=95, seed=42) -> dict:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        scores.append(metric_fn(yt, yp))

    alpha = (100 - ci) / 2
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "ci_lower": float(np.percentile(scores, alpha)),
        "ci_upper": float(np.percentile(scores, 100 - alpha)),
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
    }


def macro_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)
