"""
Baseline models for DDI prediction (129-class classification).

Tier 1: Trivial baselines (random, majority, stratified)
Tier 2: Traditional ML on molecular fingerprints + pharmacological features
        - Morgan FP + XGBoost / Random Forest / MLP
        - MACCS keys + XGBoost
        - Pharmacological features + XGBoost
        - Combined FP+Pharma + XGBoost

All baselines evaluate on the same test set as the student model.
"""

import os
import json
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix,
)
from sklearn.preprocessing import MultiLabelBinarizer

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import MACCSkeys
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

from src.utils import load_config, setup_logging, set_seed, ensure_dirs


def _load_data(cfg):
    """Load train/test JSONL and drug profiles."""
    processed = cfg["data"]["processed_dir"]
    train_df = pd.read_json(os.path.join(processed, "train.jsonl"), lines=True)
    test_df = pd.read_json(os.path.join(processed, "test.jsonl"), lines=True)
    with open(os.path.join(processed, "drug_profiles.json")) as f:
        profiles = json.load(f)
    return train_df, test_df, profiles


def _compute_metrics(y_true, y_pred, name, logger):
    """Compute and log standard metrics."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    logger.info(f"{name} -- Acc: {acc:.4f} | Macro-F1: {macro_f1:.4f} | "
                f"Weighted-F1: {weighted_f1:.4f} | Micro-F1: {micro_f1:.4f}")

    return {
        "model": name,
        "accuracy": round(float(acc), 5),
        "macro_f1": round(float(macro_f1), 5),
        "weighted_f1": round(float(weighted_f1), 5),
        "micro_f1": round(float(micro_f1), 5),
    }


# ── Tier 1: Trivial baselines ───────────────────────────────────────

def trivial_baselines(y_train, y_test, logger):
    """Random, majority-class, and stratified-random baselines."""
    rng = np.random.RandomState(42)
    n_classes = len(set(y_train))
    results = []

    y_rand = rng.randint(0, n_classes, size=len(y_test))
    results.append(_compute_metrics(y_test, y_rand, "Random (uniform)", logger))

    majority = Counter(y_train).most_common(1)[0][0]
    y_maj = np.full(len(y_test), majority)
    results.append(_compute_metrics(y_test, y_maj, "Majority class", logger))

    counts = Counter(y_train)
    total = sum(counts.values())
    labels = sorted(counts.keys())
    probs = [counts[l] / total for l in labels]
    y_strat = rng.choice(labels, size=len(y_test), p=probs)
    results.append(_compute_metrics(y_test, y_strat, "Stratified random", logger))

    return results


# ── Feature engineering ──────────────────────────────────────────────

_morgan_gen_cache = {}

def _smiles_to_morgan(smiles, n_bits=2048, radius=2):
    if not smiles or not HAS_RDKIT:
        return np.zeros(n_bits, dtype=np.uint8)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    key = (n_bits, radius)
    if key not in _morgan_gen_cache:
        _morgan_gen_cache[key] = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits)
    return _morgan_gen_cache[key].GetFingerprintAsNumPy(mol).astype(np.uint8)


def _smiles_to_maccs(smiles):
    if not smiles or not HAS_RDKIT:
        return np.zeros(167, dtype=np.uint8)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(167, dtype=np.uint8)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp, dtype=np.uint8)


def _precompute_fp_cache(profiles, n_bits=2048, radius=2):
    """Compute fingerprints once per drug, return a lookup dict."""
    cache = {}
    for did, p in profiles.items():
        cache[did] = _smiles_to_morgan(p.get("smiles", ""), n_bits, radius)
    return cache


def build_morgan_features(df, profiles, n_bits=2048, radius=2):
    """Concatenate Morgan FP of drug1 and drug2 using cached per-drug FPs."""
    cache = _precompute_fp_cache(profiles, n_bits, radius)
    zero = np.zeros(n_bits, dtype=np.uint8)
    fp1 = np.stack([cache.get(did, zero) for did in df["drug1_id"]])
    fp2 = np.stack([cache.get(did, zero) for did in df["drug2_id"]])
    return np.concatenate([fp1, fp2], axis=1)


def build_maccs_features(df, profiles):
    """Concatenate MACCS keys of drug1 and drug2 using cached per-drug FPs."""
    cache = {}
    for did, p in profiles.items():
        cache[did] = _smiles_to_maccs(p.get("smiles", ""))
    zero = np.zeros(167, dtype=np.uint8)
    fp1 = np.stack([cache.get(did, zero) for did in df["drug1_id"]])
    fp2 = np.stack([cache.get(did, zero) for did in df["drug2_id"]])
    return np.concatenate([fp1, fp2], axis=1)


def _collect_vocab(profiles, field):
    """Collect all unique values for a pharmacological field across all drugs."""
    vocab = set()
    for p in profiles.values():
        for item in p.get(field, []):
            vocab.add(item.strip().lower())
    return sorted(vocab)


def build_pharma_features(df, profiles):
    """Multi-hot encode targets, enzymes, transporters, categories for both drugs."""
    fields = ["targets", "enzymes", "transporters", "categories"]
    vocabs = {f: _collect_vocab(profiles, f) for f in fields}
    item_to_idx = {}
    offset = 0
    for f in fields:
        for i, item in enumerate(vocabs[f]):
            item_to_idx[(f, item)] = offset + i
        offset += len(vocabs[f])

    total_dim = offset

    drug_vectors = {}
    for did, p in profiles.items():
        vec = np.zeros(total_dim, dtype=np.float32)
        for f in fields:
            for item in p.get(f, []):
                key = (f, item.strip().lower())
                if key in item_to_idx:
                    vec[item_to_idx[key]] = 1.0
        drug_vectors[did] = vec

    zero = np.zeros(total_dim, dtype=np.float32)
    v1 = np.stack([drug_vectors.get(did, zero) for did in df["drug1_id"].values])
    v2 = np.stack([drug_vectors.get(did, zero) for did in df["drug2_id"].values])
    return np.concatenate([v1, v2], axis=1)


# ── Tier 2: Traditional ML ──────────────────────────────────────────

def train_and_evaluate(X_train, y_train, X_test, y_test, model, name, logger):
    """Train a model, predict, and compute metrics."""
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    logger.info(f"{name}: trained in {train_time:.1f}s on {X_train.shape} features")

    y_pred = model.predict(X_test)
    result = _compute_metrics(y_test, y_pred, name, logger)
    result["train_seconds"] = round(train_time, 1)
    result["n_features"] = X_train.shape[1]
    return result, y_pred


def ml_baselines(cfg, train_df, test_df, profiles, logger, skip=0):
    """Run all Tier 2 ML baselines. skip=N to skip first N models."""
    ml_cfg = cfg["baseline_ml"]
    seed = cfg["project"]["seed"]
    results = []

    y_train = train_df["label"].values
    y_test = test_df["label"].values
    label_min = int(y_train.min())

    logger.info("Building Morgan fingerprint features...")
    X_train_fp = build_morgan_features(
        train_df, profiles, ml_cfg["fingerprint_bits"], ml_cfg["fingerprint_radius"])
    X_test_fp = build_morgan_features(
        test_df, profiles, ml_cfg["fingerprint_bits"], ml_cfg["fingerprint_radius"])

    logger.info("Building MACCS key features...")
    X_train_maccs = build_maccs_features(train_df, profiles)
    X_test_maccs = build_maccs_features(test_df, profiles)

    logger.info("Building pharmacological features...")
    X_train_pharma = build_pharma_features(train_df, profiles)
    X_test_pharma = build_pharma_features(test_df, profiles)

    logger.info("Building combined FP+Pharma features...")
    X_train_combined = np.concatenate([X_train_fp, X_train_pharma], axis=1)
    X_test_combined = np.concatenate([X_test_fp, X_test_pharma], axis=1)

    logger.info(f"Feature dims: Morgan={X_train_fp.shape[1]}, MACCS={X_train_maccs.shape[1]}, "
                f"Pharma={X_train_pharma.shape[1]}, Combined={X_train_combined.shape[1]}")

    configs = [
        ("Morgan FP + XGBoost", X_train_fp, X_test_fp, "xgb"),
        ("Morgan FP + RF", X_train_fp, X_test_fp, "rf"),
        ("Morgan FP + MLP", X_train_fp, X_test_fp, "mlp"),
        ("MACCS + XGBoost", X_train_maccs, X_test_maccs, "xgb"),
        ("Pharma + XGBoost", X_train_pharma, X_test_pharma, "xgb"),
        ("FP+Pharma + XGBoost", X_train_combined, X_test_combined, "xgb"),
    ]

    all_preds = {}
    for idx, (name, X_tr, X_te, model_type) in enumerate(configs, 1):
        if idx <= skip:
            logger.info(f"\n[{idx}/{len(configs)}] Skipping {name} (already completed)")
            continue
        logger.info(f"\n[{idx}/{len(configs)}] Training {name}  ({X_tr.shape[0]:,} x {X_tr.shape[1]:,}) ...")
        if model_type == "xgb":
            if not HAS_XGBOOST:
                logger.warning(f"Skipping {name}: xgboost not installed")
                continue
            n_classes = len(np.unique(y_train))
            model = XGBClassifier(
                n_estimators=ml_cfg["n_estimators"],
                max_depth=ml_cfg["max_depth"],
                learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                objective="multi:softmax", num_class=n_classes,
                n_jobs=ml_cfg["n_jobs"], random_state=seed, tree_method="hist",
                verbosity=1,
            )
            y_tr = y_train - label_min
            r, y_pred_shifted = train_and_evaluate(X_tr, y_tr, X_te, y_test - label_min, model, name, logger)
            all_preds[name] = y_pred_shifted + label_min
            r_actual = _compute_metrics(y_test, all_preds[name], name, logger)
            r_actual["train_seconds"] = r["train_seconds"]
            r_actual["n_features"] = r["n_features"]
            results.append(r_actual)
        elif model_type == "rf":
            model = RandomForestClassifier(
                n_estimators=ml_cfg["n_estimators"], max_depth=ml_cfg["max_depth"],
                n_jobs=ml_cfg["n_jobs"], random_state=seed, class_weight="balanced",
                verbose=1,
            )
            r, y_pred = train_and_evaluate(X_tr, y_train, X_te, y_test, model, name, logger)
            all_preds[name] = y_pred
            results.append(r)
        elif model_type == "mlp":
            model = MLPClassifier(
                hidden_layer_sizes=(512, 256), max_iter=50, early_stopping=True,
                random_state=seed, verbose=True,
            )
            r, y_pred = train_and_evaluate(X_tr, y_train, X_te, y_test, model, name, logger)
            all_preds[name] = y_pred
            results.append(r)

    return results, all_preds


# ── Main ─────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip the first N ML models (use to resume)")
    args = parser.parse_args()

    cfg = load_config(args.config if hasattr(args, 'config') else None)
    logger = setup_logging("baselines")
    set_seed(cfg["project"]["seed"])
    ensure_dirs(cfg)

    logger.info("=== DDI Baseline Evaluation ===")
    train_df, test_df, profiles = _load_data(cfg)
    logger.info(f"Train: {len(train_df):,} | Test: {len(test_df):,} | "
                f"Profiles: {len(profiles):,} | Classes: {train_df['label'].nunique()}")

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    all_results = []

    logger.info("\n--- Tier 1: Trivial baselines ---")
    trivial = trivial_baselines(y_train, y_test, logger)
    all_results.extend(trivial)

    if HAS_RDKIT:
        logger.info("\n--- Tier 2: ML baselines ---")
        ml_results, ml_preds = ml_baselines(cfg, train_df, test_df, profiles, logger, skip=args.skip)
        all_results.extend(ml_results)
    else:
        logger.warning("rdkit not installed -- skipping ML baselines")
        ml_preds = {}

    res_dir = os.path.join(cfg["project"]["output_dir"], "results")
    os.makedirs(res_dir, exist_ok=True)

    res_path = os.path.join(res_dir, "baseline_results.json")
    with open(res_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {res_path}")

    if ml_preds:
        pred_path = os.path.join(res_dir, "baseline_predictions.pkl")
        with open(pred_path, "wb") as f:
            pickle.dump({"y_test": y_test, "predictions": ml_preds}, f)
        logger.info(f"Predictions saved to {pred_path}")

    logger.info("\n=== Summary ===")
    logger.info(f"{'Method':<30} {'Acc':>8} {'Macro-F1':>10} {'Wtd-F1':>10}")
    logger.info("-" * 60)
    for r in all_results:
        logger.info(f"{r['model']:<30} {r['accuracy']:>8.4f} {r['macro_f1']:>10.4f} {r['weighted_f1']:>10.4f}")


if __name__ == "__main__":
    main()
