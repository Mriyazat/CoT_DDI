"""
Baseline models: zero-shot LLM, Random Forest, XGBoost on Morgan fingerprints.
"""

import os
import json
import re
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.utils import load_config, setup_logging, set_seed, ensure_dirs, gpu_info
from src.data_preparation import SYSTEM_PROMPT, build_student_input
from src.evaluation import extract_label


def evaluate_zero_shot(cfg: dict):
    from vllm import LLM, SamplingParams

    logger = setup_logging("baseline_zeroshot")
    set_seed(cfg["project"]["seed"])
    ensure_dirs(cfg)

    test_path = os.path.join(cfg["data"]["processed_dir"], "test.jsonl")
    test_df = pd.read_json(test_path, lines=True)
    logger.info(f"Test set: {len(test_df):,} pairs")

    model_name = cfg["student"]["model_name"]
    logger.info(f"Loading {model_name} via vLLM for zero-shot evaluation")

    llm = LLM(
        model=model_name, dtype="bfloat16", max_model_len=2048,
        gpu_memory_utilization=0.85, trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    params = SamplingParams(
        temperature=0.1, top_p=0.9,
        max_tokens=cfg["evaluation"]["max_new_tokens"],
    )

    label_map_path = os.path.join(cfg["data"]["processed_dir"], "label_map.json")
    label_hint = ""
    if os.path.exists(label_map_path):
        with open(label_map_path) as f:
            label_map = json.load(f)
        label_hint = "\n\nValid interaction types (Y values): " + ", ".join(
            f"Y={k}" for k in sorted(label_map.keys(), key=int)
        )

    prompts = []
    for _, row in test_df.iterrows():
        user_msg = (
            build_student_input(row)
            + "\n\nIMPORTANT: End your response with exactly this format: "
            "Classification: Y=<number> — \"<interaction description>\""
            + label_hint
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    logger.info(f"Generating responses for {len(prompts):,} pairs …")
    batch_size = cfg["evaluation"]["batch_size"]
    all_outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Zero-shot"):
        all_outputs.extend(llm.generate(prompts[i:i + batch_size], params))

    results = []
    for (_, row), out in zip(test_df.iterrows(), all_outputs):
        text = out.outputs[0].text.strip()
        results.append({
            "drug1_id": row["drug1_id"],
            "drug2_id": row["drug2_id"],
            "true_label": int(row["label"]),
            "pred_label": extract_label(text),
            "response": text,
        })

    out_path = os.path.join(cfg["project"]["output_dir"], "results", "zeroshot_predictions.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    preds = [r["pred_label"] for r in results]
    trues = [r["true_label"] for r in results]
    valid = [(t, p) for t, p in zip(trues, preds) if p >= 0]
    if valid:
        t_valid, p_valid = zip(*valid)
        macro = f1_score(t_valid, p_valid, average="macro", zero_division=0)
        micro = f1_score(t_valid, p_valid, average="micro", zero_division=0)
        logger.info(f"Zero-shot — Macro F1: {macro:.4f} | Micro F1: {micro:.4f}")
        logger.info(f"  Valid: {len(valid)} / {len(results)} ({100*len(valid)/len(results):.1f}%)")

    logger.info(f"Saved to {out_path}")
    return out_path


def _smiles_to_fingerprint(smiles: str, n_bits: int = 2048, radius: int = 2):
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError:
        return np.zeros(n_bits, dtype=np.int8)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.int8)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return gen.GetFingerprintAsNumPy(mol).astype(np.int8)


def _encode_fingerprints(df, n_bits, radius):
    fp1 = np.stack([_smiles_to_fingerprint(s, n_bits, radius) for s in tqdm(df["drug1_smiles"], desc="FP1")])
    fp2 = np.stack([_smiles_to_fingerprint(s, n_bits, radius) for s in tqdm(df["drug2_smiles"], desc="FP2")])
    return np.concatenate([fp1, fp2], axis=1)


def train_ml_baseline(cfg: dict):
    logger = setup_logging("baseline_ml")
    try:
        from rdkit import Chem
    except ImportError:
        logger.warning("rdkit not installed — skipping ML baseline")
        return {"model": "RandomForest", "macro_f1": 0, "skipped": True}

    set_seed(cfg["project"]["seed"])
    ml_cfg = cfg["baseline_ml"]

    train_df = pd.read_json(os.path.join(cfg["data"]["processed_dir"], "train.jsonl"), lines=True)
    test_df = pd.read_json(os.path.join(cfg["data"]["processed_dir"], "test.jsonl"), lines=True)

    X_train = _encode_fingerprints(train_df, ml_cfg["fingerprint_bits"], ml_cfg["fingerprint_radius"])
    X_test = _encode_fingerprints(test_df, ml_cfg["fingerprint_bits"], ml_cfg["fingerprint_radius"])
    y_train, y_test = train_df["label"].values, test_df["label"].values

    rf = RandomForestClassifier(
        n_estimators=ml_cfg["n_estimators"], max_depth=ml_cfg["max_depth"],
        n_jobs=ml_cfg["n_jobs"], random_state=cfg["project"]["seed"],
        class_weight="balanced",
    )
    t0 = time.time()
    rf.fit(X_train, y_train)
    logger.info(f"RF trained in {time.time()-t0:.1f}s")

    y_pred = rf.predict(X_test)
    macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    micro = f1_score(y_test, y_pred, average="micro", zero_division=0)
    logger.info(f"Random Forest — Macro F1: {macro:.4f} | Micro F1: {micro:.4f}")

    results = {"model": "RandomForest", "macro_f1": float(macro), "micro_f1": float(micro)}
    res_path = os.path.join(cfg["project"]["output_dir"], "results", "ml_baseline_results.json")
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def train_xgboost_baseline(cfg: dict):
    logger = setup_logging("baseline_xgb")
    if not HAS_XGBOOST:
        logger.warning("xgboost not installed")
        return {"model": "XGBoost", "macro_f1": 0, "skipped": True}
    try:
        from rdkit import Chem
    except ImportError:
        logger.warning("rdkit not installed")
        return {"model": "XGBoost", "macro_f1": 0, "skipped": True}

    set_seed(cfg["project"]["seed"])
    ml_cfg = cfg["baseline_ml"]

    train_df = pd.read_json(os.path.join(cfg["data"]["processed_dir"], "train.jsonl"), lines=True)
    test_df = pd.read_json(os.path.join(cfg["data"]["processed_dir"], "test.jsonl"), lines=True)

    X_train = _encode_fingerprints(train_df, ml_cfg["fingerprint_bits"], ml_cfg["fingerprint_radius"])
    X_test = _encode_fingerprints(test_df, ml_cfg["fingerprint_bits"], ml_cfg["fingerprint_radius"])
    y_train, y_test = train_df["label"].values, test_df["label"].values

    label_min = int(y_train.min())
    y_train_z, y_test_z = y_train - label_min, y_test - label_min

    xgb = XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
        num_class=len(np.unique(y_train_z)), n_jobs=-1,
        random_state=cfg["project"]["seed"], tree_method="hist",
    )
    t0 = time.time()
    xgb.fit(X_train, y_train_z)
    logger.info(f"XGBoost trained in {time.time()-t0:.1f}s")

    y_pred = xgb.predict(X_test) + label_min
    macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    micro = f1_score(y_test, y_pred, average="micro", zero_division=0)
    logger.info(f"XGBoost — Macro F1: {macro:.4f} | Micro F1: {micro:.4f}")

    results = {"model": "XGBoost", "macro_f1": float(macro), "micro_f1": float(micro)}
    res_path = os.path.join(cfg["project"]["output_dir"], "results", "xgb_baseline_results.json")
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    return results
