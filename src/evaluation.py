"""
Phase 4 – Multi-task evaluation for DDI CoT Distillation V3.

Tasks:
  1. Fine-grained classification (150 classes): Macro/Micro F1, accuracy
  2. Severity prediction (3-class): on DDInter-labeled subset
  3. Coarse category classification (12 categories): post-hoc mapping
  4. Mechanism entity extraction: precision/recall vs DrugBank ground truth

Reasoning quality:
  - BERTScore, ROUGE-L against teacher traces
  - ECE (Expected Calibration Error)
  - Prompt robustness (3 paraphrased variants)

Supports JSONL-based inference resume for crash recovery.
"""

import os
import re
import gc
import json
import logging
import time
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix,
)

from src.utils import load_config, setup_logging, set_seed, ensure_dirs
from src.data_preparation import SYSTEM_PROMPT, build_student_input


# ── Label / severity / entity extraction from output ──────────────────

def extract_label(text: str) -> int:
    matches = re.findall(r"Y\s*=\s*(\d+)", text)
    if matches:
        return int(matches[-1])
    m = re.search(r"Classification\s*:\s*(\d+)", text, re.IGNORECASE)
    return int(m.group(1)) if m else -1


def extract_severity(text: str) -> str:
    m = re.search(r"Severity\s*:\s*(Major|Moderate|Minor|Unknown)", text, re.IGNORECASE)
    return m.group(1).capitalize() if m else "Unknown"


MECH_ENTITY_PATTERN = re.compile(
    r"(CYP\d[A-Z]\d{1,2}|CYP\d+|P-glycoprotein|P-gp|OATP\w+|OCT\d|"
    r"BCRP|MRP\d|MATE\d|UGT\w+|SULT\w+|NAT\d|"
    r"[A-Z]{2,4}\d?[A-Z]?\d?\s+receptor|"
    r"serotonin|dopamine|norepinephrine|GABA|glutamate|acetylcholine|"
    r"beta-\d|alpha-\d|mu-opioid|kappa-opioid|"
    r"MAO-[AB]|COX-[12]|PDE\d|HMGCR|ACE)",
    re.IGNORECASE,
)


def extract_mechanism_entities(text: str) -> set[str]:
    return {m.upper() for m in MECH_ENTITY_PATTERN.findall(text)}


# ── Prediction generation with resume ─────────────────────────────────

def predict_finetuned(cfg: dict, checkpoint_dir: str, condition_name: str,
                      test_path: str = None):
    """Generate predictions for all test pairs with JSONL resume."""
    from vllm import LLM, SamplingParams

    logger = setup_logging(f"eval_{condition_name}")
    set_seed(cfg["project"]["seed"])

    processed = cfg["data"]["processed_dir"]
    if test_path is None:
        test_path = os.path.join(processed, "test.jsonl")
    test_df = pd.read_json(test_path, lines=True)

    with open(os.path.join(processed, "drug_profiles.json")) as f:
        profiles = json.load(f)

    retrieved = {}
    retr_path = os.path.join(processed, "retrieved_examples_test.json")
    if os.path.exists(retr_path):
        with open(retr_path) as f:
            raw = json.load(f)
        for k, v in raw.items():
            retrieved[int(k)] = v
        logger.info(f"  Loaded {len(retrieved):,} test retrievals")
    else:
        logger.warning("  No retrieved_examples_test.json found -- prompts will lack few-shot examples")

    results_dir = os.path.join(cfg["project"]["output_dir"], "results")
    os.makedirs(results_dir, exist_ok=True)
    pred_file = os.path.join(results_dir, f"predictions_{condition_name}.jsonl")

    done = set()
    if os.path.exists(pred_file):
        with open(pred_file) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["idx"])
                except (json.JSONDecodeError, KeyError):
                    continue

    remaining = test_df[~test_df.index.isin(done)]
    logger.info(f"Condition: {condition_name}")
    logger.info(f"  Test pairs: {len(test_df):,}, done: {len(done):,}, "
                f"remaining: {len(remaining):,}")

    if len(remaining) == 0:
        logger.info("All predictions complete.")
        return pred_file

    ecfg = cfg["evaluation"]
    scfg = cfg["student"]
    model_name = scfg["model_name"]
    lora_r = int(scfg.get("lora", {}).get("r", 64))
    tp_size = int(os.environ.get("VLLM_TP_SIZE", ecfg.get("tensor_parallel_size", 1)))
    logger.info(f"  vLLM tensor_parallel_size={tp_size}")

    llm = LLM(
        model=model_name,
        enable_lora=True,
        max_lora_rank=lora_r,
        tensor_parallel_size=tp_size,
        dtype=scfg["dtype"],
        max_model_len=scfg["max_length"],
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    from vllm.lora.request import LoRARequest
    lora_req = LoRARequest("student", 1, checkpoint_dir)

    tokenizer = llm.get_tokenizer()
    params = SamplingParams(
        temperature=0.1, top_p=0.95,
        max_tokens=ecfg.get("max_new_tokens", 512),
    )

    batch_size = ecfg.get("batch_size", 64)
    t_start = time.time()

    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining.iloc[batch_start:batch_start + batch_size]
        prompts = []
        for orig_idx, row in batch.iterrows():
            retr_examples = retrieved.get(orig_idx)
            user_msg = build_student_input(row, profiles, retr_examples)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            try:
                prompts.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                ))
            except TypeError:
                prompts.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                ))

        outputs = llm.generate(prompts, params, lora_request=lora_req)

        with open(pred_file, "a") as f:
            for (orig_idx, row), out in zip(batch.iterrows(), outputs):
                text = out.outputs[0].text.strip()
                pred_label = extract_label(text)
                pred_severity = extract_severity(text)
                pred_entities = list(extract_mechanism_entities(text))
                f.write(json.dumps({
                    "idx": int(orig_idx),
                    "label": int(row["label"]),
                    "pred_label": pred_label,
                    "drug1_id": str(row.get("drug1_id", "")),
                    "drug2_id": str(row.get("drug2_id", "")),
                    "drug1_name": str(row.get("drug1_name", "")),
                    "drug2_name": str(row.get("drug2_name", "")),
                    "severity": str(row.get("severity", "Unknown")),
                    "pred_severity": pred_severity,
                    "pred_entities": pred_entities,
                    "output": text,
                }) + "\n")

        n_done = len(done) + batch_start + len(batch)
        if (batch_start // batch_size + 1) % 20 == 0:
            elapsed = time.time() - t_start
            rate = (batch_start + len(batch)) / elapsed
            logger.info(f"  {n_done:,}/{len(test_df):,} | {rate:.1f} pairs/s")

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"  Predictions saved to {pred_file}")
    return pred_file


def _iter_prediction_rows(pred_file: str, logger=None):
    """Yield valid JSON prediction rows, skipping malformed lines safely."""
    skipped = 0
    with open(pred_file) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            yield obj
    if skipped and logger is not None:
        logger.warning(f"  Skipped {skipped} malformed prediction rows in {pred_file}")


# ── Task 1: Fine-grained classification ──────────────────────────────

def evaluate_classification(pred_file: str, label_map: dict, logger):
    """Compute Macro/Micro F1, accuracy for 150-class classification."""
    y_true, y_pred = [], []
    n_total = 0
    for obj in _iter_prediction_rows(pred_file, logger):
        n_total += 1
        if obj["pred_label"] >= 0:
            y_true.append(obj["label"])
            y_pred.append(obj["pred_label"])
    n_parsed = len(y_true)
    parse_rate = 100 * n_parsed / n_total if n_total else 0

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    logger.info(f"  Classification (150-class):")
    logger.info(f"    Parse rate: {parse_rate:.1f}% ({n_parsed}/{n_total})")
    logger.info(f"    Macro F1:   {macro_f1:.4f}")
    logger.info(f"    Micro F1:   {micro_f1:.4f}")
    logger.info(f"    Accuracy:   {acc:.4f}")

    return {
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "accuracy": round(acc, 4),
        "parse_rate": round(parse_rate, 1),
        "n_parsed": n_parsed,
        "n_total": n_total,
    }


# ── Task 2: Severity prediction ──────────────────────────────────────

def evaluate_severity(pred_file: str, logger):
    """Evaluate severity prediction on DDInter-labeled subset."""
    y_true, y_pred = [], []
    sev_labels = {"Major": 0, "Moderate": 1, "Minor": 2}

    for obj in _iter_prediction_rows(pred_file, logger):
        gt_sev = obj.get("severity", "Unknown")
        if gt_sev in sev_labels:
            pred_sev = obj.get("pred_severity", "Unknown")
            if pred_sev in sev_labels:
                y_true.append(sev_labels[gt_sev])
                y_pred.append(sev_labels[pred_sev])

    if not y_true:
        logger.warning("  Severity: no DDInter-labeled predictions found")
        return {}

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    label_names = ["Major", "Moderate", "Minor"]

    logger.info(f"  Severity prediction ({len(y_true)} pairs):")
    logger.info(f"    Macro F1:  {macro_f1:.4f}")
    logger.info(f"    Accuracy:  {acc:.4f}")

    per_class = f1_score(y_true, y_pred, average=None, zero_division=0,
                         labels=[0, 1, 2])
    for i, name in enumerate(label_names):
        logger.info(f"    {name}: F1={per_class[i]:.4f}")

    return {
        "severity_macro_f1": round(macro_f1, 4),
        "severity_accuracy": round(acc, 4),
        "severity_n": len(y_true),
        "per_class_f1": {name: round(per_class[i], 4) for i, name in enumerate(label_names)},
    }


# ── Task 3: Coarse category classification ───────────────────────────

def evaluate_coarse(pred_file: str, coarse_map: dict, logger):
    """Evaluate coarse category accuracy via label -> coarse mapping."""
    y_true_coarse, y_pred_coarse = [], []
    all_categories = sorted(set(coarse_map.values()))
    cat_to_id = {c: i for i, c in enumerate(all_categories)}

    for obj in _iter_prediction_rows(pred_file, logger):
        if obj["pred_label"] < 0:
            continue
        gt_cat = coarse_map.get(str(obj["label"]))
        pred_cat = coarse_map.get(str(obj["pred_label"]))
        if gt_cat and pred_cat:
            y_true_coarse.append(cat_to_id[gt_cat])
            y_pred_coarse.append(cat_to_id[pred_cat])

    if not y_true_coarse:
        logger.warning("  Coarse: no valid predictions")
        return {}

    macro_f1 = f1_score(y_true_coarse, y_pred_coarse, average="macro",
                        zero_division=0)
    acc = accuracy_score(y_true_coarse, y_pred_coarse)

    logger.info(f"  Coarse category ({len(all_categories)} categories):")
    logger.info(f"    Macro F1:  {macro_f1:.4f}")
    logger.info(f"    Accuracy:  {acc:.4f}")

    return {
        "coarse_macro_f1": round(macro_f1, 4),
        "coarse_accuracy": round(acc, 4),
        "n_categories": len(all_categories),
    }


# ── Task 4: Mechanism entity extraction ───────────────────────────────

def evaluate_entities(pred_file: str, profiles: dict, logger):
    """Compare extracted entities against DrugBank ground truth."""
    total_precision_num, total_precision_den = 0, 0
    total_recall_num, total_recall_den = 0, 0
    n_pairs = 0

    for obj in _iter_prediction_rows(pred_file, logger):
        pred_ents = set(e.upper() for e in obj.get("pred_entities", []))
        if not pred_ents:
            continue

        gt_ents = set()
        for did_key in ("drug1_id", "drug2_id"):
            did = obj.get(did_key, "")
            prof = profiles.get(did, {})
            for field in ("enzymes", "transporters", "targets"):
                for item in prof.get(field, []):
                    for token in re.findall(r"CYP\w+|P-gp|P-glycoprotein|\w+", item):
                        if len(token) > 2:
                            gt_ents.add(token.upper())

        if not gt_ents:
            continue

        hits = pred_ents & gt_ents
        total_precision_num += len(hits)
        total_precision_den += len(pred_ents)
        total_recall_num += len(hits)
        total_recall_den += len(gt_ents)
        n_pairs += 1

    precision = total_precision_num / total_precision_den if total_precision_den else 0
    recall = total_recall_num / total_recall_den if total_recall_den else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    logger.info(f"  Mechanism entity extraction ({n_pairs} pairs):")
    logger.info(f"    Precision: {precision:.4f}")
    logger.info(f"    Recall:    {recall:.4f}")
    logger.info(f"    F1:        {f1:.4f}")

    return {
        "entity_precision": round(precision, 4),
        "entity_recall": round(recall, 4),
        "entity_f1": round(f1, 4),
        "entity_n_pairs": n_pairs,
    }


def evaluate_grounded_entity_precision(pred_file: str, profiles: dict, logger):
    """Grounded Entity Precision: fraction of entity mentions in reasoning
    that are actually present in the drug pair's KB profile.

    Unlike evaluate_entities which uses regex extraction, this metric scans
    the full output text for any drug/enzyme/transporter name from the
    specific pair's profile and checks if the model hallucinated entities
    not grounded in the knowledge base.
    """
    grounded_total, mentioned_total = 0, 0
    n_pairs = 0
    per_profile_results = {"rich": [], "sparse": [], "empty": []}

    for obj in _iter_prediction_rows(pred_file, logger):
        output = obj.get("output", "")
        if not output or obj.get("pred_label", -1) < 0:
            continue

        pair_entities = set()
        profile_field_count = 0
        for did_key in ("drug1_id", "drug2_id"):
            did = obj.get(did_key, "")
            prof = profiles.get(did, {})
            for field in ("enzymes", "transporters", "targets"):
                items = prof.get(field, [])
                profile_field_count += len(items)
                for raw in items:
                    name = raw.split("(")[0].split(":")[0].strip()
                    if len(name) >= 3:
                        pair_entities.add(name.upper())

        output_upper = output.upper()
        mentioned = set()
        for ent in pair_entities:
            if ent in output_upper:
                mentioned.add(ent)

        regex_ents = extract_mechanism_entities(output)
        all_mentioned = mentioned | regex_ents
        if not all_mentioned:
            continue

        grounded = sum(1 for e in all_mentioned if e in pair_entities)
        grounded_total += grounded
        mentioned_total += len(all_mentioned)
        n_pairs += 1

        gep = grounded / len(all_mentioned)
        if profile_field_count >= 4:
            per_profile_results["rich"].append(gep)
        elif profile_field_count >= 1:
            per_profile_results["sparse"].append(gep)
        else:
            per_profile_results["empty"].append(gep)

    overall_gep = grounded_total / mentioned_total if mentioned_total > 0 else 0.0

    logger.info(f"  Grounded Entity Precision ({n_pairs} pairs):")
    logger.info(f"    Overall GEP: {overall_gep:.4f}")
    for tier, vals in per_profile_results.items():
        if vals:
            logger.info(f"    {tier} profile ({len(vals)}): "
                        f"GEP={np.mean(vals):.4f}")

    result = {
        "grounded_entity_precision": round(overall_gep, 4),
        "gep_n_pairs": n_pairs,
    }
    for tier, vals in per_profile_results.items():
        if vals:
            result[f"gep_{tier}_profile"] = round(float(np.mean(vals)), 4)
            result[f"gep_{tier}_n"] = len(vals)
    return result


# ── Reasoning quality (BERTScore, ROUGE-L) ────────────────────────────

def evaluate_reasoning(pred_file: str, cot_traces: dict, cfg: dict, logger):
    """BERTScore and ROUGE-L against teacher traces."""
    pairs = []
    for obj in _iter_prediction_rows(pred_file, logger):
        idx = obj["idx"]
        if idx in cot_traces and obj.get("output"):
            pairs.append((obj["output"], cot_traces[idx]))

    if not pairs:
        logger.warning("  No matched pairs for reasoning evaluation")
        return {}

    sample_size = min(5000, len(pairs))
    import random
    random.seed(cfg["project"]["seed"])
    pairs = random.sample(pairs, sample_size)
    preds, refs = zip(*pairs)

    results = {}

    try:
        from bert_score import score as bert_score
        P, R, F = bert_score(
            list(preds), list(refs),
            model_type=cfg["evaluation"].get("bertscore_model",
                                             "microsoft/deberta-xlarge-mnli"),
            batch_size=32,
        )
        results["bertscore_f1"] = round(F.mean().item(), 4)
        logger.info(f"  BERTScore F1: {results['bertscore_f1']}")
    except ImportError:
        logger.warning("  bert_score not installed, skipping")

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rl_scores = [scorer.score(r, p)["rougeL"].fmeasure
                     for p, r in zip(preds, refs)]
        results["rouge_l"] = round(np.mean(rl_scores), 4)
        logger.info(f"  ROUGE-L:      {results['rouge_l']}")
    except ImportError:
        logger.warning("  rouge_score not installed, skipping")

    return results


# ── ECE (Expected Calibration Error) ─────────────────────────────────

def evaluate_ece(pred_file: str, n_bins: int = 15, logger=None):
    """ECE from structural confidence: how many expected output fields are present."""
    confidences, correct = [], []
    for obj in _iter_prediction_rows(pred_file, logger):
        if obj["pred_label"] < 0:
            continue
        output = obj.get("output", "")
        signals = 0
        total_signals = 4
        if re.search(r"Y\s*=\s*\d+", output):
            signals += 1
        if re.search(r"Severity\s*:\s*(Major|Moderate|Minor)", output, re.IGNORECASE):
            signals += 1
        if re.search(r"Summary\s*:", output, re.IGNORECASE):
            signals += 1
        if re.search(r"Classification\s*:", output, re.IGNORECASE):
            signals += 1
        conf = signals / total_signals
        confidences.append(conf)
        correct.append(1 if obj["pred_label"] == obj["label"] else 0)

    if not confidences:
        return {"ece": None}

    confidences = np.array(confidences)
    correct = np.array(correct)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() > 0:
            acc = correct[mask].mean()
            conf = confidences[mask].mean()
            ece += mask.sum() / len(confidences) * abs(acc - conf)

    if logger:
        logger.info(f"  ECE ({n_bins} bins): {ece:.4f}")
    return {"ece": round(ece, 4)}


def evaluate_ece_calibrated(cfg: dict, checkpoint_dir: str,
                            condition_name: str, n_bins: int = 15,
                            logger=None):
    """Temperature-scaled ECE using vLLM logprobs on the predicted label token.

    Learns an optimal temperature T on a calibration split, then reports
    ECE on the test split with softmax(logits/T).
    """
    if logger is None:
        logger = logging.getLogger("ece_calibrated")

    scfg = cfg["student"]
    model_name = scfg["model_name"]
    processed = cfg["data"]["processed_dir"]

    output_dir = os.path.join(cfg["evaluation"]["output_dir"], condition_name)
    logprob_path = os.path.join(output_dir, "label_logprobs.jsonl")

    test_path = os.path.join(processed, "test_cot.jsonl")
    test_df = pd.read_json(test_path, lines=True)

    if os.path.exists(logprob_path):
        existing = sum(1 for _ in open(logprob_path))
        logger.info(f"Resuming logprob collection: {existing} rows exist")
    else:
        existing = 0

    if existing < len(test_df):
        try:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
            from src.data_preparation import SYSTEM_PROMPT, build_student_input

            lora_r = scfg["lora"]["r"]
            llm = LLM(
                model=model_name, enable_lora=True,
                max_lora_rank=lora_r, tensor_parallel_size=1,
                trust_remote_code=True, max_model_len=scfg["max_length"],
            )
            lora_req = LoRARequest("student_lora", 1, checkpoint_dir)
            sampling_params = SamplingParams(
                temperature=0, max_tokens=512, logprobs=5,
            )

            with open(os.path.join(processed, "drug_profiles.json")) as f:
                profiles = json.load(f)
            retrieved = {}
            retr_path = os.path.join(processed, "retrieved_examples_test.json")
            if os.path.exists(retr_path):
                with open(retr_path) as f:
                    retrieved = json.load(f)

            tokenizer = llm.get_tokenizer()
            label_re = re.compile(r"Y\s*=\s*(\d+)")

            with open(logprob_path, "a") as fout:
                for idx, row in test_df.iterrows():
                    if idx < existing:
                        continue
                    retr_idx = retrieved.get(str(idx), [])
                    retr_examples = [r for r in retr_idx if isinstance(r, dict)] if retr_idx else None
                    user_msg = build_student_input(row, profiles, retr_examples)
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ]
                    try:
                        prompt = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    except TypeError:
                        prompt = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True,
                        )
                    outputs = llm.generate([prompt], sampling_params, lora_request=lora_req)
                    text = outputs[0].outputs[0].text

                    m = label_re.search(text)
                    pred_label = int(m.group(1)) if m else -1

                    max_logprob = None
                    if outputs[0].outputs[0].logprobs:
                        for step_lp in outputs[0].outputs[0].logprobs:
                            if step_lp:
                                for tok_id, lp_obj in step_lp.items():
                                    decoded = lp_obj.decoded_token if hasattr(lp_obj, 'decoded_token') else tokenizer.decode([tok_id])
                                    if decoded.strip().isdigit():
                                        lp_val = lp_obj.logprob if hasattr(lp_obj, 'logprob') else lp_obj
                                        if max_logprob is None or lp_val > max_logprob:
                                            max_logprob = lp_val

                    fout.write(json.dumps({
                        "idx": int(idx),
                        "gold_label": int(row["label"]),
                        "pred_label": pred_label,
                        "label_logprob": max_logprob,
                    }) + "\n")

        except ImportError:
            logger.error("vLLM not available for calibrated ECE")
            return {}

    gold_labels, pred_labels, logprobs = [], [], []
    with open(logprob_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj["pred_label"] < 0 or obj["label_logprob"] is None:
                continue
            gold_labels.append(obj["gold_label"])
            pred_labels.append(obj["pred_label"])
            logprobs.append(obj["label_logprob"])

    if len(logprobs) < 10:
        logger.warning(f"Only {len(logprobs)} valid logprobs; skipping calibrated ECE")
        return {}

    logprobs = np.array(logprobs)
    correct = np.array([1 if g == p else 0 for g, p in zip(gold_labels, pred_labels)])

    n = len(logprobs)
    cal_size = n // 5
    cal_lp, test_lp = logprobs[:cal_size], logprobs[cal_size:]
    cal_correct, test_correct = correct[:cal_size], correct[cal_size:]

    best_T, best_nll = 1.0, float("inf")
    for T in np.linspace(0.1, 5.0, 50):
        probs = np.exp(cal_lp / T)
        probs = np.clip(probs, 1e-10, 1.0)
        nll = -np.mean(cal_correct * np.log(probs) + (1 - cal_correct) * np.log(1 - probs))
        if nll < best_nll:
            best_nll = nll
            best_T = T

    calibrated_probs = np.exp(test_lp / best_T)
    calibrated_probs = np.clip(calibrated_probs, 0, 1)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece_cal = 0.0
    for i in range(n_bins):
        mask = (calibrated_probs > bin_edges[i]) & (calibrated_probs <= bin_edges[i + 1])
        if mask.sum() > 0:
            acc = test_correct[mask].mean()
            conf = calibrated_probs[mask].mean()
            ece_cal += mask.sum() / len(test_correct) * abs(acc - conf)

    result = {
        "ece_calibrated": round(float(ece_cal), 4),
        "calibration_temperature": round(float(best_T), 3),
    }
    logger.info(f"  Calibrated ECE: {result['ece_calibrated']:.4f} (T={result['calibration_temperature']:.3f})")
    return result


def evaluate_self_consistency(cfg: dict, checkpoint_dir: str,
                              condition_name: str, n_samples: int = 5,
                              temperature: float = 0.7, logger=None):
    """Self-consistency: generate N samples per test input, measure agreement.

    Returns agreement rate (fraction of inputs where majority vote == mode)
    and majority-vote accuracy (where mode == gold label).
    """
    if logger is None:
        logger = logging.getLogger("self_consistency")

    scfg = cfg["student"]
    model_name = scfg["model_name"]
    processed = cfg["data"]["processed_dir"]

    output_dir = os.path.join(cfg["evaluation"]["output_dir"], condition_name)
    sc_path = os.path.join(output_dir, "self_consistency.jsonl")

    if os.path.exists(sc_path):
        existing = sum(1 for _ in open(sc_path))
        logger.info(f"Resuming self-consistency: {existing} rows exist")
    else:
        existing = 0

    test_path = os.path.join(processed, "test_cot.jsonl")
    test_df = pd.read_json(test_path, lines=True)

    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        from src.data_preparation import SYSTEM_PROMPT, build_student_input

        lora_r = scfg["lora"]["r"]
        llm = LLM(
            model=model_name, enable_lora=True,
            max_lora_rank=lora_r, tensor_parallel_size=1,
            trust_remote_code=True, max_model_len=scfg["max_length"],
        )
        lora_req = LoRARequest("student_lora", 1, checkpoint_dir)
        sampling_params = SamplingParams(
            n=n_samples, temperature=temperature,
            max_tokens=512, top_p=0.95,
        )

        with open(os.path.join(processed, "drug_profiles.json")) as f:
            profiles = json.load(f)
        retrieved = {}
        retr_path = os.path.join(processed, "retrieved_examples_test.json")
        if os.path.exists(retr_path):
            with open(retr_path) as f:
                retrieved = json.load(f)

        tokenizer = llm.get_tokenizer()
        label_re = re.compile(r"Y\s*=\s*(\d+)")

        with open(sc_path, "a") as fout:
            for idx, row in test_df.iterrows():
                if idx < existing:
                    continue
                retr_idx = retrieved.get(str(idx), [])
                retr_examples = [r for r in retr_idx if isinstance(r, dict)] if retr_idx else None
                user_msg = build_student_input(row, profiles, retr_examples)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ]
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )

                outputs = llm.generate([prompt], sampling_params, lora_request=lora_req)

                preds = []
                for out in outputs[0].outputs:
                    m = label_re.search(out.text)
                    preds.append(int(m.group(1)) if m else -1)

                row_data = {
                    "idx": int(idx),
                    "gold_label": int(row["label"]),
                    "predictions": preds,
                }
                fout.write(json.dumps(row_data) + "\n")

    except ImportError:
        logger.error("vLLM not available for self-consistency evaluation")
        return {}

    agreements, majority_correct = [], []
    with open(sc_path) as f:
        for line in f:
            obj = json.loads(line)
            preds = [p for p in obj["predictions"] if p >= 0]
            if not preds:
                continue
            counts = Counter(preds)
            mode_label, mode_count = counts.most_common(1)[0]
            agreements.append(mode_count / len(preds))
            majority_correct.append(1 if mode_label == obj["gold_label"] else 0)

    result = {}
    if agreements:
        result["sc_agreement"] = round(float(np.mean(agreements)), 4)
        result["sc_majority_acc"] = round(float(np.mean(majority_correct)), 4)
        logger.info(f"  Self-consistency agreement: {result['sc_agreement']:.4f}")
        logger.info(f"  Majority-vote accuracy:     {result['sc_majority_acc']:.4f}")
    return result


def evaluate_efficiency(cfg: dict, checkpoint_dir: str,
                        condition_name: str, n_warmup: int = 5,
                        n_bench: int = 50, logger=None):
    """Benchmark inference latency and throughput."""
    if logger is None:
        logger = logging.getLogger("efficiency")

    scfg = cfg["student"]
    model_name = scfg["model_name"]
    processed = cfg["data"]["processed_dir"]

    test_path = os.path.join(processed, "test_cot.jsonl")
    test_df = pd.read_json(test_path, lines=True).head(n_warmup + n_bench)

    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        from src.data_preparation import SYSTEM_PROMPT, build_student_input

        lora_r = scfg["lora"]["r"]
        llm = LLM(
            model=model_name, enable_lora=True,
            max_lora_rank=lora_r, tensor_parallel_size=1,
            trust_remote_code=True, max_model_len=scfg["max_length"],
        )
        lora_req = LoRARequest("student_lora", 1, checkpoint_dir)
        sampling_params = SamplingParams(
            temperature=0, max_tokens=512,
        )

        with open(os.path.join(processed, "drug_profiles.json")) as f:
            profiles = json.load(f)
        retrieved = {}
        retr_path = os.path.join(processed, "retrieved_examples_test.json")
        if os.path.exists(retr_path):
            with open(retr_path) as f:
                retrieved = json.load(f)

        tokenizer = llm.get_tokenizer()
        prompts = []
        for idx, row in test_df.iterrows():
            retr_idx = retrieved.get(str(idx), [])
            retr_examples = [r for r in retr_idx if isinstance(r, dict)] if retr_idx else None
            user_msg = build_student_input(row, profiles, retr_examples)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            try:
                prompts.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                ))
            except TypeError:
                prompts.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                ))

        for p in prompts[:n_warmup]:
            llm.generate([p], sampling_params, lora_request=lora_req)

        bench_prompts = prompts[n_warmup:]
        t0 = time.time()
        outputs = llm.generate(bench_prompts, sampling_params, lora_request=lora_req)
        elapsed = time.time() - t0

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        result = {
            "latency_per_example_ms": round(elapsed / len(bench_prompts) * 1000, 1),
            "throughput_tok_per_sec": round(total_tokens / elapsed, 1),
            "avg_output_tokens": round(total_tokens / len(bench_prompts), 1),
            "n_bench": len(bench_prompts),
        }
        logger.info(f"  Latency: {result['latency_per_example_ms']} ms/example")
        logger.info(f"  Throughput: {result['throughput_tok_per_sec']} tok/s")
        logger.info(f"  Avg output: {result['avg_output_tokens']} tokens")
        return result

    except ImportError:
        logger.error("vLLM not available for efficiency benchmark")
        return {}


# ── Main evaluation pipeline ─────────────────────────────────────────

def run_evaluation(cfg: dict, condition_name: str, checkpoint_dir: str):
    """Full multi-task evaluation for one condition."""
    logger = setup_logging(f"evaluation_{condition_name}")
    processed = cfg["data"]["processed_dir"]

    logger.info(f"{'=' * 60}")
    logger.info(f"Evaluation: {condition_name}")
    logger.info(f"{'=' * 60}")

    # Always call prediction with resume logic so partial files continue
    # instead of being silently treated as complete.
    pred_file = predict_finetuned(cfg, checkpoint_dir, condition_name)

    with open(os.path.join(processed, "label_map.json")) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    coarse_map = {}
    cm_path = os.path.join(processed, "coarse_category_map.json")
    if os.path.exists(cm_path):
        with open(cm_path) as f:
            coarse_map = json.load(f)

    profiles = {}
    prof_path = os.path.join(processed, "drug_profiles.json")
    if os.path.exists(prof_path):
        with open(prof_path) as f:
            profiles = json.load(f)

    cot_traces = {}
    trace_sources = [
        os.path.join(processed, "test_cot.jsonl"),
        os.path.join(processed, "cot_traces.jsonl"),
    ]
    loaded_trace_source = None
    for p in trace_sources:
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    obj = json.loads(line)
                    idx = obj.get("idx", -1)
                    trace = obj.get("teacher_cot", "") or obj.get("reasoning", "")
                    if idx not in cot_traces and trace:
                        cot_traces[idx] = trace
            loaded_trace_source = os.path.basename(p)
            break
    if loaded_trace_source:
        logger.info(f"  Loaded {len(cot_traces):,} reasoning references from {loaded_trace_source}")
    else:
        logger.warning("  No test-side reasoning references found; skipping BERTScore/ROUGE.")

    all_results = {"condition": condition_name}

    all_results.update(evaluate_classification(pred_file, label_map, logger))
    all_results.update(evaluate_severity(pred_file, logger))
    all_results.update(evaluate_coarse(pred_file, coarse_map, logger))
    all_results.update(evaluate_entities(pred_file, profiles, logger))
    if profiles and condition_name != "B_label":
        all_results.update(evaluate_grounded_entity_precision(pred_file, profiles, logger))
    all_results.update(evaluate_ece(pred_file, logger=logger))

    if cot_traces and condition_name != "B_label":
        all_results.update(evaluate_reasoning(pred_file, cot_traces, cfg, logger))

    try:
        logger.info("\n--- Calibrated ECE ---")
        all_results.update(evaluate_ece_calibrated(
            cfg, checkpoint_dir, condition_name, logger=logger))
    except Exception as e:
        logger.warning(f"Calibrated ECE failed: {e}")

    try:
        logger.info("\n--- Efficiency benchmark ---")
        all_results.update(evaluate_efficiency(
            cfg, checkpoint_dir, condition_name, logger=logger))
    except Exception as e:
        logger.warning(f"Efficiency benchmark failed: {e}")

    results_dir = os.path.join(cfg["project"]["output_dir"], "results")
    report_path = os.path.join(results_dir, f"eval_report_{condition_name}.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nFull report saved to {report_path}")

    return all_results


def compare_conditions(cfg: dict, conditions: list[str]):
    """Print comparison table across all evaluated conditions and save CSV."""
    logger = setup_logging("eval_comparison")
    results_dir = os.path.join(cfg["project"]["output_dir"], "results")

    rows = []
    for cond in conditions:
        path = os.path.join(results_dir, f"eval_report_{cond}.json")
        if os.path.exists(path):
            with open(path) as f:
                rows.append(json.load(f))

    if not rows:
        logger.warning("No evaluation reports found")
        return

    cols = [
        ("Macro F1", "macro_f1"), ("Micro F1", "micro_f1"),
        ("Accuracy", "accuracy"), ("Sev F1", "severity_macro_f1"),
        ("Coarse F1", "coarse_macro_f1"), ("Ent F1", "entity_f1"),
        ("GEP", "grounded_entity_precision"),
        ("ECE", "ece"), ("ECE-Cal", "ece_calibrated"),
        ("BERTSc", "bertscore_f1"), ("ROUGE", "rouge_l"),
        ("Latency", "latency_per_example_ms"),
    ]

    header = f"{'Condition':<22}" + "".join(f"{c[0]:>10}" for c in cols)
    logger.info(f"\n{header}")
    logger.info("-" * (22 + 10 * len(cols)))
    for r in rows:
        line = f"{r.get('condition', '?'):<22}"
        for _, key in cols:
            val = r.get(key, "-")
            if isinstance(val, float):
                line += f"{val:>10.4f}"
            else:
                line += f"{str(val):>10}"
        logger.info(line)

    csv_path = os.path.join(results_dir, "ablation_comparison.csv")
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    logger.info(f"\nComparison CSV saved to {csv_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-task evaluation")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--compare", nargs="*",
                        help="Compare multiple conditions")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Override student model (for multi-scale experiments)")
    parser.add_argument("--self-consistency", action="store_true",
                        help="Run self-consistency evaluation (N=5 samples)")
    parser.add_argument("--sc-samples", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.model_name:
        cfg["student"]["model_name"] = args.model_name

    if args.compare:
        compare_conditions(cfg, args.compare)
    else:
        results = run_evaluation(cfg, args.condition, args.checkpoint)
        if args.self_consistency:
            sc_results = evaluate_self_consistency(
                cfg, args.checkpoint, args.condition,
                n_samples=args.sc_samples,
            )
            results.update(sc_results)
            results_dir = os.path.join(cfg["project"]["output_dir"], "results")
            report_path = os.path.join(results_dir, f"eval_report_{args.condition}.json")
            with open(report_path, "w") as f:
                json.dump(results, f, indent=2)
