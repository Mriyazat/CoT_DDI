"""

Classification : Macro/Micro F1, accuracy, per-category
Reasoning      : BERTScore, ROUGE-L, step structure, mechanism coverage
Calibration    : Expected Calibration Error (ECE), reliability diagram
Prompt         : Robustness across paraphrased prompts
Efficiency     : Tokens/sec, latency, VRAM
Two-stage      : Real D_real inference (reasoning → classification)
Judge          : LLM-as-a-judge with multi-dimensional rubric
"""

import os
import re
import gc
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score

from src.utils import (
    load_config, setup_logging, set_seed, ensure_dirs,
    categorize_interaction, LABEL_CATEGORY_GROUPS,
)
from src.data_preparation import SYSTEM_PROMPT, build_student_input




def extract_label(text: str) -> int:
    """Extract predicted label from model output. Prefers the LAST match
    to handle cases where the model mentions labels in reasoning."""
    matches = re.findall(r"Y\s*=\s*(\d+)", text)
    if matches:
        return int(matches[-1])
    m = re.search(r"Classification\s*:\s*(\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"[Tt]ype\s*[:=]\s*(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"[Ll]abel\s*[:=]\s*(\d+)", text)
    if m:
        return int(m.group(1))
    return -1



def predict_finetuned(cfg: dict, checkpoint_dir: str, condition_name: str,
                      custom_prompt_fn=None):
    """Run inference with LoRA-tuned student model on the test set."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    logger = setup_logging(f"eval_{condition_name}")
    set_seed(cfg["project"]["seed"])

    test_path = os.path.join(cfg["data"]["processed_dir"], "test.jsonl")
    test_df = pd.read_json(test_path, lines=True)

    base_model = cfg["student"]["model_name"]
    logger.info(f"Loading {base_model} + LoRA from {checkpoint_dir}")

    llm = LLM(
        model=base_model, dtype="bfloat16", max_model_len=4096,
        gpu_memory_utilization=0.85, enable_lora=True,
        max_lora_rank=cfg["student"]["lora"]["r"],
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    lora_req = LoRARequest(condition_name, 1, checkpoint_dir)

    params = SamplingParams(
        temperature=0.1, top_p=0.9,
        max_tokens=cfg["evaluation"]["max_new_tokens"],
    )

    prompt_fn = custom_prompt_fn or _default_prompt_fn
    prompts = [prompt_fn(row, tokenizer) for _, row in test_df.iterrows()]

    logger.info(f"Inference on {len(prompts):,} test pairs …")
    batch_size = cfg["evaluation"]["batch_size"]
    all_outputs = []
    total_tokens = 0
    t0 = time.time()

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        outs = llm.generate(batch, params, lora_request=lora_req)
        all_outputs.extend(outs)
        total_tokens += sum(len(o.outputs[0].token_ids) for o in outs)
        logger.info(f"  {min(i+batch_size, len(prompts))}/{len(prompts)}")

    elapsed = time.time() - t0
    tps = total_tokens / elapsed if elapsed > 0 else 0

    results = []
    for (_, row), out in zip(test_df.iterrows(), all_outputs):
        text = out.outputs[0].text.strip()
        results.append({
            "drug1_id": row["drug1_id"],
            "drug2_id": row["drug2_id"],
            "drug1_name": row.get("drug1_name", row["drug1_id"]),
            "drug2_name": row.get("drug2_name", row["drug2_id"]),
            "true_label": int(row["label"]),
            "pred_label": extract_label(text),
            "response": text,
        })

    out_dir = os.path.join(cfg["project"]["output_dir"], "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{condition_name}_predictions.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    logger.info(f"Throughput: {tps:.1f} tok/s | Elapsed: {elapsed:.1f}s | → {out_path}")

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out_path, {"tokens_per_sec": tps, "elapsed_sec": elapsed,
                      "total_tokens": total_tokens}


def _default_prompt_fn(row, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_student_input(row)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )



def compute_classification_metrics(predictions_path: str, label_map: dict,
                                   condition_name: str) -> dict:
    logger = setup_logging(f"metrics_{condition_name}")
    records = _load_records(predictions_path)

    trues = [r["true_label"] for r in records]
    preds = [r["pred_label"] for r in records]
    valid = [(t, p) for t, p in zip(trues, preds) if p >= 0]

    if not valid:
        return {"condition": condition_name, "macro_f1": 0, "micro_f1": 0,
                "accuracy": 0, "valid_pct": 0}

    t_v, p_v = zip(*valid)
    macro = f1_score(t_v, p_v, average="macro", zero_division=0)
    micro = f1_score(t_v, p_v, average="micro", zero_division=0)
    acc = accuracy_score(t_v, p_v)
    valid_pct = 100 * len(valid) / len(trues)

    logger.info(f"{condition_name}: Macro F1={macro:.4f} | Micro F1={micro:.4f} | "
                f"Acc={acc:.4f} | Valid={valid_pct:.1f}%")

    cat_scores = {}
    for t, p in valid:
        template = label_map.get(t, "")
        cat = categorize_interaction(template)
        if cat not in cat_scores:
            cat_scores[cat] = {"true": [], "pred": []}
        cat_scores[cat]["true"].append(t)
        cat_scores[cat]["pred"].append(p)

    cat_f1 = {}
    for cat, vals in cat_scores.items():
        cat_f1[cat] = float(f1_score(vals["true"], vals["pred"],
                                     average="micro", zero_division=0))

    return {
        "condition": condition_name,
        "macro_f1": float(macro), "micro_f1": float(micro),
        "accuracy": float(acc), "valid_pct": float(valid_pct),
        "category_f1": cat_f1,
    }



def evaluate_reasoning_quality(predictions_path: str, condition_name: str,
                               teacher_traces_path: str = None) -> dict:
    """Evaluate reasoning: structure, mechanism, BERTScore, ROUGE."""
    logger = setup_logging(f"reasoning_{condition_name}")
    records = _load_records(predictions_path)

    # Structure and mechanism analysis
    has_steps, has_mechanism = 0, 0
    step_counts, word_counts = [], []
    for r in records:
        text = r.get("response", "")
        steps = len(re.findall(r"[Ss]tep\s*\d", text))
        if steps > 0:
            has_steps += 1
            step_counts.append(steps)
        if re.search(r"(CYP|enzyme|receptor|inhibit|induc|metabol|pathway|transporter|P-gp|clearance)",
                     text, re.IGNORECASE):
            has_mechanism += 1
        word_counts.append(len(text.split()))

    n = len(records)
    metrics = {
        "condition": condition_name,
        "total": n,
        "step_structure_pct": 100 * has_steps / n if n else 0,
        "mechanism_mention_pct": 100 * has_mechanism / n if n else 0,
        "avg_steps": float(np.mean(step_counts)) if step_counts else 0,
        "avg_word_count": float(np.mean(word_counts)) if word_counts else 0,
    }

    # BERTScore and ROUGE against teacher reasoning
    if teacher_traces_path and os.path.exists(teacher_traces_path):
        teacher_map = _load_teacher_map(teacher_traces_path)
        student_texts, teacher_texts = [], []
        for r in records:
            key = f"{r['drug1_id']}_{r['drug2_id']}"
            if key in teacher_map:
                s_reason = _extract_reasoning(r.get("response", ""))
                t_reason = _extract_reasoning(teacher_map[key])
                if s_reason and t_reason:
                    student_texts.append(s_reason)
                    teacher_texts.append(t_reason)

        if student_texts:
            logger.info(f"Computing BERTScore on {len(student_texts):,} pairs …")
            try:
                from bert_score import score as bertscore_fn
                P, R, F1 = bertscore_fn(
                    student_texts, teacher_texts,
                    model_type="microsoft/deberta-xlarge-mnli",
                    batch_size=32, verbose=False,
                )
                metrics["bertscore_precision"] = float(P.mean())
                metrics["bertscore_recall"] = float(R.mean())
                metrics["bertscore_f1"] = float(F1.mean())
                logger.info(f"BERTScore F1: {metrics['bertscore_f1']:.4f}")
            except ImportError:
                logger.warning("bert-score not installed, skipping BERTScore")

            logger.info(f"Computing ROUGE-L on {len(student_texts):,} pairs …")
            try:
                from rouge_score import rouge_scorer
                scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
                rouge_scores = [scorer.score(t, s)["rougeL"].fmeasure
                                for s, t in zip(student_texts, teacher_texts)]
                metrics["rouge_l"] = float(np.mean(rouge_scores))
                logger.info(f"ROUGE-L: {metrics['rouge_l']:.4f}")
            except ImportError:
                logger.warning("rouge-score not installed, skipping ROUGE")

    logger.info(f"{condition_name}: steps={metrics['step_structure_pct']:.1f}% | "
                f"mechanism={metrics['mechanism_mention_pct']:.1f}%")
    return metrics


def _extract_reasoning(text: str) -> str:
    """Extract reasoning portion (before Classification: line)."""
    return re.sub(r"\n*Classification\s*:.*$", "", text,
                  flags=re.IGNORECASE).strip()


def _load_teacher_map(path: str) -> dict:
    teacher = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            key = f"{rec['drug1_id']}_{rec['drug2_id']}"
            teacher[key] = rec.get("teacher_cot", rec.get("cot", ""))
    return teacher


def compute_ece(predictions_path: str, cfg: dict, condition_name: str,
                fig_dir: str = None) -> dict:
    """Compute Expected Calibration Error from token log-probabilities.
    Requires predictions with 'confidence' field (added during inference)."""
    logger = setup_logging(f"ece_{condition_name}")
    records = _load_records(predictions_path)

    confidences = [r.get("confidence", -1) for r in records]
    if all(c < 0 for c in confidences):
        logger.info("No confidence scores available, computing from label frequency proxy")
        return _compute_ece_frequency_proxy(records, cfg, condition_name, fig_dir)

    n_bins = cfg["evaluation"].get("ece_n_bins", 15)
    valid = [(r["true_label"] == r["pred_label"], r["confidence"])
             for r in records if r["pred_label"] >= 0 and r.get("confidence", -1) >= 0]

    if not valid:
        return {"condition": condition_name, "ece": -1}

    corrects, confs = zip(*valid)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = [(lo <= c < hi) for c in confs]
        n_bin = sum(mask)
        if n_bin == 0:
            continue
        bin_acc = sum(c for c, m in zip(corrects, mask) if m) / n_bin
        bin_conf = sum(c for c, m in zip(confs, mask) if m) / n_bin
        ece += (n_bin / len(valid)) * abs(bin_acc - bin_conf)
        bin_data.append({"bin_mid": (lo + hi) / 2, "acc": bin_acc,
                         "conf": bin_conf, "count": n_bin})

    logger.info(f"{condition_name} ECE: {ece:.4f}")

    if fig_dir:
        _plot_reliability_diagram(bin_data, condition_name, fig_dir, ece)

    return {"condition": condition_name, "ece": float(ece), "bins": bin_data}


def _compute_ece_frequency_proxy(records, cfg, condition_name, fig_dir):
    """Proxy ECE using prediction frequency as confidence."""
    valid = [r for r in records if r["pred_label"] >= 0]
    if not valid:
        return {"condition": condition_name, "ece": -1}

    pred_counts = Counter(r["pred_label"] for r in valid)
    total = len(valid)
    for r in valid:
        r["confidence"] = pred_counts[r["pred_label"]] / total

    n_bins = cfg["evaluation"].get("ece_n_bins", 15)
    corrects = [r["true_label"] == r["pred_label"] for r in valid]
    confs = [r["confidence"] for r in valid]
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        idxs = [j for j, c in enumerate(confs) if lo <= c < hi]
        if not idxs:
            continue
        bin_acc = sum(corrects[j] for j in idxs) / len(idxs)
        bin_conf = sum(confs[j] for j in idxs) / len(idxs)
        ece += (len(idxs) / len(valid)) * abs(bin_acc - bin_conf)
        bin_data.append({"bin_mid": (lo + hi) / 2, "acc": bin_acc,
                         "conf": bin_conf, "count": len(idxs)})

    if fig_dir:
        _plot_reliability_diagram(bin_data, condition_name, fig_dir, ece)

    return {"condition": condition_name, "ece_proxy": float(ece), "bins": bin_data}


def _plot_reliability_diagram(bin_data, condition_name, fig_dir, ece):
    os.makedirs(fig_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    mids = [b["bin_mid"] for b in bin_data]
    accs = [b["acc"] for b in bin_data]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.bar(mids, accs, width=0.06, alpha=0.7, label=f"{condition_name} (ECE={ece:.3f})")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram — {condition_name}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"reliability_{condition_name}.png"), dpi=150)
    plt.close()



PROMPT_VARIANTS = [
    "Analyze the pharmacological interaction between these two drugs step-by-step and classify the interaction type.",
    "Given the molecular structures below, describe the drug-drug interaction mechanism and provide the classification.",
    "As a pharmacologist, examine these two drugs and their potential interaction. Explain your reasoning and classify.",
]


def evaluate_prompt_robustness(cfg: dict, checkpoint_dir: str,
                               condition_name: str) -> dict:
    """Test model with paraphrased prompts, measure F1 variance."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    logger = setup_logging(f"robustness_{condition_name}")
    test_path = os.path.join(cfg["data"]["processed_dir"], "test.jsonl")
    test_df = pd.read_json(test_path, lines=True)

    n_variants = min(cfg["evaluation"].get("prompt_variants", 3), len(PROMPT_VARIANTS))
    variants = PROMPT_VARIANTS[:n_variants]

    base_model = cfg["student"]["model_name"]
    llm = LLM(
        model=base_model, dtype="bfloat16", max_model_len=4096,
        gpu_memory_utilization=0.85, enable_lora=True,
        max_lora_rank=cfg["student"]["lora"]["r"], trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    lora_req = LoRARequest(condition_name, 1, checkpoint_dir)
    params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=256)

    all_f1s = []
    for vi, variant_instruction in enumerate(variants):
        prompts = []
        for _, row in test_df.iterrows():
            user_msg = (
                f"Drug 1: {row['drug1_name']} ({row['drug1_id']})\n"
                f"SMILES: {row['drug1_smiles']}\n"
                f"Drug 2: {row['drug2_name']} ({row['drug2_id']})\n"
                f"SMILES: {row['drug2_smiles']}\n\n"
                f"{variant_instruction}"
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))

        batch_sz = cfg["evaluation"]["batch_size"]
        preds = []
        for i in range(0, len(prompts), batch_sz):
            outs = llm.generate(prompts[i:i+batch_sz], params, lora_request=lora_req)
            for out in outs:
                preds.append(extract_label(out.outputs[0].text.strip()))

        y_true = test_df["label"].tolist()
        valid = [(t, p) for t, p in zip(y_true, preds) if p >= 0]
        if valid:
            vt, vp = zip(*valid)
            macro = f1_score(vt, vp, average="macro", zero_division=0)
        else:
            macro = 0.0
        all_f1s.append(macro)
        logger.info(f"  Variant {vi}: Macro F1={macro:.4f}")

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mean_f1 = float(np.mean(all_f1s))
    std_f1 = float(np.std(all_f1s))
    sensitivity = std_f1 / mean_f1 if mean_f1 > 0 else float("inf")

    logger.info(f"Prompt robustness: mean={mean_f1:.4f} std={std_f1:.4f} "
                f"sensitivity={sensitivity:.4f}")

    return {
        "condition": condition_name,
        "variant_f1s": all_f1s,
        "mean_f1": mean_f1, "std_f1": std_f1,
        "sensitivity_score": sensitivity,
    }



def predict_two_stage_real(cfg: dict, reasoning_checkpoint: str,
                           classify_checkpoint: str = None,
                           condition_name: str = "D_real",
                           n_samples: int = 1) -> str:
    """Two-stage inference: reasoning then classification, with self-consistency.

    The model generates reasoning sequentially followed by a classification.
    The two stages are coupled within a single generation pass, meaning the
    classification is explicitly conditioned on the preceding reasoning tokens.

    n_samples=1  → Single generation per test pair (equivalent to C_seq).
    n_samples>1  → Self-consistency: generate N diverse reasoning paths with
                   higher temperature, extract the classification from each,
                   and take majority vote.  This exploits reasoning diversity —
                   something pure label-only models (B) cannot benefit from.
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    logger = setup_logging(f"eval_{condition_name}")
    set_seed(cfg["project"]["seed"])

    test_path = os.path.join(cfg["data"]["processed_dir"], "test.jsonl")
    test_df = pd.read_json(test_path, lines=True)

    base_model = cfg["student"]["model_name"]
    lora_r = cfg["student"]["lora"]["r"]
    batch_sz = cfg["evaluation"]["batch_size"]
    max_tokens = cfg["evaluation"]["max_new_tokens"]

    llm = LLM(
        model=base_model, dtype="bfloat16", max_model_len=4096,
        gpu_memory_utilization=0.85, enable_lora=True,
        max_lora_rank=lora_r, trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    lora_req = LoRARequest("reasoning", 1, reasoning_checkpoint)

    prompts = [_default_prompt_fn(row, tokenizer) for _, row in test_df.iterrows()]

    if n_samples == 1:
        # Single pass — same as C_seq but saves reasoning + classification separately
        logger.info(f"Single-pass two-stage on {len(prompts):,} pairs …")
        params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=max_tokens)

        all_outputs = []
        for i in range(0, len(prompts), batch_sz):
            outs = llm.generate(prompts[i:i+batch_sz], params, lora_request=lora_req)
            all_outputs.extend(outs)
            logger.info(f"  {min(i+batch_sz, len(prompts))}/{len(prompts)}")

        results = []
        for (_, row), out in zip(test_df.iterrows(), all_outputs):
            text = out.outputs[0].text.strip()
            reasoning = _extract_reasoning(text)
            pred = extract_label(text)
            results.append({
                "drug1_id": row["drug1_id"],
                "drug2_id": row["drug2_id"],
                "drug1_name": row.get("drug1_name", row["drug1_id"]),
                "drug2_name": row.get("drug2_name", row["drug2_id"]),
                "true_label": int(row["label"]),
                "pred_label": pred,
                "response": reasoning,
                "classification_response": text[len(reasoning):].strip(),
            })

    else:
        # Self-consistency: N samples per pair, majority vote
        logger.info(f"Self-consistency (n={n_samples}) on {len(prompts):,} pairs …")
        params = SamplingParams(
            temperature=0.7, top_p=0.95, max_tokens=max_tokens, n=n_samples,
        )

        all_multi_outputs = []
        for i in range(0, len(prompts), batch_sz):
            outs = llm.generate(prompts[i:i+batch_sz], params, lora_request=lora_req)
            all_multi_outputs.extend(outs)
            logger.info(f"  {min(i+batch_sz, len(prompts))}/{len(prompts)}")

        results = []
        agreement_counts = []
        for (_, row), multi_out in zip(test_df.iterrows(), all_multi_outputs):
            sample_labels = []
            sample_texts = []
            for sample in multi_out.outputs:
                text = sample.text.strip()
                sample_texts.append(text)
                sample_labels.append(extract_label(text))

            valid_labels = [l for l in sample_labels if l >= 0]
            if valid_labels:
                vote_counter = Counter(valid_labels)
                pred, count = vote_counter.most_common(1)[0]
                agreement_counts.append(count / len(valid_labels))
            else:
                pred = -1
                agreement_counts.append(0.0)

            # Pick the sample whose label matches the majority vote for reasoning
            best_idx = 0
            for idx, lbl in enumerate(sample_labels):
                if lbl == pred:
                    best_idx = idx
                    break
            best_text = sample_texts[best_idx]
            reasoning = _extract_reasoning(best_text)

            results.append({
                "drug1_id": row["drug1_id"],
                "drug2_id": row["drug2_id"],
                "drug1_name": row.get("drug1_name", row["drug1_id"]),
                "drug2_name": row.get("drug2_name", row["drug2_id"]),
                "true_label": int(row["label"]),
                "pred_label": pred,
                "response": reasoning,
                "classification_response": best_text[len(reasoning):].strip(),
                "n_samples": n_samples,
                "sample_labels": sample_labels,
                "agreement": count / len(valid_labels) if valid_labels else 0.0,
            })

        avg_agreement = float(np.mean(agreement_counts))
        logger.info(f"Self-consistency agreement: {avg_agreement:.3f} "
                    f"(N={n_samples})")

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_dir = os.path.join(cfg["project"]["output_dir"], "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{condition_name}_predictions.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    valid = [(r["true_label"], r["pred_label"]) for r in results if r["pred_label"] >= 0]
    if valid:
        vt, vp = zip(*valid)
        macro = f1_score(vt, vp, average="macro", zero_division=0)
        micro = f1_score(vt, vp, average="micro", zero_division=0)
        logger.info(f"{condition_name}: Macro F1={macro:.4f} | Micro F1={micro:.4f}")

    logger.info(f"Saved to {out_path}")
    return out_path




JUDGE_SYSTEM_MSG = "You are a senior pharmacologist evaluating the quality of drug interaction explanations."

STUDENT_JUDGE_PROMPT = """You are a senior pharmacologist conducting a rigorous peer review. \
A student model wrote the explanation below to reason about a drug-drug interaction. \
Determine if this explanation demonstrates CORRECT pharmacological reasoning.

=== DRUG PAIR ===
Drug 1: {drug1_name}
Drug 2: {drug2_name}
Known interaction: {label_text}

=== STUDENT'S EXPLANATION ===
{cot}

=== SCORING CRITERIA (1-5, strict: 3=acceptable, 4=good, 5=excellent) ===

1. DRUG_SPECIFICITY: Names specific enzymes, receptors, transporters for each drug?
2. MECHANISM_ACCURACY: Mechanism is pharmacologically plausible and accurate?
3. CAUSAL_CHAIN: Steps build logically toward the stated interaction?
4. TEACHING_VALUE: Explanation teaches transferable reasoning?
5. FACTUAL_ERRORS: Any errors or hallucinations? (1=major, 5=none)

=== OUTPUT ===
Brief analysis (2-4 sentences). Then scores, each on its own line:

DRUG_SPECIFICITY: <score>
MECHANISM_ACCURACY: <score>
CAUSAL_CHAIN: <score>
TEACHING_VALUE: <score>
FACTUAL_ERRORS: <score>
OVERALL: <score>
VERDICT: <PASS if OVERALL >= 3, else FAIL>"""


def judge_student_reasoning(cfg: dict, predictions_path: str,
                            sample_size: int = 99999):
    """Score student reasoning with judge models. Uses drug NAMES (not IDs)."""
    from vllm import LLM, SamplingParams

    logger = setup_logging("student_judge")
    records = _load_records(predictions_path)

    if sample_size < len(records):
        import random
        random.seed(cfg["project"]["seed"])
        records = random.sample(records, sample_size)

    label_map = _load_label_map(cfg)

    res_dir = os.path.join(cfg["project"]["output_dir"], "results")
    os.makedirs(res_dir, exist_ok=True)
    all_judge_results = {}
    BATCH_SIZE = 64

    for jcfg in cfg["judge"]["models"]:
        model_name = jcfg["model_name"]
        short_name = model_name.split("/")[-1]
        no_sys = jcfg.get("no_system_prompt", False)
        m_temp = jcfg.get("temperature", 0.3)

        score_path = os.path.join(res_dir, f"student_judge_scores_{short_name}.jsonl")
        ckpt_path = score_path + ".ckpt"

        scored = {}
        if os.path.exists(ckpt_path):
            with open(ckpt_path) as f:
                for line in f:
                    s = json.loads(line)
                    scored[s["idx"]] = s
            logger.info(f"Resuming {short_name}: {len(scored)} already scored")

        remaining = []
        for rec in records:
            idx = f"{rec['drug1_id']}_{rec['drug2_id']}"
            if idx not in scored:
                remaining.append(rec)

        if not remaining:
            logger.info(f"{short_name}: all scored — skipping")
            all_judge_results[short_name] = list(scored.values())
            continue

        logger.info(f"Loading judge: {model_name} ({len(remaining)} to score)")

        llm = LLM(
            model=model_name,
            tensor_parallel_size=jcfg["tensor_parallel_size"],
            dtype=jcfg["dtype"],
            max_model_len=jcfg.get("max_model_len", 4096),
            gpu_memory_utilization=jcfg.get("gpu_memory_utilization", 0.90),
            trust_remote_code=True,
        )
        tokenizer = llm.get_tokenizer()
        max_tokens = 1200 if no_sys else 800
        params = SamplingParams(temperature=m_temp, top_p=0.9, max_tokens=max_tokens)

        ckpt_f = open(ckpt_path, "a")
        for batch_start in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[batch_start:batch_start + BATCH_SIZE]
            prompts = []
            for rec in batch:
                reasoning = _extract_reasoning(rec.get("response", ""))[:2000]
                true_label = rec["true_label"]
                template = label_map.get(true_label, f"interaction type {true_label}")
                d1_name = rec.get("drug1_name", rec["drug1_id"])
                d2_name = rec.get("drug2_name", rec["drug2_id"])
                label_text = template.replace("#Drug1", d1_name).replace("#Drug2", d2_name)

                user_msg = STUDENT_JUDGE_PROMPT.format(
                    drug1_name=d1_name, drug2_name=d2_name,
                    label_text=label_text, cot=reasoning,
                )
                if no_sys:
                    messages = [{"role": "user",
                                 "content": JUDGE_SYSTEM_MSG + "\n\n" + user_msg}]
                else:
                    messages = [
                        {"role": "system", "content": JUDGE_SYSTEM_MSG},
                        {"role": "user", "content": user_msg},
                    ]
                prompts.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ))

            outputs = llm.generate(prompts, params)
            for rec, out in zip(batch, outputs):
                resp = out.outputs[0].text.strip()
                scores = _parse_judge_response(resp)
                scores["idx"] = f"{rec['drug1_id']}_{rec['drug2_id']}"
                scores["true_label"] = rec["true_label"]
                scores["pred_label"] = rec["pred_label"]
                scores["judge"] = short_name
                scored[scores["idx"]] = scores
                ckpt_f.write(json.dumps(scores) + "\n")

            ckpt_f.flush()
            logger.info(f"  {short_name}: {len(scored)}/{len(records)} scored")

        ckpt_f.close()

        with open(score_path, "w") as f:
            for s in scored.values():
                f.write(json.dumps(s) + "\n")

        all_judge_results[short_name] = list(scored.values())

        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_judge_results


def _parse_judge_response(resp: str) -> dict:
    resp_clean = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()
    dims = ["drug_specificity", "mechanism_accuracy", "causal_chain",
            "teaching_value", "factual_errors", "overall"]
    scores = {}
    for dim in dims:
        pattern = dim.replace("_", "[_ ]") + r"\s*:\s*(\d)"
        m = re.search(pattern, resp_clean, re.IGNORECASE)
        scores[dim] = int(m.group(1)) if m else 0

    verdict_m = re.search(r"VERDICT\s*:\s*(PASS|FAIL)", resp_clean, re.IGNORECASE)
    scores["verdict"] = verdict_m.group(1).upper() if verdict_m else (
        "PASS" if scores.get("overall", 0) >= 3 else "FAIL"
    )
    return scores




def create_comparison_report(all_metrics: list, fig_dir: str):
    logger = setup_logging("comparison")
    os.makedirs(fig_dir, exist_ok=True)

    df = pd.DataFrame([{
        "Condition": m["condition"],
        "Macro F1": m["macro_f1"],
        "Micro F1": m["micro_f1"],
        "Accuracy": m["accuracy"],
        "Valid %": m["valid_pct"],
    } for m in all_metrics])

    df.to_csv(os.path.join(fig_dir, "comparison_table.csv"), index=False)
    logger.info(f"\nComparison:\n{df.to_string(index=False)}")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(df))
    w = 0.3
    ax.bar([i - w/2 for i in x], df["Macro F1"], w, label="Macro F1", color="#2196F3")
    ax.bar([i + w/2 for i in x], df["Micro F1"], w, label="Micro F1", color="#FF9800")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["Condition"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score")
    ax.set_title("DDI Classification — All Conditions")
    ax.legend()
    ax.set_ylim(0, 1.05)
    for i, row in df.iterrows():
        ax.text(i - w/2, row["Macro F1"] + 0.02, f"{row['Macro F1']:.3f}",
                ha="center", fontsize=7)
        ax.text(i + w/2, row["Micro F1"] + 0.02, f"{row['Micro F1']:.3f}",
                ha="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "comparison_chart.png"), dpi=200)
    plt.close()

    report_path = os.path.join(fig_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    logger.info(f"Report saved to {report_path}")




def _load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def _load_label_map(cfg: dict) -> dict:
    path = os.path.join(cfg["data"]["processed_dir"], "label_map.json")
    with open(path) as f:
        return {int(k): v for k, v in json.load(f).items()}
