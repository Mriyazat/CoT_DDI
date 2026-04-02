"""
Phase 5b – Independent API-based evaluation of reasoning quality.

Uses GPT-4o and Gemini 2.5 Flash to score a stratified sample of traces
(teacher and/or student) on a pharmacological reasoning rubric. Judges
receive the full drug profiles so they can properly verify factual claims.

This is an EVALUATION tool for the paper, not a filtering step.

Rubric dimensions (scored 1-5):
  1. Mechanism Accuracy   — pharmacological mechanisms correct?
  2. Factual Grounding    — cited entities real and relevant?
  3. Causal Chain Quality — reasoning logically connects mechanism to outcome?
  4. Drug Specificity     — reasoning specific to these drugs (not generic)?
  5. Summary Quality      — summary accurate and concise?

Supports resume via JSON checkpoint.
"""

import os
import re
import json
import time
import random
from collections import defaultdict
from typing import Dict, List, Optional

from src.utils import load_config, setup_logging, set_seed


def _format_full_drug_profile(profile: dict) -> str:
    """Format a COMPLETE drug profile — no truncation.

    Unlike _format_drug_profile() in data_preparation.py (which truncates
    mechanism to 200 chars, enzymes to 5, transporters to 3, targets to 3),
    this shows everything so the API judge can properly verify all claims.
    """
    lines = []

    if profile.get("description"):
        lines.append(f"  Description: {profile['description']}")

    if profile.get("mechanism_of_action"):
        lines.append(f"  Mechanism of action: {profile['mechanism_of_action']}")

    if profile.get("pharmacodynamics"):
        lines.append(f"  Pharmacodynamics: {profile['pharmacodynamics']}")

    if profile.get("enzymes"):
        enz_str = "; ".join(profile["enzymes"])
        lines.append(f"  Enzymes (all): {enz_str}")

    if profile.get("transporters"):
        trans_str = "; ".join(profile["transporters"])
        lines.append(f"  Transporters (all): {trans_str}")

    if profile.get("targets"):
        tgt_str = "; ".join(profile["targets"])
        lines.append(f"  Targets (all): {tgt_str}")

    if profile.get("categories"):
        lines.append(f"  Categories: {'; '.join(profile['categories'][:8])}")

    if profile.get("metabolism"):
        lines.append(f"  Metabolism: {profile['metabolism']}")

    if profile.get("smiles"):
        lines.append(f"  SMILES: {profile['smiles'][:200]}")

    return "\n".join(lines)

# ── Evaluation rubric ────────────────────────────────────────────────

RUBRIC_DIMENSIONS = [
    "mechanism_accuracy",
    "factual_grounding",
    "causal_chain",
    "drug_specificity",
    "summary_quality",
]

JUDGE_SYSTEM_PROMPT = """\
You are an expert pharmacologist evaluating drug-drug interaction (DDI) analyses.
You will be given an analysis of a DDI along with the complete pharmacological
profiles of both drugs from DrugBank. Score the analysis on five dimensions.

For EACH dimension, provide a score from 1 (poor) to 5 (excellent):

1. MECHANISM_ACCURACY (1-5): Are the pharmacological mechanisms described correct?
   - 5: All mechanisms are accurate and well-explained
   - 3: Most mechanisms correct but some minor inaccuracies
   - 1: Major errors in mechanism description

2. FACTUAL_GROUNDING (1-5): Are cited enzymes, transporters, targets supported by the profiles?
   - 5: All entities match the drug profiles provided
   - 3: Most entities are real but some are not in the profiles
   - 1: Multiple fabricated or incorrect entities

3. CAUSAL_CHAIN (1-5): Does the reasoning logically connect mechanism to interaction outcome?
   - 5: Clear, logical chain from mechanism to clinical effect
   - 3: Logical but with gaps or weak connections
   - 1: No coherent causal chain

4. DRUG_SPECIFICITY (1-5): Is the reasoning specific to these two drugs?
   - 5: Highly specific, references unique properties of each drug
   - 3: Somewhat specific but could apply to similar drugs
   - 1: Generic reasoning that could apply to any drug pair

5. SUMMARY_QUALITY (1-5): Is the summary accurate and concise?
   - 5: Accurate, concise, captures the key mechanism
   - 3: Mostly accurate but verbose or missing key points
   - 1: Inaccurate or not a useful summary

Respond ONLY with a JSON object in this exact format:
{
  "mechanism_accuracy": <1-5>,
  "factual_grounding": <1-5>,
  "causal_chain": <1-5>,
  "drug_specificity": <1-5>,
  "summary_quality": <1-5>,
  "brief_justification": "<1-2 sentences explaining your scores>"
}"""


def _build_judge_prompt(trace: dict, profiles: dict) -> str:
    """Build the user prompt for API judge evaluation."""
    d1_id = trace["drug1_id"]
    d2_id = trace["drug2_id"]
    d1_name = trace.get("drug1_name", d1_id)
    d2_name = trace.get("drug2_name", d2_id)
    p1 = profiles.get(d1_id, {})
    p2 = profiles.get(d2_id, {})

    parts = []
    parts.append(f"## Drug Profiles\n")
    parts.append(f"### Drug 1: {d1_name} ({d1_id})")
    if p1:
        parts.append(_format_full_drug_profile(p1))
    parts.append("")

    parts.append(f"### Drug 2: {d2_name} ({d2_id})")
    if p2:
        parts.append(_format_full_drug_profile(p2))
    parts.append("")

    label_text = trace.get("label_text", "")
    parts.append(f"## Known Interaction")
    parts.append(f"Type: Y={trace['label']} -- \"{label_text}\"")
    severity = trace.get("severity", trace.get("teacher_severity", "Unknown"))
    parts.append(f"Severity: {severity}")
    parts.append("")

    cot_field = "teacher_cot" if "teacher_cot" in trace else "student_cot"
    parts.append(f"## Analysis to Evaluate\n")
    parts.append(trace.get(cot_field, trace.get("output", "")))
    parts.append("")
    parts.append("Score this analysis on the five rubric dimensions (1-5 each). "
                 "Respond with JSON only.")

    return "\n".join(parts)


# ── API clients ──────────────────────────────────────────────────────

def _call_openai(prompt: str, system: str, model: str = "gpt-4o",
                 max_retries: int = 3) -> Optional[dict]:
    """Call OpenAI API and parse JSON response."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content.strip()
            return json.loads(text)
        except json.JSONDecodeError:
            json_match = re.search(r"\{[^}]+\}", text)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
            else:
                return {"error": str(e)}

    return None


def _call_gemini(prompt: str, system: str, model: str = "gemini-2.5-flash",
                 max_retries: int = 3) -> Optional[dict]:
    """Call Google Gemini API and parse JSON response."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("pip install google-generativeai")

    gen_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system,
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=300,
            response_mime_type="application/json",
        ),
    )

    for attempt in range(max_retries):
        try:
            response = gen_model.generate_content(prompt)
            text = response.text.strip()
            return json.loads(text)
        except json.JSONDecodeError:
            json_match = re.search(r"\{[^}]+\}", text)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
            else:
                return {"error": str(e)}

    return None


API_CALLERS = {
    "gpt-4o": _call_openai,
    "gemini-2.5-flash": _call_gemini,
}


# ── Stratified sampling ──────────────────────────────────────────────

def stratified_sample(traces: List[dict], n: int, seed: int = 42) -> List[dict]:
    """Sample n traces stratified by label class (head/mid/tail represented)."""
    rng = random.Random(seed)

    by_label = defaultdict(list)
    for t in traces:
        by_label[t["label"]].append(t)

    n_classes = len(by_label)
    per_class = max(1, n // n_classes)
    remainder = n - per_class * n_classes

    sampled = []
    labels_sorted = sorted(by_label.keys(), key=lambda l: -len(by_label[l]))

    for i, label in enumerate(labels_sorted):
        pool = by_label[label]
        take = per_class + (1 if i < remainder else 0)
        take = min(take, len(pool))
        sampled.extend(rng.sample(pool, take))

    if len(sampled) < n:
        remaining = [t for t in traces if t not in sampled]
        extra = min(n - len(sampled), len(remaining))
        sampled.extend(rng.sample(remaining, extra))

    rng.shuffle(sampled)
    return sampled[:n]


# ── Main evaluation pipeline ─────────────────────────────────────────

def evaluate_traces(cfg: dict, trace_source: str = "teacher",
                    trace_file: str = None):
    """Run API judge evaluation on a stratified sample of traces."""
    logger = setup_logging("api_judge_eval")
    set_seed(cfg["project"]["seed"])

    api_cfg = cfg.get("api_eval", {})
    sample_size = api_cfg.get("sample_size", 1000)
    judge_models = api_cfg.get("models", ["gpt-4o", "gemini-2.5-flash"])

    proc_dir = cfg["data"]["processed_dir"]
    with open(os.path.join(proc_dir, "drug_profiles.json")) as f:
        profiles = json.load(f)

    if trace_file is None:
        trace_dir = os.path.join(cfg["project"]["output_dir"], "teacher_traces")
        if trace_source == "teacher":
            trace_file = os.path.join(trace_dir, "full_traces_final.jsonl")
            if not os.path.exists(trace_file):
                trace_file = os.path.join(trace_dir, "full_traces_hard_filtered.jsonl")
        else:
            trace_file = os.path.join(
                cfg["project"]["output_dir"], "results",
                f"student_{trace_source}_outputs.jsonl"
            )

    if not os.path.exists(trace_file):
        logger.error(f"Trace file not found: {trace_file}")
        return

    traces = []
    with open(trace_file) as f:
        for line in f:
            traces.append(json.loads(line))
    logger.info(f"Loaded {len(traces):,} traces from {trace_file}")

    sample = stratified_sample(traces, sample_size, cfg["project"]["seed"])
    logger.info(f"Sampled {len(sample):,} traces (stratified by class)")

    results_dir = os.path.join(cfg["project"]["output_dir"], "results")
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_file = os.path.join(
        results_dir, f"api_judge_{trace_source}_checkpoint.json"
    )

    completed = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            completed = json.load(f)
        logger.info(f"Resume: {sum(len(v) for v in completed.values())} evaluations loaded")

    all_scores = {model: completed.get(model, {}) for model in judge_models}

    for model_name in judge_models:
        logger.info(f"\n--- Evaluating with {model_name} ---")

        caller = API_CALLERS.get(model_name)
        if caller is None:
            logger.warning(f"No API caller for {model_name}, skipping")
            continue

        pending = [
            t for t in sample
            if str(t["idx"]) not in all_scores.get(model_name, {})
        ]
        logger.info(f"  Pending: {len(pending):,} / {len(sample):,}")

        for i, trace in enumerate(pending):
            prompt = _build_judge_prompt(trace, profiles)
            result = caller(prompt, JUDGE_SYSTEM_PROMPT, model=model_name)

            if result is None:
                result = {"error": "no response"}

            all_scores[model_name][str(trace["idx"])] = result

            if (i + 1) % 25 == 0 or i == len(pending) - 1:
                with open(checkpoint_file, "w") as f:
                    json.dump(all_scores, f)
                logger.info(f"  [{model_name}] {i+1}/{len(pending)} evaluated")

            time.sleep(0.5)

    with open(checkpoint_file, "w") as f:
        json.dump(all_scores, f, indent=2)

    _compute_summary(cfg, all_scores, trace_source, sample, logger)


def _compute_summary(cfg, all_scores, trace_source, sample, logger):
    """Compute aggregate scores and inter-judge agreement."""
    results_dir = os.path.join(cfg["project"]["output_dir"], "results")
    summary = {
        "trace_source": trace_source,
        "sample_size": len(sample),
        "models": {},
        "inter_judge_agreement": {},
    }

    for model_name, scores in all_scores.items():
        dim_scores = defaultdict(list)
        n_valid = 0
        n_errors = 0

        for idx, result in scores.items():
            if "error" in result:
                n_errors += 1
                continue
            n_valid += 1
            for dim in RUBRIC_DIMENSIONS:
                val = result.get(dim)
                if isinstance(val, (int, float)) and 1 <= val <= 5:
                    dim_scores[dim].append(val)

        model_summary = {
            "n_evaluated": n_valid,
            "n_errors": n_errors,
            "dimensions": {},
        }
        for dim in RUBRIC_DIMENSIONS:
            vals = dim_scores[dim]
            if vals:
                model_summary["dimensions"][dim] = {
                    "mean": round(sum(vals) / len(vals), 3),
                    "min": min(vals),
                    "max": max(vals),
                    "n": len(vals),
                }

        all_dim_scores = []
        for vals in dim_scores.values():
            all_dim_scores.extend(vals)
        if all_dim_scores:
            model_summary["overall_mean"] = round(
                sum(all_dim_scores) / len(all_dim_scores), 3
            )

        summary["models"][model_name] = model_summary
        logger.info(f"\n{model_name} results ({n_valid} valid, {n_errors} errors):")
        for dim, stats in model_summary.get("dimensions", {}).items():
            logger.info(f"  {dim}: {stats['mean']:.2f} (min={stats['min']}, max={stats['max']})")

    model_names = list(all_scores.keys())
    if len(model_names) >= 2:
        m1, m2 = model_names[0], model_names[1]
        s1, s2 = all_scores[m1], all_scores[m2]
        common_idx = set(s1.keys()) & set(s2.keys())
        common_idx = {i for i in common_idx
                      if "error" not in s1[i] and "error" not in s2[i]}

        if common_idx:
            agreements = defaultdict(list)
            for idx in common_idx:
                for dim in RUBRIC_DIMENSIONS:
                    v1 = s1[idx].get(dim)
                    v2 = s2[idx].get(dim)
                    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                        agreements[dim].append(abs(v1 - v2))

            agreement_summary = {}
            for dim, diffs in agreements.items():
                avg_diff = sum(diffs) / len(diffs)
                exact_match = sum(1 for d in diffs if d == 0) / len(diffs)
                within_1 = sum(1 for d in diffs if d <= 1) / len(diffs)
                agreement_summary[dim] = {
                    "avg_absolute_diff": round(avg_diff, 3),
                    "exact_match_rate": round(exact_match, 3),
                    "within_1_rate": round(within_1, 3),
                    "n_pairs": len(diffs),
                }

            summary["inter_judge_agreement"] = {
                "judges": [m1, m2],
                "n_common": len(common_idx),
                "per_dimension": agreement_summary,
            }

            logger.info(f"\nInter-judge agreement ({m1} vs {m2}, n={len(common_idx)}):")
            for dim, stats in agreement_summary.items():
                logger.info(
                    f"  {dim}: avg_diff={stats['avg_absolute_diff']:.2f}, "
                    f"exact={stats['exact_match_rate']:.1%}, "
                    f"within_1={stats['within_1_rate']:.1%}"
                )

    out_path = os.path.join(results_dir, f"api_judge_{trace_source}_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="API judge evaluation of reasoning traces"
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--source", default="teacher",
                        choices=["teacher", "student"],
                        help="Evaluate teacher or student traces")
    parser.add_argument("--trace-file", default=None,
                        help="Override trace file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate_traces(cfg, trace_source=args.source, trace_file=args.trace_file)
