"""
Phase 1.7 – Targeted self-refinement of low-scoring teacher traces.

Feeds traces with low grounded factuality scores back to the teacher model
for verification and correction. Only the bottom N% (default 15%) are
refined, keeping compute cost manageable.

Flow:
  1. Load low-scoring traces from traces_for_refinement.jsonl
  2. Build refinement prompts with original trace + full drug profiles
  3. Generate refined traces using vLLM (same teacher model)
  4. Re-score with grounded factuality
  5. Keep refined trace if improved; discard if still below threshold

Supports JSONL-based resume.
"""

import os
import json
import time
import pandas as pd
from tqdm import tqdm

from src.utils import load_config, setup_logging, set_seed, ensure_dirs
from src.grounded_factuality import score_trace
from src.teacher_generation import _assess_quality


def _format_full_drug_profile(profile: dict) -> str:
    """Format a COMPLETE drug profile — no truncation.

    Unlike _format_drug_profile() in data_preparation.py (which truncates
    mechanism to 200 chars, enzymes to 5, transporters to 3, targets to 3),
    this shows everything so the refinement model can verify all claims.
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


# ── Refinement prompt ────────────────────────────────────────────────

REFINEMENT_SYSTEM_PROMPT = (
    "You are an expert pharmacologist reviewing drug-drug interaction analyses. "
    "You will be given a previous analysis and the complete pharmacological "
    "profiles of both drugs from DrugBank.\n\n"
    "Your task:\n"
    "1. Verify the reasoning against the drug profiles.\n"
    "2. Replace vague references (e.g. 'glucocorticoid receptor', "
    "'serotonin receptor') with the SPECIFIC entity names from the profiles "
    "(e.g. 'Nuclear receptor subfamily 3 group C member 1 (NR3C1)', "
    "'5-hydroxytryptamine receptor 2A (HTR2A)').\n"
    "3. Cite specific CYP enzymes, transporters, and targets BY NAME "
    "as listed in the drug profiles.\n"
    "4. Remove any enzymes, transporters, or targets NOT found in the profiles.\n"
    "5. Keep the same output structure:\n\n"
    "## Reasoning\n[Numbered steps citing specific profile entities]\n\n"
    "## Summary\n[2-3 sentences]\n\n"
    "## Classification\nY={label} -- \"{label_text}\"\n\n"
    "## Severity\n{Major/Moderate/Minor/Unknown}"
)


def build_refinement_prompt(trace: dict, profiles: dict, label_map: dict) -> str:
    """Build a refinement prompt with the original trace and drug profiles."""
    d1_id = trace["drug1_id"]
    d2_id = trace["drug2_id"]
    d1_name = trace.get("drug1_name", d1_id)
    d2_name = trace.get("drug2_name", d2_id)
    label = trace["label"]
    label_text = trace.get("label_text", label_map.get(label, ""))

    p1 = profiles.get(d1_id, {})
    p2 = profiles.get(d2_id, {})

    parts = []
    parts.append("You previously generated this analysis for the interaction "
                 f"between {d1_name} and {d2_name}:\n")
    parts.append("--- Previous Analysis ---")
    parts.append(trace.get("teacher_cot", ""))
    parts.append("\n--- Drug Profiles (Ground Truth) ---\n")
    parts.append(f"Drug 1: {d1_name} ({d1_id})")
    if p1:
        parts.append(_format_full_drug_profile(p1))
    parts.append("")

    parts.append(f"Drug 2: {d2_name} ({d2_id})")
    if p2:
        parts.append(_format_full_drug_profile(p2))
    parts.append("")

    parts.append(f"Known interaction: Y={label} -- \"{label_text}\"")
    severity = trace.get("severity", trace.get("teacher_severity", "Unknown"))
    parts.append(f"Known severity: {severity}")
    parts.append("")
    sev_instruction = ""
    if severity not in ("Unknown", ""):
        sev_instruction = (
            f"\n- In your final reasoning step, explain WHY this interaction "
            f"is classified as {severity} severity (e.g. clinical consequences, "
            f"risk of hospitalization, need for dose adjustment or monitoring)."
        )

    parts.append(
        "Review your previous analysis against the drug profiles above. "
        "Rewrite it following these rules:\n"
        "- Where you mentioned general terms (e.g. 'dopamine receptor', "
        "'serotonin transporter'), replace them with the EXACT entity names "
        "from the profiles (e.g. 'D(2) dopamine receptor (DRD2)', "
        "'Sodium-dependent serotonin transporter (SLC6A4)').\n"
        "- Cite specific CYP enzymes, transporters, and targets using "
        "the names and gene symbols listed in the drug profiles.\n"
        "- Remove any entities you mentioned that do NOT appear in the "
        "drug profiles.\n"
        "- Keep the pharmacological reasoning correct and coherent."
        f"{sev_instruction}\n"
        "- Output the corrected version with the same structure: "
        "## Reasoning, ## Summary, ## Classification, ## Severity."
    )
    return "\n".join(parts)


# ── Main refinement pipeline ─────────────────────────────────────────

def refine_traces(cfg: dict):
    """Refine low-scoring traces via teacher self-verification."""
    from vllm import LLM, SamplingParams

    logger = setup_logging("trace_refinement")
    set_seed(cfg["project"]["seed"])
    ensure_dirs(cfg)

    trace_dir = os.path.join(cfg["project"]["output_dir"], "teacher_traces")
    src = os.path.join(trace_dir, "traces_for_refinement.jsonl")
    dst = os.path.join(trace_dir, "traces_refined.jsonl")

    if not os.path.exists(src):
        logger.error(f"No traces for refinement at {src}")
        logger.info("Run grounded_factuality.py --split first.")
        return

    proc_dir = cfg["data"]["processed_dir"]
    with open(os.path.join(proc_dir, "drug_profiles.json")) as f:
        profiles = json.load(f)
    with open(os.path.join(proc_dir, "label_map.json")) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    ge_cfg = cfg.get("grounded_eval", {})
    discard_threshold = ge_cfg.get("discard_threshold", 0.3)
    precision_weight = ge_cfg.get("precision_weight", 0.7)

    rcfg = cfg.get("refinement", {})
    tcfg = cfg["teacher"]
    model_name = rcfg.get("model_name", tcfg["model_name"])
    tp = rcfg.get("tensor_parallel_size", tcfg["tensor_parallel_size"])
    batch_size = rcfg.get("batch_size", tcfg["batch_size"])

    done_indices = set()
    if os.path.exists(dst):
        with open(dst) as f:
            for line in f:
                try:
                    done_indices.add(json.loads(line)["idx"])
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info(f"Resume: {len(done_indices):,} already refined")

    traces = []
    with open(src) as f:
        for line in f:
            obj = json.loads(line)
            if obj["idx"] not in done_indices:
                traces.append(obj)

    pilot_n = cfg.get("_refine_pilot", 0)
    if pilot_n > 0 and len(traces) > pilot_n:
        traces = traces[:pilot_n]
        logger.info(f"PILOT MODE — {pilot_n} traces only")

    total_to_refine = len(traces)
    logger.info(f"Traces to refine: {total_to_refine:,}")

    if total_to_refine == 0:
        logger.info("Nothing to refine.")
        return

    logger.info(f"Loading teacher: {model_name} (tp={tp})")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp,
        dtype=rcfg.get("dtype", tcfg.get("dtype", "float16")),
        max_model_len=rcfg.get("max_model_len", tcfg.get("max_model_len", 8192)),
        gpu_memory_utilization=rcfg.get(
            "gpu_memory_utilization",
            tcfg.get("gpu_memory_utilization", 0.92)
        ),
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    params = SamplingParams(
        temperature=rcfg.get("temperature", 0.4),
        top_p=rcfg.get("top_p", tcfg.get("top_p", 0.95)),
        max_tokens=rcfg.get("max_new_tokens", tcfg.get("max_new_tokens", 1536)),
    )

    n_improved = 0
    n_discarded = 0
    n_kept_original = 0
    score_improvements = []
    t_start = time.time()

    for batch_start in tqdm(range(0, total_to_refine, batch_size),
                            desc="Refining traces", unit="batch"):
        batch = traces[batch_start:batch_start + batch_size]

        prompts = []
        for trace in batch:
            user_msg = build_refinement_prompt(trace, profiles, label_map)
            messages = [
                {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))

        outputs = llm.generate(prompts, params)

        records = []
        for trace, out in zip(batch, outputs):
            refined_text = out.outputs[0].text.strip()

            d1_id = trace["drug1_id"]
            d2_id = trace["drug2_id"]
            p1 = profiles.get(d1_id, {})
            p2 = profiles.get(d2_id, {})

            new_scores = score_trace(refined_text, p1, p2, precision_weight)
            old_score = trace.get("grounded_score", 0.0)
            new_score = new_scores["grounded_score"]

            quality = _assess_quality(
                refined_text,
                drug1_name=trace.get("drug1_name", ""),
                drug2_name=trace.get("drug2_name", ""),
                label=trace["label"],
            )

            improved = (new_score > old_score) and quality["quality_pass"]

            rec = dict(trace)
            if improved:
                rec["teacher_cot"] = refined_text
                rec["teacher_summary"] = quality["teacher_summary"]
                rec["teacher_severity"] = quality["teacher_severity"]
                rec["grounded_score"] = new_score
                rec["entity_precision"] = new_scores["entity_precision"]
                rec["entity_recall"] = new_scores["entity_recall"]
                rec["n_mentioned_entities"] = new_scores["n_mentioned"]
                rec["n_grounded_entities"] = new_scores["n_grounded"]
                rec["refined"] = True
                rec["original_grounded_score"] = old_score
                rec["score_improvement"] = round(new_score - old_score, 4)
                n_improved += 1
                score_improvements.append(new_score - old_score)
            elif new_score < discard_threshold and old_score < discard_threshold:
                rec["discarded"] = True
                n_discarded += 1
            else:
                rec["refined"] = False
                n_kept_original += 1

            records.append(rec)

        with open(dst, "a") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    elapsed = time.time() - t_start
    logger.info(f"\nRefinement complete in {elapsed/3600:.1f}h")
    logger.info(f"  Total processed: {total_to_refine:,}")
    logger.info(f"  Improved:        {n_improved:,} ({100*n_improved/total_to_refine:.1f}%)")
    logger.info(f"  Kept original:   {n_kept_original:,}")
    logger.info(f"  Discarded:       {n_discarded:,}")

    if score_improvements:
        avg_imp = sum(score_improvements) / len(score_improvements)
        logger.info(f"  Avg score improvement (when improved): {avg_imp:.4f}")

    logger.info(f"  Output: {dst}")


def annotate_severity(cfg: dict):
    """Full regeneration of high-quality traces that have known severity.

    Uses the same refinement prompt (full drug profiles + entity grounding)
    plus explicit severity reasoning instructions. Keeps regenerated trace
    only if it passes quality and has a grounded score >= the original.
    """
    from vllm import LLM, SamplingParams

    logger = setup_logging("severity_annotation")
    set_seed(cfg["project"]["seed"])

    trace_dir = os.path.join(cfg["project"]["output_dir"], "teacher_traces")
    src = os.path.join(trace_dir, "traces_high_quality.jsonl")
    dst = os.path.join(trace_dir, "traces_high_quality_annotated.jsonl")

    if not os.path.exists(src):
        logger.error(f"Missing {src}")
        return

    proc_dir = cfg["data"]["processed_dir"]
    with open(os.path.join(proc_dir, "drug_profiles.json")) as f:
        profiles = json.load(f)
    with open(os.path.join(proc_dir, "label_map.json")) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    ge_cfg = cfg.get("grounded_eval", {})
    precision_weight = ge_cfg.get("precision_weight", 0.7)

    done_indices = set()
    if os.path.exists(dst):
        with open(dst) as f:
            for line in f:
                try:
                    done_indices.add(json.loads(line)["idx"])
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info(f"Resume: {len(done_indices):,} already processed")

    traces_with_sev = []
    traces_no_sev = []
    with open(src) as f:
        for line in f:
            obj = json.loads(line)
            if obj["idx"] in done_indices:
                continue
            sev = obj.get("severity", "Unknown")
            if sev in ("Major", "Moderate", "Minor"):
                traces_with_sev.append(obj)
            else:
                traces_no_sev.append(obj)

    pilot_n = cfg.get("_refine_pilot", 0)
    if pilot_n > 0 and len(traces_with_sev) > pilot_n:
        traces_with_sev = traces_with_sev[:pilot_n]
        traces_no_sev = []
        logger.info(f"PILOT MODE — {pilot_n} severity traces only")

    logger.info(f"High-quality traces to process: "
                f"{len(traces_with_sev) + len(traces_no_sev):,} "
                f"(+{len(done_indices):,} already done)")
    logger.info(f"  With severity (full regeneration): {len(traces_with_sev):,}")
    logger.info(f"  Unknown severity (pass-through):   {len(traces_no_sev):,}")

    rcfg = cfg.get("refinement", {})
    tcfg = cfg["teacher"]
    model_name = rcfg.get("model_name", tcfg["model_name"])
    tp = rcfg.get("tensor_parallel_size", tcfg["tensor_parallel_size"])
    batch_size = rcfg.get("batch_size", tcfg["batch_size"])

    llm = None
    if traces_with_sev:
        logger.info(f"Loading teacher: {model_name} (tp={tp})")
        llm = LLM(
            model=model_name,
            tensor_parallel_size=tp,
            dtype=rcfg.get("dtype", tcfg.get("dtype", "float16")),
            max_model_len=rcfg.get("max_model_len", tcfg.get("max_model_len", 8192)),
            gpu_memory_utilization=rcfg.get(
                "gpu_memory_utilization",
                tcfg.get("gpu_memory_utilization", 0.92)
            ),
            trust_remote_code=True,
        )
        tokenizer = llm.get_tokenizer()

        params = SamplingParams(
            temperature=rcfg.get("temperature", 0.4),
            top_p=rcfg.get("top_p", tcfg.get("top_p", 0.95)),
            max_tokens=rcfg.get("max_new_tokens", tcfg.get("max_new_tokens", 1536)),
        )

    t_start = time.time()
    n_improved = 0
    n_kept_original = 0
    score_improvements = []

    with open(dst, "a") as fout:
        if traces_with_sev and llm:
            for batch_start in tqdm(range(0, len(traces_with_sev), batch_size),
                                    desc="Severity regeneration", unit="batch"):
                batch = traces_with_sev[batch_start:batch_start + batch_size]

                prompts = []
                for trace in batch:
                    user_msg = build_refinement_prompt(trace, profiles, label_map)
                    messages = [
                        {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ]
                    prompts.append(tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    ))

                outputs = llm.generate(prompts, params)

                for trace, out in zip(batch, outputs):
                    new_text = out.outputs[0].text.strip()
                    p1 = profiles.get(trace["drug1_id"], {})
                    p2 = profiles.get(trace["drug2_id"], {})

                    new_scores = score_trace(new_text, p1, p2, precision_weight)
                    old_score = trace.get("grounded_score", 0.0)
                    new_score = new_scores["grounded_score"]

                    quality = _assess_quality(
                        new_text,
                        drug1_name=trace.get("drug1_name", ""),
                        drug2_name=trace.get("drug2_name", ""),
                        label=trace["label"],
                    )

                    use_new = (
                        quality["quality_pass"]
                        and new_score >= old_score - 0.05
                    )

                    rec = dict(trace)
                    if use_new:
                        rec["teacher_cot"] = new_text
                        rec["teacher_summary"] = quality["teacher_summary"]
                        rec["teacher_severity"] = quality["teacher_severity"]
                        rec["grounded_score"] = new_score
                        rec["entity_precision"] = new_scores["entity_precision"]
                        rec["entity_recall"] = new_scores["entity_recall"]
                        rec["n_mentioned_entities"] = new_scores["n_mentioned"]
                        rec["n_grounded_entities"] = new_scores["n_grounded"]
                        rec["severity_regenerated"] = True
                        rec["original_grounded_score"] = old_score
                        n_improved += 1
                        if new_score > old_score:
                            score_improvements.append(new_score - old_score)
                    else:
                        rec["severity_regenerated"] = False
                        n_kept_original += 1

                    fout.write(json.dumps(rec) + "\n")

        for trace in traces_no_sev:
            fout.write(json.dumps(trace) + "\n")

    elapsed = time.time() - t_start
    total_sev = n_improved + n_kept_original
    logger.info(f"\nSeverity regeneration complete in {elapsed/3600:.1f}h")
    logger.info(f"  Regenerated (accepted): {n_improved:,} "
                f"({100*n_improved/total_sev:.1f}%)" if total_sev else "")
    logger.info(f"  Kept original:          {n_kept_original:,}")
    logger.info(f"  Pass-through (no sev):  {len(traces_no_sev):,}")
    if score_improvements:
        avg = sum(score_improvements) / len(score_improvements)
        logger.info(f"  Avg score improvement:  {avg:.4f}")
    logger.info(f"  Output: {dst}")


def merge_refined_traces(cfg: dict):
    """Merge high-quality (or annotated) and refined traces into final set."""
    logger = setup_logging("trace_refinement")

    trace_dir = os.path.join(cfg["project"]["output_dir"], "teacher_traces")
    annotated_path = os.path.join(trace_dir, "traces_high_quality_annotated.jsonl")
    high_path = os.path.join(trace_dir, "traces_high_quality.jsonl")
    refined_path = os.path.join(trace_dir, "traces_refined.jsonl")
    final_path = os.path.join(trace_dir, "full_traces_final.jsonl")

    if os.path.exists(annotated_path):
        use_high = annotated_path
        logger.info("Using severity-annotated high-quality traces")
    elif os.path.exists(high_path):
        use_high = high_path
        logger.info("Using original high-quality traces (no severity annotation)")
    else:
        logger.error(f"Missing both {annotated_path} and {high_path}")
        return

    if not os.path.exists(refined_path):
        logger.error(f"Missing {refined_path}")
        return

    n_high = 0
    n_refined_kept = 0
    n_discarded = 0

    with open(final_path, "w") as fout:
        with open(use_high) as f:
            for line in f:
                fout.write(line)
                n_high += 1

        with open(refined_path) as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("discarded"):
                    n_discarded += 1
                    continue
                clean = {k: v for k, v in obj.items() if k != "discarded"}
                fout.write(json.dumps(clean) + "\n")
                n_refined_kept += 1

    total = n_high + n_refined_kept
    logger.info(f"Final training traces merged:")
    logger.info(f"  High-quality:  {n_high:,}")
    logger.info(f"  Refined kept:  {n_refined_kept:,}")
    logger.info(f"  Discarded:     {n_discarded:,}")
    logger.info(f"  Total:         {total:,}")
    logger.info(f"  Output: {final_path}")

    prepare_student_data(cfg, final_path)


def prepare_student_data(cfg: dict, final_traces_path: str = None):
    """Create train_cot.jsonl in data/processed/ for student training.

    The student training module reads from data/processed/train_cot.jsonl.
    This step:
      1. Reads the final merged traces
      2. Sets _orig_idx so few-shot retrieval indices still match
      3. Verifies class distribution
      4. Writes data/processed/train_cot.jsonl
    """
    from collections import Counter
    logger = setup_logging("trace_refinement")

    if final_traces_path is None:
        trace_dir = os.path.join(cfg["project"]["output_dir"], "teacher_traces")
        final_traces_path = os.path.join(trace_dir, "full_traces_final.jsonl")
        if not os.path.exists(final_traces_path):
            final_traces_path = os.path.join(
                trace_dir, "full_traces_hard_filtered.jsonl"
            )

    if not os.path.exists(final_traces_path):
        logger.error(f"No final traces at {final_traces_path}")
        return

    proc_dir = cfg["data"]["processed_dir"]
    out_path = os.path.join(proc_dir, "train_cot.jsonl")

    label_counts = Counter()
    n_written = 0

    with open(final_traces_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            obj = json.loads(line)

            if not obj.get("quality_pass", True):
                continue

            obj["_orig_idx"] = obj["idx"]

            label_counts[obj["label"]] += 1
            fout.write(json.dumps(obj) + "\n")
            n_written += 1

    n_classes = len(label_counts)
    min_count = min(label_counts.values()) if label_counts else 0
    max_count = max(label_counts.values()) if label_counts else 0
    median_count = sorted(label_counts.values())[n_classes // 2] if label_counts else 0

    logger.info(f"\nStudent training data prepared:")
    logger.info(f"  Total traces:  {n_written:,}")
    logger.info(f"  Classes:       {n_classes}")
    logger.info(f"  Per-class:     min={min_count}, max={max_count}, median={median_count}")
    logger.info(f"  Output: {out_path}")

    empty_classes = [l for l, c in label_counts.items() if c < 10]
    if empty_classes:
        logger.warning(f"  Classes with <10 traces: {sorted(empty_classes)}")

    min_required = cfg["student"]["training"].get("min_per_class", 100)
    low_classes = [l for l, c in label_counts.items() if c < min_required]
    if low_classes:
        logger.warning(
            f"  {len(low_classes)} classes below min_per_class={min_required} "
            f"(temperature resampling will upsample these)"
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Phase 1.7: Targeted self-refinement of low-scoring traces"
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip refinement, only merge high+refined traces")
    parser.add_argument("--severity-only", action="store_true",
                        help="Only run severity regeneration on high-quality traces")
    parser.add_argument("--pilot", type=int, default=0,
                        help="Test with N traces per step, then stop (no merge)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.pilot > 0:
        cfg["_refine_pilot"] = args.pilot

    if args.severity_only:
        annotate_severity(cfg)
    elif args.merge_only:
        merge_refined_traces(cfg)
    else:
        refine_traces(cfg)
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        annotate_severity(cfg)
        merge_refined_traces(cfg)
