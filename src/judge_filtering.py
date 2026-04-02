"""
Phase 2 – Multi-judge filtering with weighted scoring, hard gates, and
tiered class-aware thresholds.

Three judge models score all hard-filtered traces sequentially:
  - Llama3-OpenBioLLM-70B (weight 0.35)
  - TxGemma-27B-Chat (weight 0.35)
  - Qwen2.5-72B-Instruct (weight 0.30)

Rubric dimensions (1-5):
  - MECHANISM_ACCURACY (HARD GATE: must be >= 3)
  - FACTUAL_ERRORS (HARD GATE: must be >= 3)
  - DRUG_SPECIFICITY
  - CAUSAL_CHAIN
  - SUMMARY_QUALITY

Tiered class-aware filtering:
  - Head (>2000 traces): full scoring + hard gates, weighted >= 3.0
  - Mid (500-2000 traces): hard gates only
  - Tail (<500 traces): keep best min(N, 50), never below 20

Each judge uses per-judge JSONL checkpoints for cluster-safe resume.
"""

import os
import re
import gc
import json
import time
from collections import defaultdict

from src.utils import load_config, setup_logging, set_seed

# ── Judge prompt / rubric ─────────────────────────────────────────────

JUDGE_SYSTEM_MSG = (
    "You are an expert pharmacologist reviewing drug interaction explanations. "
    "Score strictly — 3 means acceptable, 4 means good, 5 means excellent."
)

JUDGE_PROMPT = """=== DRUG PAIR ===
Drug 1: {drug1_name} ({drug1_id})
Drug 2: {drug2_name} ({drug2_id})
Known interaction: Y={label} -- "{label_text}"

=== TEACHER EXPLANATION ===
{cot}

=== SCORING CRITERIA (1-5) ===

1. MECHANISM_ACCURACY: Is the pharmacological mechanism correct for these drugs?
   (1=fabricated, 3=plausible but vague, 5=textbook-accurate)

2. FACTUAL_ERRORS: Any fabricated mechanisms, wrong identities, contradictions?
   (1=major errors, 3=minor inaccuracies, 5=no detectable errors)

3. DRUG_SPECIFICITY: Names specific CYP isoforms, receptors, transporters?
   (1=generic, 5=names exact molecular targets)

4. CAUSAL_CHAIN: Steps build logically: drug mechanisms -> combined effect -> interaction?
   (1=disconnected, 3=partial flow, 5=clear causal chain)

5. SUMMARY_QUALITY: Is the summary accurate and complete?
   (1=missing/wrong, 3=adequate, 5=concise and complete)

=== OUTPUT FORMAT (exactly this) ===
MECHANISM_ACCURACY: <score>
FACTUAL_ERRORS: <score>
DRUG_SPECIFICITY: <score>
CAUSAL_CHAIN: <score>
SUMMARY_QUALITY: <score>"""

JUDGE_DIMS = [
    "mechanism_accuracy", "factual_errors", "drug_specificity",
    "causal_chain", "summary_quality",
]
HARD_GATE_DIMS = ["mechanism_accuracy", "factual_errors"]

SCORE_PATTERNS = {
    dim: re.compile(rf"{dim.upper().replace('_', '.?')}[\s:=]+(\d)", re.IGNORECASE)
    for dim in JUDGE_DIMS
}

THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def _parse_judge_response(resp: str, idx: int) -> dict:
    """Extract dimension scores from a judge response."""
    cleaned = THINK_PATTERN.sub("", resp).strip()
    result = {"idx": idx}
    parsed_any = False

    for dim in JUDGE_DIMS:
        m = SCORE_PATTERNS[dim].search(cleaned)
        if not m:
            m = SCORE_PATTERNS[dim].search(resp)
        score = int(m.group(1)) if m else 0
        result[dim] = score
        if score > 0:
            parsed_any = True

    result["parse_ok"] = parsed_any
    return result


def _build_judge_prompts(tokenizer, traces, no_system_prompt=False):
    """Build tokenized judge prompts."""
    prompts = []
    for t in traces:
        user_msg = JUDGE_PROMPT.format(
            drug1_name=t.get("drug1_name", ""),
            drug1_id=t.get("drug1_id", ""),
            drug2_name=t.get("drug2_name", ""),
            drug2_id=t.get("drug2_id", ""),
            label=t["label"],
            label_text=t.get("label_text", f"Y={t['label']}"),
            cot=t["teacher_cot"][:3000],
        )
        if no_system_prompt:
            messages = [{"role": "user", "content": JUDGE_SYSTEM_MSG + "\n\n" + user_msg}]
        else:
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM_MSG},
                {"role": "user", "content": user_msg},
            ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        ))
    return prompts


# ── Per-judge scoring ─────────────────────────────────────────────────

def _score_with_judge(traces, mcfg, trace_dir, suffix, logger):
    """Score all traces with a single judge model, with checkpoint resume."""
    from vllm import LLM, SamplingParams

    model_name = mcfg["model_name"]
    short_name = model_name.split("/")[-1]
    no_sys = mcfg.get("no_system_prompt", False)

    score_path = os.path.join(
        trace_dir, f"{suffix}_judge_scores_{short_name}.jsonl"
    )

    done_idx = set()
    scores_by_idx = {}
    if os.path.exists(score_path):
        with open(score_path) as f:
            for line in f:
                try:
                    s = json.loads(line)
                    done_idx.add(s["idx"])
                    scores_by_idx[s["idx"]] = s
                except (json.JSONDecodeError, KeyError):
                    continue

    remaining = [t for t in traces if t["idx"] not in done_idx]

    logger.info(f"\n{'─' * 60}")
    logger.info(f"Judge: {short_name}  (tp={mcfg.get('tensor_parallel_size', 2)})")
    logger.info(f"  Already scored: {len(done_idx):,}  |  "
                f"Remaining: {len(remaining):,}")

    if not remaining:
        logger.info(f"  [{short_name}] All traces already scored.")
        return scores_by_idx

    logger.info(f"  Loading model: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=mcfg.get("tensor_parallel_size", 4),
        dtype=mcfg.get("dtype", "float16"),
        max_model_len=mcfg.get("max_model_len", 4096),
        gpu_memory_utilization=mcfg.get("gpu_memory_utilization", 0.90),
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    judge_params = SamplingParams(
        temperature=0.1, top_p=0.9,
        max_tokens=400,
    )

    batch_size = 256
    t_start = time.time()
    n_done = 0

    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]
        prompts = _build_judge_prompts(tokenizer, batch, no_sys)
        outputs = llm.generate(prompts, judge_params)

        with open(score_path, "a") as sf:
            for t, out in zip(batch, outputs):
                resp = out.outputs[0].text.strip()
                score = _parse_judge_response(resp, t["idx"])
                score["judge"] = short_name
                score["label"] = t["label"]
                sf.write(json.dumps(score) + "\n")
                scores_by_idx[t["idx"]] = score

        n_done += len(batch)
        if n_done % (batch_size * 4) < batch_size or n_done >= len(remaining):
            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - n_done) / rate if rate > 0 else 0
            parsed = [s for s in scores_by_idx.values() if s["parse_ok"]]
            avg_mech = (sum(s["mechanism_accuracy"] for s in parsed) / len(parsed)
                        if parsed else 0)
            logger.info(
                f"  [{short_name}] {len(done_idx)+n_done:,}/{len(traces):,} | "
                f"avg mech_acc: {avg_mech:.2f} | {rate:.1f}/s | "
                f"ETA {eta/60:.0f}min"
            )

    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    parsed = [s for s in scores_by_idx.values() if s["parse_ok"]]
    logger.info(f"  [{short_name}] Done — scored: {len(parsed):,}, "
                f"parse errors: {len(scores_by_idx)-len(parsed):,}")
    return scores_by_idx


# ── Tiered consensus filtering ────────────────────────────────────────

def _apply_tiered_filtering(traces, all_judge_scores, judge_weights,
                            cfg, logger):
    """Apply hard gates + weighted scoring + tiered class thresholds."""
    jcfg = cfg.get("judge", {})
    hard_gate_min = jcfg.get("hard_gate_min", 3)
    weighted_min = jcfg.get("weighted_score_min", 3.0)
    tiered = jcfg.get("tiered_thresholds", {})
    head_thresh = tiered.get("head_min_traces", 2000)
    mid_thresh = tiered.get("mid_min_traces", 500)
    tail_min_keep = tiered.get("tail_min_keep", 50)
    absolute_min = tiered.get("absolute_min", 20)

    class_traces = defaultdict(list)
    for t in traces:
        class_traces[t["label"]].append(t)

    class_tier = {}
    for label, ts in class_traces.items():
        n = len(ts)
        if n >= head_thresh:
            class_tier[label] = "head"
        elif n >= mid_thresh:
            class_tier[label] = "mid"
        else:
            class_tier[label] = "tail"

    tier_counts = defaultdict(int)
    for t in class_tier.values():
        tier_counts[t] += 1
    logger.info(f"  Class tiers: head={tier_counts['head']}, "
                f"mid={tier_counts['mid']}, tail={tier_counts['tail']}")

    kept_traces = []
    label_stats = {}

    for label, ts in class_traces.items():
        tier = class_tier[label]

        scored_traces = []
        for t in ts:
            idx = t["idx"]
            gate_fail = False
            weighted_sum = 0.0
            weight_total = 0.0
            all_parsed = True

            for judge_name, weight in judge_weights.items():
                scores = all_judge_scores.get(judge_name, {})
                s = scores.get(idx)
                if not s or not s.get("parse_ok"):
                    all_parsed = False
                    continue

                for dim in HARD_GATE_DIMS:
                    if s.get(dim, 0) < hard_gate_min:
                        gate_fail = True

                non_gate = [s.get(d, 0) for d in JUDGE_DIMS if d not in HARD_GATE_DIMS]
                avg_score = sum(non_gate) / len(non_gate) if non_gate else 0
                weighted_sum += weight * avg_score
                weight_total += weight

            weighted_avg = weighted_sum / weight_total if weight_total > 0 else 0

            scored_traces.append({
                "trace": t,
                "gate_fail": gate_fail,
                "weighted_avg": weighted_avg,
                "all_parsed": all_parsed,
            })

        if tier == "head":
            passed = [
                st for st in scored_traces
                if not st["gate_fail"] and st["weighted_avg"] >= weighted_min
            ]
        elif tier == "mid":
            passed = [st for st in scored_traces if not st["gate_fail"]]
        else:
            scored_traces.sort(key=lambda x: x["weighted_avg"], reverse=True)
            n_keep = min(tail_min_keep, len(scored_traces))
            n_keep = max(n_keep, min(absolute_min, len(scored_traces)))
            passed = scored_traces[:n_keep]

        kept_for_class = [st["trace"] for st in passed]
        kept_traces.extend(kept_for_class)

        label_stats[label] = {
            "tier": tier,
            "total": len(ts),
            "kept": len(kept_for_class),
        }

    return kept_traces, label_stats


# ── Main entry ────────────────────────────────────────────────────────

def judge_filter_traces(cfg: dict):
    """Score traces with all configured judges, then apply tiered filtering."""
    logger = setup_logging("judge_filter")

    jcfg = cfg.get("judge", {})
    judge_models = jcfg.get("models", [])
    if not judge_models:
        logger.error("No judge models configured in config.yaml")
        return

    trace_dir = os.path.join(cfg["project"]["output_dir"], "teacher_traces")
    src = os.path.join(trace_dir, "full_traces_hard_filtered.jsonl")
    dst = os.path.join(trace_dir, "full_traces_judge_filtered.jsonl")

    if not os.path.exists(src):
        src_alt = os.path.join(trace_dir, "full_traces.jsonl")
        if os.path.exists(src_alt):
            logger.warning(f"No hard-filtered file; falling back to {src_alt}")
            src = src_alt
        else:
            logger.error("No trace files found")
            return

    traces = []
    with open(src) as f:
        for line in f:
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    logger.info(f"Loaded {len(traces):,} traces from {src}")
    logger.info(f"Judges: {[m['model_name'].split('/')[-1] for m in judge_models]}")

    suffix = "full"
    all_judge_scores = {}
    judge_weights = {}

    for mcfg in judge_models:
        short_name = mcfg["model_name"].split("/")[-1]
        weight = mcfg.get("weight", 1.0 / len(judge_models))
        judge_weights[short_name] = weight

        scores = _score_with_judge(traces, mcfg, trace_dir, suffix, logger)
        all_judge_scores[short_name] = scores

    logger.info(f"\n{'=' * 60}")
    logger.info("TIERED CLASS-AWARE FILTERING")
    logger.info(f"{'=' * 60}")

    kept_traces, label_stats = _apply_tiered_filtering(
        traces, all_judge_scores, judge_weights, cfg, logger
    )

    with open(dst, "w") as fout:
        for t in kept_traces:
            fout.write(json.dumps(t) + "\n")

    n_kept = len(kept_traces)
    n_total = len(traces)
    logger.info(f"  Kept: {n_kept:,} / {n_total:,} ({100*n_kept/n_total:.1f}%)")

    n_classes = len(label_stats)
    empty = [l for l, s in label_stats.items() if s["kept"] == 0]
    heavy_loss = [l for l, s in label_stats.items()
                  if s["total"] > 0 and s["kept"] / s["total"] < 0.5]

    logger.info(f"  Classes: {n_classes}")
    logger.info(f"  Empty classes: {len(empty)}")
    logger.info(f"  Classes with >50% loss: {len(heavy_loss)}")

    for tier in ("head", "mid", "tail"):
        t_classes = {l: s for l, s in label_stats.items() if s["tier"] == tier}
        if not t_classes:
            continue
        total_in = sum(s["total"] for s in t_classes.values())
        total_out = sum(s["kept"] for s in t_classes.values())
        pct = 100 * total_out / total_in if total_in else 0
        logger.info(f"  {tier.upper()}: {len(t_classes)} classes, "
                    f"{total_out:,}/{total_in:,} traces kept ({pct:.1f}%)")

    if empty:
        logger.warning(f"  Emergency: classes with 0 traces: {sorted(empty)}")
        logger.warning("  Applying emergency fallback for empty classes...")
        with open(dst, "a") as fout:
            for label in empty:
                class_t = [t for t in traces if t["label"] == label]
                class_t_scored = []
                for t in class_t:
                    idx = t["idx"]
                    avg = 0
                    n = 0
                    for jname, scores in all_judge_scores.items():
                        s = scores.get(idx)
                        if s and s.get("parse_ok"):
                            avg += sum(s.get(d, 0) for d in JUDGE_DIMS)
                            n += len(JUDGE_DIMS)
                    class_t_scored.append((avg / n if n > 0 else 0, t))

                class_t_scored.sort(key=lambda x: -x[0])
                for _, t in class_t_scored:
                    fout.write(json.dumps(t) + "\n")
                logger.warning(f"    Label {label}: restored all {len(class_t)} traces")

    logger.info(f"  Output: {dst}")


def merge_traces_with_train(cfg: dict):
    """Merge judge-filtered traces with training data for student training."""
    logger = setup_logging("trace_merge")

    processed = cfg["data"]["processed_dir"]
    train_df = pd.read_json(os.path.join(processed, "train.jsonl"), lines=True)

    trace_dir = os.path.join(cfg["project"]["output_dir"], "teacher_traces")
    jf = os.path.join(trace_dir, "full_traces_judge_filtered.jsonl")
    hf = os.path.join(trace_dir, "full_traces_hard_filtered.jsonl")
    trace_path = jf if os.path.exists(jf) else hf

    out_path = os.path.join(processed, "train_cot.jsonl")

    traces = {}
    summaries = {}
    severities = {}
    with open(trace_path) as f:
        for line in f:
            obj = json.loads(line)
            traces[obj["idx"]] = obj["teacher_cot"]
            summaries[obj["idx"]] = obj.get("teacher_summary", "")
            severities[obj["idx"]] = obj.get("teacher_severity", "")

    train_df["teacher_cot"] = train_df.index.map(lambda i: traces.get(i))
    train_df["teacher_summary"] = train_df.index.map(lambda i: summaries.get(i, ""))
    train_df["teacher_severity"] = train_df.index.map(lambda i: severities.get(i, ""))
    cot_df = train_df.dropna(subset=["teacher_cot"]).reset_index(drop=True)

    cot_df.to_json(out_path, orient="records", lines=True)
    logger.info(f"Merged {len(cot_df):,} pairs with teacher traces -> {out_path}")
    logger.info(f"Covers {cot_df['label'].nunique()} / "
                f"{train_df['label'].nunique()} classes")


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Phase 2: Judge filtering")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip judging, only merge existing filtered traces")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.merge_only:
        merge_traces_with_train(cfg)
    else:
        judge_filter_traces(cfg)
        merge_traces_with_train(cfg)
