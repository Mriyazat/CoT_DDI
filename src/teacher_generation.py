"""
Phase 3 – Generate chain-of-thought reasoning traces with the 70B teacher model.

The teacher is given the CORRECT interaction label and asked to explain
the pharmacological reasoning behind it. This is standard CoT distillation:
the teacher generates rationales, not predictions.

Uses vLLM for high-throughput batched inference across multiple GPUs.
Supports checkpointing so generation can resume after a crash / timeout.
"""

import os
import re
import json
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.utils import load_config, setup_logging, set_seed, ensure_dirs
from src.data_preparation import TEACHER_SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, build_teacher_prompt

MIN_COT_LENGTH = 200
MIN_STEPS = 3
MIN_CHARS_PER_STEP = 30
STEP_PATTERN = re.compile(r"[Ss]tep\s*\d|^\d+[\.\):]|\*\*Step", re.MULTILINE)
CLASSIFICATION_PATTERN = re.compile(
    r"[Cc]lassification\s*:\s*Y\s*=\s*(\d+)", re.IGNORECASE
)
def _has_repetition_fast(text: str, min_block: int = 40, min_repeats: int = 3) -> bool:
    """Detect degenerate copy-paste loops using sliding-window hashing.

    The original backreference regex (.{40,}?)\\1{2,} is O(n^3) and takes
    ~57ms per trace (2.5 hours for 153K traces). This hash-based approach
    is O(n) per window size and finishes the full dataset in under 2 minutes.
    """
    text_len = len(text)
    if text_len < min_block * min_repeats:
        return False
    for block_len in (40, 60, 80, 120):
        if text_len < block_len * min_repeats:
            continue
        seen: dict[str, int] = {}
        step = max(1, block_len // 4)
        for start in range(0, text_len - block_len + 1, step):
            chunk = text[start:start + block_len]
            seen[chunk] = seen.get(chunk, 0) + 1
            if seen[chunk] >= min_repeats:
                return True
    return False


def _assess_quality(text: str, drug1_name: str = "", drug2_name: str = "",
                    label: int = -1, label_text: str = "") -> dict:
    """
    Multi-level quality assessment of a teacher CoT trace.

    Checks:
      1. Structure  — at least MIN_STEPS numbered reasoning steps
      2. Length     — at least MIN_COT_LENGTH chars total
      3. Step depth — average step content >= MIN_CHARS_PER_STEP
      4. Drug relevance — both drug names mentioned in the trace
      5. Label coherence — classification line matches ground-truth label
      6. No degeneration — no long repeated blocks (copy-paste loops)
    """
    text_lower = text.lower()

    step_positions = [m.start() for m in STEP_PATTERN.finditer(text)]
    n_steps = len(step_positions)
    has_structure = n_steps >= MIN_STEPS

    cot_length = len(text)
    has_length = cot_length >= MIN_COT_LENGTH

    step_depths = []
    for i, pos in enumerate(step_positions):
        end = step_positions[i + 1] if i + 1 < len(step_positions) else cot_length
        step_depths.append(end - pos)
    avg_step_depth = sum(step_depths) / len(step_depths) if step_depths else 0
    has_depth = avg_step_depth >= MIN_CHARS_PER_STEP

    d1 = drug1_name.lower().split()[-1] if drug1_name else ""
    d2 = drug2_name.lower().split()[-1] if drug2_name else ""
    drug1_mentioned = (d1 in text_lower) if d1 else True
    drug2_mentioned = (d2 in text_lower) if d2 else True
    drugs_relevant = drug1_mentioned and drug2_mentioned

    cls_match = CLASSIFICATION_PATTERN.findall(text)
    if cls_match and label >= 0:
        predicted_label = int(cls_match[-1])
        label_coherent = predicted_label == label
    else:
        label_coherent = True

    has_repetition = _has_repetition_fast(text)

    passed = (has_structure and has_length and has_depth
              and drugs_relevant and label_coherent and not has_repetition)

    return {
        "quality_pass": passed,
        "n_steps": n_steps,
        "avg_step_depth": round(avg_step_depth),
        "has_structure": has_structure,
        "has_length": has_length,
        "has_depth": has_depth,
        "drugs_relevant": drugs_relevant,
        "drug1_mentioned": drug1_mentioned,
        "drug2_mentioned": drug2_mentioned,
        "label_coherent": label_coherent,
        "has_repetition": has_repetition,
        "cot_length": cot_length,
    }


def _load_checkpoint(trace_path: str) -> set:
    """Return set of already-generated pair indices."""
    done = set()
    if os.path.exists(trace_path):
        with open(trace_path) as f:
            for line in f:
                obj = json.loads(line)
                done.add(obj["idx"])
    return done


def generate_traces(cfg: dict, pilot: bool = False):
    from vllm import LLM, SamplingParams

    logger = setup_logging("teacher_generation")
    set_seed(cfg["project"]["seed"])
    ensure_dirs(cfg)

    train_path = os.path.join(cfg["data"]["processed_dir"], "train.jsonl")
    train_df = pd.read_json(train_path, lines=True)
    label_map_path = os.path.join(cfg["data"]["processed_dir"], "label_map.json")
    with open(label_map_path) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    if pilot:
        n = cfg["teacher"]["pilot_size"]
        train_df = train_df.sample(n=n, random_state=cfg["project"]["seed"]).reset_index(drop=True)
        trace_file = os.path.join(cfg["project"]["output_dir"], "teacher_traces", "pilot_traces.jsonl")
        logger.info(f"PILOT MODE — generating traces for {n:,} pairs")
    else:
        trace_file = os.path.join(cfg["project"]["output_dir"], "teacher_traces", "full_traces.jsonl")
        logger.info(f"FULL MODE — generating traces for {len(train_df):,} pairs")

    done_indices = _load_checkpoint(trace_file)
    remaining = train_df[~train_df.index.isin(done_indices)]
    logger.info(f"  Already completed: {len(done_indices):,}  |  Remaining: {len(remaining):,}")

    if len(remaining) == 0:
        logger.info("All traces already generated — nothing to do.")
        return

    model_name = cfg["teacher"]["model_name"]
    tp = cfg["teacher"]["tensor_parallel_size"]
    logger.info(f"Loading teacher: {model_name}  (tensor_parallel={tp})")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp,
        dtype=cfg["teacher"]["dtype"],
        max_model_len=cfg["teacher"]["max_model_len"],
        gpu_memory_utilization=cfg["teacher"]["gpu_memory_utilization"],
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    params = SamplingParams(
        temperature=cfg["teacher"]["temperature"],
        top_p=cfg["teacher"]["top_p"],
        max_tokens=cfg["teacher"]["max_new_tokens"],
    )

    batch_size = cfg["teacher"]["batch_size"]
    save_every = cfg["teacher"]["save_every_n_batches"]
    verify_every = cfg["teacher"].get("verify_every_n_batches", 50)
    verify_sample = cfg["teacher"].get("verify_sample_size", 10)

    judge_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=200)

    total_generated = 0
    total_quality_pass = 0
    recent_records = []
    all_verify_scores = []
    verify_log_path = os.path.join(
        cfg["project"]["output_dir"], "teacher_traces", "verification_log.jsonl"
    )
    t_start = time.time()

    for batch_start in tqdm(range(0, len(remaining), batch_size),
                            desc="Teacher generation", unit="batch"):
        batch = remaining.iloc[batch_start:batch_start + batch_size]

        prompts = []
        for _, row in batch.iterrows():
            user_msg = build_teacher_prompt(row, label_map)
            messages = [
                {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))

        outputs = llm.generate(prompts, params)

        records = []
        for (orig_idx, row), out in zip(batch.iterrows(), outputs):
            text = out.outputs[0].text.strip()
            quality = _assess_quality(
                text,
                drug1_name=str(row.get("drug1_name", "")),
                drug2_name=str(row.get("drug2_name", "")),
                label=int(row["label"]),
                label_text=str(row.get("label_text", "")),
            )

            rec = {
                "idx": int(orig_idx),
                "drug1_id": row["drug1_id"],
                "drug2_id": row["drug2_id"],
                "drug1_name": str(row.get("drug1_name", "")),
                "drug2_name": str(row.get("drug2_name", "")),
                "label": int(row["label"]),
                "label_text": str(row.get("label_text", "")),
                "quality_pass": quality["quality_pass"],
                "n_steps": quality["n_steps"],
                "avg_step_depth": quality["avg_step_depth"],
                "drugs_relevant": quality["drugs_relevant"],
                "label_coherent": quality["label_coherent"],
                "has_repetition": quality["has_repetition"],
                "cot_length": quality["cot_length"],
                "teacher_cot": text,
            }
            records.append(rec)
            total_generated += 1
            if quality["quality_pass"]:
                total_quality_pass += 1

        recent_records.extend(records)

        with open(trace_file, "a") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        batch_num = batch_start // batch_size + 1

        if batch_num % verify_every == 0 and recent_records:
            import random as _rng
            candidates = [r for r in recent_records if r["quality_pass"]]
            n_verify = min(verify_sample, len(candidates))
            if n_verify > 0:
                sample = _rng.sample(candidates, n_verify)
                vscores = _run_inline_verify(
                    llm, tokenizer, judge_params, sample, logger
                )
                all_verify_scores.extend(vscores)
                with open(verify_log_path, "a") as vf:
                    for vs in vscores:
                        vf.write(json.dumps(vs) + "\n")
            recent_records = []

        if batch_num % save_every == 0:
            elapsed = time.time() - t_start
            rate = total_generated / elapsed * 3600
            qpass_pct = 100 * total_quality_pass / total_generated if total_generated else 0
            verify_str = ""
            if all_verify_scores:
                valid = [s for s in all_verify_scores if s["overall"] > 0]
                if valid:
                    avg = sum(s["overall"] for s in valid) / len(valid)
                    verify_str = f" | LLM judge: {avg:.1f}/5 ({len(valid)} scored)"
            logger.info(
                f"Batch {batch_num} | Generated: {total_generated:,} | "
                f"Quality pass: {qpass_pct:.1f}% | Rate: {rate:.0f} pairs/hr"
                f"{verify_str}"
            )

    elapsed = time.time() - t_start
    qpass_pct = 100 * total_quality_pass / total_generated if total_generated else 0
    logger.info(f"\nGeneration complete in {elapsed/3600:.1f}h")
    logger.info(f"  Total generated : {total_generated:,}")
    logger.info(f"  Quality pass    : {qpass_pct:.1f}%")

    if all_verify_scores:
        valid = [s for s in all_verify_scores if s["overall"] > 0]
        if valid:
            logger.info(f"  LLM verification ({len(valid)} samples):")
            for dim in JUDGE_DIMS + ["overall"]:
                avg = sum(s[dim] for s in valid) / len(valid)
                logger.info(f"    {dim:20s}: {avg:.2f} / 5")
            n_pass = sum(1 for s in valid if s["verdict"] == "PASS")
            logger.info(f"    Pass rate           : {100 * n_pass / len(valid):.1f}%")

    logger.info(f"  Traces saved to : {trace_file}")
    logger.info(f"  Verify log      : {verify_log_path}")


def filter_traces(cfg: dict, pilot: bool = False):
    """
    Re-assess and filter traces using the full quality criteria.

    This reads the raw trace file and re-runs _assess_quality with drug name
    and label information from the training data, so traces generated with an
    older/looser quality function are re-evaluated against the current criteria.
    """
    logger = setup_logging("trace_filter")

    train_path = os.path.join(cfg["data"]["processed_dir"], "train.jsonl")
    train_df = pd.read_json(train_path, lines=True)
    train_lookup = {}
    for i, row in train_df.iterrows():
        train_lookup[int(i)] = row

    if pilot:
        src = os.path.join(cfg["project"]["output_dir"], "teacher_traces", "pilot_traces.jsonl")
        dst = os.path.join(cfg["project"]["output_dir"], "teacher_traces", "pilot_traces_filtered.jsonl")
    else:
        src = os.path.join(cfg["project"]["output_dir"], "teacher_traces", "full_traces.jsonl")
        dst = os.path.join(cfg["project"]["output_dir"], "teacher_traces", "full_traces_filtered.jsonl")

    src_lines = sum(1 for _ in open(src))
    logger.info(f"Filtering {src_lines:,} traces from {src}")

    total, kept = 0, 0
    label_kept = {}
    label_total = {}
    fail_counts = {"structure": 0, "length": 0, "depth": 0,
                   "drugs": 0, "label": 0, "repetition": 0}
    t_start = time.time()

    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            total += 1
            obj = json.loads(line)
            label = obj["label"]
            label_total[label] = label_total.get(label, 0) + 1

            idx = obj["idx"]
            row = train_lookup.get(idx)
            d1 = str(row.get("drug1_name", "")) if row is not None else ""
            d2 = str(row.get("drug2_name", "")) if row is not None else ""
            lt = str(row.get("label_text", "")) if row is not None else ""

            quality = _assess_quality(
                obj.get("teacher_cot", ""),
                drug1_name=d1, drug2_name=d2,
                label=label, label_text=lt,
            )

            if quality["quality_pass"]:
                obj["quality_pass"] = True
                fout.write(json.dumps(obj) + "\n")
                kept += 1
                label_kept[label] = label_kept.get(label, 0) + 1
            else:
                if not quality["has_structure"]:  fail_counts["structure"] += 1
                if not quality["has_length"]:     fail_counts["length"] += 1
                if not quality["has_depth"]:      fail_counts["depth"] += 1
                if not quality["drugs_relevant"]: fail_counts["drugs"] += 1
                if not quality["label_coherent"]: fail_counts["label"] += 1
                if quality["has_repetition"]:     fail_counts["repetition"] += 1

            if total % 10000 == 0:
                elapsed = time.time() - t_start
                rate = total / elapsed
                eta = (src_lines - total) / rate if rate > 0 else 0
                pct_kept = 100 * kept / total if total else 0
                logger.info(
                    f"  Progress: {total:,}/{src_lines:,} "
                    f"({100*total/src_lines:.0f}%) | "
                    f"kept {kept:,} ({pct_kept:.1f}%) | "
                    f"{rate:.0f} traces/s | ETA {eta:.0f}s"
                )

    elapsed = time.time() - t_start
    n_classes_covered = len(label_kept)
    n_classes_total = len(label_total)
    logger.info(f"Filtered: {kept:,} / {total:,} traces kept "
                f"({100*kept/total:.1f}%) in {elapsed:.1f}s")
    logger.info(f"Classes covered: {n_classes_covered} / {n_classes_total}")

    rejected = total - kept
    if rejected > 0:
        logger.info(f"Rejection breakdown ({rejected:,} traces):")
        for reason, count in sorted(fail_counts.items(), key=lambda x: -x[1]):
            if count:
                logger.info(f"  {reason}: {count:,} ({100*count/rejected:.1f}% of rejections)")

    missing = set(label_total.keys()) - set(label_kept.keys())
    if missing:
        logger.warning(f"Classes with 0 quality traces: {sorted(missing)}")
        for l in sorted(missing):
            logger.warning(f"  Label {l}: {label_total[l]} pairs, 0 kept")

    logger.info(f"Saved to {dst}")


JUDGE_PROMPT = """You are a senior pharmacologist conducting a rigorous peer review. \
A junior researcher wrote the explanation below to teach a student how to reason \
about drug-drug interactions. Determine if this explanation teaches CORRECT \
pharmacological reasoning the student can apply to NEW drug pairs.

=== DRUG PAIR ===
Drug 1: {drug1_name}
Drug 2: {drug2_name}
Known interaction: {label_text}

=== EXPLANATION TO REVIEW ===
{cot}

=== SCORING CRITERIA (1-5, strict: 3=acceptable, 4=good, 5=excellent) ===

1. DRUG_SPECIFICITY: Names specific enzymes, receptors, transporters, or metabolic \
pathways for each drug? (1=generic "Drug A affects Drug B", 5=names CYP isoforms, \
specific receptors, transporters)

2. MECHANISM_ACCURACY: Mechanism is pharmacologically plausible and consistent with \
established knowledge? (1=fabricated mechanism, 3=plausible but vague, 5=textbook-accurate)

3. CAUSAL_CHAIN: Steps build logically: individual drug mechanisms → combined effect \
→ stated interaction type? (1=disconnected statements, 3=partial flow, 5=clear causal chain)

4. TEACHING_VALUE: Student learns a transferable reasoning pattern applicable to \
similar drug pairs? (1=rote memorization, 5=teaches generalizable pharmacological reasoning)

5. FACTUAL_ERRORS: Any errors, contradictions, or hallucinated mechanisms? \
(1=major errors, 3=minor inaccuracies, 5=no detectable errors)

=== OUTPUT ===
Write a brief analysis (2-4 sentences max). Then output scores in EXACTLY this \
format, each on its own line:

DRUG_SPECIFICITY: <score>
MECHANISM_ACCURACY: <score>
CAUSAL_CHAIN: <score>
TEACHING_VALUE: <score>
FACTUAL_ERRORS: <score>
OVERALL: <score>
VERDICT: <PASS if OVERALL >= 3, else FAIL>"""

JUDGE_DIMS = ["drug_specificity", "mechanism_accuracy", "causal_chain",
              "teaching_value", "factual_errors"]
SCORE_PATTERN = re.compile(
    r"DRUG_SPECIFICITY:\s*(\d).*MECHANISM_ACCURACY:\s*(\d).*CAUSAL_CHAIN:\s*(\d).*"
    r"TEACHING_VALUE:\s*(\d).*FACTUAL_ERRORS:\s*(\d).*OVERALL:\s*(\d).*VERDICT:\s*(PASS|FAIL)",
    re.DOTALL | re.IGNORECASE,
)


THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def _parse_judge_response(resp: str, rec: dict) -> dict:
    """Parse a structured judge response into a score dict.

    Handles DeepSeek-R1 models that wrap reasoning in <think>...</think>
    tags before outputting the actual scores.  Falls back to individual
    score extraction when the full regex doesn't match (e.g. slightly
    different formatting or one missing field).
    """
    cleaned = THINK_PATTERN.sub("", resp).strip()

    m = SCORE_PATTERN.search(cleaned)
    if not m:
        m = SCORE_PATTERN.search(resp)

    if m:
        return {
            "idx": rec["idx"],
            "label": rec["label"],
            "drug_specificity": int(m.group(1)),
            "mechanism_accuracy": int(m.group(2)),
            "causal_chain": int(m.group(3)),
            "teaching_value": int(m.group(4)),
            "factual_errors": int(m.group(5)),
            "overall": int(m.group(6)),
            "verdict": m.group(7).upper(),
        }

    result = {"idx": rec["idx"], "label": rec["label"]}
    for key, dim_name in [
        ("drug_specificity", "DRUG_SPECIFICITY"),
        ("mechanism_accuracy", "MECHANISM_ACCURACY"),
        ("causal_chain", "CAUSAL_CHAIN"),
        ("teaching_value", "TEACHING_VALUE"),
        ("factual_errors", "FACTUAL_ERRORS"),
        ("overall", "OVERALL"),
    ]:
        dm = re.search(rf"{dim_name}\s*[:=]\s*(\d)", cleaned, re.IGNORECASE)
        result[key] = int(dm.group(1)) if dm else 0

    vm = re.search(r"VERDICT\s*[:=]\s*(PASS|FAIL)", cleaned, re.IGNORECASE)
    result["verdict"] = vm.group(1).upper() if vm else (
        "PASS" if result.get("overall", 0) >= 3 else "FAIL"
    )

    if result["overall"] > 0:
        return result

    return {
        "idx": rec["idx"], "label": rec["label"],
        "drug_specificity": 0, "mechanism_accuracy": 0, "causal_chain": 0,
        "teaching_value": 0, "factual_errors": 0,
        "overall": 0, "verdict": "PARSE_ERROR",
    }


JUDGE_SYSTEM_MSG = "You are a pharmacology expert evaluating explanations. Be strict."


def _build_judge_prompts(tokenizer, sample_records, no_system_prompt=False):
    """Build judge prompts, handling models that don't support system prompts."""
    prompts = []
    for rec in sample_records:
        user_msg = JUDGE_PROMPT.format(
            drug1_name=rec.get("drug1_name", rec.get("drug1_id", "")),
            drug2_name=rec.get("drug2_name", rec.get("drug2_id", "")),
            label_text=rec.get("label_text", f"Y={rec['label']}"),
            cot=rec["teacher_cot"][:2000],
        )
        if no_system_prompt:
            messages = [{"role": "user", "content": JUDGE_SYSTEM_MSG + "\n\n" + user_msg}]
        else:
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM_MSG},
                {"role": "user", "content": user_msg},
            ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))
    return prompts


def _run_inline_verify(llm, tokenizer, judge_params, sample_records, logger,
                       no_system_prompt=False):
    """Run judge prompts through an already-loaded LLM."""
    prompts = _build_judge_prompts(tokenizer, sample_records, no_system_prompt)
    outputs = llm.generate(prompts, judge_params)

    scores = []
    for rec, out in zip(sample_records, outputs):
        resp = out.outputs[0].text.strip()
        score = _parse_judge_response(resp, rec)
        scores.append(score)

    valid = [s for s in scores if s["overall"] > 0]
    if valid:
        avg = sum(s["overall"] for s in valid) / len(valid)
        n_pass = sum(1 for s in valid if s["verdict"] == "PASS")
        logger.info(
            f"  Verify: {len(valid)} scored | "
            f"avg {avg:.1f}/5 | {n_pass}/{len(valid)} PASS"
        )
    return scores


def _log_judge_report(scores, judge_name, logger):
    """Print a summary report for one judge's scores."""
    valid = [s for s in scores if s["overall"] > 0]
    if not valid:
        logger.warning(f"  [{judge_name}] Could not parse any responses")
        return valid

    logger.info(f"\n  [{judge_name}] — {len(valid)} scored:")
    for dim in JUDGE_DIMS + ["overall"]:
        avg = sum(s[dim] for s in valid) / len(valid)
        logger.info(f"    {dim:20s}: {avg:.2f} / 5")
    n_pass = sum(1 for s in valid if s["verdict"] == "PASS")
    n_fail = sum(1 for s in valid if s["verdict"] == "FAIL")
    logger.info(f"    {'pass_rate':20s}: {100 * n_pass / len(valid):.1f}% "
                 f"({n_pass} pass / {n_fail} fail)")

    low_dims = {}
    for dim in JUDGE_DIMS:
        low = [s for s in valid if s[dim] <= 2]
        if low:
            low_dims[dim] = len(low)
    if low_dims:
        logger.warning(f"    Weak dimensions (score <= 2):")
        for dim, cnt in sorted(low_dims.items(), key=lambda x: -x[1]):
            logger.warning(f"      {dim}: {cnt} traces ({100*cnt/len(valid):.1f}%)")

    parse_errors = len(scores) - len(valid)
    if parse_errors:
        logger.warning(f"    Parse errors: {parse_errors}")

    return valid


def _sample_traces_for_verify(traces, n_sample, seed):
    """Stratified sample across all label classes."""
    import random
    random.seed(seed)

    by_label = {}
    for t in traces:
        by_label.setdefault(t["label"], []).append(t)

    sample = []
    per_class = max(1, n_sample // len(by_label))
    for label, class_traces in by_label.items():
        sample.extend(random.sample(class_traces, min(per_class, len(class_traces))))
    if len(sample) > n_sample:
        sample = random.sample(sample, n_sample)
    return sample, by_label


def verify_traces(cfg: dict, n_sample: int = 0, pilot: bool = False):
    """
    Independent verification using judge model(s) from config.

    Loads each judge model sequentially, scores the same sample of traces,
    then compares results across judges. Uses models from cfg['judge']['models']
    which should be DIFFERENT from the teacher for unbiased evaluation.
    """
    from vllm import LLM, SamplingParams
    import gc

    logger = setup_logging("trace_verifier")

    jcfg = cfg.get("judge", {})
    judge_model_list = jcfg.get("models", [])
    if not judge_model_list:
        judge_model_list = [{"model_name": cfg["teacher"]["model_name"],
                             "tensor_parallel_size": cfg["teacher"]["tensor_parallel_size"]}]
    normalized = []
    for m in judge_model_list:
        if isinstance(m, str):
            m = {"model_name": m}
        normalized.append(m)
    judge_model_list = normalized

    if n_sample <= 0:
        n_sample = jcfg.get("sample_size", 300)

    if pilot:
        trace_path = os.path.join(cfg["project"]["output_dir"],
                                  "teacher_traces", "pilot_traces_filtered.jsonl")
    else:
        trace_path = os.path.join(cfg["project"]["output_dir"],
                                  "teacher_traces", "full_traces_filtered.jsonl")

    if not os.path.exists(trace_path):
        logger.error(f"No filtered traces found at {trace_path}")
        return

    traces = []
    with open(trace_path) as f:
        for line in f:
            traces.append(json.loads(line))

    train_path = os.path.join(cfg["data"]["processed_dir"], "train.jsonl")
    train_df = pd.read_json(train_path, lines=True)
    train_lookup = {int(i): row for i, row in train_df.iterrows()}
    for t in traces:
        row = train_lookup.get(t["idx"])
        if row is not None:
            t.setdefault("drug1_name", str(row.get("drug1_name", "")))
            t.setdefault("drug2_name", str(row.get("drug2_name", "")))
            t.setdefault("label_text", str(row.get("label_text", "")))

    sample, by_label = _sample_traces_for_verify(
        traces, n_sample, cfg["project"]["seed"]
    )
    logger.info(f"Sampled {len(sample):,} traces across {len(by_label)} classes")

    all_judge_results = {}

    for mcfg in judge_model_list:
        model_name = mcfg["model_name"]
        m_tp = mcfg.get("tensor_parallel_size", 4)
        m_dtype = mcfg.get("dtype", "float16")
        m_mml = mcfg.get("max_model_len", 4096)
        m_gpu = mcfg.get("gpu_memory_utilization", 0.90)
        no_sys = mcfg.get("no_system_prompt", False)
        m_temp = mcfg.get("temperature", 0.1)

        short_name = model_name.split("/")[-1]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Loading judge: {model_name}  (tp={m_tp}, no_sys={no_sys})")
        logger.info(f"{'=' * 60}")

        llm = LLM(
            model=model_name,
            tensor_parallel_size=m_tp,
            dtype=m_dtype,
            max_model_len=m_mml,
            gpu_memory_utilization=m_gpu,
            trust_remote_code=True,
        )
        tokenizer = llm.get_tokenizer()
        max_judge_tokens = 1200 if no_sys else 800
        judge_params = SamplingParams(temperature=m_temp, top_p=0.9,
                                      max_tokens=max_judge_tokens)

        scores = _run_inline_verify(
            llm, tokenizer, judge_params, sample, logger,
            no_system_prompt=no_sys,
        )
        valid = _log_judge_report(scores, short_name, logger)
        all_judge_results[short_name] = scores

        del llm
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    report = {
        "sample_size": len(sample),
        "n_classes": len(by_label),
        "judges": {},
    }
    for jname, scores in all_judge_results.items():
        valid = [s for s in scores if s["overall"] > 0]
        report["judges"][jname] = {
            "n_scored": len(valid),
            "n_parse_errors": len(scores) - len(valid),
            "scores": scores,
        }
        if valid:
            for dim in JUDGE_DIMS + ["overall"]:
                report["judges"][jname][f"avg_{dim}"] = round(
                    sum(s[dim] for s in valid) / len(valid), 2
                )
            report["judges"][jname]["pass_rate"] = round(
                100 * sum(1 for s in valid if s["verdict"] == "PASS") / len(valid), 1
            )

    if len(all_judge_results) > 1:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"CROSS-JUDGE COMPARISON")
        logger.info(f"{'=' * 60}")

        idx_to_scores = {}
        for jname, scores in all_judge_results.items():
            for s in scores:
                if s["overall"] > 0:
                    idx_to_scores.setdefault(s["idx"], {})[jname] = s

        agree = 0
        disagree = 0
        disagree_indices = []
        for idx, judge_scores in idx_to_scores.items():
            if len(judge_scores) < 2:
                continue
            verdicts = [s["verdict"] for s in judge_scores.values()]
            if len(set(verdicts)) == 1:
                agree += 1
            else:
                disagree += 1
                disagree_indices.append(idx)

        total_compared = agree + disagree
        if total_compared > 0:
            logger.info(f"  Traces scored by all judges: {total_compared}")
            logger.info(f"  Agreement: {agree} ({100*agree/total_compared:.1f}%)")
            logger.info(f"  Disagreement: {disagree} ({100*disagree/total_compared:.1f}%)")

            if disagree_indices:
                report["disagreed_indices"] = disagree_indices[:50]
                logger.info(f"  Disagreed trace indices (first 20): "
                             f"{disagree_indices[:20]}")

    report_path = os.path.join(cfg["project"]["output_dir"],
                               "teacher_traces", "verification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nFull report saved to {report_path}")


def judge_filter_traces(cfg: dict, pilot: bool = False):
    """Score ALL filtered traces with every configured judge model, then apply
    consensus filtering to keep only high-quality traces for student training.

    Architecture:
      1. For each judge model: load → score all traces → checkpoint → unload GPU
      2. Combine per-judge scores using consensus strategy
      3. Write only consensus-passing traces to output file

    Consensus strategies (judge.consensus_strategy in config):
      - "all_pass":    Every judge must give overall >= min_score (strictest)
      - "majority":    More than half of judges give overall >= min_score
      - "average":     Mean overall across judges >= min_score
      - "either_pass": At least one judge gives overall >= min_score (most lenient)

    Each judge has its own checkpoint file for cluster-safe resumability.
    If the cluster kills the job mid-judge, rerunning picks up exactly where
    each judge left off without re-scoring already-scored traces.
    """
    from vllm import LLM, SamplingParams
    import gc

    logger = setup_logging("judge_filter")

    jcfg = cfg.get("judge", {})
    judge_models = jcfg.get("models", [])
    if not judge_models:
        logger.error("No judge models configured")
        return

    normalized = []
    for m in judge_models:
        if isinstance(m, str):
            m = {"model_name": m}
        normalized.append(m)
    judge_models = normalized

    min_score = jcfg.get("min_overall_score", 3)
    consensus = jcfg.get("consensus_strategy", "both_pass")

    trace_dir = os.path.join(cfg["project"]["output_dir"], "teacher_traces")
    suffix = "pilot" if pilot else "full"
    trace_path = os.path.join(trace_dir, f"{suffix}_traces_filtered.jsonl")
    out_path = os.path.join(trace_dir, f"{suffix}_traces_judge_filtered.jsonl")

    if not os.path.exists(trace_path):
        logger.error(f"No filtered traces at {trace_path}")
        return

    train_path = os.path.join(cfg["data"]["processed_dir"], "train.jsonl")
    train_df = pd.read_json(train_path, lines=True)
    train_lookup = {int(i): row for i, row in train_df.iterrows()}

    traces = []
    with open(trace_path) as f:
        for line in f:
            t = json.loads(line)
            row = train_lookup.get(t["idx"])
            if row is not None:
                t.setdefault("drug1_name", str(row.get("drug1_name", "")))
                t.setdefault("drug2_name", str(row.get("drug2_name", "")))
                t.setdefault("label_text", str(row.get("label_text", "")))
            traces.append(t)

    judge_names = [m["model_name"].split("/")[-1] for m in judge_models]
    logger.info(f"Consensus judge-filter: strategy={consensus}, "
                f"min_overall={min_score}")
    logger.info(f"  Judges: {judge_names}")
    logger.info(f"  Filtered traces to score: {len(traces):,}")

    all_judge_scores = {}

    for mcfg in judge_models:
        model_name = mcfg["model_name"]
        short_name = model_name.split("/")[-1]
        no_sys = mcfg.get("no_system_prompt", False)
        m_temp = mcfg.get("temperature", 0.1)

        score_path = os.path.join(
            trace_dir, f"{suffix}_judge_scores_{short_name}.jsonl"
        )

        done_idx = set()
        scores_by_idx = {}
        if os.path.exists(score_path):
            with open(score_path) as f:
                for line in f:
                    s = json.loads(line)
                    done_idx.add(s["idx"])
                    scores_by_idx[s["idx"]] = s

        remaining = [t for t in traces if t["idx"] not in done_idx]

        logger.info(f"\n{'─' * 60}")
        logger.info(f"Judge: {short_name}  (tp={mcfg.get('tensor_parallel_size', 2)}, "
                     f"no_sys={no_sys})")
        logger.info(f"  Already scored: {len(done_idx):,}  |  "
                     f"Remaining: {len(remaining):,}")

        if remaining:
            logger.info(f"  Loading model: {model_name}")
            llm = LLM(
                model=model_name,
                tensor_parallel_size=mcfg.get("tensor_parallel_size", 2),
                dtype=mcfg.get("dtype", "float16"),
                max_model_len=mcfg.get("max_model_len", 4096),
                gpu_memory_utilization=mcfg.get("gpu_memory_utilization", 0.90),
                trust_remote_code=True,
            )
            tokenizer = llm.get_tokenizer()
            max_tokens = 1200 if no_sys else 800
            judge_params = SamplingParams(
                temperature=m_temp, top_p=0.9, max_tokens=max_tokens,
            )

            batch_size = 256
            t_start = time.time()
            n_done = 0

            for batch_start in range(0, len(remaining), batch_size):
                batch = remaining[batch_start:batch_start + batch_size]
                prompts = _build_judge_prompts(tokenizer, batch, no_sys)
                outputs = llm.generate(prompts, judge_params)

                with open(score_path, "a") as sf:
                    for rec, out in zip(batch, outputs):
                        resp = out.outputs[0].text.strip()
                        score = _parse_judge_response(resp, rec)
                        score["judge"] = short_name
                        sf.write(json.dumps(score) + "\n")
                        scores_by_idx[rec["idx"]] = score

                n_done += len(batch)
                elapsed = time.time() - t_start
                rate = n_done / elapsed
                eta = (len(remaining) - n_done) / rate if rate > 0 else 0
                scored = [s for s in scores_by_idx.values()
                          if s["overall"] > 0]
                avg = (sum(s["overall"] for s in scored) / len(scored)
                       if scored else 0)
                logger.info(
                    f"  [{short_name}] "
                    f"{len(done_idx)+n_done:,}/{len(traces):,} | "
                    f"batch {batch_start//batch_size+1} | "
                    f"avg {avg:.2f}/5 | {rate:.1f}/s | "
                    f"ETA {eta/60:.0f}min"
                )

            del llm
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        valid = [s for s in scores_by_idx.values() if s["overall"] > 0]
        parse_errors = len(scores_by_idx) - len(valid)
        if valid:
            avg_overall = sum(s["overall"] for s in valid) / len(valid)
            pass_ct = sum(1 for s in valid if s["overall"] >= min_score)
            logger.info(f"  [{short_name}] Done — scored: {len(valid):,} | "
                        f"parse errors: {parse_errors:,} | "
                        f"avg overall: {avg_overall:.2f} | "
                        f"pass rate: {100*pass_ct/len(valid):.1f}%")

        all_judge_scores[short_name] = scores_by_idx

    logger.info(f"\n{'=' * 60}")
    logger.info(f"CONSENSUS FILTERING  (strategy: {consensus}, "
                f"min_overall: {min_score})")
    logger.info(f"{'=' * 60}")

    passed_idx = set()
    n_all_parsed = 0
    n_partial = 0
    n_no_parse = 0

    for t in traces:
        idx = t["idx"]
        valid_scores = []
        for jname, jscores in all_judge_scores.items():
            if idx in jscores and jscores[idx]["overall"] > 0:
                valid_scores.append(jscores[idx]["overall"])

        if not valid_scores:
            n_no_parse += 1
            continue

        if len(valid_scores) == len(judge_models):
            n_all_parsed += 1
        else:
            n_partial += 1

        if consensus == "all_pass":
            if (len(valid_scores) == len(judge_models)
                    and all(s >= min_score for s in valid_scores)):
                passed_idx.add(idx)
        elif consensus == "majority":
            n_pass = sum(1 for s in valid_scores if s >= min_score)
            if n_pass > len(judge_models) / 2:
                passed_idx.add(idx)
        elif consensus == "average":
            if sum(valid_scores) / len(valid_scores) >= min_score:
                passed_idx.add(idx)
        elif consensus == "either_pass":
            if any(s >= min_score for s in valid_scores):
                passed_idx.add(idx)

    with open(out_path, "w") as fout:
        for t in traces:
            if t["idx"] in passed_idx:
                fout.write(json.dumps(t) + "\n")

    by_label = {}
    for t in traces:
        if t["idx"] in passed_idx:
            by_label.setdefault(t["label"], []).append(t)

    logger.info(f"  Scored by ALL judges:  {n_all_parsed:,}")
    logger.info(f"  Partial scores only:   {n_partial:,}")
    logger.info(f"  No valid scores:       {n_no_parse:,}")
    logger.info(f"  Consensus PASS: {len(passed_idx):,} / {len(traces):,} "
                f"({100*len(passed_idx)/len(traces):.1f}%)")
    logger.info(f"  Classes covered: {len(by_label)} / 86")

    missing = set(range(1, 87)) - set(by_label.keys())
    if missing:
        logger.warning(f"  Classes with 0 traces after filtering: "
                       f"{sorted(missing)}")

    for jname, jscores in all_judge_scores.items():
        valid = [s for s in jscores.values() if s["overall"] > 0]
        if valid:
            logger.info(f"\n  [{jname}] dimension averages:")
            for dim in JUDGE_DIMS + ["overall"]:
                avg = sum(s[dim] for s in valid) / len(valid)
                logger.info(f"    {dim:20s}: {avg:.2f} / 5")

    logger.info(f"\n  Output: {out_path}")


def merge_traces_with_train(cfg: dict, pilot: bool = False,
                            use_judge_filtered: bool = False):
    """Join filtered traces back with training data for student training."""
    logger = setup_logging("trace_merge")

    train_path = os.path.join(cfg["data"]["processed_dir"], "train.jsonl")
    train_df = pd.read_json(train_path, lines=True)

    trace_dir = os.path.join(cfg["project"]["output_dir"], "teacher_traces")
    suffix = "pilot" if pilot else "full"

    if use_judge_filtered:
        jf = os.path.join(trace_dir, f"{suffix}_traces_judge_filtered.jsonl")
        if os.path.exists(jf):
            trace_path = jf
            logger.info("Using judge-filtered traces")
        else:
            trace_path = os.path.join(trace_dir, f"{suffix}_traces_filtered.jsonl")
            logger.warning("Judge-filtered file not found, falling back to structural filter")
    else:
        trace_path = os.path.join(trace_dir, f"{suffix}_traces_filtered.jsonl")

    out_path = os.path.join(cfg["data"]["processed_dir"],
                            "train_cot_pilot.jsonl" if pilot else "train_cot.jsonl")

    traces = {}
    with open(trace_path) as f:
        for line in f:
            obj = json.loads(line)
            traces[obj["idx"]] = obj["teacher_cot"]

    train_df["teacher_cot"] = train_df.index.map(lambda i: traces.get(i, None))
    cot_df = train_df.dropna(subset=["teacher_cot"]).reset_index(drop=True)

    n_classes = cot_df["label"].nunique()
    cot_df.to_json(out_path, orient="records", lines=True)
    logger.info(f"Merged {len(cot_df):,} pairs with CoT traces → {out_path}")
    logger.info(f"Covers {n_classes} / {train_df['label'].nunique()} classes")
