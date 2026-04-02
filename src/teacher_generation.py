"""
Phase 1 – Generate structured teacher reasoning traces with Llama-3.3-70B-Instruct.

The teacher receives enriched drug profiles (names, mechanisms, enzymes, targets,
transporters, SMILES) from DrugBank XML, dynamic few-shot examples via Tanimoto
retrieval, and the ground-truth interaction label. It produces:

  ## Reasoning   (numbered steps)
  ## Summary     (2-3 sentences)
  ## Classification  (Y={label} -- "{label_text}")
  ## Severity    (Major/Moderate/Minor/Unknown)

Supports JSONL-based resume: on restart, loads already-generated traces and
continues from where it stopped.
"""

import os
import re
import json
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.utils import load_config, setup_logging, set_seed, ensure_dirs
from src.data_preparation import TEACHER_SYSTEM_PROMPT, build_teacher_prompt

# ── Quality checks ────────────────────────────────────────────────────

MIN_COT_LENGTH = 200
MIN_STEPS = 2
MIN_CHARS_PER_STEP = 30

STEP_PATTERN = re.compile(r"[Ss]tep\s*\d|^\d+[\.\):]|\*\*Step", re.MULTILINE)
CLASSIFICATION_PATTERN = re.compile(r"Y\s*=\s*(\d+)", re.IGNORECASE)
SECTION_REASONING = re.compile(r"##\s*Reasoning", re.IGNORECASE)
SECTION_SUMMARY = re.compile(r"##\s*Summary", re.IGNORECASE)
SECTION_CLASSIFICATION = re.compile(r"##\s*Classification", re.IGNORECASE)
SECTION_SEVERITY = re.compile(r"##\s*Severity", re.IGNORECASE)


def _has_repetition_fast(text: str, min_block: int = 40, min_repeats: int = 3) -> bool:
    """Detect degenerate copy-paste loops via sliding-window hashing."""
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


def _extract_summary(text: str) -> str:
    """Extract the Summary section from structured teacher output."""
    m_start = SECTION_SUMMARY.search(text)
    if not m_start:
        return ""
    after = text[m_start.end():]
    m_end = re.search(r"##\s*", after)
    summary = after[:m_end.start()].strip() if m_end else after.strip()
    return summary


def _extract_severity(text: str) -> str:
    """Extract the severity label from teacher output."""
    m_start = SECTION_SEVERITY.search(text)
    if not m_start:
        return ""
    after = text[m_start.end():].strip()
    first_line = after.split("\n")[0].strip()
    for sev in ("Major", "Moderate", "Minor", "Unknown"):
        if sev.lower() in first_line.lower():
            return sev
    return first_line[:20]


def _assess_quality(text: str, drug1_name: str = "", drug2_name: str = "",
                    label: int = -1) -> dict:
    """Multi-level quality assessment of a structured teacher trace.

    Checks:
      1. Sections  — Reasoning, Summary, Classification sections present
      2. Structure — at least MIN_STEPS numbered steps in Reasoning
      3. Length    — at least MIN_COT_LENGTH chars total
      4. Step depth — average step >= MIN_CHARS_PER_STEP
      5. Drug relevance — both drug names mentioned
      6. Label coherence — classification matches ground-truth
      7. Degeneration — no copy-paste loops
    """
    text_lower = text.lower()

    has_reasoning = bool(SECTION_REASONING.search(text))
    has_summary = bool(SECTION_SUMMARY.search(text))
    has_classification = bool(SECTION_CLASSIFICATION.search(text))
    has_sections = has_reasoning and has_summary and has_classification

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

    teacher_summary = _extract_summary(text)
    teacher_severity = _extract_severity(text)

    passed = (has_sections and has_structure and has_length and has_depth
              and drugs_relevant and label_coherent and not has_repetition)

    return {
        "quality_pass": passed,
        "has_sections": has_sections,
        "has_reasoning": has_reasoning,
        "has_summary": has_summary,
        "has_classification": has_classification,
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
        "teacher_summary": teacher_summary,
        "teacher_severity": teacher_severity,
    }


# ── Checkpoint handling ───────────────────────────────────────────────

def _load_checkpoint(trace_path: str) -> set:
    """Return set of already-generated pair indices."""
    done = set()
    if os.path.exists(trace_path):
        with open(trace_path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done.add(obj["idx"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


# ── Main generation ───────────────────────────────────────────────────

def generate_traces(cfg: dict):
    """Generate structured teacher traces for all training pairs."""
    from vllm import LLM, SamplingParams

    logger = setup_logging("teacher_generation")
    set_seed(cfg["project"]["seed"])
    ensure_dirs(cfg)

    processed = cfg["data"]["processed_dir"]
    train_df = pd.read_json(os.path.join(processed, "train.jsonl"), lines=True)
    with open(os.path.join(processed, "label_map.json")) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(processed, "drug_profiles.json")) as f:
        profiles = json.load(f)

    retrievals = {}
    retr_path = os.path.join(processed, "retrieved_examples_train.json")
    if os.path.exists(retr_path):
        with open(retr_path) as f:
            raw_retr = json.load(f)
        for k, v in raw_retr.items():
            retrievals[int(k)] = v
        logger.info(f"  Loaded retrieved examples for {len(retrievals):,} pairs")

    trace_dir = os.path.join(cfg["project"]["output_dir"], "teacher_traces")
    os.makedirs(trace_dir, exist_ok=True)

    pilot_n = cfg.get("_pilot_n", 0)
    if pilot_n > 0:
        trace_file = os.path.join(trace_dir, "pilot_traces.jsonl")
        train_df = train_df.sample(n=min(pilot_n, len(train_df)),
                                   random_state=cfg["project"]["seed"]).reset_index(drop=True)
        logger.info(f"PILOT MODE — {len(train_df):,} pairs sampled")
    else:
        trace_file = os.path.join(trace_dir, "full_traces.jsonl")

    done_indices = _load_checkpoint(trace_file)
    remaining = train_df[~train_df.index.isin(done_indices)]
    logger.info(f"Total training pairs: {len(train_df):,}")
    logger.info(f"  Already completed: {len(done_indices):,}")
    logger.info(f"  Remaining: {len(remaining):,}")

    if len(remaining) == 0:
        logger.info("All traces already generated — nothing to do.")
        return

    tcfg = cfg["teacher"]
    model_name = tcfg["model_name"]
    tp = tcfg["tensor_parallel_size"]
    logger.info(f"Loading teacher: {model_name}  (tp={tp})")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp,
        dtype=tcfg["dtype"],
        max_model_len=tcfg["max_model_len"],
        gpu_memory_utilization=tcfg["gpu_memory_utilization"],
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    params = SamplingParams(
        temperature=tcfg["temperature"],
        top_p=tcfg["top_p"],
        max_tokens=tcfg["max_new_tokens"],
    )

    batch_size = tcfg["batch_size"]
    save_every = tcfg["save_every_n_batches"]

    total_generated = 0
    total_quality_pass = 0
    fail_reasons = {
        "no_sections": 0, "no_structure": 0, "too_short": 0,
        "shallow_steps": 0, "drugs_missing": 0, "label_mismatch": 0,
        "repetition": 0,
    }
    t_start = time.time()

    for batch_start in tqdm(range(0, len(remaining), batch_size),
                            desc="Teacher generation", unit="batch"):
        batch = remaining.iloc[batch_start:batch_start + batch_size]

        prompts = []
        for _, row in batch.iterrows():
            idx = int(row.name) if hasattr(row, 'name') else int(_)
            retr_examples = retrievals.get(idx, [])
            user_msg = build_teacher_prompt(row, label_map, profiles, retr_examples)
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
            )

            rec = {
                "idx": int(orig_idx),
                "drug1_id": row["drug1_id"],
                "drug2_id": row["drug2_id"],
                "drug1_name": str(row.get("drug1_name", "")),
                "drug2_name": str(row.get("drug2_name", "")),
                "label": int(row["label"]),
                "label_text": str(row.get("label_text", "")),
                "severity": str(row.get("severity", "Unknown")),
                "quality_pass": quality["quality_pass"],
                "has_sections": quality["has_sections"],
                "n_steps": quality["n_steps"],
                "avg_step_depth": quality["avg_step_depth"],
                "drugs_relevant": quality["drugs_relevant"],
                "label_coherent": quality["label_coherent"],
                "has_repetition": quality["has_repetition"],
                "cot_length": quality["cot_length"],
                "teacher_cot": text,
                "teacher_summary": quality["teacher_summary"],
                "teacher_severity": quality["teacher_severity"],
            }
            records.append(rec)
            total_generated += 1

            if quality["quality_pass"]:
                total_quality_pass += 1
            else:
                if not quality["has_sections"]:   fail_reasons["no_sections"] += 1
                if not quality["has_structure"]:   fail_reasons["no_structure"] += 1
                if not quality["has_length"]:      fail_reasons["too_short"] += 1
                if not quality["has_depth"]:       fail_reasons["shallow_steps"] += 1
                if not quality["drugs_relevant"]:  fail_reasons["drugs_missing"] += 1
                if not quality["label_coherent"]:  fail_reasons["label_mismatch"] += 1
                if quality["has_repetition"]:      fail_reasons["repetition"] += 1

        with open(trace_file, "a") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        batch_num = batch_start // batch_size + 1
        if batch_num % save_every == 0:
            elapsed = time.time() - t_start
            rate = total_generated / elapsed * 3600
            qpass_pct = (100 * total_quality_pass / total_generated
                         if total_generated else 0)
            logger.info(
                f"Batch {batch_num} | Generated: {total_generated:,} | "
                f"Quality pass: {qpass_pct:.1f}% | Rate: {rate:.0f} pairs/hr"
            )

    elapsed = time.time() - t_start
    qpass_pct = 100 * total_quality_pass / total_generated if total_generated else 0
    logger.info(f"\nGeneration complete in {elapsed/3600:.1f}h")
    logger.info(f"  Total generated : {total_generated:,}")
    logger.info(f"  Quality pass    : {total_quality_pass:,} ({qpass_pct:.1f}%)")
    logger.info(f"  Traces saved to : {trace_file}")

    rejected = total_generated - total_quality_pass
    if rejected > 0:
        logger.info(f"  Rejection breakdown ({rejected:,} traces):")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            if count:
                logger.info(f"    {reason}: {count:,} "
                            f"({100*count/rejected:.1f}% of rejections)")


def filter_traces(cfg: dict):
    """Re-assess raw traces with current quality criteria, write filtered output."""
    logger = setup_logging("trace_filter")

    processed = cfg["data"]["processed_dir"]
    train_df = pd.read_json(os.path.join(processed, "train.jsonl"), lines=True)
    train_lookup = {int(i): row for i, row in train_df.iterrows()}

    trace_dir = os.path.join(cfg["project"]["output_dir"], "teacher_traces")
    src = os.path.join(trace_dir, "full_traces.jsonl")
    dst = os.path.join(trace_dir, "full_traces_filtered.jsonl")

    if not os.path.exists(src):
        logger.error(f"No raw traces found at {src}")
        return

    src_lines = sum(1 for _ in open(src))
    logger.info(f"Filtering {src_lines:,} traces from {src}")

    total, kept = 0, 0
    label_kept = {}
    label_total = {}
    fail_counts = {
        "sections": 0, "structure": 0, "length": 0, "depth": 0,
        "drugs": 0, "label": 0, "repetition": 0,
    }
    t_start = time.time()

    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            total += 1
            obj = json.loads(line)
            label = obj["label"]
            label_total[label] = label_total.get(label, 0) + 1

            row = train_lookup.get(obj["idx"])
            d1 = str(row.get("drug1_name", "")) if row is not None else ""
            d2 = str(row.get("drug2_name", "")) if row is not None else ""

            quality = _assess_quality(
                obj.get("teacher_cot", ""),
                drug1_name=d1, drug2_name=d2,
                label=label,
            )

            if quality["quality_pass"]:
                obj["quality_pass"] = True
                obj["teacher_summary"] = quality["teacher_summary"]
                obj["teacher_severity"] = quality["teacher_severity"]
                fout.write(json.dumps(obj) + "\n")
                kept += 1
                label_kept[label] = label_kept.get(label, 0) + 1
            else:
                if not quality["has_sections"]:    fail_counts["sections"] += 1
                if not quality["has_structure"]:    fail_counts["structure"] += 1
                if not quality["has_length"]:       fail_counts["length"] += 1
                if not quality["has_depth"]:        fail_counts["depth"] += 1
                if not quality["drugs_relevant"]:   fail_counts["drugs"] += 1
                if not quality["label_coherent"]:   fail_counts["label"] += 1
                if quality["has_repetition"]:       fail_counts["repetition"] += 1

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
                logger.info(f"  {reason}: {count:,} "
                            f"({100*count/rejected:.1f}% of rejections)")

    missing = set(label_total.keys()) - set(label_kept.keys())
    if missing:
        logger.warning(f"Classes with 0 quality traces: {sorted(missing)}")

    logger.info(f"Saved to {dst}")


# ── CLI entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Teacher trace generation")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--filter-only", action="store_true",
                        help="Skip generation, only re-filter existing traces")
    parser.add_argument("--pilot", type=int, default=0,
                        help="Generate traces for N random pairs only (for testing)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.pilot > 0:
        cfg["_pilot_n"] = args.pilot

    if args.filter_only:
        filter_traces(cfg)
    else:
        generate_traces(cfg)
        filter_traces(cfg)
