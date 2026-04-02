"""
Phase 1.6 – Grounded factuality verification.

Verifies that pharmacological entities mentioned in teacher reasoning traces
are actually present in the DrugBank profiles of the relevant drugs. This
provides an automated, source-grounded quality metric.

Scoring:
  entity_precision  — fraction of trace-mentioned entities found in either drug's profile
  entity_recall     — fraction of profile entities that the trace actually discusses
  grounded_score    — 0.7 * precision + 0.3 * recall (precision-biased: penalise hallucination)

Supports JSONL-based resume.
"""

import os
import re
import json
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from src.utils import load_config, setup_logging, set_seed

# ── Entity parsing from DrugBank profiles ────────────────────────────

PROFILE_ENTRY_RE = re.compile(r"^(.+?)\s*\(([^)]+)\):\s*(\w+)", re.IGNORECASE)

TRANSPORTER_ALIASES = {
    "abcb1": {"p-glycoprotein", "p-gp", "pgp", "mdr1", "abcb1"},
    "abcg2": {"bcrp", "abcg2"},
    "abcc1": {"mrp1", "abcc1"},
    "abcc2": {"mrp2", "abcc2"},
    "abcc3": {"mrp3", "abcc3"},
    "slco1b1": {"oatp1b1", "slco1b1"},
    "slco1b3": {"oatp1b3", "slco1b3"},
    "slc22a1": {"oct1", "slc22a1"},
    "slc22a2": {"oct2", "slc22a2"},
    "slc22a6": {"oat1", "slc22a6"},
    "slc22a8": {"oat3", "slc22a8"},
    "slc47a1": {"mate1", "slc47a1"},
    "slc47a2": {"mate2-k", "mate2k", "slc47a2"},
}

CYP_TRACE_RE = re.compile(r"\bCYP\d[A-Z]\d+\b", re.IGNORECASE)
GENE_SYMBOL_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,6}\b")


def _parse_profile_entry(entry: str) -> Tuple[str, str, str]:
    """Parse 'Full name (GENE): role' into (full_name, gene_symbol, role)."""
    m = PROFILE_ENTRY_RE.match(entry.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    return entry.strip(), "", ""


def _build_profile_entity_set(profile: dict) -> Set[str]:
    """Build a normalised set of all entity names from a drug profile."""
    entities = set()

    for entry in profile.get("enzymes", []):
        full, gene, _ = _parse_profile_entry(entry)
        if full:
            entities.add(full.lower())
        if gene:
            entities.add(gene.lower())
            cyp_m = CYP_TRACE_RE.search(full) or CYP_TRACE_RE.search(gene)
            if cyp_m:
                entities.add(cyp_m.group().lower())

    for entry in profile.get("transporters", []):
        full, gene, _ = _parse_profile_entry(entry)
        if full:
            entities.add(full.lower())
        if gene:
            entities.add(gene.lower())
            for alias_key, alias_set in TRANSPORTER_ALIASES.items():
                if gene.lower() in alias_set or full.lower() in alias_set:
                    entities.update(alias_set)

    for entry in profile.get("targets", []):
        full, gene, _ = _parse_profile_entry(entry)
        if full:
            entities.add(full.lower())
        if gene:
            entities.add(gene.lower())

    return entities


def _build_profile_structured(profile: dict) -> Dict[str, List[dict]]:
    """Build structured entity lists from a drug profile for detailed breakdown."""
    structured = {"enzymes": [], "transporters": [], "targets": []}

    for entry in profile.get("enzymes", []):
        full, gene, role = _parse_profile_entry(entry)
        names = {full.lower()} if full else set()
        if gene:
            names.add(gene.lower())
        structured["enzymes"].append({"names": names, "role": role, "raw": entry})

    for entry in profile.get("transporters", []):
        full, gene, role = _parse_profile_entry(entry)
        names = {full.lower()} if full else set()
        if gene:
            names.add(gene.lower())
            for alias_key, alias_set in TRANSPORTER_ALIASES.items():
                if gene.lower() in alias_set or full.lower() in alias_set:
                    names.update(alias_set)
        structured["transporters"].append({"names": names, "role": role, "raw": entry})

    for entry in profile.get("targets", []):
        full, gene, role = _parse_profile_entry(entry)
        names = {full.lower()} if full else set()
        if gene:
            names.add(gene.lower())
        structured["targets"].append({"names": names, "role": role, "raw": entry})

    return structured


# ── Entity extraction from trace text ────────────────────────────────

KNOWN_TRANSPORTERS = {
    "p-glycoprotein", "p-gp", "pgp", "mdr1", "abcb1",
    "bcrp", "abcg2",
    "mrp1", "abcc1", "mrp2", "abcc2", "mrp3", "abcc3",
    "oatp1b1", "slco1b1", "oatp1b3", "slco1b3",
    "oct1", "slc22a1", "oct2", "slc22a2",
    "oat1", "slc22a6", "oat3", "slc22a8",
    "mate1", "slc47a1", "mate2-k", "mate2k", "slc47a2",
    "oatp", "oct", "oat", "mate",
}

KNOWN_RECEPTOR_KEYWORDS = {
    "gaba", "gaba-a", "gaba-b", "gabaa", "gabab",
    "nmda", "ampa", "5-ht", "serotonin receptor",
    "dopamine receptor", "adrenergic receptor",
    "muscarinic", "nicotinic", "opioid receptor",
    "histamine receptor", "angiotensin receptor",
    "glucocorticoid receptor", "mineralocorticoid receptor",
    "ppar", "estrogen receptor", "androgen receptor",
    "thyroid receptor",
}

RECEPTOR_ALIAS_PREFIXES = {
    "dopamine receptor": {"drd", "d(1", "d(2", "d(3", "d(4", "dopamine receptor"},
    "serotonin receptor": {"htr", "5-hydroxytryptamine", "serotonin"},
    "5-ht": {"htr", "5-hydroxytryptamine", "serotonin"},
    "adrenergic receptor": {"adra", "adrb", "adrenergic"},
    "muscarinic": {"chrm", "muscarinic"},
    "nicotinic": {"chrn", "nicotinic"},
    "opioid receptor": {"oprm", "oprd", "oprk", "opioid"},
    "histamine receptor": {"hrh", "histamine"},
    "gaba": {"gabr", "gaba"},
    "gaba-a": {"gabra", "gabrb", "gabrg", "gaba"},
    "gaba-b": {"gabbr", "gaba"},
    "nmda": {"grin", "nmda"},
    "ampa": {"gria", "ampa"},
    "angiotensin receptor": {"agtr", "angiotensin"},
    "glucocorticoid receptor": {"nr3c1", "glucocorticoid"},
    "mineralocorticoid receptor": {"nr3c2", "mineralocorticoid"},
    "ppar": {"ppar"},
    "estrogen receptor": {"esr", "estrogen"},
    "androgen receptor": {"ar", "androgen"},
    "thyroid receptor": {"thr", "thyroid"},
}


def _extract_trace_entities(text: str, profile_entities_d1: Set[str],
                            profile_entities_d2: Set[str]) -> Set[str]:
    """Extract pharmacological entity mentions from a trace."""
    text_lower = text.lower()
    mentioned = set()

    for m in CYP_TRACE_RE.finditer(text):
        mentioned.add(m.group().lower())

    for trans in KNOWN_TRANSPORTERS:
        if len(trans) >= 3 and trans in text_lower:
            mentioned.add(trans)

    all_profile = profile_entities_d1 | profile_entities_d2
    for ent in all_profile:
        if len(ent) >= 3 and ent in text_lower:
            mentioned.add(ent)

    for kw in KNOWN_RECEPTOR_KEYWORDS:
        if kw in text_lower:
            mentioned.add(kw)

    return mentioned


# ── Scoring ──────────────────────────────────────────────────────────

def score_trace(trace_text: str, profile_d1: dict, profile_d2: dict,
                precision_weight: float = 0.7) -> dict:
    """Compute grounded factuality scores for a single trace.

    Returns dict with entity_precision, entity_recall, grounded_score,
    and detailed entity breakdown.
    """
    ent_set_d1 = _build_profile_entity_set(profile_d1)
    ent_set_d2 = _build_profile_entity_set(profile_d2)
    all_profile_entities = ent_set_d1 | ent_set_d2

    mentioned = _extract_trace_entities(trace_text, ent_set_d1, ent_set_d2)

    specific_mentioned = set()
    for ent in mentioned:
        if ent in all_profile_entities:
            specific_mentioned.add(ent)
        elif CYP_TRACE_RE.match(ent):
            specific_mentioned.add(ent)
        elif ent in KNOWN_TRANSPORTERS:
            specific_mentioned.add(ent)
        elif ent in KNOWN_RECEPTOR_KEYWORDS:
            specific_mentioned.add(ent)

    grounded = set()
    for ent in specific_mentioned:
        if ent in all_profile_entities:
            grounded.add(ent)
        elif ent in RECEPTOR_ALIAS_PREFIXES:
            prefixes = RECEPTOR_ALIAS_PREFIXES[ent]
            if any(any(px in pe for px in prefixes) for pe in all_profile_entities):
                grounded.add(ent)
        elif any(ent in pe or pe in ent for pe in all_profile_entities if len(pe) >= 4):
            grounded.add(ent)

    covered_profile = set()
    for pe in all_profile_entities:
        if pe in specific_mentioned:
            covered_profile.add(pe)
            continue
        for ent in grounded:
            if ent in RECEPTOR_ALIAS_PREFIXES:
                prefixes = RECEPTOR_ALIAS_PREFIXES[ent]
                if any(px in pe for px in prefixes):
                    covered_profile.add(pe)
                    break
            elif ent in pe or pe in ent:
                covered_profile.add(pe)
                break

    ungrounded_cyp = set()
    for ent in specific_mentioned - grounded:
        if CYP_TRACE_RE.match(ent):
            ungrounded_cyp.add(ent)

    total_mentioned = len(specific_mentioned)
    total_grounded = len(grounded)
    total_profile = len(all_profile_entities)

    if total_mentioned > 0:
        entity_precision = total_grounded / total_mentioned
    else:
        entity_precision = 1.0

    if total_profile > 0:
        entity_recall = min(len(covered_profile) / total_profile, 1.0)
    else:
        entity_recall = 1.0

    recall_weight = 1.0 - precision_weight
    grounded_score = precision_weight * entity_precision + recall_weight * entity_recall

    return {
        "entity_precision": round(entity_precision, 4),
        "entity_recall": round(entity_recall, 4),
        "grounded_score": round(grounded_score, 4),
        "n_mentioned": total_mentioned,
        "n_grounded": total_grounded,
        "n_profile_entities": total_profile,
        "n_ungrounded_cyp": len(ungrounded_cyp),
        "grounded_entities": sorted(grounded),
        "ungrounded_cyp": sorted(ungrounded_cyp),
        "ungrounded_other": sorted(
            (specific_mentioned - all_profile_entities) - ungrounded_cyp
        ),
    }


# ── Main pipeline ────────────────────────────────────────────────────

def score_all_traces(cfg: dict):
    """Score all hard-filtered traces with grounded factuality metrics."""
    logger = setup_logging("grounded_factuality")

    trace_dir = os.path.join(cfg["project"]["output_dir"], "teacher_traces")
    src = os.path.join(trace_dir, "full_traces_hard_filtered.jsonl")
    dst = os.path.join(trace_dir, "full_traces_scored.jsonl")

    if not os.path.exists(src):
        logger.error(f"No hard-filtered traces at {src}")
        return

    proc_dir = cfg["data"]["processed_dir"]
    with open(os.path.join(proc_dir, "drug_profiles.json")) as f:
        profiles = json.load(f)
    logger.info(f"Loaded {len(profiles):,} drug profiles")

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
        logger.info(f"Resume: {len(done_indices):,} already scored")

    src_lines = sum(1 for _ in open(src))
    logger.info(f"Scoring {src_lines:,} hard-filtered traces")

    total = 0
    score_sum = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    score_buckets = defaultdict(int)
    t_start = time.time()

    with open(src) as fin, open(dst, "a") as fout:
        for line in fin:
            obj = json.loads(line)
            total += 1

            if obj["idx"] in done_indices:
                score_val = obj.get("grounded_score", 0)
                score_sum += score_val
                bucket = int(score_val * 10) / 10
                score_buckets[bucket] += 1
                continue

            d1_id = obj.get("drug1_id", "")
            d2_id = obj.get("drug2_id", "")
            p1 = profiles.get(d1_id, {})
            p2 = profiles.get(d2_id, {})
            text = obj.get("teacher_cot", "")

            scores = score_trace(text, p1, p2, precision_weight)

            obj["entity_precision"] = scores["entity_precision"]
            obj["entity_recall"] = scores["entity_recall"]
            obj["grounded_score"] = scores["grounded_score"]
            obj["n_mentioned_entities"] = scores["n_mentioned"]
            obj["n_grounded_entities"] = scores["n_grounded"]
            obj["n_profile_entities"] = scores["n_profile_entities"]
            obj["n_ungrounded_cyp"] = scores["n_ungrounded_cyp"]
            obj["ungrounded_cyp"] = scores["ungrounded_cyp"]

            fout.write(json.dumps(obj) + "\n")

            score_sum += scores["grounded_score"]
            precision_sum += scores["entity_precision"]
            recall_sum += scores["entity_recall"]
            bucket = int(scores["grounded_score"] * 10) / 10
            score_buckets[bucket] += 1

            if total % 25000 == 0:
                elapsed = time.time() - t_start
                rate = total / elapsed if elapsed > 0 else 0
                avg = score_sum / total
                logger.info(
                    f"  Progress: {total:,}/{src_lines:,} | "
                    f"avg score: {avg:.3f} | {rate:.0f}/s"
                )

    elapsed = time.time() - t_start
    avg_score = score_sum / total if total else 0
    avg_prec = precision_sum / total if total else 0
    avg_rec = recall_sum / total if total else 0

    logger.info(f"\nGrounded factuality scoring complete in {elapsed:.1f}s")
    logger.info(f"  Total traces scored: {total:,}")
    logger.info(f"  Avg grounded_score:  {avg_score:.4f}")
    logger.info(f"  Avg entity_precision:{avg_prec:.4f}")
    logger.info(f"  Avg entity_recall:   {avg_rec:.4f}")

    logger.info("  Score distribution:")
    for bucket in sorted(score_buckets.keys()):
        cnt = score_buckets[bucket]
        pct = 100 * cnt / total if total else 0
        bar = "#" * int(pct / 2)
        logger.info(f"    [{bucket:.1f}-{bucket+0.1:.1f}): {cnt:>6,} ({pct:5.1f}%) {bar}")

    logger.info(f"  Output: {dst}")

    _write_summary(cfg, dst, total, avg_score, avg_prec, avg_rec, score_buckets)


def _write_summary(cfg, scored_path, total, avg_score, avg_prec, avg_rec,
                   score_buckets):
    """Write a JSON summary file for downstream modules."""
    results_dir = os.path.join(cfg["project"]["output_dir"], "results")
    os.makedirs(results_dir, exist_ok=True)

    summary = {
        "total_traces": total,
        "avg_grounded_score": round(avg_score, 4),
        "avg_entity_precision": round(avg_prec, 4),
        "avg_entity_recall": round(avg_rec, 4),
        "score_distribution": {
            f"{k:.1f}": v for k, v in sorted(score_buckets.items())
        },
        "scored_file": scored_path,
    }

    out = os.path.join(results_dir, "grounded_factuality_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)


def identify_low_scoring(cfg: dict) -> str:
    """Identify traces below the score percentile threshold.

    Writes the low-scoring traces to a separate file for refinement and
    returns the output path.
    """
    logger = setup_logging("grounded_factuality")

    trace_dir = os.path.join(cfg["project"]["output_dir"], "teacher_traces")
    scored = os.path.join(trace_dir, "full_traces_scored.jsonl")
    low_out = os.path.join(trace_dir, "traces_for_refinement.jsonl")
    high_out = os.path.join(trace_dir, "traces_high_quality.jsonl")

    if not os.path.exists(scored):
        logger.error(f"No scored traces at {scored}")
        return ""

    ge_cfg = cfg.get("grounded_eval", {})
    percentile = ge_cfg.get("low_score_percentile", 15)

    scores = []
    with open(scored) as f:
        for line in f:
            obj = json.loads(line)
            scores.append(obj.get("grounded_score", 0.0))

    if not scores:
        logger.error("No scores found")
        return ""

    scores_sorted = sorted(scores)
    threshold_idx = int(len(scores_sorted) * percentile / 100)
    threshold = scores_sorted[threshold_idx]

    logger.info(f"Percentile threshold: bottom {percentile}% = score < {threshold:.4f}")
    logger.info(f"  Total traces: {len(scores):,}")

    n_low = 0
    n_high = 0

    with open(scored) as fin, \
         open(low_out, "w") as flow, \
         open(high_out, "w") as fhigh:
        for line in fin:
            obj = json.loads(line)
            if obj.get("grounded_score", 0.0) < threshold:
                flow.write(line)
                n_low += 1
            else:
                fhigh.write(line)
                n_high += 1

    logger.info(f"  Low-scoring (for refinement): {n_low:,}")
    logger.info(f"  High-quality (direct to training): {n_high:,}")
    logger.info(f"  Refinement file: {low_out}")

    return low_out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Phase 1.6: Grounded factuality verification"
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", action="store_true",
                        help="After scoring, split into high/low quality files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    score_all_traces(cfg)
    if args.split:
        identify_low_scoring(cfg)
