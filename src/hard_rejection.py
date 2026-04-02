"""
Phase 1.5 – Hard trace rejection (deterministic pre-judge filter).

Catches the exact problems found in v2 at zero compute cost before sending
traces to expensive judge LLMs. Four rules:

  1. Drug misidentification: trace says "also known as <wrong drug>"
  2. Unresolved DrugBank IDs: raw DB##### still present in reasoning
  3. No mechanism content: no pharmacological entity at all
  4. Wrong classification: extracted Y=label != ground truth

For tail classes (<500 traces), rule 3 is relaxed because some interactions
genuinely don't involve CYP enzymes. Rules 1, 2, 4 always apply.

Supports JSONL-based resume.
"""

import os
import re
import json
import time
from collections import defaultdict

from src.utils import load_config, setup_logging, set_seed

# ── Rule patterns ─────────────────────────────────────────────────────

MISID_PATTERN = re.compile(
    r"(?:also\s+known\s+as|identified\s+as|referred\s+to\s+as)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    re.IGNORECASE,
)

DBID_IN_BODY = re.compile(r"DB\d{5}")

MECH_ENTITIES = re.compile(
    r"CYP\d|enzyme|receptor|transporter|pathway|substrate|inhibit|induc|"
    r"agonist|antagonist|reuptake|metaboli[sz]|clearance|absorption|"
    r"P-glycoprotein|P-gp|efflux|affinity|binding|kinase|oxidase|reductase|"
    r"dehydrogenase|transferase|channel|excret|bioavailab|half-life|"
    r"pharmacokinet|pharmacodynam|plasma\s+(?:level|concentration)",
    re.IGNORECASE,
)

CLASSIFICATION_PATTERN = re.compile(r"Y\s*=\s*(\d+)", re.IGNORECASE)


# ── Rule functions ────────────────────────────────────────────────────

def _check_drug_misidentification(text: str, drug1_name: str,
                                   drug2_name: str,
                                   drug1_synonyms: set = None,
                                   drug2_synonyms: set = None,
                                   all_drug_names: set = None) -> bool:
    """Return True if trace misidentifies either drug (REJECT).

    Two checks:
      1. Neither drug name (or synonym) appears in the trace at all.
      2. An "also known as <other drug>" appears in the first 600 chars
         where drugs are identified — but NOT deeper in the reasoning
         where the teacher legitimately references endogenous compounds.
    """
    if not drug1_name or not drug2_name:
        return False

    text_lower = text.lower()
    valid_names = set()
    for name in (drug1_name, drug2_name):
        valid_names.add(name.lower())
        for word in name.lower().split():
            if len(word) > 3:
                valid_names.add(word)

    for syn_set in (drug1_synonyms, drug2_synonyms):
        if syn_set:
            for syn in syn_set:
                valid_names.add(syn.lower())

    d1_found = any(v in text_lower for v in [drug1_name.lower()] +
                   [s.lower() for s in (drug1_synonyms or set())])
    d2_found = any(v in text_lower for v in [drug2_name.lower()] +
                   [s.lower() for s in (drug2_synonyms or set())])
    if not d1_found or not d2_found:
        return True

    intro = text[:600]
    for match in MISID_PATTERN.finditer(intro):
        mentioned = match.group(1).strip().lower()
        if any(v in mentioned or mentioned in v for v in valid_names):
            continue
        if all_drug_names and mentioned in all_drug_names:
            return True
    return False


def _check_unresolved_ids(text: str, drug1_id: str, drug2_id: str) -> bool:
    """Return True if raw DrugBank IDs appear in the reasoning body (REJECT).

    The two input drug IDs in the prompt header are allowed; only reject
    if ADDITIONAL unrecognized IDs appear in the reasoning text.
    """
    reasoning_start = text.find("## Reasoning")
    if reasoning_start == -1:
        reasoning_start = 0
    reasoning_text = text[reasoning_start:]

    found_ids = set(DBID_IN_BODY.findall(reasoning_text))
    allowed = {drug1_id, drug2_id}
    unexpected = found_ids - allowed
    return len(unexpected) > 0


def _check_no_mechanism(text: str) -> bool:
    """Return True if trace contains zero pharmacological entities (REJECT)."""
    return not bool(MECH_ENTITIES.search(text))


def _check_wrong_classification(text: str, label: int) -> bool:
    """Return True if extracted Y=label doesn't match ground truth (REJECT)."""
    matches = CLASSIFICATION_PATTERN.findall(text)
    if not matches:
        return True
    predicted = int(matches[-1])
    return predicted != label


# ── Main rejection pipeline ───────────────────────────────────────────

def _load_synonym_map(cfg: dict):
    """Load drug synonyms. Returns (id_to_names, all_drug_names)."""
    syn_path = os.path.join(cfg["data"]["processed_dir"], "drug_synonyms.json")
    if not os.path.exists(syn_path):
        return {}, set()
    with open(syn_path) as f:
        raw = json.load(f)
    id_to_names = defaultdict(set)
    all_drug_names = set()
    for name, ids in raw.items():
        all_drug_names.add(name.lower())
        for did in ids:
            id_to_names[did].add(name)
    return dict(id_to_names), all_drug_names


def hard_reject_traces(cfg: dict):
    """Apply deterministic rejection rules to teacher traces.

    Reads full_traces.jsonl, writes full_traces_hard_filtered.jsonl.
    """
    logger = setup_logging("hard_rejection")

    trace_dir = os.path.join(cfg["project"]["output_dir"], "teacher_traces")
    src = os.path.join(trace_dir, "full_traces.jsonl")
    dst = os.path.join(trace_dir, "full_traces_hard_filtered.jsonl")

    if not os.path.exists(src):
        logger.error(f"No raw traces found at {src}")
        return

    synonym_map, all_drug_names = _load_synonym_map(cfg)
    logger.info(f"Loaded synonyms for {len(synonym_map):,} drugs "
                f"({len(all_drug_names):,} known names)")

    class_counts = defaultdict(int)
    with open(src) as f:
        for line in f:
            obj = json.loads(line)
            class_counts[obj["label"]] += 1

    tail_threshold = cfg.get("judge", {}).get("tiered_thresholds", {}).get(
        "mid_min_traces", 500
    )

    done_indices = set()
    if os.path.exists(dst):
        with open(dst) as f:
            for line in f:
                try:
                    done_indices.add(json.loads(line)["idx"])
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info(f"  Resume: {len(done_indices):,} already filtered")

    src_lines = sum(1 for _ in open(src))
    logger.info(f"Hard rejection on {src_lines:,} traces")

    total = 0
    kept = 0
    rule_rejections = {
        "drug_misid": 0,
        "unresolved_ids": 0,
        "no_mechanism": 0,
        "wrong_classification": 0,
    }
    label_stats = defaultdict(lambda: {"total": 0, "kept": 0})
    t_start = time.time()

    with open(src) as fin, open(dst, "a") as fout:
        for line in fin:
            obj = json.loads(line)
            total += 1

            if obj["idx"] in done_indices:
                continue

            text = obj.get("teacher_cot", "")
            label = obj["label"]
            d1_name = obj.get("drug1_name", "")
            d2_name = obj.get("drug2_name", "")
            d1_id = obj.get("drug1_id", "")
            d2_id = obj.get("drug2_id", "")

            is_tail = class_counts[label] < tail_threshold

            rejected = False
            reasons = []

            d1_syns = synonym_map.get(d1_id, set())
            d2_syns = synonym_map.get(d2_id, set())
            if _check_drug_misidentification(text, d1_name, d2_name,
                                              d1_syns, d2_syns, all_drug_names):
                rejected = True
                reasons.append("drug_misid")
                rule_rejections["drug_misid"] += 1

            if _check_unresolved_ids(text, d1_id, d2_id):
                rejected = True
                reasons.append("unresolved_ids")
                rule_rejections["unresolved_ids"] += 1

            if not is_tail and _check_no_mechanism(text):
                rejected = True
                reasons.append("no_mechanism")
                rule_rejections["no_mechanism"] += 1

            if _check_wrong_classification(text, label):
                rejected = True
                reasons.append("wrong_classification")
                rule_rejections["wrong_classification"] += 1

            label_stats[label]["total"] += 1
            if not rejected:
                fout.write(json.dumps(obj) + "\n")
                kept += 1
                label_stats[label]["kept"] += 1

            if total % 25000 == 0:
                elapsed = time.time() - t_start
                rate = total / elapsed if elapsed > 0 else 0
                pct = 100 * kept / total if total else 0
                logger.info(
                    f"  Progress: {total:,}/{src_lines:,} | "
                    f"kept {kept:,} ({pct:.1f}%) | {rate:.0f}/s"
                )

    elapsed = time.time() - t_start
    logger.info(f"\nHard rejection complete in {elapsed:.1f}s")
    logger.info(f"  Input:    {total:,} traces")
    logger.info(f"  Kept:     {kept:,} ({100*kept/total:.1f}%)")
    logger.info(f"  Rejected: {total-kept:,}")

    if total - kept > 0:
        logger.info("  Rule breakdown:")
        for rule, count in sorted(rule_rejections.items(), key=lambda x: -x[1]):
            if count:
                logger.info(f"    {rule}: {count:,}")

    empty_classes = [l for l, s in label_stats.items() if s["kept"] == 0]
    if empty_classes:
        logger.warning(f"  Classes with 0 traces after hard rejection: "
                       f"{sorted(empty_classes)}")

    heavy_loss = [
        (l, s) for l, s in label_stats.items()
        if s["total"] > 0 and s["kept"] / s["total"] < 0.5
    ]
    if heavy_loss:
        logger.warning(f"  Classes losing >50% of traces: {len(heavy_loss)}")
        for l, s in sorted(heavy_loss, key=lambda x: x[1]["kept"]/max(x[1]["total"],1)):
            logger.warning(f"    Label {l}: {s['kept']}/{s['total']} kept "
                          f"({100*s['kept']/s['total']:.0f}%)")

    logger.info(f"  Output: {dst}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1.5: Hard trace rejection")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    hard_reject_traces(cfg)
