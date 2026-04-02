"""
Data preparation pipeline for DDI CoT Distillation V3.

Handles:
  1. Class filtering (keep >= min_pairs_per_class)
  2. Label remapping to contiguous IDs
  3. Coarse category mapping
  4. Per-class training cap
  5. Stratified 80/20 split
  6. Drug profile enrichment
  7. Severity label attachment
  8. Dynamic few-shot retrieval precomputation
  9. Prompt construction (teacher + student)
"""
import os
import json
import pickle
import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.utils import load_config, setup_logging, set_seed, ensure_dirs, categorize_interaction


# ── Prompt templates ──────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert pharmacologist specialising in drug-drug interactions. "
    "Given two drugs with their pharmacological profiles, analyse their "
    "mechanisms step-by-step and predict their interaction type. "
    "Include the severity if known."
)

TEACHER_SYSTEM_PROMPT = (
    "You are an expert pharmacologist specialising in drug-drug interactions. "
    "Given two drugs and their known interaction type, explain the "
    "pharmacological mechanisms step-by-step. Then provide a concise summary. "
    "Structure your response as:\n\n"
    "## Reasoning\n"
    "[Numbered steps explaining the mechanism]\n\n"
    "## Summary\n"
    "[2-3 sentence summary of the key interaction mechanism]\n\n"
    "## Classification\n"
    "Y={label} -- \"{label_text}\"\n\n"
    "## Severity\n"
    "{Major/Moderate/Minor/Unknown}"
)


def _format_drug_profile(profile: dict) -> str:
    """Format a drug profile into a compact text block for prompts."""
    lines = [f"  Description: {profile['description']}" if profile.get("description") else None]

    if profile.get("mechanism_of_action"):
        lines.append(f"  Mechanism: {profile['mechanism_of_action'][:200]}")

    if profile.get("enzymes"):
        enz_str = "; ".join(profile["enzymes"][:5])
        lines.append(f"  Key enzymes: {enz_str}")

    if profile.get("transporters"):
        trans_str = "; ".join(profile["transporters"][:3])
        lines.append(f"  Transporters: {trans_str}")

    if profile.get("targets"):
        tgt_str = "; ".join(profile["targets"][:3])
        lines.append(f"  Targets: {tgt_str}")

    if profile.get("smiles"):
        lines.append(f"  SMILES: {profile['smiles'][:200]}")

    return "\n".join(l for l in lines if l)


def build_teacher_prompt(row, label_map, profiles, retrieved_examples=None):
    """Construct the enriched teacher prompt with drug profiles and retrieved examples."""
    parts = []

    if retrieved_examples:
        for i, ex in enumerate(retrieved_examples, 1):
            p1 = profiles.get(ex["drug1_id"], {})
            p2 = profiles.get(ex["drug2_id"], {})
            parts.append(f"--- Example {i} ---")
            parts.append(f"Drug 1: {ex['drug1_name']} ({ex['drug1_id']})")
            if p1:
                parts.append(_format_drug_profile(p1))
            parts.append(f"Drug 2: {ex['drug2_name']} ({ex['drug2_id']})")
            if p2:
                parts.append(_format_drug_profile(p2))
            ex_label_text = label_map.get(ex["label"], "")
            if "#Drug1" in ex_label_text:
                ex_label_text = ex_label_text.replace("#Drug1", ex["drug1_name"]).replace("#Drug2", ex["drug2_name"])
            parts.append(f"Interaction: Y={ex['label']} -- \"{ex_label_text}\"")
            parts.append("")

    parts.append("--- Your turn ---")
    p1 = profiles.get(row["drug1_id"], {})
    p2 = profiles.get(row["drug2_id"], {})

    parts.append(f"Drug 1: {row['drug1_name']} ({row['drug1_id']})")
    if p1:
        parts.append(_format_drug_profile(p1))
    parts.append(f"Drug 2: {row['drug2_name']} ({row['drug2_id']})")
    if p2:
        parts.append(_format_drug_profile(p2))

    severity = row.get("severity", "Unknown")
    parts.append(f"Known interaction: Y={row['label']} -- \"{row['label_text']}\"")
    parts.append(f"Known severity: {severity}")
    parts.append("")
    parts.append(
        "Explain step-by-step the pharmacological mechanisms behind this "
        "drug-drug interaction. Discuss each drug's mechanism of action and "
        "how they combine to produce this effect. Then provide a concise summary. "
        "End with the classification and severity."
    )
    return "\n".join(parts)


def build_student_input(row, profiles, retrieved_examples=None):
    """Build the student prompt (no answer, task instruction to predict)."""
    parts = []

    if retrieved_examples:
        for i, ex in enumerate(retrieved_examples, 1):
            p1 = profiles.get(ex["drug1_id"], {})
            p2 = profiles.get(ex["drug2_id"], {})
            parts.append(f"--- Example {i} ---")
            parts.append(f"Drug 1: {ex['drug1_name']} ({ex['drug1_id']})")
            if p1:
                parts.append(_format_drug_profile(p1))
            parts.append(f"Drug 2: {ex['drug2_name']} ({ex['drug2_id']})")
            if p2:
                parts.append(_format_drug_profile(p2))
            parts.append(f"Interaction: Y={ex['label']} -- \"{ex.get('label_text', '')}\"")
            sev = ex.get("severity", "Unknown")
            parts.append(f"Severity: {sev}")
            parts.append("")

    p1 = profiles.get(row["drug1_id"], {})
    p2 = profiles.get(row["drug2_id"], {})

    parts.append(f"Drug 1: {row['drug1_name']} ({row['drug1_id']})")
    if p1:
        parts.append(_format_drug_profile(p1))
    parts.append(f"Drug 2: {row['drug2_name']} ({row['drug2_id']})")
    if p2:
        parts.append(_format_drug_profile(p2))
    parts.append("")
    parts.append("Predict the interaction type, explain the mechanism briefly, "
                 "and state the severity.")
    return "\n".join(parts)


# ── Few-shot retrieval ────────────────────────────────────────────────

def precompute_retrievals(train_df, profiles, drug_id_order, sim_matrix,
                          fingerprints, top_k=5, min_diverse=2, seed=42,
                          batch_size=500):
    """Precompute top-k retrieved examples using vectorized numpy operations.

    Pairs where either drug lacks fingerprints get no retrieved examples
    (empty list) rather than random fallback, to avoid noise.
    """
    id_to_idx = {did: i for i, did in enumerate(drug_id_order)}
    n_total = len(train_df)

    fp_mask = np.array([
        id_to_idx.get(row["drug1_id"]) is not None and
        id_to_idx.get(row["drug2_id"]) is not None
        for _, row in train_df.iterrows()
    ])

    all_indices = np.array(train_df.index.tolist())
    all_labels = np.array(train_df["label"].tolist())

    d1_sim_idx = np.full(n_total, -1, dtype=np.int32)
    d2_sim_idx = np.full(n_total, -1, dtype=np.int32)
    for i, (_, row) in enumerate(train_df.iterrows()):
        i1 = id_to_idx.get(row["drug1_id"])
        i2 = id_to_idx.get(row["drug2_id"])
        if i1 is not None and i2 is not None:
            d1_sim_idx[i] = i1
            d2_sim_idx[i] = i2

    fp_positions = np.where(fp_mask)[0]
    n_with_fp = len(fp_positions)
    n_skip = n_total - n_with_fp
    print(f"  Pairs with fingerprints: {n_with_fp:,}, without: {n_skip:,}")

    cand_d1 = d1_sim_idx[fp_mask]
    cand_d2 = d2_sim_idx[fp_mask]
    cand_labels = all_labels[fp_mask]
    cand_orig_idx = all_indices[fp_mask]

    sim_dense = sim_matrix
    if hasattr(sim_matrix, 'toarray'):
        sim_dense = sim_matrix.toarray()
    elif hasattr(sim_matrix, 'A'):
        sim_dense = np.asarray(sim_matrix.A)
    else:
        sim_dense = np.asarray(sim_matrix)

    retrievals = {}
    for i in range(n_total):
        if not fp_mask[i]:
            retrievals[all_indices[i]] = []

    processed = 0
    for batch_start in range(0, n_with_fp, batch_size):
        batch_end = min(batch_start + batch_size, n_with_fp)
        batch_pos = fp_positions[batch_start:batch_end]
        b_size = len(batch_pos)

        b_d1 = d1_sim_idx[batch_pos]
        b_d2 = d2_sim_idx[batch_pos]

        s_d1_c1 = sim_dense[b_d1][:, cand_d1]
        s_d2_c2 = sim_dense[b_d2][:, cand_d2]
        s_d1_c2 = sim_dense[b_d1][:, cand_d2]
        s_d2_c1 = sim_dense[b_d2][:, cand_d1]

        pair_sim = np.maximum(
            (s_d1_c1 + s_d2_c2) / 2.0,
            (s_d1_c2 + s_d2_c1) / 2.0
        )

        for bi in range(b_size):
            pos_in_fp = batch_start + bi
            orig_idx = all_indices[batch_pos[bi]]
            sims = pair_sim[bi].copy()
            sims[pos_in_fp] = -1.0

            top_n = min(top_k * 10, n_with_fp)
            top_positions = np.argpartition(sims, -top_n)[-top_n:]
            top_positions = top_positions[np.argsort(-sims[top_positions])]

            selected = []
            classes_seen = set()
            for cp in top_positions:
                if len(selected) >= top_k:
                    break
                lbl = cand_labels[cp]
                if len(selected) >= top_k - min_diverse or lbl not in classes_seen:
                    selected.append(int(cand_orig_idx[cp]))
                    classes_seen.add(lbl)

            if len(selected) < top_k:
                for cp in top_positions:
                    if int(cand_orig_idx[cp]) not in selected:
                        selected.append(int(cand_orig_idx[cp]))
                        if len(selected) >= top_k:
                            break

            retrievals[orig_idx] = selected

        processed += b_size
        if processed % 10000 < batch_size or batch_end == n_with_fp:
            print(f"  Retrieval: {processed:,}/{n_with_fp:,} pairs with FP computed...")

    n_with = sum(1 for v in retrievals.values() if v)
    print(f"  Retrieval complete: {n_with:,}/{n_total:,} pairs have retrieved examples, "
          f"{n_skip:,} skipped (no fingerprints)")
    return retrievals


# ── Main preparation pipeline ─────────────────────────────────────────

def prepare_data(cfg: dict):
    logger = setup_logging("data_preparation")
    set_seed(cfg["project"]["seed"])
    ensure_dirs(cfg)

    proc_dir = Path(cfg["data"]["processed_dir"])
    min_pairs = cfg["data"]["min_pairs_per_class"]
    max_train = cfg["data"]["max_train_per_class"]
    train_ratio = cfg["data"]["train_ratio"]
    seed = cfg["project"]["seed"]

    logger.info("Loading extracted data...")
    interactions = []
    with open(proc_dir / "interactions_full.jsonl") as f:
        for line in f:
            interactions.append(json.loads(line))
    logger.info(f"  Total interaction pairs: {len(interactions):,}")

    raw_lm_path = proc_dir / "raw_label_map.json"
    if raw_lm_path.exists():
        with open(raw_lm_path) as f:
            raw_label_map = {int(k): v for k, v in json.load(f).items()}
        logger.info(f"  Raw label classes (from raw_label_map.json): {len(raw_label_map)}")
    else:
        with open(proc_dir / "label_map.json") as f:
            raw_label_map = {int(k): v for k, v in json.load(f).items()}
        with open(raw_lm_path, "w") as f:
            json.dump({str(k): v for k, v in raw_label_map.items()}, f, indent=2)
        logger.info(f"  Raw label classes: {len(raw_label_map)} (backed up to raw_label_map.json)")

    with open(proc_dir / "drug_profiles.json") as f:
        profiles = json.load(f)
    logger.info(f"  Drug profiles: {len(profiles):,}")

    with open(proc_dir / "severity_map.json") as f:
        severity_map = json.load(f)
    logger.info(f"  Severity labels: {len(severity_map):,}")

    # Step 1: Filter to classes with >= min_pairs
    label_counts = Counter(ix["label"] for ix in interactions)
    kept_labels = {lbl for lbl, cnt in label_counts.items() if cnt >= min_pairs}
    filtered = [ix for ix in interactions if ix["label"] in kept_labels]
    logger.info(f"  After filtering (>= {min_pairs} pairs): "
                f"{len(kept_labels)} classes, {len(filtered):,} pairs")

    # Step 1b: Remove pairs where EITHER drug lacks useful pharmacological info.
    # Both drugs must have at least one of: description, mechanism, enzymes, targets, transporters.
    # This prevents the teacher from hallucinating mechanisms for unknown drugs.
    useful_fields = ["description", "mechanism_of_action", "enzymes", "targets", "transporters"]
    before_quality = len(filtered)
    quality_filtered = []
    for ix in filtered:
        p1 = profiles.get(ix["drug1_id"], {})
        p2 = profiles.get(ix["drug2_id"], {})
        d1_has = p1 and any(p1.get(f) for f in useful_fields)
        d2_has = p2 and any(p2.get(f) for f in useful_fields)
        if d1_has and d2_has:
            quality_filtered.append(ix)
    n_removed = before_quality - len(quality_filtered)
    filtered = quality_filtered
    logger.info(f"  Profile quality filter: removed {n_removed:,} pairs "
                f"(at least one drug missing description/mechanism/enzymes/targets/transporters)")
    logger.info(f"  Remaining: {len(filtered):,} pairs")

    # Re-check class counts after quality filter -- some classes may now be below min_pairs
    label_counts_post = Counter(ix["label"] for ix in filtered)
    dropped_classes = {lbl for lbl in kept_labels if label_counts_post.get(lbl, 0) < min_pairs}
    if dropped_classes:
        kept_labels -= dropped_classes
        filtered = [ix for ix in filtered if ix["label"] in kept_labels]
        logger.info(f"  Dropped {len(dropped_classes)} classes below {min_pairs} after quality filter")
        logger.info(f"  Final: {len(kept_labels)} classes, {len(filtered):,} pairs")

    # Step 2: Remap labels to contiguous IDs (1..N)
    old_to_new = {}
    new_label_map = {}
    for new_id, (old_id, template) in enumerate(
        sorted([(lbl, raw_label_map[lbl]) for lbl in kept_labels],
               key=lambda x: -label_counts[x[0]]),
        start=1,
    ):
        old_to_new[old_id] = new_id
        new_label_map[new_id] = template

    for ix in filtered:
        ix["label"] = old_to_new[ix["label"]]

    # Step 3: Build coarse category mapping
    coarse_map = {}
    for label_id, template in new_label_map.items():
        coarse_map[label_id] = categorize_interaction(template)

    coarse_counts = Counter(coarse_map.values())
    logger.info(f"  Coarse categories: {len(coarse_counts)}")
    for cat, cnt in coarse_counts.most_common():
        logger.info(f"    {cat}: {cnt} fine-grained classes")

    # Step 4: Fill label_text with real drug names
    for ix in filtered:
        template = new_label_map[ix["label"]]
        ix["label_text"] = template.replace("#Drug1", ix["drug1_name"]).replace("#Drug2", ix["drug2_name"])
        ix["coarse_category"] = coarse_map[ix["label"]]

    # Step 5: Attach severity labels
    severity_attached = 0
    for ix in filtered:
        pair_key = "_".join(sorted([ix["drug1_id"], ix["drug2_id"]]))
        sev = severity_map.get(pair_key, "Unknown")
        if sev in ("Major", "Moderate", "Minor"):
            ix["severity"] = sev
            severity_attached += 1
        else:
            ix["severity"] = "Unknown"
    logger.info(f"  Severity attached: {severity_attached:,} / {len(filtered):,} "
                f"({100*severity_attached/len(filtered):.1f}%)")

    # Step 6: Stratified 80/20 split
    df = pd.DataFrame(filtered)
    train_df, test_df = train_test_split(
        df, train_size=train_ratio, random_state=seed, stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    logger.info(f"  Split: train={len(train_df):,}, test={len(test_df):,}")

    # Step 7: Per-class training cap
    capped_parts = []
    rng = np.random.RandomState(seed)
    for label in sorted(train_df["label"].unique()):
        group = train_df[train_df["label"] == label]
        if len(group) > max_train:
            group = group.sample(n=max_train, random_state=rng)
        capped_parts.append(group)
    train_df = pd.concat(capped_parts, ignore_index=True)
    train_df = train_df.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    logger.info(f"  After per-class cap ({max_train}): {len(train_df):,} training pairs")

    # Step 8: Log distribution stats
    train_counts = train_df["label"].value_counts()
    test_counts = test_df["label"].value_counts()
    logger.info(f"  Training class distribution:")
    logger.info(f"    Min: {train_counts.min()}, Max: {train_counts.max()}, "
                f"Median: {train_counts.median():.0f}")
    logger.info(f"  Test class distribution:")
    logger.info(f"    Min: {test_counts.min()}, Max: {test_counts.max()}, "
                f"Median: {test_counts.median():.0f}")

    sev_train = train_df[train_df["severity"] != "Unknown"]
    sev_test = test_df[test_df["severity"] != "Unknown"]
    logger.info(f"  Severity-labeled: train={len(sev_train):,}, test={len(sev_test):,}")

    # Save outputs
    train_df.to_json(proc_dir / "train.jsonl", orient="records", lines=True)
    test_df.to_json(proc_dir / "test.jsonl", orient="records", lines=True)

    with open(proc_dir / "label_map.json", "w") as f:
        json.dump(new_label_map, f, indent=2)

    with open(proc_dir / "coarse_category_map.json", "w") as f:
        json.dump(coarse_map, f, indent=2)

    logger.info(f"  Saved train.jsonl ({len(train_df):,} pairs)")
    logger.info(f"  Saved test.jsonl ({len(test_df):,} pairs)")
    logger.info(f"  Saved label_map.json ({len(new_label_map)} classes)")
    logger.info(f"  Saved coarse_category_map.json ({len(coarse_map)} mappings)")

    # Step 9: Precompute few-shot retrievals for training set
    logger.info("Precomputing few-shot retrievals for training pairs...")
    fp_path = proc_dir / "drug_fingerprints.pkl"
    sim_path = proc_dir / "drug_similarity_matrix.npz"
    id_path = proc_dir / "drug_id_order.json"

    if fp_path.exists() and sim_path.exists() and id_path.exists():
        with open(fp_path, "rb") as f:
            fingerprints = pickle.load(f)
        sim_data = np.load(sim_path)
        sim_matrix = sim_data["matrix"]
        with open(id_path) as f:
            drug_id_order = json.load(f)

        top_k = cfg.get("retrieval", {}).get("top_k", 5)
        min_diverse = cfg.get("retrieval", {}).get("min_diverse_classes", 2)

        train_retrievals = precompute_retrievals(
            train_df, profiles, drug_id_order, sim_matrix, fingerprints,
            top_k=top_k, min_diverse=min_diverse, seed=seed,
        )

        retrieval_out = {}
        for idx, selected_indices in train_retrievals.items():
            examples = []
            for sel_idx in selected_indices:
                sel_row = train_df.iloc[sel_idx] if sel_idx < len(train_df) else None
                if sel_row is not None:
                    examples.append({
                        "drug1_id": sel_row["drug1_id"],
                        "drug2_id": sel_row["drug2_id"],
                        "drug1_name": sel_row["drug1_name"],
                        "drug2_name": sel_row["drug2_name"],
                        "label": int(sel_row["label"]),
                        "label_text": sel_row["label_text"],
                        "severity": sel_row.get("severity", "Unknown"),
                    })
            retrieval_out[str(idx)] = examples

        with open(proc_dir / "retrieved_examples_train.json", "w") as f:
            json.dump(retrieval_out, f)
        logger.info(f"  Saved retrieved_examples_train.json ({len(retrieval_out):,} entries)")
    else:
        logger.warning("Fingerprint/similarity files not found. Skipping retrieval precomputation.")

    logger.info("Data preparation complete.")
    return train_df, test_df, new_label_map


def precompute_test_retrievals(cfg: dict):
    """Precompute few-shot retrievals for test set using training candidates."""
    logger = setup_logging("test_retrieval")
    proc_dir = Path(cfg["data"]["processed_dir"])

    train_df = pd.read_json(proc_dir / "train.jsonl", lines=True)
    test_df = pd.read_json(proc_dir / "test.jsonl", lines=True)

    fp_path = proc_dir / "drug_fingerprints.pkl"
    sim_path = proc_dir / "drug_similarity_matrix.npz"
    id_path = proc_dir / "drug_id_order.json"

    if not (fp_path.exists() and sim_path.exists() and id_path.exists()):
        logger.error("Fingerprint/similarity files not found. Run prepare_data first.")
        return

    with open(fp_path, "rb") as f:
        fingerprints = pickle.load(f)
    sim_data = np.load(sim_path)
    sim_matrix = sim_data["matrix"]
    with open(id_path) as f:
        drug_id_order = json.load(f)

    id_to_idx = {did: i for i, did in enumerate(drug_id_order)}
    top_k = cfg.get("retrieval", {}).get("top_k", 5)
    min_diverse = cfg.get("retrieval", {}).get("min_diverse_classes", 2)
    batch_size = cfg.get("retrieval", {}).get("test_retrieval_batch_size", 128)

    sim_dense = sim_matrix
    if hasattr(sim_matrix, 'toarray'):
        sim_dense = sim_matrix.toarray()
    elif hasattr(sim_matrix, 'A'):
        sim_dense = np.asarray(sim_matrix.A)
    else:
        sim_dense = np.asarray(sim_matrix)

    train_d1_idx = np.array([id_to_idx.get(r["drug1_id"], -1) for _, r in train_df.iterrows()], dtype=np.int32)
    train_d2_idx = np.array([id_to_idx.get(r["drug2_id"], -1) for _, r in train_df.iterrows()], dtype=np.int32)
    train_fp_mask = (train_d1_idx >= 0) & (train_d2_idx >= 0)
    train_labels = np.array(train_df["label"].tolist())

    cand_d1 = train_d1_idx[train_fp_mask]
    cand_d2 = train_d2_idx[train_fp_mask]
    cand_labels = train_labels[train_fp_mask]
    cand_orig_idx = np.array(train_df.index.tolist())[train_fp_mask]
    if len(cand_d1) == 0:
        logger.error("No fingerprint-backed training candidates; cannot build test retrievals.")
        out_path = proc_dir / "retrieved_examples_test.json"
        with open(out_path, "w") as f:
            json.dump({}, f)
        return

    test_d1_idx = np.array([id_to_idx.get(r["drug1_id"], -1) for _, r in test_df.iterrows()], dtype=np.int32)
    test_d2_idx = np.array([id_to_idx.get(r["drug2_id"], -1) for _, r in test_df.iterrows()], dtype=np.int32)
    test_fp_mask = (test_d1_idx >= 0) & (test_d2_idx >= 0)
    test_indices = np.array(test_df.index.tolist())

    n_total = len(test_df)
    n_with_fp = int(test_fp_mask.sum())
    n_skip = n_total - n_with_fp
    logger.info(
        f"Test pairs: {n_total:,} | with fingerprints: {n_with_fp:,} | "
        f"without: {n_skip:,} | train candidates: {len(cand_d1):,}"
    )

    retrieval_out = {str(idx): [] for idx in test_indices}
    fp_positions = np.where(test_fp_mask)[0]
    processed = 0

    for batch_start in range(0, n_with_fp, batch_size):
        batch_end = min(batch_start + batch_size, n_with_fp)
        batch_pos = fp_positions[batch_start:batch_end]
        b_size = len(batch_pos)

        b_d1 = test_d1_idx[batch_pos]
        b_d2 = test_d2_idx[batch_pos]

        s_d1_c1 = sim_dense[b_d1][:, cand_d1]
        s_d2_c2 = sim_dense[b_d2][:, cand_d2]
        s_d1_c2 = sim_dense[b_d1][:, cand_d2]
        s_d2_c1 = sim_dense[b_d2][:, cand_d1]

        pair_sim = np.maximum(
            (s_d1_c1 + s_d2_c2) / 2.0,
            (s_d1_c2 + s_d2_c1) / 2.0,
        )

        top_n = min(top_k * 10, len(cand_d1))
        for bi in range(b_size):
            sims = pair_sim[bi]
            top_positions = np.argpartition(sims, -top_n)[-top_n:]
            top_positions = top_positions[np.argsort(-sims[top_positions])]

            selected = []
            classes_seen = set()
            for cp in top_positions:
                if len(selected) >= top_k:
                    break
                lbl = cand_labels[cp]
                if len(selected) >= top_k - min_diverse or lbl not in classes_seen:
                    sel_idx = int(cand_orig_idx[cp])
                    sel_row = train_df.iloc[sel_idx]
                    selected.append({
                        "drug1_id": sel_row["drug1_id"],
                        "drug2_id": sel_row["drug2_id"],
                        "drug1_name": sel_row["drug1_name"],
                        "drug2_name": sel_row["drug2_name"],
                        "label": int(sel_row["label"]),
                        "label_text": sel_row.get("label_text", ""),
                        "severity": sel_row.get("severity", "Unknown"),
                    })
                    classes_seen.add(lbl)

            retrieval_out[str(int(test_indices[batch_pos[bi]]))] = selected

        processed += b_size
        if processed % 10000 < b_size or batch_end == n_with_fp:
            logger.info(f"  Test retrieval: {processed:,}/{n_with_fp:,} with-fp pairs")

    out_path = proc_dir / "retrieved_examples_test.json"
    with open(out_path, "w") as f:
        json.dump(retrieval_out, f)
    logger.info(f"Saved {out_path} ({len(retrieval_out):,} entries)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--test-retrieval", action="store_true",
                        help="Only precompute test set retrievals")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.test_retrieval:
        precompute_test_retrievals(cfg)
    else:
        prepare_data(cfg)
