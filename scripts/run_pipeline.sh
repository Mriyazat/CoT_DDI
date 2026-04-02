#!/bin/bash
# Master pipeline for DDI CoT Distillation V3
# Run phases sequentially; each phase has resume support.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# Output dir: $SCRATCH/ddi_v3_outputs on cluster, ./outputs locally
if [ -n "${SCRATCH:-}" ]; then
    OUT_DIR="$SCRATCH/ddi_v3_outputs"
    CKPT_BASE="$SCRATCH/ddi_checkpoints_v3"
else
    OUT_DIR="$PROJECT_DIR/outputs"
    CKPT_BASE="$PROJECT_DIR/outputs/checkpoints"
fi

echo "============================================"
echo "DDI CoT Distillation V3 Pipeline"
echo "Project dir: $PROJECT_DIR"
echo "Output dir:  $OUT_DIR"
echo "Checkpoint:  $CKPT_BASE"
echo "============================================"

# ── Phase 0: Data preparation (run locally before cluster) ───────────
phase0() {
    echo ""
    echo "=== Phase 0a: Extract dataset from DrugBank XML ==="
    python3 scripts/extract_dataset_from_xml.py

    echo ""
    echo "=== Phase 0b: Build fingerprints ==="
    python3 scripts/build_fingerprints.py

    echo ""
    echo "=== Phase 0c: Data preparation ==="
    python3 -m src.data_preparation

    echo ""
    echo "=== Phase 0: Verification ==="
    echo "Checking output files..."
    for f in data/processed/{interactions_full.jsonl,drug_profiles.json,label_map.json,severity_map.json,train.jsonl,test.jsonl,coarse_category_map.json,retrieved_examples_train.json}; do
        if [ -f "$f" ]; then
            lines=$(wc -l < "$f")
            echo "  OK: $f ($lines lines)"
        else
            echo "  MISSING: $f"
            exit 1
        fi
    done
    echo "Phase 0 complete."
}

# ── Phase 1: Teacher generation (cluster, 4x H100) ──────────────────
phase1() {
    echo ""
    echo "=== Phase 1: Teacher trace generation ==="
    python3 -m src.teacher_generation --config configs/config.yaml

    echo ""
    echo "=== Phase 1: Verification ==="
    trace_file="$OUT_DIR/teacher_traces/full_traces.jsonl"
    if [ -f "$trace_file" ]; then
        n=$(wc -l < "$trace_file")
        echo "  Traces generated: $n"
    else
        echo "  MISSING: $trace_file"
        exit 1
    fi
}

# ── Phase 1.5: Hard trace rejection ──────────────────────────────────
phase1_5() {
    echo ""
    echo "=== Phase 1.5: Hard trace rejection ==="
    python3 -m src.hard_rejection --config configs/config.yaml

    echo ""
    echo "=== Phase 1.5: Verification ==="
    filtered="$OUT_DIR/teacher_traces/full_traces_hard_filtered.jsonl"
    if [ -f "$filtered" ]; then
        n=$(wc -l < "$filtered")
        echo "  Traces after hard rejection: $n"
    else
        echo "  MISSING: $filtered"
        exit 1
    fi
}

# ── Phase 1.6: Grounded factuality scoring ───────────────────────────
phase1_6() {
    echo ""
    echo "=== Phase 1.6: Grounded factuality verification ==="
    python3 -m src.grounded_factuality --config configs/config.yaml --split

    echo ""
    echo "=== Phase 1.6: Verification ==="
    scored="$OUT_DIR/teacher_traces/full_traces_scored.jsonl"
    if [ -f "$scored" ]; then
        n=$(wc -l < "$scored")
        echo "  Scored traces: $n"
    else
        echo "  MISSING: $scored"
        exit 1
    fi

    low="$OUT_DIR/teacher_traces/traces_for_refinement.jsonl"
    high="$OUT_DIR/teacher_traces/traces_high_quality.jsonl"
    if [ -f "$low" ] && [ -f "$high" ]; then
        n_low=$(wc -l < "$low")
        n_high=$(wc -l < "$high")
        echo "  High-quality: $n_high | For refinement: $n_low"
    fi
}

# ── Phase 1.7: Targeted self-refinement ──────────────────────────────
phase1_7() {
    echo ""
    echo "=== Phase 1.7: Targeted self-refinement ==="
    python3 -m src.trace_refinement --config configs/config.yaml

    echo ""
    echo "=== Phase 1.7: Verification ==="
    final="$OUT_DIR/teacher_traces/full_traces_final.jsonl"
    if [ -f "$final" ]; then
        n=$(wc -l < "$final")
        echo "  Final training traces: $n"
    else
        echo "  MISSING: $final"
        exit 1
    fi

    cot_file="data/processed/train_cot.jsonl"
    if [ -f "$cot_file" ]; then
        n=$(wc -l < "$cot_file")
        echo "  Student training data: $n traces"
    else
        echo "  MISSING: $cot_file"
        exit 1
    fi
}

# ── Phase 3: Student training (cluster, 4x H100 DDP) ────────────────
phase3() {
    echo ""
    echo "=== Phase 3: Student training ==="

    echo "--- Condition B: Label-only ---"
    torchrun --nproc_per_node=4 -m src.student_training \
        --mode label --config configs/config.yaml

    echo "--- Condition C_naive: Naive CoT ---"
    torchrun --nproc_per_node=4 -m src.student_training \
        --mode cot_naive --config configs/config.yaml

    echo "--- Condition C_seq: Sequential ---"
    torchrun --nproc_per_node=4 -m src.student_training \
        --mode sequential --config configs/config.yaml

    echo "--- Condition C_compact: Compact (ablation) ---"
    torchrun --nproc_per_node=4 -m src.student_training \
        --mode compact --config configs/config.yaml

    echo "--- Condition C_summary: Summary (PRIMARY) ---"
    torchrun --nproc_per_node=4 -m src.student_training \
        --mode summary --config configs/config.yaml

    echo ""
    echo "=== Phase 3: Verification ==="
    for cond in B_label C_naive C_seq C_compact100 C_summary; do
        d="$CKPT_BASE/${cond}_s42/final"
        if [ -d "$d" ]; then
            echo "  OK: $d"
        else
            echo "  MISSING: $d"
        fi
    done
}

# ── Phase 4: Evaluation ──────────────────────────────────────────────
phase4() {
    echo ""
    echo "=== Phase 4: Multi-task evaluation ==="
    for cond in B_label C_naive C_seq C_compact100 C_summary; do
        ckpt_dir="$CKPT_BASE/${cond}_s42/final"
        if [ -d "$ckpt_dir" ]; then
            echo "--- Evaluating $cond ---"
            python3 -m src.evaluation --condition "$cond" \
                --checkpoint "$ckpt_dir" --config configs/config.yaml
        else
            echo "  Skipping $cond (no checkpoint at $ckpt_dir)"
        fi
    done

    echo ""
    echo "--- Comparison ---"
    python3 -m src.evaluation --condition dummy --checkpoint dummy \
        --compare B_label C_naive C_seq C_compact100 C_summary \
        --config configs/config.yaml
}

# ── Phase 4b: API judge evaluation (requires internet) ───────────────
phase4b() {
    echo ""
    echo "=== Phase 4b: API judge evaluation ==="
    echo "--- Evaluating teacher traces ---"
    python3 -m src.api_judge_eval --config configs/config.yaml --source teacher

    echo ""
    echo "--- Evaluating student traces ---"
    python3 -m src.api_judge_eval --config configs/config.yaml --source student

    echo ""
    echo "=== Phase 4b: Verification ==="
    for src_name in teacher student; do
        f="$OUT_DIR/results/api_judge_${src_name}_summary.json"
        if [ -f "$f" ]; then
            echo "  OK: $f"
        else
            echo "  MISSING: $f"
        fi
    done
}

# ── Phase 5: Chatbot demo ────────────────────────────────────────────
phase5() {
    echo ""
    echo "=== Phase 5: Clinical DDI Chatbot ==="
    python3 app/chatbot.py
}

# ── CLI ───────────────────────────────────────────────────────────────
case "${1:-all}" in
    phase0)   phase0 ;;
    phase1)   phase1 ;;
    phase1.5) phase1_5 ;;
    phase1.6) phase1_6 ;;
    phase1.7) phase1_7 ;;
    phase3)   phase3 ;;
    phase4)   phase4 ;;
    phase4b)  phase4b ;;
    phase5)   phase5 ;;
    all)
        phase0
        phase1
        phase1_5
        phase1_6
        phase1_7
        phase3
        phase4
        phase4b
        ;;
    *)
        echo "Usage: $0 {phase0|phase1|phase1.5|phase1.6|phase1.7|phase3|phase4|phase4b|phase5|all}"
        exit 1
        ;;
esac
