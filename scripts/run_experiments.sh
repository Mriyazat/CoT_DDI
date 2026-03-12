#!/bin/bash
# Run baselines and ablations — after main pipeline completes.
# Usage: sbatch scripts/run_experiments.sh  (or bash scripts/run_experiments.sh)
#
# IMPORTANT: DDP training uses torchrun, evaluation uses plain python.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
source activate_env.sh

NPROC=${NPROC:-4}

echo "==============================================="
echo "DDI V2 — Experiments (Baselines + Ablations)"
echo "Started: $(date)"
echo "==============================================="

echo ""
echo "=== Baseline: Zero-shot ==="
python -c "
from src.utils import load_config
from src.baselines import evaluate_zero_shot
evaluate_zero_shot(load_config())
"

echo ""
echo "=== Baseline: ML (RF + XGBoost) ==="
python -c "
from src.utils import load_config
from src.baselines import train_ml_baseline, train_xgboost_baseline
cfg = load_config()
train_ml_baseline(cfg)
train_xgboost_baseline(cfg)
"

echo ""
echo "=== Baseline: BERT variants ==="
for MODEL in pubmedbert biobert scibert; do
    echo "--- $MODEL ---"
    python experiments/baselines/run_bert_baseline.py --model $MODEL --epochs 10 --batch-size 64
done

echo ""
echo "=== Baseline: Few-shot (5-shot) ==="
python experiments/baselines/run_few_shot.py --n-shots 5

echo ""
echo "=== Baseline: CoT-no-filter ==="
torchrun --nproc_per_node=$NPROC experiments/baselines/run_cot_no_filter.py
python experiments/baselines/run_cot_no_filter.py --eval-only

echo ""
echo "=== Ablation: Multi-seed ==="
for MODE in label cot_naive sequential mixed; do
    for SEED in 0 1 2; do
        echo "--- $MODE seed=$SEED ---"
        torchrun --nproc_per_node=$NPROC experiments/ablations/run_multi_seed.py --mode $MODE --seed $SEED
        python experiments/ablations/run_multi_seed.py --mode $MODE --seed $SEED --eval-only
    done
done

echo ""
echo "=== Ablation: LoRA rank ==="
for RANK in 16 32 128; do
    echo "--- r=$RANK ---"
    torchrun --nproc_per_node=$NPROC experiments/ablations/run_lora_rank.py --rank $RANK
    python experiments/ablations/run_lora_rank.py --rank $RANK --eval-only
done

echo ""
echo "=== Ablation: Data efficiency ==="
for FRAC in 0.10 0.25 0.50; do
    echo "--- ${FRAC}x data ---"
    torchrun --nproc_per_node=$NPROC experiments/ablations/run_data_efficiency.py --fraction $FRAC
    python experiments/ablations/run_data_efficiency.py --fraction $FRAC --eval-only
done

echo ""
echo "=== Ablation: Sequential LR ratio ==="
for RATIO in 0.1 0.25 0.5; do
    echo "--- ratio=$RATIO ---"
    torchrun --nproc_per_node=$NPROC experiments/ablations/run_seq_lr_ratio.py --ratio $RATIO
    python experiments/ablations/run_seq_lr_ratio.py --ratio $RATIO --eval-only
done

echo ""
echo "=== Ablation: Mix ratio ==="
for RATIO in 0.25 0.50 0.75; do
    echo "--- label=${RATIO} / CoT=$(echo "1 - $RATIO" | bc) ---"
    torchrun --nproc_per_node=$NPROC experiments/ablations/run_mix_ratio.py --ratio $RATIO
    python experiments/ablations/run_mix_ratio.py --ratio $RATIO --eval-only
done

echo ""
echo "=== Ablation: Judge filtering ==="
python experiments/ablations/run_judge_ablation.py

echo ""
echo "=== Analysis: Statistical tests ==="
python experiments/analysis/run_statistical_tests.py

echo ""
echo "=== Analysis: BERTScore comparison ==="
python experiments/analysis/run_bertscore_comparison.py

echo ""
echo "=== Analysis: ECE / Calibration ==="
python experiments/analysis/run_ece_calibration.py

echo ""
echo "=== Analysis: Category analysis ==="
python experiments/analysis/run_category_analysis.py

echo ""
echo "=== Analysis: Case studies ==="
python experiments/analysis/run_case_studies.py

echo ""
echo "=== Analysis: Human eval prep ==="
python experiments/analysis/run_human_eval_prep.py

echo ""
echo "==============================================="
echo "All experiments complete: $(date)"
echo "==============================================="
