#!/bin/bash
# Master pipeline — run on compute node (4x H100, ~48h total).
# Usage: sbatch scripts/run_pipeline.sh  (or bash scripts/run_pipeline.sh)
#
# All DDP training uses torchrun, all inference uses plain python (vLLM).
# These MUST be separate processes to avoid CUDA fork issues.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
source activate_env.sh

NPROC=${NPROC:-4}  # GPUs per node
SEED=42

echo "==============================================="
echo "DDI CoT Distillation V2 — Full Pipeline"
echo "Project: $PROJECT_DIR"
echo "GPUs: $NPROC"
echo "Started: $(date)"
echo "==============================================="

echo ""
echo "=== Phase 0: Verifying data ==="
python -c "
import json, os
cfg_data = '$PROJECT_DIR/configs/config.yaml'
from src.utils import load_config
cfg = load_config(cfg_data)
d = cfg['data']['processed_dir']
for f in ['train.jsonl', 'test.jsonl', 'label_map.json', 'train_cot.jsonl']:
    p = os.path.join(d, f)
    if not os.path.exists(p):
        print(f'MISSING: {p}')
        exit(1)
import pandas as pd
train = pd.read_json(os.path.join(d, 'train.jsonl'), lines=True)
test = pd.read_json(os.path.join(d, 'test.jsonl'), lines=True)
cot = pd.read_json(os.path.join(d, 'train_cot.jsonl'), lines=True)
with open(os.path.join(d, 'label_map.json')) as f_:
    lm = json.load(f_)
print(f'Train: {len(train):,}  Test: {len(test):,}  CoT: {len(cot):,}  Labels: {len(lm)}')
print('Phase 0: OK')
"

echo ""
echo "=== Phase 1: Model smoke test ==="
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
m = 'Qwen/Qwen2.5-7B-Instruct'
tok = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.bfloat16, trust_remote_code=True)
ids = tok('Hello world', return_tensors='pt')
with torch.no_grad(): out = mdl(**ids)
print(f'Logits shape: {out.logits.shape}')
print('Phase 1: OK')
del mdl; import gc; gc.collect(); torch.cuda.empty_cache()
"

echo ""
echo "=== Phase 4a: Training B (label-only) ==="
torchrun --nproc_per_node=$NPROC -m src.student_training --mode label --seed $SEED

echo "=== Phase 4a: Evaluating B ==="
python -c "
from src.utils import load_config
from src.evaluation import predict_finetuned, compute_classification_metrics
import json, os

cfg = load_config()
scratch = os.environ.get('SCRATCH', '')
if scratch:
    ckpt = os.path.join(scratch, 'ddi_checkpoints_v2', 'B_label_s42', 'final')
else:
    ckpt = 'outputs/checkpoints/B_label_s42/final'

pred_path, eff = predict_finetuned(cfg, ckpt, 'B_label_only')
with open(os.path.join(cfg['data']['processed_dir'], 'label_map.json')) as f:
    lm = {int(k): v for k, v in json.load(f).items()}
metrics = compute_classification_metrics(pred_path, lm, 'B_label_only')
print(f\"B: Macro F1={metrics['macro_f1']:.4f} | Micro F1={metrics['micro_f1']:.4f}\")
"

echo ""
echo "=== Phase 4b: Training C_naive (reproducing failure) ==="
torchrun --nproc_per_node=$NPROC -m src.student_training --mode cot_naive --seed $SEED

echo "=== Phase 4b: Evaluating C_naive ==="
python -c "
from src.utils import load_config
from src.evaluation import predict_finetuned, compute_classification_metrics
import json, os

cfg = load_config()
scratch = os.environ.get('SCRATCH', '')
if scratch:
    ckpt = os.path.join(scratch, 'ddi_checkpoints_v2', 'C_naive_s42', 'final')
else:
    ckpt = 'outputs/checkpoints/C_naive_s42/final'

pred_path, eff = predict_finetuned(cfg, ckpt, 'C_naive')
with open(os.path.join(cfg['data']['processed_dir'], 'label_map.json')) as f:
    lm = {int(k): v for k, v in json.load(f).items()}
metrics = compute_classification_metrics(pred_path, lm, 'C_naive')
print(f\"C_naive: Macro F1={metrics['macro_f1']:.4f} | Micro F1={metrics['micro_f1']:.4f}\")
"

echo ""
echo "=== Phase 4c: Training C_seq (sequential from B checkpoint) ==="
torchrun --nproc_per_node=$NPROC -m src.student_training --mode sequential --seed $SEED

echo "=== Phase 4c: Evaluating C_seq ==="
python -c "
from src.utils import load_config
from src.evaluation import predict_finetuned, compute_classification_metrics
import json, os

cfg = load_config()
scratch = os.environ.get('SCRATCH', '')
if scratch:
    ckpt = os.path.join(scratch, 'ddi_checkpoints_v2', 'C_seq_s42', 'final')
else:
    ckpt = 'outputs/checkpoints/C_seq_s42/final'

pred_path, eff = predict_finetuned(cfg, ckpt, 'C_seq')
with open(os.path.join(cfg['data']['processed_dir'], 'label_map.json')) as f:
    lm = {int(k): v for k, v in json.load(f).items()}
metrics = compute_classification_metrics(pred_path, lm, 'C_seq')
print(f\"C_seq: Macro F1={metrics['macro_f1']:.4f} | Micro F1={metrics['micro_f1']:.4f}\")
"

echo ""
echo "=== Phase 4d: Training C_mix (mixed label+CoT) ==="
torchrun --nproc_per_node=$NPROC -m src.student_training --mode mixed --seed $SEED

echo "=== Phase 4d: Evaluating C_mix ==="
python -c "
from src.utils import load_config
from src.evaluation import predict_finetuned, compute_classification_metrics
import json, os

cfg = load_config()
scratch = os.environ.get('SCRATCH', '')
if scratch:
    ckpt = os.path.join(scratch, 'ddi_checkpoints_v2', 'C_mix_s42', 'final')
else:
    ckpt = 'outputs/checkpoints/C_mix_s42/final'

pred_path, eff = predict_finetuned(cfg, ckpt, 'C_mix')
with open(os.path.join(cfg['data']['processed_dir'], 'label_map.json')) as f:
    lm = {int(k): v for k, v in json.load(f).items()}
metrics = compute_classification_metrics(pred_path, lm, 'C_mix')
print(f\"C_mix: Macro F1={metrics['macro_f1']:.4f} | Micro F1={metrics['micro_f1']:.4f}\")
"

echo ""
echo "=== Phase 4e: Training C_wt (sequential + weighted cls loss) ==="
torchrun --nproc_per_node=$NPROC -m src.student_training --mode weighted --seed $SEED --cls-weight 10.0

echo "=== Phase 4e: Evaluating C_wt ==="
python -c "
from src.utils import load_config
from src.evaluation import predict_finetuned, compute_classification_metrics
import json, os

cfg = load_config()
scratch = os.environ.get('SCRATCH', '')
if scratch:
    ckpt = os.path.join(scratch, 'ddi_checkpoints_v2', 'C_wt_s42', 'final')
else:
    ckpt = 'outputs/checkpoints/C_wt_s42/final'

pred_path, eff = predict_finetuned(cfg, ckpt, 'C_wt')
with open(os.path.join(cfg['data']['processed_dir'], 'label_map.json')) as f:
    lm = {int(k): v for k, v in json.load(f).items()}
metrics = compute_classification_metrics(pred_path, lm, 'C_wt')
print(f\"C_wt: Macro F1={metrics['macro_f1']:.4f} | Micro F1={metrics['micro_f1']:.4f}\")
"

echo ""
echo "=== Phase 5: Training C_compact (all three fixes combined) ==="
torchrun --nproc_per_node=$NPROC -m src.student_training \
    --mode compact --seed $SEED \
    --cls-weight 10.0 --cot-max-words 100 --min-per-class 50 --sampling-temperature 0.25

echo "=== Phase 5: Evaluating C_compact ==="
python -c "
from src.utils import load_config
from src.evaluation import predict_finetuned, compute_classification_metrics
import json, os

cfg = load_config()
scratch = os.environ.get('SCRATCH', '')
if scratch:
    ckpt = os.path.join(scratch, 'ddi_checkpoints_v2', 'C_compact_s42', 'final')
else:
    ckpt = 'outputs/checkpoints/C_compact_s42/final'

pred_path, eff = predict_finetuned(cfg, ckpt, 'C_compact')
with open(os.path.join(cfg['data']['processed_dir'], 'label_map.json')) as f:
    lm = {int(k): v for k, v in json.load(f).items()}
metrics = compute_classification_metrics(pred_path, lm, 'C_compact')
print(f\"C_compact: Macro F1={metrics['macro_f1']:.4f} | Micro F1={metrics['micro_f1']:.4f}\")
"

echo ""
echo "=== Phase 6: Self-consistency (n=5) on C_compact ==="
python -c "
from src.utils import load_config
from src.evaluation import predict_two_stage_real, compute_classification_metrics
import json, os

cfg = load_config()
scratch = os.environ.get('SCRATCH', '')
if scratch:
    r_ckpt = os.path.join(scratch, 'ddi_checkpoints_v2', 'C_compact_s42', 'final')
else:
    r_ckpt = 'outputs/checkpoints/C_compact_s42/final'

pred_path = predict_two_stage_real(cfg, r_ckpt, condition_name='D_compact_SC', n_samples=5)
with open(os.path.join(cfg['data']['processed_dir'], 'label_map.json')) as f:
    lm = {int(k): v for k, v in json.load(f).items()}
metrics = compute_classification_metrics(pred_path, lm, 'D_compact_SC')
print(f\"D_compact_SC (n=5): Macro F1={metrics['macro_f1']:.4f} | Micro F1={metrics['micro_f1']:.4f}\")
"

echo ""
echo "=== Phase 7: Judge evaluation of best reasoning model ==="
python -c "
from src.utils import load_config
from src.evaluation import judge_student_reasoning

cfg = load_config()
import os
for pred in ['C_compact', 'C_wt', 'C_seq']:
    pred_path = f'outputs/results/{pred}_predictions.jsonl'
    if os.path.exists(pred_path):
        print(f'Judging {pred} ...')
        judge_student_reasoning(cfg, pred_path)
        print(f'Judge evaluation for {pred} complete.')
        break
else:
    print('No reasoning predictions found for judge evaluation.')
"

echo ""
echo "==============================================="
echo "Pipeline complete: $(date)"
echo "==============================================="
