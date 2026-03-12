#!/bin/bash
# Run on LOGIN NODE only (needs internet access).
# Downloads all model files into $HF_HOME without loading into RAM.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
source activate_env.sh

# Allow downloads
unset HF_HUB_OFFLINE
export HF_HUB_DISABLE_XET=1

echo "=== Downloading models to $HF_HOME ==="

# Student models (Qwen family)
for MODEL in "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct"; do
    echo ">>> $MODEL"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', resume_download=True)
print('  OK: $MODEL')
"
done

# BERT baselines
for MODEL in "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext" "dmis-lab/biobert-v1.1" "allenai/scibert_scivocab_uncased"; do
    echo ">>> $MODEL"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', resume_download=True)
print('  OK: $MODEL')
"
done

# BERTScore model (DeBERTa)
echo ">>> microsoft/deberta-xlarge-mnli (for BERTScore)"
python -c "
from huggingface_hub import snapshot_download
snapshot_download('microsoft/deberta-xlarge-mnli', resume_download=True)
print('  OK: deberta-xlarge-mnli')
"

echo ""
echo "=== All models downloaded ==="
echo "Cache contents:"
du -sh $HF_HOME/hub/models--* 2>/dev/null || echo "No models cached"
