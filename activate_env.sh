
module load StdEnv/2023
module load gcc/12.3 python/3.11 cuda/12.2 arrow opencv/4.12.0
source $HOME/ddi_venv/bin/activate
export HF_HOME=$SCRATCH/.cache/huggingface
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "Environment ready — $(python -c 'import torch; print(torch.cuda.device_count(), "GPUs")' 2>/dev/null || echo 'no GPU node')"
