#!/usr/bin/env python3
"""Sequential LR ratio ablation — test different Phase 2 learning rates.
Usage:
  torchrun --nproc_per_node=4 experiments/ablations/run_seq_lr_ratio.py --ratio 0.1
  python experiments/ablations/run_seq_lr_ratio.py --ratio 0.1 --eval-only
"""

import argparse, os, sys, json, copy
import torch, pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.exp_utils import (
    get_config, get_exp_output_dir, setup_exp_logging,
    compute_classification_metrics, save_results, load_label_map,
)
from src.student_training import train_sequential
from src.data_preparation import SYSTEM_PROMPT, build_student_input
from src.evaluation import extract_label


def run(ratio: float, eval_only=False):
    exp_name = f"seq_lr_{str(ratio).replace('.','')}"
    out_dir = get_exp_output_dir(exp_name)
    logger = setup_exp_logging(exp_name, out_dir)
    cfg = get_config()
    scfg = cfg["student"]

    # Override the LR ratio in config
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy["student"]["training"]["sequential_lr_ratio"] = ratio

    scratch = os.environ.get("SCRATCH", "")
    ckpt_dir = os.path.join(scratch, "ddi_checkpoints_v2", exp_name) if scratch \
        else str(out_dir / "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    final_dir = os.path.join(ckpt_dir, "final")

    if not eval_only:
        logger.info(f"Sequential LR ratio ablation: ratio={ratio}")
        result = train_sequential(cfg_copy, seed=cfg["project"]["seed"])
        if result is None:
            return None
        final_dir = result

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=scfg["model_name"], dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.85, enable_lora=True,
              max_lora_rank=scfg["lora"]["r"], trust_remote_code=True)
    tokenizer = llm.get_tokenizer()
    lora_req = LoRARequest(exp_name, 1, final_dir)
    sp = SamplingParams(temperature=0.1, top_p=0.9,
                        max_tokens=cfg["evaluation"]["max_new_tokens"])

    test_df = pd.read_json(os.path.join(cfg["data"]["processed_dir"], "test.jsonl"), lines=True)
    prompts = []
    for _, row in test_df.iterrows():
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_student_input(row)}]
        prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    y_pred = []
    for i in range(0, len(prompts), cfg["evaluation"]["batch_size"]):
        outs = llm.generate(prompts[i:i+cfg["evaluation"]["batch_size"]], sp, lora_request=lora_req)
        for out in outs:
            y_pred.append(extract_label(out.outputs[0].text.strip()))

    metrics = compute_classification_metrics(test_df["label"].tolist(), y_pred, load_label_map())
    metrics["sequential_lr_ratio"] = ratio
    logger.info(f"LR ratio={ratio}: Macro F1={metrics['macro_f1']:.4f}")
    save_results(metrics, out_dir)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    run(args.ratio, args.eval_only)
