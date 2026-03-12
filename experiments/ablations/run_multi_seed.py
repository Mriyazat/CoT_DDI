#!/usr/bin/env python3
"""Multi-seed ablation — train each condition with different seeds.
Usage:
  torchrun --nproc_per_node=4 experiments/ablations/run_multi_seed.py --mode label --seed 0
  python experiments/ablations/run_multi_seed.py --mode label --seed 0 --eval-only

Supports modes: label, cot_naive, sequential, mixed
"""

import argparse, os, sys, json
import torch, pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.exp_utils import (
    get_config, get_exp_output_dir, setup_exp_logging,
    compute_classification_metrics, save_results, load_label_map,
)
from src.student_training import (
    train_label_only, train_cot_naive, train_sequential, train_mixed,
)
from src.data_preparation import SYSTEM_PROMPT, build_student_input
from src.evaluation import extract_label


TRAIN_FNS = {
    "label": train_label_only,
    "cot_naive": train_cot_naive,
    "sequential": train_sequential,
    "mixed": train_mixed,
}


def run(mode: str, seed: int, eval_only: bool = False):
    exp_name = f"seed_{mode}_s{seed}"
    out_dir = get_exp_output_dir(exp_name)
    logger = setup_exp_logging(exp_name, out_dir)
    cfg = get_config()
    scfg = cfg["student"]

    scratch = os.environ.get("SCRATCH", "")
    if scratch:
        ckpt_dir = os.path.join(scratch, "ddi_checkpoints_v2", exp_name)
    else:
        ckpt_dir = str(out_dir / "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    final_dir = os.path.join(ckpt_dir, "final")

    if not eval_only:
        logger.info(f"=== Multi-seed: mode={mode}, seed={seed} ===")
        train_fn = TRAIN_FNS[mode]
        result = train_fn(cfg, seed=seed)
        if result is None:
            return None
        final_dir = result

    logger.info(f"Evaluating from {final_dir} …")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=scfg["model_name"], dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.85, enable_lora=True,
              max_lora_rank=scfg["lora"]["r"], trust_remote_code=True)
    tokenizer = llm.get_tokenizer()
    lora_req = LoRARequest(exp_name, 1, final_dir)
    sp = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=256)

    test_df = pd.read_json(os.path.join(cfg["data"]["processed_dir"], "test.jsonl"), lines=True)
    prompts = []
    for _, row in test_df.iterrows():
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_student_input(row)}]
        prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False,
                                                      add_generation_prompt=True))

    y_pred = []
    batch_sz = cfg["evaluation"]["batch_size"]
    for i in range(0, len(prompts), batch_sz):
        outs = llm.generate(prompts[i:i+batch_sz], sp, lora_request=lora_req)
        for out in outs:
            y_pred.append(extract_label(out.outputs[0].text.strip()))
    y_true = test_df["label"].tolist()

    metrics = compute_classification_metrics(y_true, y_pred, load_label_map())
    metrics["mode"] = mode
    metrics["seed"] = seed
    logger.info(f"Seed {seed} ({mode}): Macro F1={metrics['macro_f1']:.4f}")
    save_results(metrics, out_dir)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=list(TRAIN_FNS.keys()))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    run(args.mode, args.seed, args.eval_only)
