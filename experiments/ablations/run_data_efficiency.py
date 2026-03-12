#!/usr/bin/env python3
"""Data efficiency ablation — train Condition B with {10%, 25%, 50%} of data.
Usage:
  torchrun --nproc_per_node=4 experiments/ablations/run_data_efficiency.py --fraction 0.10
  python experiments/ablations/run_data_efficiency.py --fraction 0.10 --eval-only
"""

import argparse, os, sys, json
import torch, numpy as np, pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.exp_utils import (
    get_config, get_exp_output_dir, setup_exp_logging,
    compute_classification_metrics, save_results, load_label_map,
)
from src.data_preparation import SYSTEM_PROMPT, build_student_input
from src.student_training import _format_chat, temperature_resample
from src.evaluation import extract_label


def run(fraction: float, eval_only=False):
    pct = int(fraction * 100)
    exp_name = f"data_{pct}pct"
    out_dir = get_exp_output_dir(exp_name)
    logger = setup_exp_logging(exp_name, out_dir)
    cfg = get_config()
    scfg = cfg["student"]
    tcfg = scfg["training"]
    model_name = scfg["model_name"]
    seed = cfg["project"]["seed"]

    scratch = os.environ.get("SCRATCH", "")
    ckpt_dir = os.path.join(scratch, "ddi_checkpoints_v2", exp_name) if scratch \
        else str(out_dir / "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    if not eval_only:
        logger.info(f"Data efficiency: {pct}% of training data")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        df = pd.read_json(os.path.join(cfg["data"]["processed_dir"], "train.jsonl"), lines=True)
        # Stratified subsample
        _, df_sub = train_test_split(df, test_size=fraction, random_state=seed,
                                     stratify=df["label"])
        df_sub = temperature_resample(df_sub, temperature=tcfg.get("sampling_temperature", 0.5),
                                      seed=seed)
        logger.info(f"Subsampled: {len(df_sub):,} ({pct}% of {len(df):,})")

        texts, skipped = [], 0
        for _, row in df_sub.iterrows():
            messages = _format_chat(row, mode="label")
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            if len(tokenizer.encode(text, add_special_tokens=False)) > scfg["max_length"]:
                skipped += 1
                continue
            texts.append(text)

        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(texts))
        n_eval = max(1, int(0.05 * len(texts)))
        train_ds = Dataset.from_dict({"text": [texts[i] for i in indices[n_eval:]]})
        eval_ds = Dataset.from_dict({"text": [texts[i] for i in indices[:n_eval]]})

        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                         trust_remote_code=True, attn_implementation="flash_attention_2")
        except (ImportError, ValueError):
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                         trust_remote_code=True, attn_implementation="sdpa")

        lora_cfg = LoraConfig(r=scfg["lora"]["r"], lora_alpha=scfg["lora"]["alpha"],
                              lora_dropout=scfg["lora"]["dropout"],
                              target_modules=scfg["lora"]["target_modules"],
                              task_type=TaskType.CAUSAL_LM, bias="none")
        model = get_peft_model(model, lora_cfg)

        n_epochs = tcfg["num_epochs"] + (2 if fraction <= 0.25 else 0)
        args = TrainingArguments(
            output_dir=ckpt_dir, num_train_epochs=min(n_epochs, 8),
            per_device_train_batch_size=tcfg["per_device_batch_size"],
            gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
            learning_rate=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"],
            warmup_ratio=tcfg["warmup_ratio"], lr_scheduler_type=tcfg["lr_scheduler_type"],
            bf16=tcfg["bf16"], logging_steps=tcfg["logging_steps"],
            save_steps=tcfg["save_steps"], eval_strategy="steps",
            eval_steps=tcfg["eval_steps"], save_total_limit=3,
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, gradient_checkpointing=tcfg["gradient_checkpointing"],
            gradient_checkpointing_kwargs={"use_reentrant": False},
            remove_unused_columns=False, dataloader_num_workers=4,
            ddp_find_unused_parameters=False, report_to="none",
        )
        args.max_seq_length = scfg["max_length"]
        args.dataset_text_field = "text"

        trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds,
                             eval_dataset=eval_ds, processing_class=tokenizer)
        last = get_last_checkpoint(ckpt_dir)
        if last:
            logger.info(f"Resuming from {last}")
        trainer.train(resume_from_checkpoint=last)

        final_dir = os.path.join(ckpt_dir, "final")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            if torch.distributed.get_rank() != 0:
                return None
            torch.distributed.destroy_process_group()
            logger.info("DDP done. Run with --eval-only.")
            return None
    else:
        final_dir = os.path.join(ckpt_dir, "final")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=model_name, dtype="bfloat16", max_model_len=4096,
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
        prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    y_pred = []
    for i in range(0, len(prompts), cfg["evaluation"]["batch_size"]):
        outs = llm.generate(prompts[i:i+cfg["evaluation"]["batch_size"]], sp, lora_request=lora_req)
        for out in outs:
            y_pred.append(extract_label(out.outputs[0].text.strip()))

    metrics = compute_classification_metrics(test_df["label"].tolist(), y_pred, load_label_map())
    metrics["data_fraction"] = fraction
    logger.info(f"{pct}%: Macro F1={metrics['macro_f1']:.4f}")
    save_results(metrics, out_dir)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    run(args.fraction, args.eval_only)
