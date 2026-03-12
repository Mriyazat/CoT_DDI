#!/usr/bin/env python3
"""CoT distillation WITHOUT judge filtering baseline.
Usage:
  torchrun --nproc_per_node=4 experiments/baselines/run_cot_no_filter.py
  python experiments/baselines/run_cot_no_filter.py --eval-only
"""

import argparse, os, sys, json, re, logging
import torch, numpy as np, pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.exp_utils import (
    get_config, get_exp_output_dir, setup_exp_logging,
    compute_classification_metrics, save_results, load_label_map,
)
from src.data_preparation import SYSTEM_PROMPT, build_student_input
from src.evaluation import extract_label
from src.student_training import _strip_trailing_classification, temperature_resample


def _prepare_unfiltered_data(cfg):
    traces_path = os.path.join(cfg["project"]["output_dir"], "teacher_traces", "full_traces_filtered.jsonl")
    train_path = os.path.join(cfg["data"]["processed_dir"], "train.jsonl")
    out_dir = get_exp_output_dir("cot_no_filter")
    out_path = str(out_dir / "train_cot_unfiltered.jsonl")

    if os.path.exists(out_path):
        return out_path

    train_df = pd.read_json(train_path, lines=True)
    train_keys = set(zip(train_df["drug1_id"], train_df["drug2_id"]))

    traces = {}
    with open(traces_path) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["drug1_id"], rec["drug2_id"])
            if key in train_keys and key not in traces:
                traces[key] = rec.get("teacher_cot", rec.get("cot", ""))

    rows = []
    for _, row in train_df.iterrows():
        key = (row["drug1_id"], row["drug2_id"])
        if key in traces:
            r = row.to_dict()
            r["teacher_cot"] = traces[key]
            rows.append(r)

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")

    logging.getLogger("cot_no_filter").info(f"Built unfiltered CoT data: {len(rows):,}")
    return out_path


def run(eval_only=False):
    out_dir = get_exp_output_dir("cot_no_filter")
    logger = setup_exp_logging("cot_no_filter", out_dir)
    cfg = get_config()
    scfg = cfg["student"]
    tcfg = scfg["training"]
    model_name = scfg["model_name"]

    scratch = os.environ.get("SCRATCH", "")
    ckpt_dir = os.path.join(scratch, "ddi_checkpoints_v2", "cot_no_filter") if scratch \
        else str(out_dir / "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    if not eval_only:
        data_path = _prepare_unfiltered_data(cfg)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        df = pd.read_json(data_path, lines=True)
        df = temperature_resample(df, temperature=tcfg.get("sampling_temperature", 0.5))

        texts, skipped = [], 0
        for _, row in df.iterrows():
            cot = _strip_trailing_classification(str(row["teacher_cot"]))
            assistant = f"{cot}\n\nClassification: Y={row['label']} — \"{row['label_text']}\""
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_student_input(row)},
                    {"role": "assistant", "content": assistant}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False)
            if len(tokenizer.encode(text, add_special_tokens=False)) > scfg["max_length"]:
                skipped += 1
                continue
            texts.append(text)

        rng = np.random.RandomState(cfg["project"]["seed"])
        indices = rng.permutation(len(texts))
        n_eval = max(1, int(0.05 * len(texts)))
        train_ds = Dataset.from_dict({"text": [texts[i] for i in indices[n_eval:]]})
        eval_ds = Dataset.from_dict({"text": [texts[i] for i in indices[:n_eval]]})
        logger.info(f"Train: {len(train_ds):,} | Eval: {len(eval_ds):,} | Skipped: {skipped:,}")

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

        args = TrainingArguments(
            output_dir=ckpt_dir, num_train_epochs=tcfg["num_epochs"],
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
            logger.info("DDP done. Run with --eval-only for evaluation.")
            return None
    else:
        final_dir = os.path.join(ckpt_dir, "final")

    logger.info("Evaluating …")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=model_name, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.85, enable_lora=True,
              max_lora_rank=scfg["lora"]["r"], trust_remote_code=True)
    tokenizer = llm.get_tokenizer()
    lora_req = LoRARequest("cot_no_filter", 1, final_dir)
    sp = SamplingParams(temperature=0.1, top_p=0.9,
                        max_tokens=cfg["evaluation"]["max_new_tokens"])

    test_df = pd.read_json(os.path.join(cfg["data"]["processed_dir"], "test.jsonl"), lines=True)
    prompts = []
    for _, row in test_df.iterrows():
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_student_input(row)}]
        prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False,
                                                      add_generation_prompt=True))

    y_true, y_pred, records = [], [], []
    batch_sz = cfg["evaluation"]["batch_size"]
    for i in range(0, len(prompts), batch_sz):
        outs = llm.generate(prompts[i:i+batch_sz], sp, lora_request=lora_req)
        for (_, row), out in zip(list(test_df.iloc[i:i+len(outs)].iterrows()), outs):
            resp = out.outputs[0].text.strip()
            pred = extract_label(resp)
            y_true.append(int(row["label"]))
            y_pred.append(pred)
            records.append({"drug1_id": row["drug1_id"], "drug2_id": row["drug2_id"],
                            "true_label": int(row["label"]), "pred_label": pred,
                            "response": resp})

    metrics = compute_classification_metrics(y_true, y_pred, load_label_map())
    metrics["condition"] = "cot_no_filter"
    logger.info(f"CoT (no filter): Macro F1={metrics['macro_f1']:.4f}")
    save_results(metrics, out_dir)
    with open(out_dir / "cot_no_filter_predictions.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    run(args.eval_only)
