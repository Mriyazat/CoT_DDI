"""
Student fine-tuning with six training strategies:

  B        : Label-only (classification baseline)
  C_naive  : Naive CoT distillation (reproduces the failure)
  C_seq    : Sequential training (B checkpoint → CoT with lower LR)
  C_mix    : Mixed training (label-only + CoT interleaved)
  C_wt     : Sequential with classification-weighted loss
  C_compact: Sequential with compact CoT + weighted loss + aggressive balancing

All strategies use:
  - Stratified train/eval split
  - Temperature-scaled class sampling with optional min-per-class floor
  - DDP via torchrun for multi-GPU training
  - Checkpoint resume
"""

import os
import re
import json
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datasets import Dataset

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from trl import SFTTrainer

from src.utils import load_config, setup_logging, set_seed, ensure_dirs, gpu_info
from src.data_preparation import SYSTEM_PROMPT, build_student_input




def _strip_trailing_classification(text: str) -> str:
    return re.sub(
        r"\n*Classification\s*:\s*Y\s*=\s*\d+.*$", "", text, flags=re.IGNORECASE
    ).rstrip()


def _format_chat(row, mode: str = "cot", cot_max_words: int = 0) -> list[dict]:
    """Build a chat-format training example.
    mode="cot"   → reasoning + classification (Condition C)
    mode="label" → classification only (Condition B)
    cot_max_words → if > 0, truncate teacher reasoning to first N words.
                    Shorter CoT aligns better with 7B model capacity
                    (ACL Findings 2025: weaker SLMs learn better from simpler CoT).
    """
    user_msg = build_student_input(row)

    if mode == "cot":
        cot = _strip_trailing_classification(str(row["teacher_cot"]))
        if cot_max_words > 0:
            words = cot.split()
            if len(words) > cot_max_words:
                cot = " ".join(words[:cot_max_words])
        assistant_msg = (
            f"{cot}\n\n"
            f"Classification: Y={row['label']} — \"{row['label_text']}\""
        )
    else:
        assistant_msg = f"Classification: Y={row['label']} — \"{row['label_text']}\""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]




def temperature_resample(df: pd.DataFrame, temperature: float = 0.5,
                         seed: int = 42, min_per_class: int = 0) -> pd.DataFrame:
    """Resample training data with temperature-scaled class probabilities.
    T=1.0 → original distribution, T→0 → uniform, T=0.5 → square-root sampling.
    min_per_class → if > 0, every class gets at least this many examples
                    (with replacement). Critical for 8,837:1 imbalance where
                    24 classes have < 50 examples.
    """
    if temperature >= 1.0 and min_per_class <= 0:
        return df

    counts = df["label"].value_counts()
    rng = np.random.RandomState(seed)

    if temperature < 1.0:
        weights = counts.apply(lambda c: c ** temperature)
        weights = weights / weights.sum()
        target_total = len(df)
    else:
        weights = counts / counts.sum()
        target_total = len(df)

    sampled = []
    for label, weight in weights.items():
        n_target = max(1, int(round(weight * target_total)))
        if min_per_class > 0:
            n_target = max(n_target, min_per_class)
        group = df[df["label"] == label]
        sampled.append(group.sample(n=n_target, replace=True, random_state=rng))

    result = pd.concat(sampled, ignore_index=True)
    return result.sample(frac=1.0, random_state=rng).reset_index(drop=True)


def _stratified_split(texts: list, labels: list, eval_fraction: float,
                      seed: int, logger) -> tuple[list, list]:
    """Guarantee every class has at least 1 example in both train and eval.
    For classes with only 1 example: duplicate it so both splits see the class.
    Then do a proper stratified split on the remainder.
    """
    rng = np.random.RandomState(seed)

    by_class = defaultdict(list)
    for i, lbl in enumerate(labels):
        by_class[lbl].append(i)

    forced_train, forced_eval = [], []
    remainder_idx = []

    for lbl, idxs in by_class.items():
        rng.shuffle(idxs)
        if len(idxs) == 1:
            forced_train.append(idxs[0])
            forced_eval.append(idxs[0])
        elif len(idxs) == 2:
            forced_train.append(idxs[0])
            forced_eval.append(idxs[1])
        else:
            forced_eval.append(idxs[0])
            remainder_idx.extend(idxs[1:])

    # Split remainder proportionally
    n_eval_target = max(0, int(len(texts) * eval_fraction) - len(forced_eval))
    rng.shuffle(remainder_idx)
    extra_eval = remainder_idx[:n_eval_target]
    extra_train = remainder_idx[n_eval_target:]

    train_set = set(forced_train + extra_train)
    eval_set = set(forced_eval + extra_eval)

    train_texts = [texts[i] for i in sorted(train_set)]
    eval_texts = [texts[i] for i in sorted(eval_set)]

    n_classes = len(by_class)
    train_classes = len(set(labels[i] for i in train_set))
    eval_classes = len(set(labels[i] for i in eval_set))
    logger.info(f"Stratified split: {n_classes} classes → "
                f"train covers {train_classes}, eval covers {eval_classes}")

    return train_texts, eval_texts




def _prepare_dataset(data_path: str, tokenizer, mode: str, max_length: int,
                     seed: int = 42, eval_fraction: float = 0.05,
                     sampling_temperature: float = 0.5,
                     min_per_class: int = 0,
                     cot_max_words: int = 0):
    """Load JSONL, apply temperature sampling, class-aware stratified eval split,
    convert to chat-templated text."""
    df = pd.read_json(data_path, lines=True)
    logger = logging.getLogger(f"student_training_{mode}")

    n_before = len(df)
    df = temperature_resample(df, temperature=sampling_temperature, seed=seed,
                              min_per_class=min_per_class)
    logger.info(f"Temperature sampling (T={sampling_temperature}, "
                f"min_per_class={min_per_class}): {n_before:,} → {len(df):,}")

    texts, labels = [], []
    skipped = 0
    for _, row in df.iterrows():
        messages = _format_chat(row, mode=mode, cot_max_words=cot_max_words)
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if n_tokens > max_length:
            skipped += 1
            continue
        texts.append(text)
        labels.append(int(row["label"]))

    if skipped:
        logger.warning(f"Skipped {skipped:,} examples > max_length={max_length}")

    train_texts, eval_texts = _stratified_split(
        texts, labels, eval_fraction, seed, logger,
    )
    logger.info(f"Dataset: {len(texts):,} total → train {len(train_texts):,} / "
                f"eval {len(eval_texts):,}")

    return (Dataset.from_dict({"text": train_texts}),
            Dataset.from_dict({"text": eval_texts}))


def _prepare_mixed_dataset(cfg: dict, tokenizer, max_length: int, seed: int = 42,
                           eval_fraction: float = 0.05):
    """Prepare a mixed dataset: label-only + CoT examples interleaved."""
    logger = logging.getLogger("student_training_mixed")
    mix_ratio = cfg["student"]["training"].get("mix_ratio", 0.5)
    temp = cfg["student"]["training"].get("sampling_temperature", 0.5)

    label_path = os.path.join(cfg["data"]["processed_dir"], "train.jsonl")
    cot_path = os.path.join(cfg["data"]["processed_dir"], "train_cot.jsonl")

    label_df = pd.read_json(label_path, lines=True)
    cot_df = pd.read_json(cot_path, lines=True)

    label_df = temperature_resample(label_df, temperature=temp, seed=seed)
    cot_df = temperature_resample(cot_df, temperature=temp, seed=seed)

    n_label = int(len(label_df) * mix_ratio)
    n_cot = len(label_df) - n_label

    rng = np.random.RandomState(seed)
    label_sub = label_df.sample(n=min(n_label, len(label_df)),
                                random_state=rng, replace=False)
    cot_sub = cot_df.sample(n=min(n_cot, len(cot_df)),
                            random_state=rng, replace=False)

    logger.info(f"Mixed dataset: {len(label_sub):,} label-only + "
                f"{len(cot_sub):,} CoT (ratio={mix_ratio})")

    texts, labels = [], []
    skipped = 0
    for _, row in label_sub.iterrows():
        messages = _format_chat(row, mode="label")
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        if len(tokenizer.encode(text, add_special_tokens=False)) > max_length:
            skipped += 1
            continue
        texts.append(text)
        labels.append(int(row["label"]))

    for _, row in cot_sub.iterrows():
        messages = _format_chat(row, mode="cot")
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        if len(tokenizer.encode(text, add_special_tokens=False)) > max_length:
            skipped += 1
            continue
        texts.append(text)
        labels.append(int(row["label"]))

    if skipped:
        logger.warning(f"Skipped {skipped:,} examples > max_length")

    train_texts, eval_texts = _stratified_split(
        texts, labels, eval_fraction, seed, logger,
    )
    logger.info(f"Mixed total: {len(texts):,} → train {len(train_texts):,} / "
                f"eval {len(eval_texts):,}")

    return (Dataset.from_dict({"text": train_texts}),
            Dataset.from_dict({"text": eval_texts}))




def _load_base_model(model_name: str, bf16: bool = True):
    dtype = torch.bfloat16 if bf16 else torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True,
            attn_implementation="sdpa",
        )
    return model


def _apply_lora(model, cfg: dict) -> tuple:
    scfg = cfg["student"]["lora"]
    lora_cfg = LoraConfig(
        r=scfg["r"], lora_alpha=scfg["alpha"], lora_dropout=scfg["dropout"],
        target_modules=scfg["target_modules"],
        task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    t, tot = model.get_nb_trainable_parameters()
    return model, t, tot


def _get_checkpoint_dir(cfg: dict, name: str) -> str:
    scratch = os.environ.get("SCRATCH", "")
    if scratch:
        d = os.path.join(scratch, "ddi_checkpoints_v2", name)
    else:
        d = os.path.join(cfg["project"]["output_dir"], "checkpoints", name)
    os.makedirs(d, exist_ok=True)
    return d


def _build_training_args(cfg: dict, output_dir: str, **overrides) -> TrainingArguments:
    tcfg = cfg["student"]["training"]
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=overrides.get("num_epochs", tcfg["num_epochs"]),
        per_device_train_batch_size=tcfg["per_device_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=overrides.get("learning_rate", tcfg["learning_rate"]),
        weight_decay=tcfg["weight_decay"],
        warmup_ratio=tcfg["warmup_ratio"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        bf16=tcfg["bf16"],
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        eval_strategy="steps",
        eval_steps=tcfg["eval_steps"],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        report_to="none",
        seed=overrides.get("seed", cfg["project"]["seed"]),
        data_seed=overrides.get("seed", cfg["project"]["seed"]),
    )
    args.max_seq_length = cfg["student"]["max_length"]
    args.dataset_text_field = "text"
    return args


def _run_training(model, tokenizer, train_ds, eval_ds, args, ckpt_dir, logger,
                  cls_weight: float = 1.0):
    TrainerClass = SFTTrainer
    extra_kwargs = {}

    if cls_weight > 1.0:
        logger.info(f"Using ClassificationWeightedSFTTrainer (cls_weight={cls_weight})")
        cls_marker_ids = tokenizer.encode(
            "\n\nClassification:", add_special_tokens=False,
        )
        TrainerClass = ClassificationWeightedSFTTrainer
        extra_kwargs["cls_weight"] = cls_weight
        extra_kwargs["cls_marker_ids"] = cls_marker_ids

    trainer = TrainerClass(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tokenizer,
        **extra_kwargs,
    )
    last = get_last_checkpoint(ckpt_dir)
    if last:
        logger.info(f"Resuming from {last}")
    else:
        logger.info("Training from scratch")
    trainer.train(resume_from_checkpoint=last)

    final_dir = os.path.join(ckpt_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Model saved to {final_dir}")
    return final_dir


class ClassificationWeightedSFTTrainer(SFTTrainer):
    """SFTTrainer that upweights loss on classification tokens in CoT sequences.

    During CoT training, ~95% of tokens are reasoning and ~5% are
    classification. This imbalance causes the model to optimize primarily
    for reasoning, degrading classification accuracy. This trainer applies
    a multiplier (cls_weight) to the cross-entropy loss of tokens that
    fall after the "Classification:" marker, rebalancing the gradient signal.
    """

    def __init__(self, *args, cls_weight: float = 5.0,
                 cls_marker_ids: list = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_weight = cls_weight
        self.cls_marker_ids = cls_marker_ids or []

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.shape)

        weights = self._build_cls_weights(inputs["input_ids"], shift_labels)

        mask = (shift_labels != -100).float()
        weighted_loss = (per_token_loss * weights * mask).sum() / (weights * mask).sum()

        return (weighted_loss, outputs) if return_outputs else weighted_loss

    def _build_cls_weights(self, input_ids: torch.Tensor,
                           shift_labels: torch.Tensor) -> torch.Tensor:
        weights = torch.ones_like(shift_labels, dtype=torch.float32)
        if not self.cls_marker_ids:
            return weights

        marker_len = len(self.cls_marker_ids)
        for b in range(input_ids.shape[0]):
            seq = input_ids[b].tolist()
            # Find last occurrence of the classification marker
            pos = -1
            for i in range(len(seq) - marker_len, -1, -1):
                if seq[i:i + marker_len] == self.cls_marker_ids:
                    pos = i
                    break
            if pos >= 0:
                # Upweight everything from the marker onward (shifted by 1 for labels)
                start = max(0, pos - 1)
                weights[b, start:] = self.cls_weight

        return weights


def _ddp_cleanup(logger):
    """After DDP training, barrier + destroy for non-rank-0, return True if should exit."""
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        if torch.distributed.get_rank() != 0:
            return True
        torch.distributed.destroy_process_group()
        logger.info("DDP cleanup done. Run with --eval-only for evaluation.")
    return False



def train_label_only(cfg: dict, seed: int = None) -> str:
    """Condition B: Label-only fine-tuning."""
    seed = seed or cfg["project"]["seed"]
    logger = setup_logging("train_B_label")
    set_seed(seed)
    ensure_dirs(cfg)

    scfg = cfg["student"]
    model_name = scfg["model_name"]
    logger.info(f"=== Condition B: Label-only | seed={seed} ===")
    logger.info(f"GPU setup:\n{gpu_info()}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data_path = os.path.join(cfg["data"]["processed_dir"], "train.jsonl")
    temp = cfg["student"]["training"].get("sampling_temperature", 0.5)
    train_ds, eval_ds = _prepare_dataset(
        data_path, tokenizer, "label", scfg["max_length"],
        seed=seed, sampling_temperature=temp,
    )

    model = _load_base_model(model_name, scfg["training"]["bf16"])
    model, t, tot = _apply_lora(model, cfg)
    logger.info(f"LoRA: {t:,} / {tot:,} trainable ({100*t/tot:.2f}%)")

    ckpt_dir = _get_checkpoint_dir(cfg, f"B_label_s{seed}")
    args = _build_training_args(cfg, ckpt_dir, seed=seed)

    final = _run_training(model, tokenizer, train_ds, eval_ds, args, ckpt_dir, logger)
    if _ddp_cleanup(logger):
        return None
    return final


def train_cot_naive(cfg: dict, seed: int = None) -> str:
    """Condition C_naive: Naive CoT distillation (reproduces the known failure)."""
    seed = seed or cfg["project"]["seed"]
    logger = setup_logging("train_C_naive")
    set_seed(seed)
    ensure_dirs(cfg)

    scfg = cfg["student"]
    model_name = scfg["model_name"]
    logger.info(f"=== Condition C_naive: Naive CoT | seed={seed} ===")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    cot_path = os.path.join(cfg["data"]["processed_dir"], "train_cot.jsonl")
    if not os.path.exists(cot_path):
        logger.error(f"CoT training data not found: {cot_path}")
        return None

    temp = cfg["student"]["training"].get("sampling_temperature", 0.5)
    train_ds, eval_ds = _prepare_dataset(
        cot_path, tokenizer, "cot", scfg["max_length"],
        seed=seed, sampling_temperature=temp,
    )

    model = _load_base_model(model_name, scfg["training"]["bf16"])
    model, t, tot = _apply_lora(model, cfg)
    logger.info(f"LoRA: {t:,} / {tot:,} trainable ({100*t/tot:.2f}%)")

    ckpt_dir = _get_checkpoint_dir(cfg, f"C_naive_s{seed}")
    args = _build_training_args(cfg, ckpt_dir, seed=seed)

    final = _run_training(model, tokenizer, train_ds, eval_ds, args, ckpt_dir, logger)
    if _ddp_cleanup(logger):
        return None
    return final


def train_sequential(cfg: dict, base_checkpoint: str = None,
                     seed: int = None) -> str:
    """Condition C_seq: Sequential fine-tuning.
    Phase 1 → label-only (B), Phase 2 → continue on CoT with lower LR.
    If base_checkpoint is None, trains B first then continues.
    """
    seed = seed or cfg["project"]["seed"]
    logger = setup_logging("train_C_seq")
    set_seed(seed)
    ensure_dirs(cfg)

    scfg = cfg["student"]
    tcfg = scfg["training"]
    model_name = scfg["model_name"]

    # Phase 1: Get or create the label-only (B) checkpoint
    if base_checkpoint is None:
        b_ckpt_dir = _get_checkpoint_dir(cfg, f"B_label_s{seed}")
        b_final = os.path.join(b_ckpt_dir, "final")
        if not os.path.exists(b_final):
            logger.info("Phase 1: Training label-only (B) first …")
            b_final = train_label_only(cfg, seed=seed)
            if b_final is None:
                return None
        else:
            logger.info(f"Phase 1: Using existing B checkpoint: {b_final}")
        base_checkpoint = b_final

    logger.info(f"=== Condition C_seq: Sequential CoT | seed={seed} ===")
    logger.info(f"Base checkpoint: {base_checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    cot_path = os.path.join(cfg["data"]["processed_dir"], "train_cot.jsonl")
    temp = cfg["student"]["training"].get("sampling_temperature", 0.5)
    train_ds, eval_ds = _prepare_dataset(
        cot_path, tokenizer, "cot", scfg["max_length"],
        seed=seed, sampling_temperature=temp,
    )

    # Phase 2: Load B checkpoint and continue on CoT with reduced LR
    logger.info(f"Loading base model + LoRA from {base_checkpoint}")
    base_model = _load_base_model(model_name, tcfg["bf16"])
    model = PeftModel.from_pretrained(base_model, base_checkpoint, is_trainable=True)
    t, tot = model.get_nb_trainable_parameters()
    logger.info(f"LoRA (resumed): {t:,} / {tot:,} trainable ({100*t/tot:.2f}%)")

    lr_ratio = tcfg.get("sequential_lr_ratio", 0.25)
    seq_epochs = tcfg.get("sequential_epochs", 2)
    seq_lr = tcfg["learning_rate"] * lr_ratio
    logger.info(f"Sequential LR: {seq_lr:.2e} (base × {lr_ratio}), epochs: {seq_epochs}")

    ckpt_dir = _get_checkpoint_dir(cfg, f"C_seq_s{seed}")
    args = _build_training_args(
        cfg, ckpt_dir, num_epochs=seq_epochs, learning_rate=seq_lr, seed=seed,
    )

    final = _run_training(model, tokenizer, train_ds, eval_ds, args, ckpt_dir, logger)
    if _ddp_cleanup(logger):
        return None
    return final


def train_sequential_weighted(cfg: dict, base_checkpoint: str = None,
                              seed: int = None, cls_weight: float = 10.0,
                              cot_max_words: int = 0,
                              min_per_class: int = 0,
                              sampling_temperature: float = None,
                              condition_name: str = "C_wt") -> str:
    """Condition C_wt / C_compact: Sequential training with weighted classification loss.

    Same as C_seq (Phase 1: label-only, Phase 2: CoT with lower LR) but
    Phase 2 applies cls_weight multiplier to the loss on classification
    tokens (everything after "Classification:"). Optionally also truncates
    CoT and uses aggressive class balancing for C_compact.
    """
    seed = seed or cfg["project"]["seed"]
    logger = setup_logging(f"train_{condition_name}")
    set_seed(seed)
    ensure_dirs(cfg)

    scfg = cfg["student"]
    tcfg = scfg["training"]
    model_name = scfg["model_name"]

    if base_checkpoint is None:
        b_ckpt_dir = _get_checkpoint_dir(cfg, f"B_label_s{seed}")
        b_final = os.path.join(b_ckpt_dir, "final")
        if not os.path.exists(b_final):
            logger.info("Phase 1: Training label-only (B) first …")
            b_final = train_label_only(cfg, seed=seed)
            if b_final is None:
                return None
        else:
            logger.info(f"Phase 1: Using existing B checkpoint: {b_final}")
        base_checkpoint = b_final

    if sampling_temperature is None:
        sampling_temperature = tcfg.get("sampling_temperature", 0.5)

    logger.info(f"=== Condition {condition_name}: Weighted sequential CoT | seed={seed} ===")
    logger.info(f"Base checkpoint: {base_checkpoint}")
    logger.info(f"Classification loss weight: {cls_weight}x")
    logger.info(f"CoT max words: {cot_max_words if cot_max_words > 0 else 'unlimited'}")
    logger.info(f"Sampling temperature: {sampling_temperature}, min_per_class: {min_per_class}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    cot_path = os.path.join(cfg["data"]["processed_dir"], "train_cot.jsonl")
    train_ds, eval_ds = _prepare_dataset(
        cot_path, tokenizer, "cot", scfg["max_length"],
        seed=seed, sampling_temperature=sampling_temperature,
        min_per_class=min_per_class, cot_max_words=cot_max_words,
    )

    logger.info(f"Loading base model + LoRA from {base_checkpoint}")
    base_model = _load_base_model(model_name, tcfg["bf16"])
    model = PeftModel.from_pretrained(base_model, base_checkpoint, is_trainable=True)
    t, tot = model.get_nb_trainable_parameters()
    logger.info(f"LoRA (resumed): {t:,} / {tot:,} trainable ({100*t/tot:.2f}%)")

    lr_ratio = tcfg.get("sequential_lr_ratio", 0.25)
    seq_epochs = tcfg.get("sequential_epochs", 2)
    seq_lr = tcfg["learning_rate"] * lr_ratio

    ckpt_dir = _get_checkpoint_dir(cfg, f"{condition_name}_s{seed}")
    args = _build_training_args(
        cfg, ckpt_dir, num_epochs=seq_epochs, learning_rate=seq_lr, seed=seed,
    )

    final = _run_training(
        model, tokenizer, train_ds, eval_ds, args, ckpt_dir, logger,
        cls_weight=cls_weight,
    )
    if _ddp_cleanup(logger):
        return None
    return final


def train_compact(cfg: dict, base_checkpoint: str = None,
                  seed: int = None, cls_weight: float = 10.0,
                  cot_max_words: int = 100, min_per_class: int = 50,
                  sampling_temperature: float = 0.25) -> str:
    """Condition C_compact: All three fixes combined.
    - Compact CoT (~100 words): matches 7B student capacity
    - Classification-weighted loss (10x): fixes 95/5 token imbalance
    - Aggressive class balancing (T=0.25, min50): addresses 8,837:1 imbalance
    """
    return train_sequential_weighted(
        cfg, base_checkpoint=base_checkpoint, seed=seed,
        cls_weight=cls_weight, cot_max_words=cot_max_words,
        min_per_class=min_per_class, sampling_temperature=sampling_temperature,
        condition_name="C_compact",
    )


def train_mixed(cfg: dict, seed: int = None) -> str:
    """Condition C_mix: Mixed training with interleaved label-only + CoT."""
    seed = seed or cfg["project"]["seed"]
    logger = setup_logging("train_C_mix")
    set_seed(seed)
    ensure_dirs(cfg)

    scfg = cfg["student"]
    model_name = scfg["model_name"]
    logger.info(f"=== Condition C_mix: Mixed training | seed={seed} ===")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_ds, eval_ds = _prepare_mixed_dataset(
        cfg, tokenizer, scfg["max_length"], seed=seed,
    )

    model = _load_base_model(model_name, scfg["training"]["bf16"])
    model, t, tot = _apply_lora(model, cfg)
    logger.info(f"LoRA: {t:,} / {tot:,} trainable ({100*t/tot:.2f}%)")

    ckpt_dir = _get_checkpoint_dir(cfg, f"C_mix_s{seed}")
    args = _build_training_args(cfg, ckpt_dir, seed=seed)

    final = _run_training(model, tokenizer, train_ds, eval_ds, args, ckpt_dir, logger)
    if _ddp_cleanup(logger):
        return None
    return final




MODES = {
    "label": train_label_only,
    "cot_naive": train_cot_naive,
    "sequential": train_sequential,
    "mixed": train_mixed,
    "weighted": train_sequential_weighted,
    "compact": train_compact,
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(MODES.keys()), required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--base-checkpoint", type=str, default=None,
                        help="Path to B checkpoint for sequential/weighted/compact")
    parser.add_argument("--cls-weight", type=float, default=10.0,
                        help="Classification loss multiplier (weighted/compact)")
    parser.add_argument("--cot-max-words", type=int, default=0,
                        help="Truncate CoT to first N words (0=no truncation)")
    parser.add_argument("--min-per-class", type=int, default=0,
                        help="Minimum examples per class after resampling")
    parser.add_argument("--sampling-temperature", type=float, default=None,
                        help="Override sampling temperature (default from config)")
    args = parser.parse_args()

    cfg = load_config()
    if args.mode in ("sequential",):
        MODES[args.mode](cfg, base_checkpoint=args.base_checkpoint, seed=args.seed)
    elif args.mode in ("weighted",):
        MODES[args.mode](
            cfg, base_checkpoint=args.base_checkpoint, seed=args.seed,
            cls_weight=args.cls_weight, cot_max_words=args.cot_max_words,
            min_per_class=args.min_per_class,
            sampling_temperature=args.sampling_temperature,
        )
    elif args.mode == "compact":
        MODES[args.mode](
            cfg, base_checkpoint=args.base_checkpoint, seed=args.seed,
            cls_weight=args.cls_weight,
            cot_max_words=args.cot_max_words or 100,
            min_per_class=args.min_per_class or 50,
            sampling_temperature=args.sampling_temperature if args.sampling_temperature is not None else 0.25,
        )
    else:
        MODES[args.mode](cfg, seed=args.seed)
