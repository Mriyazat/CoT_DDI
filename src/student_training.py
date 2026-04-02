"""
Student fine-tuning for DDI CoT Distillation V3.

Five training conditions:
  B         : Label-only classification baseline
  C_naive   : Naive CoT distillation (reproduces the known failure)
  C_seq     : Sequential (B checkpoint -> CoT with lower LR)
  C_compact : Sequential + truncated CoT (100 words) + weighted loss (ablation)
  C_summary : Sequential + teacher SUMMARY only + weighted loss (PRIMARY)

C_summary trains on the teacher's Summary + Classification + Severity —
NOT the full Reasoning. Token ratio ~70/30 vs C_naive's 95/5.

All use Qwen2.5-7B-Instruct with LoRA, temperature-scaled class sampling,
DDP via torchrun, and checkpoint resume.
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
    AutoModelForCausalLM, AutoTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from trl import SFTTrainer, SFTConfig

from src.utils import load_config, setup_logging, set_seed, ensure_dirs, gpu_info
from src.data_preparation import SYSTEM_PROMPT, build_student_input


# ── Chat formatting ───────────────────────────────────────────────────

def _format_chat(row, mode: str = "cot", cot_max_words: int = 0,
                 profiles: dict = None, retrieved: dict = None) -> list[dict]:
    """Build a chat-format training example.

    Modes:
      "label"   -> classification only (Condition B)
      "cot"     -> full reasoning + classification (C_naive / C_seq)
      "compact" -> truncated reasoning + classification (C_compact)
      "summary" -> teacher summary + classification + severity (C_summary)
    """
    idx = int(row.get("_orig_idx", row.name)) if hasattr(row, "name") else 0
    retr_indices = retrieved.get(idx, []) if retrieved else []
    retr_examples = None
    if retr_indices and profiles:
        retr_examples = []
        for ri_data in retr_indices:
            if isinstance(ri_data, dict):
                retr_examples.append(ri_data)

    user_msg = build_student_input(row, profiles or {}, retr_examples)

    severity = str(row.get("severity", "Unknown"))
    label_text = str(row.get("label_text", ""))

    if mode == "summary":
        raw_summary = row.get("teacher_summary", "")
        summary = str(raw_summary) if raw_summary is not None and str(raw_summary) not in ("nan", "None", "") else ""
        if not summary:
            raw_cot = row.get("teacher_cot", "")
            summary = str(raw_cot)[:300] if raw_cot is not None and str(raw_cot) not in ("nan", "None", "") else ""
        assistant_msg = (
            f"{summary}\n\n"
            f"Classification: Y={row['label']} -- \"{label_text}\"\n"
            f"Severity: {severity}"
        )
    elif mode in ("cot", "compact"):
        cot = re.sub(
            r"\n*##\s*Classification.*$", "", str(row["teacher_cot"]),
            flags=re.IGNORECASE | re.DOTALL,
        ).rstrip()
        if mode == "compact" or cot_max_words > 0:
            max_w = cot_max_words if cot_max_words > 0 else 100
            words = cot.split()
            if len(words) > max_w:
                cot = " ".join(words[:max_w])
        assistant_msg = (
            f"{cot}\n\n"
            f"Classification: Y={row['label']} -- \"{label_text}\"\n"
            f"Severity: {severity}"
        )
    else:
        assistant_msg = (
            f"Classification: Y={row['label']} -- \"{label_text}\"\n"
            f"Severity: {severity}"
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]


# ── Temperature-scaled class sampling ─────────────────────────────────

def temperature_resample(df: pd.DataFrame, temperature: float = 0.5,
                         seed: int = 42, min_per_class: int = 0) -> pd.DataFrame:
    """Resample with temperature-scaled class probabilities.
    T=1.0 -> original, T->0 -> uniform, T=0.5 -> square-root sampling.
    """
    if temperature >= 1.0 and min_per_class <= 0:
        return df

    counts = df["label"].value_counts()
    rng = np.random.RandomState(seed)

    weights = counts.apply(lambda c: c ** temperature)
    weights = weights / weights.sum()
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


# ── Stratified split ──────────────────────────────────────────────────

def _stratified_split(texts: list, labels: list, eval_fraction: float,
                      seed: int, logger) -> tuple[list, list]:
    """Guarantee every class appears in both train and eval splits."""
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

    n_eval_target = max(0, int(len(texts) * eval_fraction) - len(forced_eval))
    rng.shuffle(remainder_idx)
    extra_eval = remainder_idx[:n_eval_target]
    extra_train = remainder_idx[n_eval_target:]

    train_set = set(forced_train + extra_train)
    eval_set = set(forced_eval + extra_eval)

    train_texts = [texts[i] for i in sorted(train_set)]
    eval_texts = [texts[i] for i in sorted(eval_set)]
    logger.info(f"Stratified split: {len(by_class)} classes -> "
                f"train {len(train_texts):,} / eval {len(eval_texts):,}")

    return train_texts, eval_texts


# ── Severity-balanced oversampling ─────────────────────────────────────

def _severity_oversample(df, target_known_ratio: float = 0.30, seed: int = 42):
    """Oversample examples with known severity (Major/Moderate/Minor) to reach target ratio."""
    known_mask = df["severity"].isin(["Major", "Moderate", "Minor"])
    known = df[known_mask]
    unknown = df[~known_mask]

    current_ratio = len(known) / len(df) if len(df) > 0 else 0
    if current_ratio >= target_known_ratio or len(known) == 0:
        return df

    n_unknown = len(unknown)
    n_known_target = int(n_unknown * target_known_ratio / (1 - target_known_ratio))

    rng = np.random.RandomState(seed)
    known_resampled = known.sample(n=n_known_target, replace=True, random_state=rng)

    result = pd.concat([unknown, known_resampled], ignore_index=True)
    return result.sample(frac=1.0, random_state=rng).reset_index(drop=True)


# ── Dataset preparation ───────────────────────────────────────────────

def _prepare_dataset(data_path: str, tokenizer, mode: str, max_length: int,
                     seed: int = 42, eval_fraction: float = 0.05,
                     sampling_temperature: float = 0.5,
                     min_per_class: int = 0, cot_max_words: int = 0,
                     profiles: dict = None, retrieved: dict = None,
                     severity_oversample: bool = False,
                     severity_target_ratio: float = 0.30,
                     compute_entity_lookup: bool = False):
    """Load JSONL, apply temperature sampling, split, and convert to chat format.

    Returns (train_dataset, eval_dataset, entity_lookup).
    entity_lookup is populated only when compute_entity_lookup=True.
    """
    df = pd.read_json(data_path, lines=True)
    logger = logging.getLogger(f"student_training_{mode}")

    if "_orig_idx" not in df.columns:
        df["_orig_idx"] = df.index

    n_before = len(df)
    df = temperature_resample(df, temperature=sampling_temperature, seed=seed,
                              min_per_class=min_per_class)
    logger.info(f"Temperature sampling (T={sampling_temperature}): "
                f"{n_before:,} -> {len(df):,}")

    if severity_oversample and "severity" in df.columns:
        n_pre = len(df)
        df = _severity_oversample(df, target_known_ratio=severity_target_ratio,
                                  seed=seed)
        logger.info(f"Severity oversampling: {n_pre:,} -> {len(df):,}")

    texts, labels = [], []
    entity_lookup = {}
    skipped_long, skipped_noclass = 0, 0
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 10_000 == 0:
            logger.info(f"Tokenizing: {i:,}/{total:,} ({100*i/total:.0f}%) ...")
        messages = _format_chat(row, mode=mode, cot_max_words=cot_max_words,
                                profiles=profiles, retrieved=retrieved)
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if n_tokens > max_length:
            skipped_long += 1
            continue
        if "Classification:" not in text or "Severity:" not in text:
            skipped_noclass += 1
            continue
        texts.append(text)
        labels.append(int(row["label"]))

        if compute_entity_lookup and profiles:
            pair_eids = _extract_pair_entity_ids(row, profiles, tokenizer)
            if pair_eids:
                text_ids = tokenizer.encode(text, add_special_tokens=False)
                key = tuple(text_ids[:_ENTITY_LOOKUP_KEY_LEN])
                entity_lookup[key] = pair_eids

    logger.info(f"Tokenizing: {total:,}/{total:,} (100%) done.")
    if skipped_long:
        logger.warning(f"Skipped {skipped_long:,}/{total:,} examples > max_length={max_length}")
    if skipped_noclass:
        logger.error(f"CRITICAL: {skipped_noclass:,} examples missing Classification/Severity token!")
    if entity_lookup:
        logger.info(f"KAT entity lookup: {len(entity_lookup):,} examples with entity info")

    train_texts, eval_texts = _stratified_split(
        texts, labels, eval_fraction, seed, logger,
    )

    return (Dataset.from_dict({"text": train_texts}),
            Dataset.from_dict({"text": eval_texts}),
            entity_lookup)


# ── Model loading ─────────────────────────────────────────────────────

def _load_base_model(model_name: str, bf16: bool = True):
    dtype = torch.bfloat16 if bf16 else torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        )
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    return model


def _apply_lora(model, cfg: dict):
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
        d = os.path.join(scratch, "ddi_checkpoints_v3", name)
    else:
        d = os.path.join(cfg["project"]["output_dir"], "checkpoints", name)
    os.makedirs(d, exist_ok=True)
    return d


def _build_training_args(cfg: dict, output_dir: str, **overrides) -> SFTConfig:
    tcfg = cfg["student"]["training"]
    scfg = cfg["student"]

    fsdp_cfg = tcfg.get("fsdp", {})
    fsdp_enabled = fsdp_cfg.get("enabled", False)

    extra = {}
    if fsdp_enabled:
        strategy = fsdp_cfg.get("sharding_strategy", "full_shard")
        parts = [strategy]
        if fsdp_cfg.get("auto_wrap", True):
            parts.append("auto_wrap")
        extra["fsdp"] = " ".join(parts)
        extra["fsdp_config"] = {
            "min_num_params": int(fsdp_cfg.get("min_num_params", 1_000_000)),
            "backward_prefetch": fsdp_cfg.get("backward_prefetch", "backward_pre"),
            "forward_prefetch": fsdp_cfg.get("forward_prefetch", False),
            "use_orig_params": fsdp_cfg.get("use_orig_params", True),
            "cpu_ram_efficient_loading": fsdp_cfg.get("cpu_ram_efficient_loading", True),
            "sync_module_states": fsdp_cfg.get("sync_module_states", True),
        }

    args = SFTConfig(
        output_dir=output_dir,
        max_length=overrides.get("max_length", scfg["max_length"]),
        dataset_text_field="text",
        num_train_epochs=overrides.get("num_epochs", tcfg["num_epochs"]),
        per_device_train_batch_size=tcfg["per_device_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_batch_size"],
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
        **extra,
    )
    return args


# ── Classification-weighted loss ──────────────────────────────────────

class ClassificationWeightedSFTTrainer(SFTTrainer):
    """Upweights loss on classification tokens in CoT/summary sequences."""

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

        if not model.training:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            return (loss, outputs) if return_outputs else loss

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
            pos = -1
            for i in range(len(seq) - marker_len, -1, -1):
                if seq[i:i + marker_len] == self.cls_marker_ids:
                    pos = i
                    break
            if pos >= 0:
                start = max(0, pos - 1)
                weights[b, start:] = self.cls_weight
        return weights


def _extract_pair_entity_ids(row, profiles: dict, tokenizer) -> set:
    """Extract tokenized entity IDs specific to a drug pair's KB profiles.

    Returns the set of token IDs that correspond to pharmacological entities
    (drug names, enzymes, transporters, targets) for the two drugs in this pair.
    """
    entity_strings = set()
    for drug_key in ("drug1_id", "drug2_id"):
        did = str(row.get(drug_key, ""))
        prof = profiles.get(did, {})
        if prof.get("name"):
            entity_strings.add(prof["name"])
        for field in ("enzymes", "transporters", "targets"):
            for raw in prof.get(field, []):
                name = raw.split("(")[0].split(":")[0].strip()
                if len(name) >= 3:
                    entity_strings.add(name)

    token_ids = set()
    for entity in entity_strings:
        ids = tokenizer.encode(entity, add_special_tokens=False)
        for tid in ids:
            decoded = tokenizer.decode([tid]).strip()
            if len(decoded) >= 3:
                token_ids.add(tid)
    return token_ids


_ENTITY_LOOKUP_KEY_LEN = 48


class KATTrainer(ClassificationWeightedSFTTrainer):
    """Knowledge-Anchored Token weighting trainer for CoT distillation.

    Per-example dynamic weighting: for each training example, entity tokens
    from the specific drug pair's KB profile receive an upweighted loss.
    Weight is scaled inversely with sequence length to counteract gradient
    dilution in longer rationales (cf. Skip-Thinking, EMNLP 2025).
    """

    def __init__(self, *args, entity_lookup: dict = None,
                 kat_alpha: float = 2.0,
                 sev_marker_ids: list = None, sev_gamma: float = 1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.entity_lookup = entity_lookup or {}
        self.kat_alpha = kat_alpha
        self.sev_marker_ids = sev_marker_ids or []
        self.sev_gamma = sev_gamma
        self._bos_id = None
        tok = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if tok is not None:
            self._bos_id = getattr(tok, "bos_token_id", None)

    def _build_cls_weights(self, input_ids, shift_labels):
        weights = super()._build_cls_weights(input_ids, shift_labels)

        if self.entity_lookup:
            self._apply_kat_weights(input_ids, shift_labels, weights)

        if self.sev_gamma > 1.0 and self.sev_marker_ids:
            self._apply_severity_weights(input_ids, weights)

        return weights

    def _apply_kat_weights(self, input_ids, shift_labels, weights):
        for b in range(input_ids.shape[0]):
            ids = input_ids[b].tolist()
            start = 1 if (self._bos_id is not None and ids[0] == self._bos_id) else 0
            key = tuple(ids[start:start + _ENTITY_LOOKUP_KEY_LEN])
            entity_ids = self.entity_lookup.get(key)
            if not entity_ids:
                continue

            n = min(input_ids.shape[1] - 1, weights.shape[1])
            target_ids = input_ids[b, 1:n + 1]

            entity_mask = torch.zeros(n, dtype=torch.bool, device=input_ids.device)
            for tid in entity_ids:
                entity_mask |= (target_ids == tid)

            active_tokens = (shift_labels[b] != -100).sum().float()
            n_entity = entity_mask.sum().float()
            if n_entity == 0 or active_tokens == 0:
                continue

            scale_factor = (active_tokens / n_entity).clamp(min=1.0, max=20.0)
            effective_weight = (self.kat_alpha * scale_factor.sqrt()).clamp(max=8.0)

            boost_mask = entity_mask[:weights.shape[1]] & (weights[b, :n] < self.cls_weight)
            weights[b, :n] = torch.where(
                boost_mask,
                torch.full_like(weights[b, :n], effective_weight.item()),
                weights[b, :n],
            )

    def _apply_severity_weights(self, input_ids, weights):
        marker_len = len(self.sev_marker_ids)
        if marker_len == 0:
            return
        for b in range(input_ids.shape[0]):
            seq = input_ids[b].tolist()
            for i in range(len(seq) - marker_len, -1, -1):
                if seq[i:i + marker_len] == self.sev_marker_ids:
                    start = max(0, i - 1)
                    n = min(len(seq) - 1, weights.shape[1])
                    weights[b, start:n] = torch.clamp(
                        weights[b, start:n] * self.sev_gamma,
                        max=self.cls_weight,
                    )
                    break


def _verify_classification_tokens(train_ds, tokenizer, max_seq_length, logger):
    """Spot-check that Classification: and Severity: survive tokenization + truncation."""
    n_check = min(200, len(train_ds))
    cls_trunc, sev_trunc = 0, 0
    for i in range(n_check):
        text = train_ds[i]["text"]
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=True,
                               max_length=max_seq_length)
        decoded_tail = tokenizer.decode(ids[-80:])
        if "Classification:" not in decoded_tail:
            cls_trunc += 1
        if "Severity:" not in decoded_tail:
            sev_trunc += 1
    if cls_trunc > 0 or sev_trunc > 0:
        logger.error(
            f"CRITICAL: Classification truncated in {cls_trunc}/{n_check}, "
            f"Severity truncated in {sev_trunc}/{n_check} samples "
            f"at max_seq_length={max_seq_length}! "
            f"Increase max_length or shorten prompts.")
        raise ValueError(
            f"Classification/Severity tokens truncated. "
            f"Training would repeat V2 failure. Aborting.")
    logger.info(f"Safety check passed: {n_check}/{n_check} samples have "
                f"Classification: and Severity: intact after tokenization.")


def _run_training(model, tokenizer, train_ds, eval_ds, args, ckpt_dir, logger,
                  cls_weight: float = 1.0,
                  entity_lookup: dict = None, kat_alpha: float = 2.0,
                  sev_gamma: float = 1.0):
    _verify_classification_tokens(train_ds, tokenizer, args.max_length, logger)

    TrainerClass = SFTTrainer
    extra_kwargs = {}

    if cls_weight > 1.0:
        logger.info(f"Using weighted training (cls_weight={cls_weight})")
        probe = "Some reasoning text.\n\nClassification: Y=1"
        probe_ids = tokenizer.encode(probe, add_special_tokens=False)
        marker_isolated = tokenizer.encode(
            "\n\nClassification:", add_special_tokens=False,
        )
        marker_found = any(
            probe_ids[i:i + len(marker_isolated)] == marker_isolated
            for i in range(len(probe_ids) - len(marker_isolated) + 1)
        )
        if not marker_found:
            prefix_ids = tokenizer.encode("Some reasoning text.", add_special_tokens=False)
            marker_isolated = probe_ids[len(prefix_ids):]
            suffix_ids = tokenizer.encode(" Y=1", add_special_tokens=False)
            if len(suffix_ids) > 0:
                marker_isolated = marker_isolated[:-len(suffix_ids)]
            logger.warning(
                f"BPE marker mismatch: using context-derived marker IDs {marker_isolated}"
            )
        cls_marker_ids = marker_isolated
        logger.info(f"  Classification marker IDs: {cls_marker_ids}")

        use_kat = entity_lookup and len(entity_lookup) > 0
        if use_kat or sev_gamma > 1.0:
            sev_marker_ids = []
            if sev_gamma > 1.0:
                sev_probe = "text.\n\nSeverity: Major"
                sev_probe_ids = tokenizer.encode(sev_probe, add_special_tokens=False)
                sev_isolated = tokenizer.encode("\n\nSeverity:", add_special_tokens=False)
                sev_found = any(
                    sev_probe_ids[i:i + len(sev_isolated)] == sev_isolated
                    for i in range(len(sev_probe_ids) - len(sev_isolated) + 1)
                )
                if not sev_found:
                    pre = tokenizer.encode("text.", add_special_tokens=False)
                    suf = tokenizer.encode(" Major", add_special_tokens=False)
                    sev_isolated = sev_probe_ids[len(pre):]
                    if len(suf) > 0:
                        sev_isolated = sev_isolated[:-len(suf)]
                sev_marker_ids = sev_isolated
                logger.info(f"  Severity marker IDs: {sev_marker_ids}, gamma={sev_gamma}")

            TrainerClass = KATTrainer
            extra_kwargs["entity_lookup"] = entity_lookup or {}
            extra_kwargs["kat_alpha"] = kat_alpha
            extra_kwargs["sev_marker_ids"] = sev_marker_ids
            extra_kwargs["sev_gamma"] = sev_gamma
            logger.info(f"KAT: alpha={kat_alpha}, {len(entity_lookup or {}):,} pair lookups")
        else:
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


def _ddp_cleanup(logger):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        if torch.distributed.get_rank() != 0:
            return True
        logger.info("DDP barrier passed; rank 0 continues.")
    return False


def _inter_phase_cleanup(logger):
    """Barrier + free GPU memory between sequential training phases.
    All ranks survive (unlike _ddp_cleanup which exits non-rank-0).
    """
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        rank = torch.distributed.get_rank()
        logger.info(f"Inter-phase barrier passed on rank {rank}.")
    torch.cuda.empty_cache()
    import gc; gc.collect()
    torch.cuda.empty_cache()


# ── Load shared resources for enriched prompts ────────────────────────

def _load_shared_resources(cfg: dict):
    """Load drug profiles and precomputed retrievals."""
    processed = cfg["data"]["processed_dir"]
    profiles = {}
    prof_path = os.path.join(processed, "drug_profiles.json")
    if os.path.exists(prof_path):
        with open(prof_path) as f:
            profiles = json.load(f)

    retrieved = {}
    retr_path = os.path.join(processed, "retrieved_examples_train.json")
    if os.path.exists(retr_path):
        with open(retr_path) as f:
            raw = json.load(f)
        for k, v in raw.items():
            retrieved[int(k)] = v

    return profiles, retrieved


# ── Training modes ────────────────────────────────────────────────────

def train_label_only(cfg: dict, seed: int = None,
                     _continues_to_next_phase: bool = False) -> str:
    """Condition B: Label-only fine-tuning."""
    seed = seed or cfg["project"]["seed"]
    logger = setup_logging("train_B_label")
    set_seed(seed)
    ensure_dirs(cfg)

    scfg = cfg["student"]
    model_name = scfg["model_name"]
    logger.info(f"=== Condition B: Label-only | seed={seed} ===")
    logger.info(f"GPU setup:\n{gpu_info()}")

    profiles, retrieved = _load_shared_resources(cfg)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data_path = os.path.join(cfg["data"]["processed_dir"], "train.jsonl")
    temp = scfg["training"].get("sampling_temperature", 0.5)
    train_ds, eval_ds, _ = _prepare_dataset(
        data_path, tokenizer, "label", scfg["max_length"],
        seed=seed, sampling_temperature=temp,
        profiles=profiles, retrieved=retrieved,
    )

    model = _load_base_model(model_name, scfg["training"]["bf16"])
    model, t, tot = _apply_lora(model, cfg)
    logger.info(f"LoRA: {t:,} / {tot:,} trainable ({100*t/tot:.2f}%)")

    ckpt_dir = _get_checkpoint_dir(cfg, f"B_label_s{seed}")
    args = _build_training_args(cfg, ckpt_dir, seed=seed)

    final = _run_training(model, tokenizer, train_ds, eval_ds, args, ckpt_dir, logger)

    if _continues_to_next_phase:
        _inter_phase_cleanup(logger)
    else:
        if _ddp_cleanup(logger):
            return None
    return final


def train_cot_naive(cfg: dict, seed: int = None) -> str:
    """Condition C_naive: Naive CoT distillation."""
    seed = seed or cfg["project"]["seed"]
    logger = setup_logging("train_C_naive")
    set_seed(seed)
    ensure_dirs(cfg)

    scfg = cfg["student"]
    model_name = scfg["model_name"]
    logger.info(f"=== Condition C_naive: Naive CoT | seed={seed} ===")

    profiles, retrieved = _load_shared_resources(cfg)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    cot_path = os.path.join(cfg["data"]["processed_dir"], "train_cot.jsonl")
    if not os.path.exists(cot_path):
        logger.error(f"CoT data not found: {cot_path}")
        return None

    temp = scfg["training"].get("sampling_temperature", 0.5)
    train_ds, eval_ds, _ = _prepare_dataset(
        cot_path, tokenizer, "cot", scfg["max_length"],
        seed=seed, sampling_temperature=temp,
        profiles=profiles, retrieved=retrieved,
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
    """Condition C_seq: Sequential fine-tuning (B -> CoT with lower LR)."""
    seed = seed or cfg["project"]["seed"]
    logger = setup_logging("train_C_seq")
    set_seed(seed)
    ensure_dirs(cfg)

    scfg = cfg["student"]
    tcfg = scfg["training"]
    model_name = scfg["model_name"]

    if base_checkpoint is None:
        b_ckpt_dir = _get_checkpoint_dir(cfg, f"B_label_s{seed}")
        b_final = os.path.join(b_ckpt_dir, "final")
        if not os.path.exists(b_final):
            logger.info("Phase 1: Training label-only (B) first...")
            b_final = train_label_only(cfg, seed=seed,
                                       _continues_to_next_phase=True)
        else:
            logger.info(f"Phase 1: Using existing B checkpoint: {b_final}")
        base_checkpoint = b_final

    logger.info(f"=== Condition C_seq: Sequential CoT | seed={seed} ===")

    profiles, retrieved = _load_shared_resources(cfg)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    cot_path = os.path.join(cfg["data"]["processed_dir"], "train_cot.jsonl")
    temp = scfg["training"].get("sampling_temperature", 0.5)
    train_ds, eval_ds, _ = _prepare_dataset(
        cot_path, tokenizer, "cot", scfg["max_length"],
        seed=seed, sampling_temperature=temp,
        profiles=profiles, retrieved=retrieved,
    )

    base_model = _load_base_model(model_name, tcfg["bf16"])
    model = PeftModel.from_pretrained(base_model, base_checkpoint, is_trainable=True)
    t, tot = model.get_nb_trainable_parameters()
    logger.info(f"LoRA (resumed): {t:,} / {tot:,} trainable")

    lr_ratio = tcfg.get("sequential_lr_ratio", 0.25)
    seq_epochs = tcfg.get("sequential_epochs", 2)
    seq_lr = tcfg["learning_rate"] * lr_ratio

    ckpt_dir = _get_checkpoint_dir(cfg, f"C_seq_s{seed}")
    args = _build_training_args(cfg, ckpt_dir,
                                num_epochs=seq_epochs, learning_rate=seq_lr, seed=seed)

    final = _run_training(model, tokenizer, train_ds, eval_ds, args, ckpt_dir, logger)
    if _ddp_cleanup(logger):
        return None
    return final


def train_compact(cfg: dict, base_checkpoint: str = None,
                  seed: int = None, cls_weight: float = 5.0,
                  cot_max_words: int = 100) -> str:
    """Condition C_compact: Sequential + truncated CoT + weighted loss (ABLATION)."""
    seed = seed or cfg["project"]["seed"]
    logger = setup_logging("train_C_compact")
    set_seed(seed)
    ensure_dirs(cfg)

    scfg = cfg["student"]
    tcfg = scfg["training"]
    model_name = scfg["model_name"]

    if base_checkpoint is None:
        b_ckpt_dir = _get_checkpoint_dir(cfg, f"B_label_s{seed}")
        b_final = os.path.join(b_ckpt_dir, "final")
        if not os.path.exists(b_final):
            b_final = train_label_only(cfg, seed=seed,
                                       _continues_to_next_phase=True)
        base_checkpoint = b_final

    logger.info(f"=== Condition C_compact: Truncated CoT + weighted | seed={seed} ===")

    profiles, retrieved = _load_shared_resources(cfg)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    cot_path = os.path.join(cfg["data"]["processed_dir"], "train_cot.jsonl")
    temp = tcfg.get("sampling_temperature", 0.25)
    train_ds, eval_ds, _ = _prepare_dataset(
        cot_path, tokenizer, "compact", scfg["max_length"],
        seed=seed, sampling_temperature=temp,
        min_per_class=tcfg.get("min_per_class", 100),
        cot_max_words=cot_max_words,
        profiles=profiles, retrieved=retrieved,
    )

    base_model = _load_base_model(model_name, tcfg["bf16"])
    model = PeftModel.from_pretrained(base_model, base_checkpoint, is_trainable=True)

    lr_ratio = tcfg.get("sequential_lr_ratio", 0.25)
    seq_epochs = tcfg.get("sequential_epochs", 2)
    seq_lr = tcfg["learning_rate"] * lr_ratio

    ckpt_dir = _get_checkpoint_dir(cfg, f"C_compact{cot_max_words}_s{seed}")
    args = _build_training_args(cfg, ckpt_dir,
                                num_epochs=seq_epochs, learning_rate=seq_lr, seed=seed)

    final = _run_training(model, tokenizer, train_ds, eval_ds, args, ckpt_dir, logger,
                          cls_weight=cls_weight)
    if _ddp_cleanup(logger):
        return None
    return final


def train_summary(cfg: dict, base_checkpoint: str = None,
                  seed: int = None, cls_weight: float = 5.0,
                  kat_alpha: float = 0.0, sev_gamma: float = 1.0,
                  severity_oversample: bool = False) -> str:
    """Condition C_summary: Sequential + teacher summary + weighted loss.

    When kat_alpha > 0, uses Knowledge-Anchored Token weighting (KAT):
    per-example entity tokens from DrugBank profiles receive dynamic weights
    with inverse-length scaling to counteract gradient dilution.

    When sev_gamma > 1, severity output tokens receive additional weighting.
    When severity_oversample=True, known-severity examples are oversampled.
    """
    seed = seed or cfg["project"]["seed"]
    logger = setup_logging("train_C_summary")
    set_seed(seed)
    ensure_dirs(cfg)

    scfg = cfg["student"]
    tcfg = scfg["training"]
    model_name = scfg["model_name"]

    if base_checkpoint is None:
        b_ckpt_dir = _get_checkpoint_dir(cfg, f"B_label_s{seed}")
        b_final = os.path.join(b_ckpt_dir, "final")
        if not os.path.exists(b_final):
            logger.info("Phase 1: Training label-only (B) first...")
            b_final = train_label_only(cfg, seed=seed,
                                       _continues_to_next_phase=True)
        else:
            logger.info(f"Phase 1: Using existing B checkpoint: {b_final}")
        base_checkpoint = b_final

    logger.info(f"=== Condition C_summary: Summary distillation | seed={seed} ===")
    logger.info(f"Classification loss weight: {cls_weight}x")
    if kat_alpha > 0:
        logger.info(f"KAT enabled: alpha={kat_alpha}")
    if sev_gamma > 1.0:
        logger.info(f"Severity weighting: gamma={sev_gamma}")

    profiles, retrieved = _load_shared_resources(cfg)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    cot_path = os.path.join(cfg["data"]["processed_dir"], "train_cot.jsonl")
    if not os.path.exists(cot_path):
        logger.error(f"CoT data not found: {cot_path}")
        return None

    temp = tcfg.get("summary_sampling_temperature", 0.25)
    train_ds, eval_ds, entity_lookup = _prepare_dataset(
        cot_path, tokenizer, "summary", scfg["max_length"],
        seed=seed, sampling_temperature=temp,
        min_per_class=tcfg.get("min_per_class", 100),
        profiles=profiles, retrieved=retrieved,
        severity_oversample=severity_oversample,
        compute_entity_lookup=(kat_alpha > 0),
    )

    base_model = _load_base_model(model_name, tcfg["bf16"])
    model = PeftModel.from_pretrained(base_model, base_checkpoint, is_trainable=True)
    t, tot = model.get_nb_trainable_parameters()
    logger.info(f"LoRA (resumed): {t:,} / {tot:,} trainable")

    lr_ratio = tcfg.get("sequential_lr_ratio", 0.25)
    seq_epochs = tcfg.get("sequential_epochs", 2)
    seq_lr = tcfg["learning_rate"] * lr_ratio
    logger.info(f"Sequential LR: {seq_lr:.2e}, epochs: {seq_epochs}")

    cond_name = "C_summary"
    if kat_alpha > 0 and sev_gamma > 1.0:
        cond_name = "C_summary_KAT_sev"
    elif kat_alpha > 0:
        cond_name = "C_summary_KAT"
    ckpt_dir = _get_checkpoint_dir(cfg, f"{cond_name}_s{seed}")
    args = _build_training_args(cfg, ckpt_dir,
                                num_epochs=seq_epochs, learning_rate=seq_lr, seed=seed)

    final = _run_training(model, tokenizer, train_ds, eval_ds, args, ckpt_dir, logger,
                          cls_weight=cls_weight,
                          entity_lookup=entity_lookup if kat_alpha > 0 else None,
                          kat_alpha=kat_alpha, sev_gamma=sev_gamma)
    if _ddp_cleanup(logger):
        return None
    return final


# ── CLI entrypoint ────────────────────────────────────────────────────

MODES = {
    "label": train_label_only,
    "cot_naive": train_cot_naive,
    "sequential": train_sequential,
    "compact": train_compact,
    "summary": train_summary,
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Student fine-tuning")
    parser.add_argument("--mode", choices=list(MODES.keys()), required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--base-checkpoint", type=str, default=None)
    parser.add_argument("--cls-weight", type=float, default=5.0)
    parser.add_argument("--cot-max-words", type=int, default=100)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Override student model (for multi-scale experiments)")
    parser.add_argument("--kat-alpha", type=float, default=0.0,
                        help="KAT entity weight alpha (0 = disabled, 2.0 = default)")
    parser.add_argument("--sev-gamma", type=float, default=1.0,
                        help="Severity token weight gamma (1.0 = no boost)")
    parser.add_argument("--severity-oversample", action="store_true",
                        help="Oversample known-severity examples to 30%% of data")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.model_name:
        cfg["student"]["model_name"] = args.model_name
        name_lower = args.model_name.lower()
        is_small = any(s in name_lower for s in ["0.6b", "1.7b", "0.5b", "1b", "1.5b", "2b"])
        is_medium = any(s in name_lower for s in ["3b", "4b"])
        if is_small:
            cfg["student"]["training"]["fsdp"]["enabled"] = False
            cfg["student"]["training"]["per_device_batch_size"] = 4
            cfg["student"]["training"]["gradient_accumulation_steps"] = 4
        elif is_medium:
            cfg["student"]["training"]["fsdp"]["enabled"] = False
            cfg["student"]["training"]["per_device_batch_size"] = 2
            cfg["student"]["training"]["gradient_accumulation_steps"] = 8

    kwargs = {"seed": args.seed}
    if args.mode in ("sequential", "compact", "summary"):
        kwargs["base_checkpoint"] = args.base_checkpoint
    if args.mode in ("compact", "summary"):
        kwargs["cls_weight"] = args.cls_weight
    if args.mode == "compact":
        kwargs["cot_max_words"] = args.cot_max_words
    if args.mode == "summary":
        kwargs["kat_alpha"] = args.kat_alpha
        kwargs["sev_gamma"] = args.sev_gamma
        kwargs["severity_oversample"] = args.severity_oversample
    MODES[args.mode](cfg, **kwargs)
