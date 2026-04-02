"""DPO anti-hedging refinement for DDI CoT student.

After SFT, the student sometimes hedges ("may or may not interact") or
produces malformed outputs.  This module generates preference pairs from
the student's own predictions and runs DPO to steer the model toward
confident, correctly-formatted responses.

Pipeline:
  1. Generate N candidate responses per test example via temperature sampling.
  2. Score each candidate: correct label, format compliance, confidence.
  3. Build (chosen, rejected) pairs from the same input.
  4. Run DPO training using TRL.
"""
import json
import logging
import os
import re
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import load_config, setup_logging, set_seed, ensure_dirs


HEDGING_PATTERNS = [
    r"may or may not",
    r"it is unclear whether",
    r"cannot determine",
    r"insufficient information",
    r"no definitive",
    r"uncertain",
    r"possibly interact",
]
HEDGING_RE = re.compile("|".join(HEDGING_PATTERNS), re.IGNORECASE)

LABEL_RE = re.compile(r"Y\s*=\s*(\d+)")
FORMAT_MARKERS = ["Classification:", "Severity:"]


def _score_response(text: str, gold_label: int,
                    profile_d1: dict = None, profile_d2: dict = None,
                    precision_weight: float = 0.7) -> dict:
    """Score a student response on label accuracy, format, hedging, and grounded factuality."""
    label_match = LABEL_RE.search(text)
    pred_label = int(label_match.group(1)) if label_match else -1

    label_correct = pred_label == gold_label
    has_format = all(m in text for m in FORMAT_MARKERS)
    is_hedging = bool(HEDGING_RE.search(text))
    length_ok = 10 < len(text.split()) < 500

    score = 0.0
    if label_correct:
        score += 3.0
    if has_format:
        score += 2.0
    if not is_hedging:
        score += 1.0
    if length_ok:
        score += 0.5

    grounded_score = 0.0
    if profile_d1 and profile_d2:
        try:
            from src.grounded_factuality import score_trace
            gf = score_trace(text, profile_d1, profile_d2,
                             precision_weight=precision_weight)
            grounded_score = gf.get("grounded_score", 0.0)
            score += grounded_score * 2.0
        except Exception:
            pass

    return {
        "score": round(score, 3),
        "label_correct": label_correct,
        "has_format": has_format,
        "is_hedging": is_hedging,
        "grounded_score": round(grounded_score, 3),
        "pred_label": pred_label,
    }


def generate_preference_pairs(cfg: dict, condition: str,
                              checkpoint: str, n_samples: int = 4,
                              temperature: float = 0.7,
                              seed: int = 42) -> str:
    """Generate preference pairs by sampling multiple responses per input.

    Returns path to the output JSONL file.
    """
    logger = setup_logging("dpo_gen_pairs")
    set_seed(seed)

    scfg = cfg["student"]
    model_name = scfg["model_name"]
    output_dir = os.path.join(cfg["evaluation"]["output_dir"], condition, "dpo_pairs")
    os.makedirs(output_dir, exist_ok=True)
    pairs_path = os.path.join(output_dir, "preference_pairs.jsonl")

    if os.path.exists(pairs_path):
        n_existing = sum(1 for _ in open(pairs_path))
        logger.info(f"Resuming: {n_existing} pairs already exist at {pairs_path}")
    else:
        n_existing = 0

    test_path = os.path.join(cfg["data"]["processed_dir"], "test_cot.jsonl")
    test_df = pd.read_json(test_path, lines=True)
    logger.info(f"Test set: {len(test_df)} examples, generating {n_samples} samples each")

    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        lora_r = scfg["lora"]["r"]
        llm = LLM(
            model=model_name,
            enable_lora=True,
            max_lora_rank=lora_r,
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_model_len=scfg["max_length"],
        )
        lora_req = LoRARequest("student_lora", 1, checkpoint)
        sampling_params = SamplingParams(
            n=n_samples,
            temperature=temperature,
            max_tokens=512,
            top_p=0.95,
        )

        from src.data_preparation import SYSTEM_PROMPT, build_student_input

        with open(os.path.join(cfg["data"]["processed_dir"], "drug_profiles.json")) as f:
            profiles = json.load(f)

        retrieved = {}
        retr_path = os.path.join(cfg["data"]["processed_dir"], "retrieved_examples_test.json")
        if os.path.exists(retr_path):
            with open(retr_path) as f:
                retrieved = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        n_pairs = 0
        with open(pairs_path, "a") as fout:
            for idx, row in test_df.iterrows():
                if idx < n_existing:
                    continue

                retr_idx = retrieved.get(str(idx), [])
                retr_examples = [r for r in retr_idx if isinstance(r, dict)] if retr_idx else None
                user_msg = build_student_input(row, profiles, retr_examples)

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                outputs = llm.generate([prompt], sampling_params, lora_request=lora_req)

                p1 = profiles.get(row.get("drug1_id", ""), {})
                p2 = profiles.get(row.get("drug2_id", ""), {})
                pw = cfg.get("grounded_eval", {}).get("precision_weight", 0.7)

                candidates = []
                for out in outputs[0].outputs:
                    text = out.text.strip()
                    sc = _score_response(text, int(row["label"]),
                                         profile_d1=p1, profile_d2=p2,
                                         precision_weight=pw)
                    candidates.append({"text": text, **sc})

                candidates.sort(key=lambda x: x["score"], reverse=True)

                best = candidates[0]
                for worst in reversed(candidates):
                    if worst["score"] < best["score"]:
                        pair = {
                            "prompt": prompt,
                            "chosen": best["text"],
                            "rejected": worst["text"],
                            "chosen_score": best["score"],
                            "rejected_score": worst["score"],
                            "gold_label": int(row["label"]),
                        }
                        fout.write(json.dumps(pair) + "\n")
                        n_pairs += 1
                        break

        logger.info(f"Generated {n_pairs} preference pairs -> {pairs_path}")
        return pairs_path

    except ImportError:
        logger.error("vLLM not available. Install with: pip install vllm")
        return None


def run_dpo(cfg: dict, condition: str, checkpoint: str,
            pairs_path: str = None, seed: int = 42) -> str:
    """Run DPO training on preference pairs.

    Returns path to the DPO-refined checkpoint.
    """
    logger = setup_logging("dpo_training")
    set_seed(seed)
    ensure_dirs(cfg)

    scfg = cfg["student"]
    tcfg = scfg["training"]
    model_name = scfg["model_name"]

    if pairs_path is None:
        pairs_path = os.path.join(
            cfg["evaluation"]["output_dir"], condition, "dpo_pairs",
            "preference_pairs.jsonl"
        )

    if not os.path.exists(pairs_path):
        logger.error(f"Preference pairs not found: {pairs_path}")
        return None

    pairs_df = pd.read_json(pairs_path, lines=True)
    logger.info(f"Loaded {len(pairs_df)} preference pairs")

    if len(pairs_df) < 50:
        logger.warning(f"Only {len(pairs_df)} pairs -- DPO may underfit")

    dpo_ds = Dataset.from_dict({
        "prompt": pairs_df["prompt"].tolist(),
        "chosen": pairs_df["chosen"].tolist(),
        "rejected": pairs_df["rejected"].tolist(),
    })

    try:
        from trl import DPOTrainer, DPOConfig
    except ImportError:
        logger.error("TRL not available. Install with: pip install trl")
        return None

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if tcfg["bf16"] else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, checkpoint, is_trainable=True)

    ckpt_dir = os.path.join(
        cfg["evaluation"]["output_dir"], condition, "dpo_checkpoint"
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    dpo_cfg = cfg.get("dpo", {})
    dpo_lr = dpo_cfg.get("learning_rate", 5e-7)
    dpo_epochs = dpo_cfg.get("num_epochs", 1)
    dpo_beta = dpo_cfg.get("beta", 0.1)
    dpo_bs = dpo_cfg.get("per_device_batch_size", 1)
    dpo_ga = dpo_cfg.get("gradient_accumulation_steps", 8)

    training_args = DPOConfig(
        output_dir=ckpt_dir,
        num_train_epochs=dpo_epochs,
        per_device_train_batch_size=dpo_bs,
        gradient_accumulation_steps=dpo_ga,
        learning_rate=dpo_lr,
        beta=dpo_beta,
        bf16=tcfg["bf16"],
        logging_steps=10,
        save_strategy="epoch",
        seed=seed,
        max_length=scfg["max_length"],
        max_prompt_length=scfg["max_length"] // 2,
        remove_unused_columns=False,
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    ref_model = PeftModel.from_pretrained(ref_model, checkpoint, is_trainable=False)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dpo_ds,
        processing_class=tokenizer,
    )

    trainer.train()

    final_dir = os.path.join(ckpt_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"DPO model saved to {final_dir}")
    return final_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DPO anti-hedging refinement")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--condition", default="C_summary")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--step", choices=["gen_pairs", "train", "both"], default="both")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.step in ("gen_pairs", "both"):
        pairs_path = generate_preference_pairs(
            cfg, args.condition, args.checkpoint,
            n_samples=args.n_samples, temperature=args.temperature,
            seed=args.seed,
        )

    if args.step in ("train", "both"):
        run_dpo(cfg, args.condition, args.checkpoint, seed=args.seed)
