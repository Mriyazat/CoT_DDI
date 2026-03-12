#!/usr/bin/env python3
"""BERT-family baselines for DDI classification.
Usage:
  python experiments/baselines/run_bert_baseline.py --model pubmedbert
  python experiments/baselines/run_bert_baseline.py --model pubmedbert --eval-only
"""

import argparse, os, sys, json
import numpy as np, pandas as pd, torch
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from experiments.exp_utils import (
    get_config, get_exp_output_dir, setup_exp_logging,
    compute_classification_metrics, per_category_f1,
    save_results, load_label_map,
)

MODEL_MAP = {
    "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "biobert": "dmis-lab/biobert-v1.1",
    "scibert": "allenai/scibert_scivocab_uncased",
}
NUM_LABELS = 86
MAX_SEQ_LEN = 512


def _format_input(row) -> str:
    return (f"Drug 1: {row['drug1_name']} ({row['drug1_id']}). "
            f"SMILES: {str(row['drug1_smiles'])[:150]}. "
            f"Drug 2: {row['drug2_name']} ({row['drug2_id']}). "
            f"SMILES: {str(row['drug2_smiles'])[:150]}.")


def _load_and_tokenize(tokenizer, split, cfg):
    path = os.path.join(cfg["data"]["processed_dir"], f"{split}.jsonl")
    df = pd.read_json(path, lines=True)
    texts = [_format_input(row) for _, row in df.iterrows()]
    labels = [int(row["label"]) - 1 for _, row in df.iterrows()]
    enc = tokenizer(texts, truncation=True, padding="max_length",
                    max_length=MAX_SEQ_LEN, return_tensors="np")
    ds = Dataset.from_dict({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    })
    return ds, df


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "accuracy": accuracy_score(labels, preds),
    }


def train_bert(model_key: str, num_epochs=10, batch_size=64, lr=2e-5, eval_only=False):
    model_name = MODEL_MAP[model_key]
    out_dir = get_exp_output_dir(f"bert_{model_key}")
    logger = setup_exp_logging(f"bert_{model_key}", out_dir)
    cfg = get_config()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    best_dir = out_dir / "best_model"

    if not eval_only:
        train_ds, _ = _load_and_tokenize(tokenizer, "train", cfg)
        rng = np.random.RandomState(cfg["project"]["seed"])
        indices = rng.permutation(len(train_ds))
        n_val = max(1, int(0.05 * len(train_ds)))
        val_ds = train_ds.select(indices[:n_val].tolist())
        train_ds = train_ds.select(indices[n_val:].tolist())
        logger.info(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}")

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=NUM_LABELS)

        training_args = TrainingArguments(
            output_dir=str(ckpt_dir), num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=lr, weight_decay=0.01, warmup_ratio=0.06,
            lr_scheduler_type="cosine", eval_strategy="epoch",
            save_strategy="epoch", save_total_limit=3,
            load_best_model_at_end=True, metric_for_best_model="macro_f1",
            greater_is_better=True, logging_steps=100,
            bf16=torch.cuda.is_available(), dataloader_num_workers=4,
            report_to="none",
        )

        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=train_ds, eval_dataset=val_ds,
            compute_metrics=_compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        last_ckpt = get_last_checkpoint(str(ckpt_dir))
        if last_ckpt:
            logger.info(f"Resuming from {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
        trainer.save_model(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))
    else:
        logger.info(f"Loading from {best_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(str(best_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(best_dir))

    test_ds, test_df = _load_and_tokenize(tokenizer, "test", cfg)
    eval_trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(out_dir / "eval_tmp"),
                               per_device_eval_batch_size=batch_size * 2,
                               bf16=torch.cuda.is_available(), report_to="none"),
    )
    outputs = eval_trainer.predict(test_ds)
    preds = np.argmax(outputs.predictions, axis=-1) + 1

    label_map = load_label_map()
    metrics = compute_classification_metrics(test_df["label"].tolist(), preds.tolist(), label_map)
    metrics["model"] = model_name
    logger.info(f"Macro F1={metrics['macro_f1']:.4f} | Micro F1={metrics['micro_f1']:.4f}")
    save_results(metrics, out_dir, f"bert_{model_key}_results.json")

    pred_records = [{"drug1_id": row["drug1_id"], "drug2_id": row["drug2_id"],
                     "true_label": int(row["label"]), "pred_label": int(preds[i])}
                    for i, (_, row) in enumerate(test_df.iterrows())]
    pred_path = out_dir / f"bert_{model_key}_predictions.jsonl"
    with open(pred_path, "w") as f:
        for r in pred_records:
            f.write(json.dumps(r) + "\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_MAP.keys()), required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    train_bert(args.model, args.epochs, args.batch_size, args.lr, args.eval_only)
