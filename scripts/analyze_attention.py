"""Post-hoc attention pattern analysis: compare entity-token attention
in KAT vs non-KAT student models.

Connects to MoLSAKI (EMNLP 2025) insight that attention to critical tokens
matters for distillation quality, but achieves this via loss design rather
than attention transfer (no white-box teacher required).

Produces publication figures comparing:
  1. Mean attention to entity tokens vs non-entity tokens
  2. Layer-wise attention concentration on entities
  3. Attention entropy (lower = more focused)
"""

import os
import re
import json
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_model_and_tokenizer(model_name: str, checkpoint_dir: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager", output_attentions=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer


def extract_entity_positions(input_ids: list, entity_token_ids: set) -> list:
    return [i for i, tid in enumerate(input_ids) if tid in entity_token_ids]


def get_attention_stats(model, tokenizer, prompt: str, entity_token_ids: set,
                        max_length: int = 4096):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=max_length)
    except Exception:
        return None

    input_ids = inputs["input_ids"].squeeze().tolist()
    entity_positions = extract_entity_positions(input_ids, entity_token_ids)

    if len(entity_positions) < 2:
        return None

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions
    n_layers = len(attentions)

    entity_set = set(entity_positions)
    non_entity_positions = [i for i in range(len(input_ids)) if i not in entity_set]

    layer_stats = []
    for layer_idx, layer_attn in enumerate(attentions):
        attn = layer_attn.squeeze(0).float()
        avg_attn = attn.mean(dim=0)

        entity_attn = avg_attn[:, entity_positions].sum(dim=-1).mean().item()
        if non_entity_positions:
            non_entity_attn = avg_attn[:, non_entity_positions].sum(dim=-1).mean().item()
        else:
            non_entity_attn = 0.0

        n_ent = len(entity_positions)
        n_non = len(non_entity_positions)
        entity_attn_norm = entity_attn / n_ent if n_ent > 0 else 0
        non_entity_attn_norm = non_entity_attn / n_non if n_non > 0 else 0

        attn_probs = avg_attn.clamp(min=1e-10)
        entropy = -(attn_probs * attn_probs.log()).sum(dim=-1).mean().item()

        layer_stats.append({
            "layer": layer_idx,
            "entity_attn": entity_attn_norm,
            "non_entity_attn": non_entity_attn_norm,
            "entity_attn_ratio": (entity_attn_norm / (non_entity_attn_norm + 1e-10)),
            "entropy": entropy,
        })

    return {
        "n_layers": n_layers,
        "n_entity_tokens": len(entity_positions),
        "n_total_tokens": len(input_ids),
        "entity_ratio": len(entity_positions) / len(input_ids),
        "layer_stats": layer_stats,
    }


def build_entity_ids_for_pair(drug1_id, drug2_id, profiles, tokenizer):
    entity_strings = set()
    for did in [drug1_id, drug2_id]:
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


def analyze_model(model, tokenizer, test_df, profiles, n_samples=200, seed=42):
    from src.data_preparation import SYSTEM_PROMPT, build_student_input

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(test_df), size=min(n_samples, len(test_df)), replace=False)

    all_stats = []
    for idx in indices:
        row = test_df.iloc[idx]
        d1_id = str(row.get("drug1_id", ""))
        d2_id = str(row.get("drug2_id", ""))

        entity_ids = build_entity_ids_for_pair(d1_id, d2_id, profiles, tokenizer)
        if len(entity_ids) < 3:
            continue

        user_msg = build_student_input(row, profiles, None)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

        stats = get_attention_stats(model, tokenizer, prompt, entity_ids)
        if stats:
            all_stats.append(stats)

        if len(all_stats) % 20 == 0 and len(all_stats) > 0:
            print(f"  Processed {len(all_stats)} samples...")

    return all_stats


def plot_comparison(kat_stats, baseline_stats, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    def avg_layer_metric(stats_list, metric):
        n_layers = stats_list[0]["n_layers"]
        result = np.zeros(n_layers)
        for s in stats_list:
            for ls in s["layer_stats"]:
                result[ls["layer"]] += ls[metric]
        return result / len(stats_list)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    kat_ratio = avg_layer_metric(kat_stats, "entity_attn_ratio")
    base_ratio = avg_layer_metric(baseline_stats, "entity_attn_ratio")
    layers = range(len(kat_ratio))
    axes[0].plot(layers, kat_ratio, "o-", label="KAT", color="#e74c3c", markersize=3)
    axes[0].plot(layers, base_ratio, "s-", label="Baseline", color="#3498db", markersize=3)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Entity / Non-Entity Attention Ratio")
    axes[0].set_title("Entity Attention Concentration by Layer")
    axes[0].legend()
    axes[0].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    kat_ent = avg_layer_metric(kat_stats, "entity_attn")
    base_ent = avg_layer_metric(baseline_stats, "entity_attn")
    axes[1].plot(layers, kat_ent, "o-", label="KAT entity", color="#e74c3c", markersize=3)
    axes[1].plot(layers, base_ent, "s-", label="Baseline entity", color="#3498db", markersize=3)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Mean Attention per Entity Token")
    axes[1].set_title("Entity Token Attention by Layer")
    axes[1].legend()

    kat_entropy = avg_layer_metric(kat_stats, "entropy")
    base_entropy = avg_layer_metric(baseline_stats, "entropy")
    axes[2].plot(layers, kat_entropy, "o-", label="KAT", color="#e74c3c", markersize=3)
    axes[2].plot(layers, base_entropy, "s-", label="Baseline", color="#3498db", markersize=3)
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Attention Entropy")
    axes[2].set_title("Attention Entropy by Layer (lower = more focused)")
    axes[2].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "attention_kat_vs_baseline.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {path}")

    kat_mean_ratio = np.mean([s["layer_stats"][-1]["entity_attn_ratio"]
                              for s in kat_stats])
    base_mean_ratio = np.mean([s["layer_stats"][-1]["entity_attn_ratio"]
                               for s in baseline_stats])
    print(f"\nLast-layer entity attention ratio:")
    print(f"  KAT:      {kat_mean_ratio:.4f}")
    print(f"  Baseline: {base_mean_ratio:.4f}")
    print(f"  Delta:    {kat_mean_ratio - base_mean_ratio:+.4f}")


def main():
    parser = argparse.ArgumentParser(description="Attention pattern analysis")
    parser.add_argument("--model-name", required=True, help="Base model name")
    parser.add_argument("--kat-checkpoint", required=True, help="KAT student checkpoint")
    parser.add_argument("--baseline-checkpoint", required=True, help="Baseline student checkpoint")
    parser.add_argument("--test-data", required=True, help="Path to test.jsonl")
    parser.add_argument("--profiles", required=True, help="Path to drug_profiles.json")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.profiles) as f:
        profiles = json.load(f)
    test_df = pd.read_json(args.test_data, lines=True)

    print("Loading KAT model...")
    kat_model, tokenizer = load_model_and_tokenizer(args.model_name, args.kat_checkpoint)
    print(f"Analyzing KAT attention ({args.n_samples} samples)...")
    kat_stats = analyze_model(kat_model, tokenizer, test_df, profiles,
                              n_samples=args.n_samples, seed=args.seed)
    del kat_model
    torch.cuda.empty_cache()

    print("Loading baseline model...")
    base_model, tokenizer = load_model_and_tokenizer(args.model_name, args.baseline_checkpoint)
    print(f"Analyzing baseline attention ({args.n_samples} samples)...")
    base_stats = analyze_model(base_model, tokenizer, test_df, profiles,
                               n_samples=args.n_samples, seed=args.seed)
    del base_model
    torch.cuda.empty_cache()

    if kat_stats and base_stats:
        plot_comparison(kat_stats, base_stats, args.output_dir)

        stats_out = {
            "kat_n_samples": len(kat_stats),
            "baseline_n_samples": len(base_stats),
            "kat_mean_entity_ratio": round(np.mean([
                s["entity_ratio"] for s in kat_stats]), 4),
            "baseline_mean_entity_ratio": round(np.mean([
                s["entity_ratio"] for s in base_stats]), 4),
        }
        stats_path = os.path.join(args.output_dir, "attention_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats_out, f, indent=2)
        print(f"Stats saved: {stats_path}")
    else:
        print("Insufficient samples for comparison.")


if __name__ == "__main__":
    main()
