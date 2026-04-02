"""Hedging analysis: correlate hedging rate with profile completeness, severity, class frequency.

Produces publication-quality figures and statistics for EMNLP paper.
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from pathlib import Path

HEDGING_PATTERNS = [
    re.compile(r"\b(may|might|could|possibly|potentially|likely|unlikely)\b", re.IGNORECASE),
    re.compile(r"\b(it is (possible|unclear|uncertain|not clear))\b", re.IGNORECASE),
    re.compile(r"\b(further (research|investigation|studies))\b", re.IGNORECASE),
    re.compile(r"\b(not (well|fully) (established|understood|characterized))\b", re.IGNORECASE),
    re.compile(r"\b(limited (evidence|data|information))\b", re.IGNORECASE),
    re.compile(r"\b(the exact mechanism is)\b", re.IGNORECASE),
    re.compile(r"\b(mechanism .{0,20} not (fully |well )?known)\b", re.IGNORECASE),
]

VAGUE_PATTERNS = [
    re.compile(r"\b(various|several|multiple|certain|some)\s+(mechanisms?|pathways?|factors?)\b", re.IGNORECASE),
    re.compile(r"\b(complex interaction|multifactorial)\b", re.IGNORECASE),
]


def count_hedging(text: str) -> dict:
    hedge_count = sum(len(p.findall(text)) for p in HEDGING_PATTERNS)
    vague_count = sum(len(p.findall(text)) for p in VAGUE_PATTERNS)
    return {"hedge_count": hedge_count, "vague_count": vague_count,
            "is_hedging": hedge_count > 0, "is_vague": vague_count > 0}


def compute_profile_completeness(drug_id: str, profiles: dict) -> dict:
    prof = profiles.get(drug_id, {})
    n_enzymes = len(prof.get("enzymes", []))
    n_transporters = len(prof.get("transporters", []))
    n_targets = len(prof.get("targets", []))
    total = n_enzymes + n_transporters + n_targets
    tier = "rich" if total >= 4 else ("sparse" if total >= 1 else "empty")
    return {"n_enzymes": n_enzymes, "n_transporters": n_transporters,
            "n_targets": n_targets, "total_fields": total, "tier": tier}


def analyze_predictions(pred_file: str, profiles: dict, label_freq: dict = None):
    rows = []
    with open(pred_file) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            output = obj.get("output", "")
            if not output:
                continue

            hedge = count_hedging(output)

            d1_comp = compute_profile_completeness(obj.get("drug1_id", ""), profiles)
            d2_comp = compute_profile_completeness(obj.get("drug2_id", ""), profiles)
            pair_total = d1_comp["total_fields"] + d2_comp["total_fields"]
            pair_tier = "rich" if pair_total >= 6 else ("sparse" if pair_total >= 1 else "empty")

            severity = obj.get("severity", "Unknown")
            label = obj.get("label", -1)
            freq = label_freq.get(str(label), 0) if label_freq else 0

            rows.append({
                **hedge,
                "pair_profile_total": pair_total,
                "pair_tier": pair_tier,
                "severity": severity,
                "label": label,
                "label_freq": freq,
                "pred_correct": int(obj.get("pred_label", -1) == label),
                "output_len": len(output.split()),
            })

    return pd.DataFrame(rows)


def plot_hedging_by_profile(df: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    tiers = ["empty", "sparse", "rich"]
    tier_rates = [df[df["pair_tier"] == t]["is_hedging"].mean() * 100 for t in tiers]
    colors = ["#e74c3c", "#f39c12", "#27ae60"]
    axes[0].bar(tiers, tier_rates, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Hedging Rate (%)")
    axes[0].set_xlabel("Profile Completeness Tier")
    axes[0].set_title("Hedging vs Profile Completeness")
    for i, v in enumerate(tier_rates):
        axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)

    sev_order = ["Major", "Moderate", "Minor", "Unknown"]
    sev_rates = []
    sev_counts = []
    for s in sev_order:
        sub = df[df["severity"] == s]
        sev_rates.append(sub["is_hedging"].mean() * 100 if len(sub) > 0 else 0)
        sev_counts.append(len(sub))
    sev_colors = ["#c0392b", "#e67e22", "#2ecc71", "#95a5a6"]
    axes[1].bar(sev_order, sev_rates, color=sev_colors, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Hedging Rate (%)")
    axes[1].set_xlabel("Severity Class")
    axes[1].set_title("Hedging vs Severity")
    for i, (v, n) in enumerate(zip(sev_rates, sev_counts)):
        axes[1].text(i, v + 0.5, f"{v:.1f}%\n(n={n})", ha="center", fontsize=9)

    bins = np.linspace(0, df["label_freq"].max(), 10)
    bin_labels = []
    bin_rates = []
    for i in range(len(bins) - 1):
        sub = df[(df["label_freq"] >= bins[i]) & (df["label_freq"] < bins[i + 1])]
        if len(sub) >= 10:
            bin_labels.append(f"{int(bins[i])}-{int(bins[i+1])}")
            bin_rates.append(sub["is_hedging"].mean() * 100)
    if bin_labels:
        axes[2].bar(range(len(bin_labels)), bin_rates, color="#3498db",
                     edgecolor="black", linewidth=0.5)
        axes[2].set_xticks(range(len(bin_labels)))
        axes[2].set_xticklabels(bin_labels, rotation=45, ha="right")
        axes[2].set_ylabel("Hedging Rate (%)")
        axes[2].set_xlabel("Label Frequency (train count)")
        axes[2].set_title("Hedging vs Class Frequency")

    plt.tight_layout()
    path = os.path.join(output_dir, "hedging_analysis.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_hedging_accuracy_interaction(df: pd.DataFrame, output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))

    for tier, color in [("empty", "#e74c3c"), ("sparse", "#f39c12"), ("rich", "#27ae60")]:
        sub = df[df["pair_tier"] == tier]
        hedging_sub = sub[sub["is_hedging"]]
        no_hedge_sub = sub[~sub["is_hedging"]]
        hedge_acc = hedging_sub["pred_correct"].mean() * 100 if len(hedging_sub) > 0 else 0
        no_hedge_acc = no_hedge_sub["pred_correct"].mean() * 100 if len(no_hedge_sub) > 0 else 0
        ax.bar([f"{tier}\nhedging", f"{tier}\nno hedge"],
               [hedge_acc, no_hedge_acc],
               color=[color]*2, alpha=[0.5, 1.0],
               edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_title("Hedging Impact on Accuracy by Profile Completeness")
    plt.tight_layout()
    path = os.path.join(output_dir, "hedging_accuracy_interaction.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Hedging analysis for DDI CoT predictions")
    parser.add_argument("--pred-file", required=True, help="Path to predictions JSONL")
    parser.add_argument("--profiles", required=True, help="Path to drug_profiles.json")
    parser.add_argument("--train-data", default=None, help="Path to train.jsonl for class freq")
    parser.add_argument("--output-dir", required=True, help="Output directory for figures")
    parser.add_argument("--condition", default="student", help="Condition name for labeling")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.profiles) as f:
        profiles = json.load(f)

    label_freq = {}
    if args.train_data and os.path.exists(args.train_data):
        train_df = pd.read_json(args.train_data, lines=True)
        label_freq = dict(train_df["label"].value_counts())

    df = analyze_predictions(args.pred_file, profiles, label_freq)

    print(f"\n{'='*60}")
    print(f"Hedging Analysis: {args.condition}")
    print(f"{'='*60}")
    print(f"Total predictions: {len(df):,}")
    print(f"Hedging rate:      {df['is_hedging'].mean()*100:.1f}%")
    print(f"Vague mechanism:   {df['is_vague'].mean()*100:.1f}%")
    print(f"Avg hedge count:   {df['hedge_count'].mean():.2f}")

    print(f"\nBy profile completeness:")
    for tier in ["empty", "sparse", "rich"]:
        sub = df[df["pair_tier"] == tier]
        print(f"  {tier:>6}: {sub['is_hedging'].mean()*100:5.1f}% hedging  "
              f"(n={len(sub):,}, accuracy={sub['pred_correct'].mean()*100:.1f}%)")

    print(f"\nBy severity:")
    for sev in ["Major", "Moderate", "Minor", "Unknown"]:
        sub = df[df["severity"] == sev]
        if len(sub) > 0:
            print(f"  {sev:>10}: {sub['is_hedging'].mean()*100:5.1f}% hedging  "
                  f"(n={len(sub):,}, accuracy={sub['pred_correct'].mean()*100:.1f}%)")

    fig_path = plot_hedging_by_profile(df, args.output_dir)
    print(f"\nFigure saved: {fig_path}")
    fig_path2 = plot_hedging_accuracy_interaction(df, args.output_dir)
    print(f"Figure saved: {fig_path2}")

    stats = {
        "condition": args.condition,
        "n_total": len(df),
        "hedging_rate": round(df["is_hedging"].mean(), 4),
        "vague_rate": round(df["is_vague"].mean(), 4),
        "avg_hedge_count": round(df["hedge_count"].mean(), 2),
        "by_tier": {},
        "by_severity": {},
    }
    for tier in ["empty", "sparse", "rich"]:
        sub = df[df["pair_tier"] == tier]
        stats["by_tier"][tier] = {
            "n": len(sub),
            "hedging_rate": round(sub["is_hedging"].mean(), 4) if len(sub) else 0,
            "accuracy": round(sub["pred_correct"].mean(), 4) if len(sub) else 0,
        }
    for sev in ["Major", "Moderate", "Minor", "Unknown"]:
        sub = df[df["severity"] == sev]
        if len(sub) > 0:
            stats["by_severity"][sev] = {
                "n": len(sub),
                "hedging_rate": round(sub["is_hedging"].mean(), 4),
                "accuracy": round(sub["pred_correct"].mean(), 4),
            }

    stats_path = os.path.join(args.output_dir, f"hedging_stats_{args.condition}.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved: {stats_path}")


if __name__ == "__main__":
    main()
