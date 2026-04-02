"""
Generate publication-quality figures for the PharmCoT paper.

Figures:
  1. Main comparison bar chart (Macro-F1 + Accuracy)
  2. Head / Mid / Tail class analysis
  3. Coarse category radar chart
  4. Model size vs performance scatter
  5. Confusion matrix heatmap (best model)
  6. Severity prediction breakdown
  7. Per-class F1 distribution (box plot)

Run after all baselines and student evaluations complete.
Reads from $OUTPUT_DIR/results/ and saves figures to $OUTPUT_DIR/figures/.
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import Counter

from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report,
)

from src.utils import load_config, setup_logging, ensure_dirs


plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "Random (uniform)": "#bdc3c7",
    "Majority class": "#95a5a6",
    "Stratified random": "#7f8c8d",
    "Morgan FP + XGBoost": "#3498db",
    "Morgan FP + RF": "#2980b9",
    "Morgan FP + MLP": "#1abc9c",
    "MACCS + XGBoost": "#9b59b6",
    "Pharma + XGBoost": "#e67e22",
    "FP+Pharma + XGBoost": "#e74c3c",
    "Zero-shot Qwen2.5-7B-Instruct": "#f39c12",
    "Zero-shot Llama-3.3-70B-Instruct": "#d35400",
    "Qwen2.5-7B label-only (B)": "#27ae60",
    "PharmCoT C_summary (ours)": "#c0392b",
}

OURS_COLOR = "#c0392b"
ACCENT = "#1a5276"


def load_all_results(res_dir):
    """Load all baseline results JSONs into a single list."""
    all_results = []
    for fname in ["baseline_results.json", "zeroshot_all_metrics.json",
                   "student_eval_results.json"]:
        path = os.path.join(res_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                all_results.extend(data)
            else:
                all_results.append(data)
    return all_results


# ── Figure 1: Main comparison bar chart ──────────────────────────────

def plot_main_comparison(results, fig_dir):
    """Side-by-side bars for Accuracy and Macro-F1 across all methods."""
    results = sorted(results, key=lambda r: r.get("macro_f1", 0))

    names = [r["model"] for r in results]
    acc = [r.get("accuracy", 0) * 100 for r in results]
    f1 = [r.get("macro_f1", 0) * 100 for r in results]

    fig, ax = plt.subplots(figsize=(12, max(5, len(names) * 0.45)))
    y = np.arange(len(names))
    h = 0.35

    bars1 = ax.barh(y - h/2, f1, h, label="Macro F1", color=ACCENT, alpha=0.85)
    bars2 = ax.barh(y + h/2, acc, h, label="Accuracy", color="#148f77", alpha=0.85)

    for bar, val in zip(bars1, f1):
        if val > 3:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}", va="center", fontsize=9)
    for bar, val in zip(bars2, acc):
        if val > 3:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Score (%)")
    ax.set_title("DDI Classification: All Methods Comparison")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.3)

    path = os.path.join(fig_dir, "main_comparison.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Figure 2: Head / Mid / Tail analysis ─────────────────────────────

def plot_head_mid_tail(predictions_path, train_path, fig_dir):
    """F1 breakdown by class frequency tier."""
    if not os.path.exists(predictions_path):
        return None

    with open(predictions_path, "rb") as f:
        data = pickle.load(f)
    y_test = data["y_test"]
    preds = data["predictions"]

    train_df = pd.read_json(train_path, lines=True)
    counts = Counter(train_df["label"])
    sorted_labels = sorted(counts.keys(), key=lambda l: counts[l], reverse=True)
    n = len(sorted_labels)
    head = set(sorted_labels[:n//3])
    mid = set(sorted_labels[n//3:2*n//3])
    tail = set(sorted_labels[2*n//3:])

    tiers = {"Head (top 33%)": head, "Mid (33-66%)": mid, "Tail (bottom 33%)": tail}

    methods = list(preds.keys())[:5]
    tier_names = list(tiers.keys())
    method_scores = {m: [] for m in methods}

    for tier_name, tier_labels in tiers.items():
        mask = np.isin(y_test, list(tier_labels))
        for m in methods:
            y_p = preds[m]
            if mask.sum() > 0:
                f1 = f1_score(y_test[mask], y_p[mask], average="macro", zero_division=0)
            else:
                f1 = 0
            method_scores[m].append(f1 * 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(tier_names))
    width = 0.8 / len(methods)

    for i, m in enumerate(methods):
        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, method_scores[m], width, label=m, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(tier_names)
    ax.set_ylabel("Macro F1 (%)")
    ax.set_title("Performance by Class Frequency Tier")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(fig_dir, "head_mid_tail.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Figure 3: Coarse category radar ──────────────────────────────────

def plot_category_radar(predictions_path, test_path, fig_dir):
    """Radar chart of per-category performance for top methods."""
    if not os.path.exists(predictions_path):
        return None

    with open(predictions_path, "rb") as f:
        data = pickle.load(f)
    y_test = data["y_test"]
    preds = data["predictions"]

    test_df = pd.read_json(test_path, lines=True)
    categories = test_df["coarse_category"].values

    cat_names = sorted(set(categories))
    if len(cat_names) > 12:
        top_cats = Counter(categories).most_common(10)
        cat_names = [c for c, _ in top_cats]

    methods = list(preds.keys())[:4]
    scores = {m: [] for m in methods}

    for cat in cat_names:
        mask = categories == cat
        for m in methods:
            f1 = f1_score(y_test[mask], preds[m][mask], average="macro", zero_division=0)
            scores[m].append(f1 * 100)

    angles = np.linspace(0, 2 * np.pi, len(cat_names), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for m in methods:
        vals = scores[m] + scores[m][:1]
        ax.plot(angles, vals, "o-", linewidth=1.5, label=m, markersize=4)
        ax.fill(angles, vals, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace("_", "\n") for c in cat_names], fontsize=8)
    ax.set_title("Per-Category Macro F1 (%)", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    path = os.path.join(fig_dir, "category_radar.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Figure 4: Model size vs performance ──────────────────────────────

def plot_size_vs_performance(results, fig_dir):
    """Scatter: model parameters vs Macro-F1."""
    size_map = {
        "Morgan FP + XGBoost": 0.01,
        "FP+Pharma + XGBoost": 0.01,
        "Zero-shot Qwen2.5-7B-Instruct": 7.0,
        "Zero-shot Llama-3.3-70B-Instruct": 70.0,
        "Qwen2.5-7B label-only (B)": 7.0,
        "PharmCoT C_summary (ours)": 7.0,
    }

    filtered = [r for r in results if r["model"] in size_map]
    if len(filtered) < 3:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    for r in filtered:
        size = size_map[r["model"]]
        f1 = r["macro_f1"] * 100
        color = OURS_COLOR if "ours" in r["model"] else ACCENT
        marker = "*" if "ours" in r["model"] else "o"
        ms = 200 if "ours" in r["model"] else 100
        ax.scatter(size, f1, c=color, s=ms, marker=marker, zorder=5)
        ax.annotate(r["model"].replace("Zero-shot ", "ZS "),
                    (size, f1), textcoords="offset points",
                    xytext=(10, 5), fontsize=8)

    ax.set_xscale("log")
    ax.set_xlabel("Model Parameters (B)")
    ax.set_ylabel("Macro F1 (%)")
    ax.set_title("Model Size vs DDI Classification Performance")
    ax.grid(alpha=0.3)

    path = os.path.join(fig_dir, "size_vs_performance.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Figure 5: Confusion matrix ──────────────────────────────────────

def plot_confusion_matrix(predictions_path, fig_dir, method_name=None):
    """Heatmap of top-N most confused class pairs."""
    if not os.path.exists(predictions_path):
        return None

    with open(predictions_path, "rb") as f:
        data = pickle.load(f)
    y_test = data["y_test"]
    preds = data["predictions"]

    if method_name is None:
        method_name = list(preds.keys())[-1]
    if method_name not in preds:
        return None

    y_pred = preds[method_name]
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    np.fill_diagonal(cm, 0)
    top_confused = np.unravel_index(np.argsort(cm.ravel())[-20:], cm.shape)
    active_labels = sorted(set(top_confused[0].tolist() + top_confused[1].tolist()))

    if len(active_labels) > 25:
        active_labels = active_labels[:25]

    sub_cm = cm[np.ix_(active_labels, active_labels)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sub_cm, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(active_labels)))
    ax.set_yticks(range(len(active_labels)))
    label_strs = [str(labels[i]) for i in active_labels]
    ax.set_xticklabels(label_strs, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(label_strs, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Most Confused Class Pairs: {method_name}")
    fig.colorbar(im, ax=ax, shrink=0.8)

    path = os.path.join(fig_dir, "confusion_matrix.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Figure 6: Severity prediction ────────────────────────────────────

def plot_severity_breakdown(res_dir, fig_dir):
    """Bar chart of severity accuracy by level for LLM methods."""
    severity_data = {}

    for fname in os.listdir(res_dir):
        if fname.startswith("zeroshot_") and fname.endswith("_predictions.jsonl"):
            key = fname.replace("_predictions.jsonl", "")
            preds = []
            with open(os.path.join(res_dir, fname)) as f:
                for line in f:
                    preds.append(json.loads(line))
            severity_data[key] = preds

    if not severity_data:
        return None

    levels = ["Major", "Moderate", "Minor"]
    methods = list(severity_data.keys())
    scores = {m: [] for m in methods}

    for level in levels:
        for m in methods:
            matching = [(r["true_severity"], r["pred_severity"])
                        for r in severity_data[m] if r["true_severity"] == level]
            if matching:
                correct = sum(1 for t, p in matching if t == p)
                scores[m].append(100 * correct / len(matching))
            else:
                scores[m].append(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(levels))
    width = 0.8 / max(len(methods), 1)

    for i, m in enumerate(methods):
        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, scores[m], width, label=m.replace("zeroshot_", ""), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Severity Prediction Accuracy by Level")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(fig_dir, "severity_breakdown.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Figure 7: Per-class F1 distribution ──────────────────────────────

def plot_perclass_f1_distribution(predictions_path, fig_dir):
    """Box plots of per-class F1 across methods."""
    if not os.path.exists(predictions_path):
        return None

    with open(predictions_path, "rb") as f:
        data = pickle.load(f)
    y_test = data["y_test"]
    preds = data["predictions"]

    labels = sorted(set(y_test))
    methods = list(preds.keys())[:5]

    all_f1s = []
    method_names = []
    for m in methods:
        report = classification_report(y_test, preds[m], labels=labels,
                                       output_dict=True, zero_division=0)
        f1s = [report[str(l)]["f1-score"] * 100 for l in labels if str(l) in report]
        all_f1s.append(f1s)
        method_names.append(m)

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(all_f1s, labels=method_names, patch_artist=True, showmeans=True)

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Per-class F1 (%)")
    ax.set_title("Distribution of Per-class F1 Scores")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(fig_dir, "perclass_f1_box.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate all result figures")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging("plot_results")
    ensure_dirs(cfg)

    res_dir = os.path.join(cfg["project"]["output_dir"], "results")
    fig_dir = os.path.join(cfg["project"]["output_dir"], "figures")
    os.makedirs(fig_dir, exist_ok=True)

    processed = cfg["data"]["processed_dir"]
    train_path = os.path.join(processed, "train.jsonl")
    test_path = os.path.join(processed, "test.jsonl")
    pred_path = os.path.join(res_dir, "baseline_predictions.pkl")

    results = load_all_results(res_dir)
    if not results:
        logger.warning("No results found -- run baselines first.")
        return

    logger.info(f"Loaded {len(results)} result entries")

    figs = []

    logger.info("Generating Figure 1: Main comparison...")
    figs.append(("Main comparison", plot_main_comparison(results, fig_dir)))

    logger.info("Generating Figure 2: Head/Mid/Tail...")
    figs.append(("Head/Mid/Tail", plot_head_mid_tail(pred_path, train_path, fig_dir)))

    logger.info("Generating Figure 3: Category radar...")
    figs.append(("Category radar", plot_category_radar(pred_path, test_path, fig_dir)))

    logger.info("Generating Figure 4: Size vs performance...")
    figs.append(("Size vs perf", plot_size_vs_performance(results, fig_dir)))

    logger.info("Generating Figure 5: Confusion matrix...")
    figs.append(("Confusion matrix", plot_confusion_matrix(pred_path, fig_dir)))

    logger.info("Generating Figure 6: Severity breakdown...")
    figs.append(("Severity", plot_severity_breakdown(res_dir, fig_dir)))

    logger.info("Generating Figure 7: Per-class F1 distribution...")
    figs.append(("Per-class F1", plot_perclass_f1_distribution(pred_path, fig_dir)))

    logger.info("\n=== Generated Figures ===")
    for name, path in figs:
        status = "OK" if path else "SKIPPED (data missing)"
        logger.info(f"  {name}: {status}")
        if path:
            logger.info(f"    -> {path}")


if __name__ == "__main__":
    main()
