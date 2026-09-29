"""
Utilities and Visualization Module for Fake News Detection System.

This module provides:
- Automated generation of professional EDA charts for Phase 3:
    1. Class distribution (Real vs Fake)
    2. Word cloud for Fake news
    3. Word cloud for Real news
    4. Top 20 most frequent terms
    5. News length distributions (character & word count)
    6. Subject distributions across categories
    7. Correlation heatmap of handcrafted linguistic features
- Model evaluation plotting utilities for Phase 7:
    - Confusion Matrix heatmap
    - ROC Curve with AUC score
    - Precision-Recall Curve with average precision
    - 5-Fold Cross Validation performance bar chart
    - Comprehensive Model Comparison comparison chart
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI environments
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)

# Premium, modern visual palette
PALETTE = {
    "fake": "#E63946",       # Crimson Red
    "real": "#2A9D8F",       # Emerald Teal
    "neutral": "#457B9D",    # Steel Blue
    "dark": "#1D3557",       # Deep Navy
    "light": "#F8F9FA",      # Clean White/Gray
    "accent": "#F4A261"      # Warm Coral
}

# Apply global styling with multilingual font fallback
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8


def setup_report_directories():
    """Ensure output reporting and figures directories exist."""
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/metrics", exist_ok=True)


def plot_class_distribution(df: pd.DataFrame, output_path: str = "reports/figures/class_distribution.png"):
    """Plot target binary class distribution (Real vs Fake)."""
    setup_report_directories()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    counts = df["label"].value_counts().sort_index()
    labels = ["Fake News (0)", "Real News (1)"]
    colors = [PALETTE["fake"], PALETTE["real"]]
    bars = ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor="none", zorder=3)

    # Annotate bar tops with count and percentage
    total = len(df)
    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        ax.annotate(
            f"{height:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#333333"
        )

    ax.set_title("Ground-Truth Class Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Number of Articles", fontsize=12)
    ax.set_ylim(0, max(counts.values) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved class distribution figure to: {output_path}")


def plot_subject_distribution(df: pd.DataFrame, output_path: str = "reports/figures/subject_distribution.png"):
    """Plot distribution of news topics/subjects partitioned by class."""
    if "subject" not in df.columns:
        return
    setup_report_directories()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    subject_counts = df.groupby(["subject", "label"]).size().unstack(fill_value=0)
    subject_counts.rename(columns={0: "Fake", 1: "Real"}, inplace=True)
    
    # Sort by total volume
    subject_counts["Total"] = subject_counts.sum(axis=1)
    subject_counts = subject_counts.sort_values(by="Total", ascending=True)
    plot_data = subject_counts[["Fake", "Real"]]

    plot_data.plot(
        kind="barh",
        stacked=True,
        color=[PALETTE["fake"], PALETTE["real"]],
        ax=ax,
        width=0.65,
        edgecolor="none"
    )

    ax.set_title("News Distribution by Subject / Category", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Articles", fontsize=12)
    ax.set_ylabel("Subject Category", fontsize=12)
    ax.legend(title="Classification", loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved subject distribution figure to: {output_path}")


def plot_wordclouds(
    df: pd.DataFrame,
    fake_output: str = "reports/figures/wordcloud_fake.png",
    real_output: str = "reports/figures/wordcloud_real.png",
    max_words: int = 150
):
    """Generate high-resolution word clouds for Fake and Real news corpora."""
    setup_report_directories()
    preprocessor = TextPreprocessor()

    for label_val, name, color_map, out_path in [
        (0, "Fake News", "Reds", fake_output),
        (1, "Real News", "Greens", real_output)
    ]:
        subset = df[df["label"] == label_val]
        # Sample text for fast, memory-efficient wordcloud generation
        sample_texts = subset["title"].fillna("") + " " + subset["text"].fillna("").str[:400]
        sample_subset = sample_texts.sample(n=min(3000, len(subset)), random_state=42)
        cleaned_blob = " ".join(preprocessor.transform(sample_subset))

        font_path = None
        for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/arial.ttf"]:
            if os.path.exists(fp):
                font_path = fp
                break

        wc = WordCloud(
            width=1000,
            height=550,
            background_color="white",
            colormap=color_map,
            max_words=max_words,
            contour_width=1,
            contour_color="#EEEEEE",
            collocations=False,
            font_path=font_path,
            random_state=42
        ).generate(cleaned_blob)

        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Frequent Terms Word Cloud — {name}", fontsize=15, fontweight="bold", pad=15)

        plt.tight_layout(pad=1.5)
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Saved {name} word cloud to: {out_path}")


def plot_top_words(df: pd.DataFrame, output_path: str = "reports/figures/top_20_words.png", top_n: int = 20):
    """Plot top 20 most frequent words in Fake vs Real news."""
    setup_report_directories()
    preprocessor = TextPreprocessor()

    def get_top_tokens(series: pd.Series, n=20):
        cleaned = preprocessor.transform(series.sample(min(2500, len(series)), random_state=42))
        tokens = [token for doc in cleaned for token in doc.split()]
        return Counter(tokens).most_common(n)

    fake_top = get_top_tokens(df[df["label"] == 0]["title"].fillna("") + " " + df[df["label"] == 0]["text"].fillna("").str[:300], top_n)
    real_top = get_top_tokens(df[df["label"] == 1]["title"].fillna("") + " " + df[df["label"] == 1]["text"].fillna("").str[:300], top_n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=300)

    # Fake news top words
    f_words, f_counts = zip(*fake_top[::-1])
    ax1.barh(f_words, f_counts, color=PALETTE["fake"], height=0.65)
    ax1.set_title(f"Top {top_n} Frequent Words in Fake News", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Term Frequency", fontsize=11)
    ax1.grid(axis="x", linestyle="--", alpha=0.6)

    # Real news top words
    r_words, r_counts = zip(*real_top[::-1])
    ax2.barh(r_words, r_counts, color=PALETTE["real"], height=0.65)
    ax2.set_title(f"Top {top_n} Frequent Words in Real News", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Term Frequency", fontsize=11)
    ax2.grid(axis="x", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved top words figure to: {output_path}")


def plot_length_distribution(df: pd.DataFrame, output_path: str = "reports/figures/news_length_distribution.png"):
    """Plot character length and word count distributions for Fake vs Real news."""
    setup_report_directories()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Character count
    fake_chars = df[df["label"] == 0]["text"].fillna("").str.len()
    real_chars = df[df["label"] == 1]["text"].fillna("").str.len()

    # Clip for clean visualization without extreme outliers
    fake_chars_clipped = np.clip(fake_chars, 0, 10000)
    real_chars_clipped = np.clip(real_chars, 0, 10000)

    sns.kdeplot(fake_chars_clipped, label="Fake News", color=PALETTE["fake"], fill=True, alpha=0.35, ax=ax1, linewidth=2)
    sns.kdeplot(real_chars_clipped, label="Real News", color=PALETTE["real"], fill=True, alpha=0.35, ax=ax1, linewidth=2)
    ax1.set_title("Article Character Length Density", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Character Length (capped at 10k)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Word/token count (handles both space-separated English and character/token-based Chinese)
    import re
    def count_tokens(s):
        st = str(s)
        if re.search(r"[\u4e00-\u9fff]", st):
            return len(st)
        return len(st.split())

    fake_words = df[df["label"] == 0]["text"].fillna("").apply(count_tokens)
    real_words = df[df["label"] == 1]["text"].fillna("").apply(count_tokens)

    fake_words_clipped = np.clip(fake_words, 0, 1500)
    real_words_clipped = np.clip(real_words, 0, 1500)

    sns.kdeplot(fake_words_clipped, label="Fake News", color=PALETTE["fake"], fill=True, alpha=0.35, ax=ax2, linewidth=2)
    sns.kdeplot(real_words_clipped, label="Real News", color=PALETTE["real"], fill=True, alpha=0.35, ax=ax2, linewidth=2)
    ax2.set_title("Article Word Count Density", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Word Count (capped at 1,500 words)", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved length distribution figure to: {output_path}")


def plot_correlation_heatmap(df: pd.DataFrame, output_path: str = "reports/figures/correlation_heatmap.png"):
    """Plot correlation heatmap across handcrafted stylometric features and target label."""
    setup_report_directories()
    from src.feature_engineering import HandcraftedLinguisticExtractor

    extractor = HandcraftedLinguisticExtractor()
    sample = df.sample(min(5000, len(df)), random_state=42).copy()
    features = extractor.transform(sample)

    feat_df = pd.DataFrame(features, columns=extractor.feature_names)
    feat_df["label"] = sample["label"].values

    corr = feat_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.5,
        vmax=1.0,
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title("Correlation Heatmap: Stylometric Features vs Target Label", fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved correlation heatmap to: {output_path}")


def plot_confusion_matrix_figure(
    cm: np.ndarray,
    output_path: str = "reports/figures/confusion_matrix.png",
    model_name: str = "Best Model"
):
    """Plot high-contrast normalized and count confusion matrix."""
    setup_report_directories()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    group_counts = [f"{val:,}" for val in cm.flatten()]
    group_percentages = [f"{val:.2%}" for val in cm.flatten() / np.sum(cm)]
    box_labels = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(2, 2)

    sns.heatmap(
        cm,
        annot=box_labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Fake (0)", "Predicted Real (1)"],
        yticklabels=["Actual Fake (0)", "Actual Real (1)"],
        ax=ax,
        annot_kws={"size": 12, "weight": "bold"}
    )

    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix figure to: {output_path}")


def plot_roc_curve_figure(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str = "reports/figures/roc_curve.png",
    model_name: str = "Best Model"
):
    """Plot Receiver Operating Characteristic (ROC) curve."""
    setup_report_directories()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
    ax.plot(fpr, tpr, color=PALETTE["dark"], lw=2.5, label=f"{model_name} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#999999", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curve figure to: {output_path}")


def plot_precision_recall_curve_figure(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str = "reports/figures/precision_recall_curve.png",
    model_name: str = "Best Model"
):
    """Plot Precision-Recall curve."""
    setup_report_directories()
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
    ax.plot(recall, precision, color=PALETTE["real"], lw=2.5, label=f"{model_name} (AP = {ap:.4f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved Precision-Recall curve figure to: {output_path}")


def plot_model_comparison_bar(
    comparison_df: pd.DataFrame,
    output_path: str = "reports/figures/model_comparison.png"
):
    """Generate multi-metric comparative bar chart across all evaluated algorithms."""
    setup_report_directories()
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    available_metrics = [m for m in metrics if m in comparison_df.columns]

    df_melted = comparison_df.melt(
        id_vars=["Model"],
        value_vars=available_metrics,
        var_name="Metric",
        value_name="Score"
    )

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    sns.barplot(
        data=df_melted,
        x="Model",
        y="Score",
        hue="Metric",
        palette="viridis",
        ax=ax
    )

    ax.set_title("Benchmarking Comparison Across Machine Learning Classifiers", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Performance Score (0.0 to 1.0)", fontsize=12)
    ax.set_xlabel("Machine Learning Algorithm", fontsize=12)
    ax.set_ylim(0.7, 1.02)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved model comparison figure to: {output_path}")


def generate_all_eda_visualizations(df: pd.DataFrame):
    """Orchestrate generation of all 7 Phase 3 EDA figures."""
    logger.info("=== Generating Phase 3 Exploratory Data Analysis Figures ===")
    setup_report_directories()
    plot_class_distribution(df)
    plot_subject_distribution(df)
    plot_length_distribution(df)
    plot_correlation_heatmap(df)
    plot_top_words(df)
    plot_wordclouds(df)
    logger.info("=== Phase 3 EDA Visualizations Successfully Generated ===")


if __name__ == "__main__":
    from src.data_loader import NewsDataLoader
    loader = NewsDataLoader()
    df, _ = loader.run_pipeline()
    generate_all_eda_visualizations(df)
