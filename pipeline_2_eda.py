# ============================================================
# 📊  pipeline_2_eda.py
#     INPUT  : processed_data.csv
#     OUTPUT : eda_plots.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

from config import PROCESSED_CSV, EDA_PLOT_PATH

COLORS      = {"Positive": "#2ecc71", "Neutral": "#3498db", "Negative": "#e74c3c"}
CLASS_ORDER = ["Positive", "Neutral", "Negative"]


def run():
    print("=" * 56)
    print("   📊  PIPELINE 2 : EDA PLOTS")
    print("=" * 56)

    df = pd.read_csv(PROCESSED_CSV)

    fig = plt.figure(figsize=(18, 7))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── Plot 1 : Sentiment Distribution ──────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#0d1117")

    counts    = df['sentiment_category'].value_counts()
    cats      = [c for c in CLASS_ORDER if c in counts.index]
    vals      = [counts[c] for c in cats]
    bar_cols  = [COLORS[c] for c in cats]
    pct_total = sum(vals)

    bars = ax1.bar(cats, vals, color=bar_cols, width=0.55,
                   edgecolor="#0d1117", linewidth=1.5, zorder=3)

    for bar, v in zip(bars, vals):
        pct = v / pct_total * 100
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + pct_total * 0.01,
                 f"{v:,}\n({pct:.1f}%)",
                 ha="center", va="bottom",
                 color="white", fontsize=11, fontweight="bold")

    ax1.set_title("Sentiment Class Distribution",
                  color="white", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Sentiment", color="#aaaaaa", fontsize=11)
    ax1.set_ylabel("Count",     color="#aaaaaa", fontsize=11)
    ax1.tick_params(colors="white")
    ax1.spines[["top","right","left","bottom"]].set_color("#333333")
    ax1.yaxis.grid(True, color="#222222", linestyle="--", linewidth=0.7, zorder=0)
    ax1.set_axisbelow(True)

    # ── Plot 2 : Top-20 Words per Sentiment ──────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#0d1117")

    TOP_N = 20
    word_freq = {}
    for cls in CLASS_ORDER:
        subset    = df[df['sentiment_category'] == cls]['processed_review']
        all_words = " ".join(subset.dropna()).split()
        word_freq[cls] = Counter(all_words).most_common(TOP_N)

    y_positions = np.arange(TOP_N)
    bar_height  = 0.25
    offsets     = [-bar_height, 0, bar_height]

    for i, cls in enumerate(CLASS_ORDER):
        if cls not in word_freq:
            continue
        words = [w for w, _ in word_freq[cls]][::-1]
        freqs = [c for _, c in word_freq[cls]][::-1]
        y_pos = y_positions + offsets[i]
        ax2.barh(y_pos, freqs, height=bar_height * 0.85,
                 color=COLORS[cls], label=cls, alpha=0.9, zorder=3)

    ref_words = [w for w, _ in word_freq.get("Positive", word_freq[CLASS_ORDER[0]])][::-1]
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(ref_words, color="white", fontsize=8)
    ax2.tick_params(axis="x", colors="white")
    ax2.set_title("Top 20 Words per Sentiment Class",
                  color="white", fontsize=14, fontweight="bold", pad=12)
    ax2.set_xlabel("Frequency", color="#aaaaaa", fontsize=11)
    ax2.spines[["top","right","left","bottom"]].set_color("#333333")
    ax2.xaxis.grid(True, color="#222222", linestyle="--", linewidth=0.7, zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend(facecolor="#1a1a2e", labelcolor="white",
               framealpha=0.8, fontsize=10, loc="lower right")

    fig.suptitle("🔍  Exploratory Data Analysis  —  Sentiment Dataset",
                 color="white", fontsize=16, fontweight="bold", y=1.01)
    plt.savefig(EDA_PLOT_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"   Saved → eda_plots.png ✅\n")


if __name__ == "__main__":
    run()
