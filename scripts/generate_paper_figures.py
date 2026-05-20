#!/usr/bin/env python3
"""
Generate publication-quality figures for IC2S2 paper.
Figures 2–4 from the Results section.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from pathlib import Path
from scipy.stats import pearsonr

# ── Global style ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "axes.edgecolor": "#333333",
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.color": "#E0E0E0",
    "grid.linewidth": 0.5,
})

# Muted, cohesive academic palette
C = {
    "blue":    "#3C78A8",
    "red":     "#C0504D",
    "green":   "#4EA072",
    "purple":  "#8B6BAE",
    "orange":  "#D48B3B",
    "gray":    "#8C8C8C",
    "lt_blue": "#A8CBE2",
    "lt_red":  "#E8ABA9",
    "lt_gray": "#D5D5D5",
}

PAL_SUB = {
    "teen_male":    C["blue"],
    "teen_female":  C["green"],
    "adult_male":   C["red"],
    "adult_female": C["purple"],
}

LABELS = {
    "teen_male":   "Teen Men",
    "teen_female":  "Teen Women",
    "adult_male":  "Adult Men",
    "adult_female": "Adult Women",
}

OUT = Path(__file__).parent.parent / "results" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)
CONF_THRESH = 0.60


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data():
    base = Path(__file__).parent.parent
    comments = pd.read_parquet(base / "Data/processed/all_comments.parquet")
    v3 = pd.read_parquet(base / "experiments/anthroscore_v3/anthroscore_v3_full.parquet")
    age = pd.read_parquet(base / "experiments/v2_correction/age_predictions_v4.parquet")
    gen = pd.read_parquet(base / "experiments/v2_correction/gender_predictions_v4.parquet")
    emo = pd.read_parquet(base / "Data/features/user_emotions.parquet")

    valid = v3[v3["score"] > 0].copy()
    valid = valid.merge(
        comments[["id", "author"]], left_on="comment_id", right_on="id", how="left"
    )

    user = valid.groupby("author").agg(
        anthro_mean=("score", "mean"),
        anthro_count=("score", "count"),
    ).reset_index()

    user = user.merge(
        age[["author", "age_predicted", "confidence"]].rename(
            columns={"confidence": "age_conf"}),
        on="author", how="left",
    )
    user = user.merge(
        gen[["author", "gender_predicted", "confidence"]].rename(
            columns={"confidence": "gender_conf"}),
        on="author", how="left",
    )
    user = user.merge(emo, on="author", how="left")

    hc = user[
        (user["age_conf"] >= CONF_THRESH) & (user["gender_conf"] >= CONF_THRESH)
    ].copy()
    hc["subgroup"] = hc["age_predicted"] + "_" + hc["gender_predicted"]

    comment_hc = valid.merge(
        age[["author", "age_predicted", "confidence"]].rename(
            columns={"confidence": "age_conf"}),
        on="author", how="left",
    )
    comment_hc = comment_hc.merge(
        gen[["author", "gender_predicted", "confidence"]].rename(
            columns={"confidence": "gender_conf"}),
        on="author", how="left",
    )
    comment_hc = comment_hc[
        (comment_hc["age_conf"] >= CONF_THRESH)
        & (comment_hc["gender_conf"] >= CONF_THRESH)
    ].copy()
    comment_hc["subgroup"] = (
        comment_hc["age_predicted"] + "_" + comment_hc["gender_predicted"]
    )

    return hc, comment_hc


# ── Figure 2: AnthroIndex distribution ───────────────────────────────────────

def fig2_distribution(comment_hc):
    fig, axes = plt.subplots(
        1, 2, figsize=(7.5, 3.4), gridspec_kw={"width_ratios": [1, 1.4]}
    )

    scores = [1, 2, 3, 4, 5]
    tick_labels = ["1\nNone", "2\nMinimal", "3\nModerate", "4\nHigh", "5\nExtreme"]

    # Panel (a): overall
    ax = axes[0]
    overall = comment_hc["score"].value_counts(normalize=True).reindex(scores, fill_value=0)
    bars = ax.bar(
        scores, overall.values * 100,
        color=C["blue"], edgecolor="white", width=0.65, linewidth=0.4,
    )
    for bar, pct in zip(bars, overall.values * 100):
        if pct >= 2:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=8,
            )
    ax.set_xticks(scores)
    ax.set_xticklabels(tick_labels, fontsize=8.5)
    ax.set_ylabel("Percentage of comments")
    ax.set_xlabel("AnthroIndex score")
    ax.set_ylim(0, 84)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title("(a)  Overall", fontweight="semibold", loc="left", pad=6)

    # Panel (b): by subgroup
    ax = axes[1]
    order = ["teen_male", "teen_female", "adult_male", "adult_female"]
    n_groups = len(order)
    bar_w = 0.16
    x = np.arange(len(scores))

    for i, sg in enumerate(order):
        sg_data = comment_hc[comment_hc["subgroup"] == sg]
        dist = sg_data["score"].value_counts(normalize=True).reindex(scores, fill_value=0)
        offset = (i - (n_groups - 1) / 2) * bar_w
        ax.bar(
            x + offset, dist.values * 100, bar_w,
            label=LABELS[sg], color=PAL_SUB[sg],
            edgecolor="white", linewidth=0.3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=8.5)
    ax.set_ylabel("Percentage of comments")
    ax.set_xlabel("AnthroIndex score")
    ax.set_ylim(0, 84)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right",
              handlelength=1.2, handletextpad=0.4, columnspacing=1.0)
    ax.set_title("(b)  By demographic subgroup", fontweight="semibold", loc="left", pad=6)

    fig.tight_layout(w_pad=2.5)
    path = OUT / "fig2_anthroindex_distribution.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 3: Age × Gender effects ──────────────────────────────────────────

def fig3_age_gender_effects(user_hc):
    df = user_hc[user_hc["anthro_mean"] > 1].copy()

    fig, axes = plt.subplots(
        1, 2, figsize=(7.5, 3.6), gridspec_kw={"width_ratios": [1, 1.3]}
    )

    # Panel (a): main effects
    ax = axes[0]
    pairs = [
        ("Teens",  df.loc[df["age_predicted"] == "teen",   "anthro_mean"], C["blue"]),
        ("Adults", df.loc[df["age_predicted"] == "adult",  "anthro_mean"], C["red"]),
        ("Men",    df.loc[df["gender_predicted"] == "male",   "anthro_mean"], "#2E6E9E"),
        ("Women",  df.loc[df["gender_predicted"] == "female", "anthro_mean"], C["purple"]),
    ]
    positions = [0, 0.55, 1.3, 1.85]

    for pos, (label, vals, color) in zip(positions, pairs):
        m = vals.mean()
        se = vals.std() / np.sqrt(len(vals))
        ci = 1.96 * se
        ax.bar(pos, m, 0.45, color=color, edgecolor="none", zorder=3)
        ax.errorbar(pos, m, yerr=ci, fmt="none", ecolor="#333333",
                    capsize=3, linewidth=0.9, zorder=4)
        ax.text(pos, m + ci + 0.006, f"{m:.2f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="semibold")

    ax.set_xticks(positions)
    ax.set_xticklabels(["Teens", "Adults", "Men", "Women"], fontsize=9)
    ax.set_ylabel("Mean AnthroIndex")
    ax.set_ylim(1.92, 2.38)
    ax.yaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)

    # Effect-size annotations
    ax.annotate(
        "", xy=(-0.02, 2.33), xytext=(0.57, 2.33),
        arrowprops=dict(arrowstyle="-", color="#666666", lw=0.7,
                        connectionstyle="bar,fraction=0.18"),
    )
    ax.text(0.275, 2.355, "d\u2009=\u20090.50", ha="center", fontsize=7.5,
            color="#555555", fontstyle="italic")

    ax.annotate(
        "", xy=(1.28, 2.27), xytext=(1.87, 2.27),
        arrowprops=dict(arrowstyle="-", color="#666666", lw=0.7,
                        connectionstyle="bar,fraction=0.22"),
    )
    ax.text(1.575, 2.295, "d\u2009=\u20090.29", ha="center", fontsize=7.5,
            color="#555555", fontstyle="italic")

    ax.set_title("(a)  Main effects", fontweight="semibold", loc="left", pad=6)

    # Panel (b): subgroup interaction
    ax = axes[1]
    order = ["teen_male", "teen_female", "adult_male", "adult_female"]
    x_pos = np.arange(len(order))

    means, cis, ns = [], [], []
    for sg in order:
        vals = df.loc[df["subgroup"] == sg, "anthro_mean"]
        m = vals.mean()
        se = vals.std() / np.sqrt(len(vals))
        means.append(m)
        cis.append(1.96 * se)
        ns.append(len(vals))

    ax.bar(x_pos, means, 0.58, color=[PAL_SUB[sg] for sg in order],
           edgecolor="none", zorder=3)
    ax.errorbar(x_pos, means, yerr=cis, fmt="none", ecolor="#333333",
                capsize=4, linewidth=0.9, zorder=4)

    for pos, m, ci, n in zip(x_pos, means, cis, ns):
        ax.text(pos, m + ci + 0.005, f"{m:.2f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="semibold")
        ax.text(pos, 1.935, f"n\u2009=\u2009{n:,}", ha="center", va="bottom",
                fontsize=7.5, color="#777777")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([LABELS[sg] for sg in order], fontsize=9)
    ax.set_ylabel("Mean AnthroIndex")
    ax.set_ylim(1.92, 2.48)
    ax.yaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)

    # Significance brackets
    def bracket(ax, x1, x2, y, text):
        ax.plot([x1, x1, x2, x2], [y, y + 0.01, y + 0.01, y],
                lw=0.7, color="#444444", zorder=5)
        ax.text((x1 + x2) / 2, y + 0.013, text, ha="center", va="bottom",
                fontsize=7.5, color="#444444")

    bracket(ax, 0, 2, 2.39, "p < .001")
    bracket(ax, 0, 3, 2.43, "p < .001")

    ax.set_title("(b)  Age \u00d7 Gender interaction", fontweight="semibold",
                 loc="left", pad=6)

    fig.tight_layout(w_pad=2.5)
    path = OUT / "fig3_age_gender_effects.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 4: Emotion–Anthropomorphization correlations ──────────────────────

def fig4_emotion_correlations(user_hc):
    """Coefficient plot (dot + whisker) — the standard for showing correlations."""

    emotion_meta = [
        ("emotion_neutral",  "Neutral"),
        ("emotion_surprise", "Surprise"),
        ("emotion_disgust",  "Disgust"),
        ("emotion_sadness",  "Sadness"),
        ("emotion_anger",    "Anger"),
        ("emotion_fear",     "Fear"),
        ("emotion_joy",      "Joy"),
    ]

    df = user_hc.dropna(
        subset=["anthro_mean"] + [e[0] for e in emotion_meta]
    ).copy()
    n = len(df)

    rows = []
    for col, label in emotion_meta:
        r, p = pearsonr(df["anthro_mean"], df[col])
        se = np.sqrt((1 - r ** 2) / (n - 2))
        rows.append({
            "col": col, "label": label,
            "r": r, "p": p,
            "ci_lo": r - 1.96 * se,
            "ci_hi": r + 1.96 * se,
        })

    res = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    y_pos = np.arange(len(res))
    colors = [C["red"] if r < 0 else C["blue"] for r in res["r"]]

    # Whiskers (CI)
    ax.hlines(y_pos, res["ci_lo"], res["ci_hi"], colors="#555555",
              linewidth=0.8, zorder=2)
    # Caps
    cap_h = 0.18
    for i, row in res.iterrows():
        ax.vlines(row["ci_lo"], i - cap_h, i + cap_h, colors="#555555",
                  linewidth=0.7, zorder=2)
        ax.vlines(row["ci_hi"], i - cap_h, i + cap_h, colors="#555555",
                  linewidth=0.7, zorder=2)

    # Dots
    ax.scatter(res["r"], y_pos, s=48, c=colors, edgecolors="white",
               linewidths=0.6, zorder=3)

    # r-value labels
    for i, row in res.iterrows():
        sig = ""
        if row["p"] < .001:
            sig = "***"
        elif row["p"] < .01:
            sig = "**"
        elif row["p"] < .05:
            sig = "*"

        nudge = 0.006 if row["r"] >= 0 else -0.006
        ha = "left" if row["r"] >= 0 else "right"
        label_x = row["ci_hi"] + 0.005 if row["r"] >= 0 else row["ci_lo"] - 0.005
        ax.text(label_x, i, f"{row['r']:.3f}{sig}", ha=ha, va="center",
                fontsize=7.5, color="#444444")

    ax.axvline(0, color="#999999", linewidth=0.6, linestyle="-", zorder=1)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(res["label"], fontsize=9.5)
    ax.set_xlabel("Pearson r with mean AnthroIndex", labelpad=6)
    ax.set_xlim(-0.19, 0.21)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.05))

    ax.text(0.98, 0.04, f"N = {n:,}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#888888")

    ax.set_title("Emotion–Anthropomorphization\nCorrelations",
                 fontweight="semibold", loc="left", pad=8, fontsize=11)

    fig.tight_layout()
    path = OUT / "fig4_emotion_correlations.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    user_hc, comment_hc = load_data()
    print(f"  Users: {len(user_hc):,}  |  Comments: {len(comment_hc):,}\n")

    print("Figure 2: AnthroIndex distribution")
    fig2_distribution(comment_hc)

    print("Figure 3: Age × Gender effects")
    fig3_age_gender_effects(user_hc)

    print("Figure 4: Emotion correlations")
    fig4_emotion_correlations(user_hc)

    print(f"\nAll figures saved to: {OUT}")
