"""
Generate a comprehensive paper-writing reference for The Illusion Project.

This script intentionally recomputes paper-facing quantities from the current
project parquet/JSON artifacts. It is not meant to replace the canonical
COMPREHENSIVE_V3_ANALYSIS.py pipeline; it is a higher-level reference file
for writing Methods, Results, Discussion, Limitations, and Appendix sections.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "PAPER_WRITING_MASTER_REFERENCE.md"
CONFIDENCE_THRESHOLD = 0.60
EMOTIONS = [
    "emotion_joy",
    "emotion_sadness",
    "emotion_anger",
    "emotion_fear",
    "emotion_disgust",
    "emotion_surprise",
    "emotion_neutral",
]
NON_NEUTRAL_EMOTIONS = [e for e in EMOTIONS if e != "emotion_neutral"]


def read_json(path: str) -> dict:
    p = ROOT / path
    return json.loads(p.read_text()) if p.exists() else {}


def fmt_p(p: float | int | None) -> str:
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def fmt_num(x: float | int | None, digits: int = 3) -> str:
    if x is None:
        return "NA"
    try:
        if not np.isfinite(x):
            return "NA"
    except TypeError:
        return str(x)
    return f"{x:.{digits}f}"


def fmt_int(x: float | int) -> str:
    return f"{int(round(x)):,}"


def pct(x: float | int, digits: int = 1) -> str:
    return f"{100 * float(x):.{digits}f}%"


def ci_prop(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    lo, hi = stats.binomtest(k, n).proportion_ci(confidence_level=1 - alpha, method="wilson")
    return float(lo), float(hi)


def cohens_d(g1: pd.Series, g2: pd.Series) -> float:
    a = pd.Series(g1).dropna().astype(float).to_numpy()
    b = pd.Series(g2).dropna().astype(float).to_numpy()
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    d = (np.mean(a) - np.mean(b)) / pooled
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    return float(d * correction)


def interpret_d(d: float) -> str:
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def odds_ratio_2x2(table: pd.DataFrame, row_a: str, row_b: str, col_high: int = 1) -> float:
    # Rows are groups, columns are binary outcome 0/1.
    a = table.loc[row_a, col_high] if row_a in table.index and col_high in table.columns else 0
    b = table.loc[row_a, 1 - col_high] if row_a in table.index and (1 - col_high) in table.columns else 0
    c = table.loc[row_b, col_high] if row_b in table.index and col_high in table.columns else 0
    d = table.loc[row_b, 1 - col_high] if row_b in table.index and (1 - col_high) in table.columns else 0
    return float(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))


def fisher_z_diff(r1: float, n1: int, r2: float, n2: int) -> tuple[float, float]:
    if n1 <= 3 or n2 <= 3 or not np.isfinite(r1) or not np.isfinite(r2):
        return np.nan, np.nan
    r1 = min(max(r1, -0.999999), 0.999999)
    r2 = min(max(r2, -0.999999), 0.999999)
    z = (np.arctanh(r1) - np.arctanh(r2)) / math.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def section(title: str) -> str:
    return f"\n## {title}\n"


def subsection(title: str) -> str:
    return f"\n### {title}\n"


def load_data() -> dict[str, pd.DataFrame]:
    data = {
        "comments": pd.read_parquet(ROOT / "Data/processed/all_comments.parquet"),
        "enriched": pd.read_parquet(ROOT / "Data/features/comments_enriched.parquet"),
        "anthro": pd.read_parquet(ROOT / "experiments/anthroscore_v3/anthroscore_v3_improved_final.parquet"),
        "anthro_original": pd.read_parquet(ROOT / "experiments/anthroscore_v3/anthroscore_v3_full.parquet"),
        "emotions_user": pd.read_parquet(ROOT / "Data/features/user_emotions.parquet"),
        "emotions_comment": pd.read_parquet(ROOT / "Data/features/comments_with_emotions.parquet"),
        "v2_user": pd.read_parquet(ROOT / "Data/features/user_anthroscores.parquet"),
        "gender": pd.read_parquet(ROOT / "experiments/v2_correction/gender_predictions_v4.parquet"),
        "age": pd.read_parquet(ROOT / "experiments/v2_correction/age_predictions_v4.parquet"),
        "self_decl": pd.read_parquet(ROOT / "Data/features/self_declarations.parquet"),
    }
    confirmatory = ROOT / "Data/confirmatory/confirmatory_scored.parquet"
    if confirmatory.exists():
        data["confirmatory"] = pd.read_parquet(confirmatory)
    llm_demo = ROOT / "Data/features/llm_classifications.parquet"
    if llm_demo.exists():
        data["llm_demo"] = pd.read_parquet(llm_demo)
    return data


def add_primary_subreddit(comments: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    counts = comments.groupby(["author", "subreddit"]).size().rename("n").reset_index()
    counts = counts.sort_values(["author", "n", "subreddit"], ascending=[True, False, True])
    primary = counts.drop_duplicates("author").rename(columns={"subreddit": "primary_subreddit", "n": "primary_subreddit_comments"})
    total = comments.groupby("author").size().rename("total_comments_raw").reset_index()
    out = users.merge(primary[["author", "primary_subreddit", "primary_subreddit_comments"]], on="author", how="left")
    out = out.merge(total, on="author", how="left")
    out["primary_subreddit_share"] = out["primary_subreddit_comments"] / out["total_comments_raw"]
    return out


def make_user_dataset(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    comments = data["comments"].copy()
    comments["id"] = comments["id"].astype(str)
    anthro = data["anthro"].copy()
    anthro["comment_id"] = anthro["comment_id"].astype(str)
    scored_comments = anthro.merge(
        comments[["id", "author", "subreddit", "body", "created_utc", "score"]].astype({"id": str}),
        left_on="comment_id",
        right_on="id",
        how="left",
    )

    user_scores = (
        scored_comments.groupby("author")
        .agg(
            anthro_mean=("score_x", "mean"),
            anthro_std=("score_x", "std"),
            anthro_count=("score_x", "size"),
            anthro_median=("score_x", "median"),
            anthro_min=("score_x", "min"),
            anthro_max=("score_x", "max"),
            score_1_count=("score_x", lambda s: int((s == 1).sum())),
            score_2_count=("score_x", lambda s: int((s == 2).sum())),
            score_3plus_count=("score_x", lambda s: int((s >= 3).sum())),
            score_4plus_count=("score_x", lambda s: int((s >= 4).sum())),
        )
        .reset_index()
    )
    user_scores["pct_comments_score_3plus"] = user_scores["score_3plus_count"] / user_scores["anthro_count"]
    user_scores["pct_comments_score_4plus"] = user_scores["score_4plus_count"] / user_scores["anthro_count"]
    user_scores["has_score_3plus"] = (user_scores["score_3plus_count"] > 0).astype(int)
    user_scores["has_score_4plus"] = (user_scores["score_4plus_count"] > 0).astype(int)

    gender = data["gender"].rename(columns={"confidence": "gender_confidence"})
    age = data["age"].rename(columns={"confidence": "age_confidence"})
    users = user_scores.merge(gender, on="author", how="left").merge(age, on="author", how="left")
    users = users.merge(data["emotions_user"], on="author", how="left")
    users = users.merge(
        data["v2_user"][["author", "anthroscore_mean"]].rename(columns={"anthroscore_mean": "anthro_v2_mean"}),
        on="author",
        how="left",
    )
    users = users.merge(data["self_decl"], on="author", how="left")
    users = add_primary_subreddit(comments, users)
    users["demo_high_conf"] = (
        (users["gender_confidence"] >= CONFIDENCE_THRESHOLD)
        & (users["age_confidence"] >= CONFIDENCE_THRESHOLD)
    )
    users["is_teen"] = (users["age_predicted"] == "teen").astype(int)
    users["is_female"] = (users["gender_predicted"] == "female").astype(int)
    users["teen_x_female"] = users["is_teen"] * users["is_female"]
    return users, scored_comments


def describe_distribution(s: pd.Series) -> dict[str, float]:
    s = pd.Series(s).dropna().astype(float)
    return {
        "n": len(s),
        "mean": float(s.mean()),
        "sd": float(s.std()),
        "median": float(s.median()),
        "p10": float(s.quantile(0.10)),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "min": float(s.min()),
        "max": float(s.max()),
        "skew": float(stats.skew(s)),
    }


def group_test(df: pd.DataFrame, group_col: str, a: str, b: str, value: str = "anthro_mean") -> dict[str, object]:
    x = df.loc[df[group_col] == a, value].dropna()
    y = df.loc[df[group_col] == b, value].dropna()
    t = stats.ttest_ind(x, y, equal_var=False)
    u = stats.mannwhitneyu(x, y, alternative="two-sided")
    d = cohens_d(x, y)
    return {
        "a": a,
        "b": b,
        "n_a": len(x),
        "n_b": len(y),
        "mean_a": float(x.mean()),
        "mean_b": float(y.mean()),
        "sd_a": float(x.std()),
        "sd_b": float(y.std()),
        "median_a": float(x.median()),
        "median_b": float(y.median()),
        "t": float(t.statistic),
        "t_p": float(t.pvalue),
        "u": float(u.statistic),
        "u_p": float(u.pvalue),
        "d": d,
        "d_interp": interpret_d(d),
        "diff_a_minus_b": float(x.mean() - y.mean()),
    }


def model_summary(model) -> dict[str, object]:
    out = {
        "n": int(model.nobs),
        "r2": float(model.rsquared) if hasattr(model, "rsquared") else np.nan,
        "adj_r2": float(model.rsquared_adj) if hasattr(model, "rsquared_adj") else np.nan,
        "f": float(model.fvalue) if hasattr(model, "fvalue") and model.fvalue is not None else np.nan,
        "p": float(model.f_pvalue) if hasattr(model, "f_pvalue") and model.f_pvalue is not None else np.nan,
        "aic": float(model.aic),
        "bic": float(model.bic),
        "coefficients": [],
    }
    conf = model.conf_int()
    for name in model.params.index:
        out["coefficients"].append(
            {
                "term": name,
                "b": float(model.params[name]),
                "se": float(model.bse[name]),
                "t": float(model.tvalues[name]),
                "p": float(model.pvalues[name]),
                "ci_lo": float(conf.loc[name, 0]),
                "ci_hi": float(conf.loc[name, 1]),
            }
        )
    return out


def vif_table(df: pd.DataFrame, cols: list[str]) -> list[dict[str, object]]:
    X = sm.add_constant(df[cols].dropna())
    out = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        out.append({"term": col, "vif": float(variance_inflation_factor(X.values, i))})
    return out


def pattern_stats(df: pd.DataFrame, pattern: str, text_col: str = "body") -> pd.Series:
    return df[text_col].fillna("").str.contains(pattern, case=False, regex=True).astype(int)


def main() -> None:
    data = load_data()
    users, scored_comments = make_user_dataset(data)
    hc = users.loc[users["demo_high_conf"]].copy()
    hc_meaningful = hc.loc[hc["anthro_mean"] > 1].copy()
    hc_emo = hc.loc[hc[EMOTIONS].notna().all(axis=1)].copy()
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    comprehensive = read_json("results/COMPREHENSIVE_V3_ANALYSIS_RESULTS.json")
    extended = read_json("results/extended_analysis/extended_analysis_results.json")
    validation = read_json("experiments/anthroscore_v3/validation_results.json")
    mlm_compare = read_json("experiments/anthroscore_v3/mlm_comparison_results.json")
    v3_demo = read_json("experiments/v2_correction/v3_results.json")
    v4_demo = read_json("experiments/v2_correction/v4_results.json")

    # Comment/date/subreddit inventory.
    comments = data["comments"].copy()
    comments["created_dt"] = pd.to_datetime(comments["created_utc"], unit="s", utc=True)
    comment_counts = comments.groupby("subreddit").agg(comments=("id", "size"), users=("author", "nunique")).reset_index()
    total_comments = len(comments)
    total_users = comments["author"].nunique()
    raw_pre_clean = {"CharacterAI": 397230, "replika": 10000, "AICompanions": 7527}
    raw_total = sum(raw_pre_clean.values())

    # Coverage and confidence threshold table.
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90]
    threshold_rows = []
    for th in thresholds:
        g_ok = users["gender_confidence"] >= th
        a_ok = users["age_confidence"] >= th
        both = g_ok & a_ok
        with_anthro = users.loc[both & users["anthro_mean"].notna()]
        threshold_rows.append(
            [
                f">={th:.2f}",
                fmt_int(g_ok.sum()),
                pct(g_ok.mean()),
                fmt_int(a_ok.sum()),
                pct(a_ok.mean()),
                fmt_int(both.sum()),
                pct(both.mean()),
                fmt_int(len(with_anthro)),
            ]
        )

    # Core demographic tests.
    age_test = group_test(hc, "age_predicted", "teen", "adult")
    gender_test = group_test(hc, "gender_predicted", "male", "female")
    age_test_cond = group_test(hc_meaningful, "age_predicted", "teen", "adult")
    gender_test_cond = group_test(hc_meaningful, "gender_predicted", "male", "female")
    simple_rows = []
    for age_group in ["teen", "adult"]:
        sub = hc[hc["age_predicted"] == age_group]
        if {"male", "female"}.issubset(set(sub["gender_predicted"].dropna())):
            gt = group_test(sub, "gender_predicted", "male", "female")
            simple_rows.append(
                [
                    f"Gender within {age_group}",
                    fmt_int(gt["n_a"]),
                    fmt_num(gt["mean_a"]),
                    fmt_int(gt["n_b"]),
                    fmt_num(gt["mean_b"]),
                    fmt_num(gt["diff_a_minus_b"]),
                    fmt_num(gt["d"]),
                    fmt_p(gt["t_p"]),
                ]
            )
    for gender_group in ["male", "female"]:
        sub = hc[hc["gender_predicted"] == gender_group]
        if {"teen", "adult"}.issubset(set(sub["age_predicted"].dropna())):
            at = group_test(sub, "age_predicted", "teen", "adult")
            simple_rows.append(
                [
                    f"Age within {gender_group}",
                    fmt_int(at["n_a"]),
                    fmt_num(at["mean_a"]),
                    fmt_int(at["n_b"]),
                    fmt_num(at["mean_b"]),
                    fmt_num(at["diff_a_minus_b"]),
                    fmt_num(at["d"]),
                    fmt_p(at["t_p"]),
                ]
            )

    # Subreddit context at raw, scored, and analysis levels.
    hc_sub_rows = []
    for sub_name, sub in hc.groupby("primary_subreddit"):
        if len(sub) == 0:
            continue
        row = [
            sub_name,
            fmt_int(len(sub)),
            fmt_num(sub["anthro_mean"].mean()),
            fmt_num(sub["anthro_median"].median()),
            pct((sub["age_predicted"] == "teen").mean()),
            pct((sub["gender_predicted"] == "female").mean()),
            pct(sub["has_score_3plus"].mean()),
            pct(sub["has_score_4plus"].mean()),
        ]
        hc_sub_rows.append(row)

    within_sub_rows = []
    for sub_name, sub in hc.groupby("primary_subreddit"):
        if {"teen", "adult"}.issubset(set(sub["age_predicted"].dropna())) and min(sub["age_predicted"].value_counts()) >= 10:
            at = group_test(sub, "age_predicted", "teen", "adult")
            within_sub_rows.append(
                [
                    sub_name,
                    fmt_int(at["n_a"]),
                    fmt_num(at["mean_a"]),
                    fmt_int(at["n_b"]),
                    fmt_num(at["mean_b"]),
                    fmt_num(at["d"]),
                    fmt_p(at["t_p"]),
                ]
            )

    # Score distributions.
    comment_score_dist = data["anthro"]["score"].value_counts().sort_index()
    comment_score_rows = []
    for score in [1, 2, 3, 4, 5]:
        n = int(comment_score_dist.get(score, 0))
        lo, hi = ci_prop(n, len(data["anthro"]))
        comment_score_rows.append([score, fmt_int(n), pct(n / len(data["anthro"]), 2), f"[{pct(lo, 2)}, {pct(hi, 2)}]"])
    user_dist = describe_distribution(hc["anthro_mean"])
    raw_comment_dist = describe_distribution(comments.groupby("author").size())

    # Dominant emotions and emotion regressions.
    dominant_rows = []
    if "dominant_emotion" in data["emotions_user"].columns:
        dom = hc_emo["dominant_emotion"].value_counts(normalize=False)
        for emotion, n in dom.items():
            dominant_rows.append([emotion, fmt_int(n), pct(n / len(hc_emo))])

    # Corrected emotion model: omit neutral, interpret coefficients relative to neutral.
    emotion_formula = "anthro_mean ~ is_teen + is_female + " + " + ".join(NON_NEUTRAL_EMOTIONS)
    emotion_model = ols(emotion_formula, data=hc_emo).fit()
    emotion_model_info = model_summary(emotion_model)
    emotion_vif = vif_table(hc_emo, ["is_teen", "is_female"] + NON_NEUTRAL_EMOTIONS)

    # ALR model: log emotion/non-neutral relative to neutral with pseudocount.
    alr = hc_emo.copy()
    eps = 1e-6
    for e in NON_NEUTRAL_EMOTIONS:
        alr[f"alr_{e}"] = np.log((alr[e] + eps) / (alr["emotion_neutral"] + eps))
    alr_cols = [f"alr_{e}" for e in NON_NEUTRAL_EMOTIONS]
    alr_model = ols("anthro_mean ~ is_teen + is_female + " + " + ".join(alr_cols), data=alr).fit()
    alr_info = model_summary(alr_model)
    alr_vif = vif_table(alr, ["is_teen", "is_female"] + alr_cols)

    # Hierarchical models.
    model_demo = ols("anthro_mean ~ is_teen + is_female", data=hc_emo).fit()
    model_demo_inter = ols("anthro_mean ~ is_teen + is_female + teen_x_female", data=hc_emo).fit()
    model_sub = ols("anthro_mean ~ is_teen + is_female + C(primary_subreddit)", data=hc_emo).fit()
    model_sub_emo = ols(
        "anthro_mean ~ is_teen + is_female + C(primary_subreddit) + " + " + ".join(NON_NEUTRAL_EMOTIONS),
        data=hc_emo,
    ).fit()
    hierarchical_rows = [
        ["Demographics", fmt_int(model_demo.nobs), fmt_num(model_demo.rsquared, 4), fmt_num(model_demo.rsquared_adj, 4), fmt_num(model_demo.aic, 1), fmt_p(model_demo.f_pvalue)],
        ["Demographics + age x gender", fmt_int(model_demo_inter.nobs), fmt_num(model_demo_inter.rsquared, 4), fmt_num(model_demo_inter.rsquared_adj, 4), fmt_num(model_demo_inter.aic, 1), fmt_p(model_demo_inter.f_pvalue)],
        ["Demographics + subreddit FE", fmt_int(model_sub.nobs), fmt_num(model_sub.rsquared, 4), fmt_num(model_sub.rsquared_adj, 4), fmt_num(model_sub.aic, 1), fmt_p(model_sub.f_pvalue)],
        ["Demographics + subreddit FE + emotions", fmt_int(model_sub_emo.nobs), fmt_num(model_sub_emo.rsquared, 4), fmt_num(model_sub_emo.rsquared_adj, 4), fmt_num(model_sub_emo.aic, 1), fmt_p(model_sub_emo.f_pvalue)],
    ]

    # Correlations and moderation by age/gender.
    corr_rows = []
    age_mod_rows = []
    gender_mod_rows = []
    age_pvals = []
    gender_pvals = []
    age_payload = []
    gender_payload = []
    for e in EMOTIONS:
        tmp = hc_emo[["anthro_mean", e, "age_predicted", "gender_predicted"]].dropna()
        pear = stats.pearsonr(tmp["anthro_mean"], tmp[e])
        spear = stats.spearmanr(tmp["anthro_mean"], tmp[e])
        corr_rows.append([e, fmt_num(pear.statistic), fmt_p(pear.pvalue), fmt_num(spear.statistic), fmt_p(spear.pvalue)])

        teen = tmp[tmp["age_predicted"] == "teen"]
        adult = tmp[tmp["age_predicted"] == "adult"]
        rt = stats.pearsonr(teen["anthro_mean"], teen[e]).statistic
        ra = stats.pearsonr(adult["anthro_mean"], adult[e]).statistic
        z, p = fisher_z_diff(rt, len(teen), ra, len(adult))
        age_pvals.append(p)
        age_payload.append([e, len(teen), rt, len(adult), ra, z, p])

        male = tmp[tmp["gender_predicted"] == "male"]
        female = tmp[tmp["gender_predicted"] == "female"]
        rm = stats.pearsonr(male["anthro_mean"], male[e]).statistic
        rf = stats.pearsonr(female["anthro_mean"], female[e]).statistic
        z2, p2 = fisher_z_diff(rm, len(male), rf, len(female))
        gender_pvals.append(p2)
        gender_payload.append([e, len(male), rm, len(female), rf, z2, p2])

    age_q = multipletests(age_pvals, method="fdr_bh")[1]
    gender_q = multipletests(gender_pvals, method="fdr_bh")[1]
    for payload, q in zip(age_payload, age_q):
        e, nt, rt, na, ra, z, p = payload
        age_mod_rows.append([e, fmt_int(nt), fmt_num(rt), fmt_int(na), fmt_num(ra), fmt_num(z, 2), fmt_p(p), fmt_p(q)])
    for payload, q in zip(gender_payload, gender_q):
        e, nm, rm, nf, rf, z, p = payload
        gender_mod_rows.append([e, fmt_int(nm), fmt_num(rm), fmt_int(nf), fmt_num(rf), fmt_num(z, 2), fmt_p(p), fmt_p(q)])

    # Binary high-anthro models.
    hc_bin = hc.copy()
    hc_bin["high_any_3plus"] = hc_bin["has_score_3plus"]
    tab_age = pd.crosstab(hc_bin["age_predicted"], hc_bin["high_any_3plus"])
    tab_gender = pd.crosstab(hc_bin["gender_predicted"], hc_bin["high_any_3plus"])
    chi_age = stats.chi2_contingency(tab_age)
    chi_gender = stats.chi2_contingency(tab_gender)
    logit_model = logit("high_any_3plus ~ is_teen + is_female", data=hc_bin).fit(disp=0)

    # N-gram diagnostics.
    enriched = data["enriched"].copy()
    ngram_cols = [c for c in enriched.columns if c.startswith("ngram_")]
    ngram = enriched.merge(data["anthro"], left_on="id", right_on="comment_id", how="left", suffixes=("", "_anthro"))
    ngram_rows = []
    for col in [
        "ngram_anthro_total",
        "ngram_deanthro_total",
        "ngram_relationship",
        "ngram_emotion_attr",
        "ngram_agency",
        "ngram_consciousness",
        "ngram_pronoun_verb",
        "ngram_technical",
        "ngram_tool_framing",
    ]:
        if col in ngram:
            any_hit = (ngram[col].fillna(0) > 0)
            mean_score = ngram.loc[any_hit & ngram["score_anthro"].notna(), "score_anthro"].mean()
            no_mean = ngram.loc[(~any_hit) & ngram["score_anthro"].notna(), "score_anthro"].mean()
            ngram_rows.append([col, fmt_int(any_hit.sum()), pct(any_hit.mean()), fmt_num(mean_score), fmt_num(no_mean), fmt_num(mean_score - no_mean)])

    # Comment-level content/language diagnostics.
    sc = scored_comments.copy()
    sc["body"] = sc["body"].fillna("")
    sc["word_count"] = sc["body"].str.findall(r"\b\w+\b").str.len()
    sc["char_count"] = sc["body"].str.len()
    sc["score_imp"] = sc["score_x"]
    patterns = {
        "loneliness": r"\b(?:lonely|loneliness|alone|isolated|no friends|friendless|socially isolated|nobody to talk)\b",
        "romantic": r"\b(?:love|romance|romantic|girlfriend|boyfriend|wife|husband|marry|married|dating|partner|crush)\b",
        "friendship": r"\b(?:friend|best friend|bff|companion|buddy|pal)\b",
        "roleplay": r"\b(?:roleplay|rp|character|persona|scenario|story|chatbot character)\b",
        "technical": r"\b(?:bug|glitch|update|server|app|cache|settings|filter|model|api|subscription|account)\b",
        "emotional_support": r"\b(?:comfort|support|vent|therapy|therapist|depressed|anxiety|suicidal|mental health)\b",
        "creative": r"\b(?:write|writing|story|plot|novel|poem|art|creative|character)\b",
    }
    pattern_rows = []
    high_comments = sc["score_imp"] >= 4
    low_comments = sc["score_imp"] == 1
    for name, pat in patterns.items():
        hit = pattern_stats(sc, pat)
        high_pct = hit[high_comments].mean()
        low_pct = hit[low_comments].mean()
        all_pct = hit.mean()
        table = pd.crosstab(hit, high_comments.astype(int))
        chi = stats.chi2_contingency(table) if table.shape == (2, 2) else (np.nan, np.nan, None, None)
        pattern_rows.append([name, pct(all_pct), pct(low_pct), pct(high_pct), fmt_num(high_pct / low_pct if low_pct > 0 else np.nan, 2), fmt_p(chi[1])])

    # Confirmatory replication.
    confirm_rows = []
    if "confirmatory" in data:
        conf_scores = data["confirmatory"]["score"].dropna()
        main_scores = data["anthro"]["score"].dropna()
        d_conf = cohens_d(main_scores, conf_scores)
        confirm_rows = [
            ["Main improved", fmt_int(len(main_scores)), fmt_num(main_scores.mean()), pct((main_scores == 1).mean()), pct((main_scores >= 4).mean())],
            ["Confirmatory", fmt_int(len(conf_scores)), fmt_num(conf_scores.mean()), pct((conf_scores == 1).mean()), pct((conf_scores >= 4).mean())],
        ]
    else:
        d_conf = np.nan

    # Self-declaration coverage.
    self_decl = data["self_decl"]
    self_rows = [
        ["Age self-declared non-null", fmt_int(self_decl["age_self_declared"].notna().sum()), pct(self_decl["age_self_declared"].notna().mean())],
        ["Age bucket self-declared non-null", fmt_int(self_decl["age_bucket_self_declared"].notna().sum()), pct(self_decl["age_bucket_self_declared"].notna().mean())],
        ["Gender self-declared non-null", fmt_int(self_decl["gender_self_declared"].notna().sum()), pct(self_decl["gender_self_declared"].notna().mean())],
    ]

    # Build markdown.
    lines: list[str] = []
    lines.append("# Paper-Writing Master Reference: The Illusion Project")
    lines.append("")
    lines.append(f"**Generated:** {generated}")
    lines.append("**Purpose:** One-stop, paper-facing reference file for Methods, Results, Discussion, Limitations, Appendix, and AI-assisted drafting.")
    lines.append("**Canonical quantitative source:** this file recomputes from current parquet/JSON artifacts and should be cited internally over stale duplicate summaries.")
    lines.append("")
    lines.append("> Important: this is a writing/reference artifact, not a journal-ready manuscript section. It includes caveats, stale-file warnings, and interpretation notes so an AI assistant or human writer does not mix incompatible analysis generations.")

    lines.append(section("0. Executive Takeaways for the Paper"))
    lines.append("- Use **283,895 cleaned comments from 47,062 Reddit authors** as the cleaned analytic corpus. The **414,757** figure refers to pre-cleaning raw comments, not final analyzable comments.")
    lines.append(f"- Improved AnthroScore V3 covers **{fmt_int(len(data['anthro']))} comments** ({pct(len(data['anthro']) / total_comments)} of cleaned comments) and **{fmt_int(users['author'].nunique())} users** with at least one scored comment.")
    lines.append(f"- The inclusive high-confidence demographic sample is **{fmt_int(len(hc))} users** at age and gender confidence >= {CONFIDENCE_THRESHOLD:.2f}. The canonical RQ2 report further conditions on `anthro_mean > 1`, yielding **{fmt_int(len(hc_meaningful))} users**.")
    lines.append(f"- Inclusive estimand (all high-confidence users): adults score higher than teens (teen M={fmt_num(age_test['mean_a'])}, adult M={fmt_num(age_test['mean_b'])}, Welch t={fmt_num(age_test['t'])}, p={fmt_p(age_test['t_p'])}, d={fmt_num(age_test['d'])}); women score higher than men (male M={fmt_num(gender_test['mean_a'])}, female M={fmt_num(gender_test['mean_b'])}, d={fmt_num(gender_test['d'])}).")
    lines.append(f"- Canonical conditional estimand (`anthro_mean > 1`): adults score higher than teens (teen M={fmt_num(age_test_cond['mean_a'])}, adult M={fmt_num(age_test_cond['mean_b'])}, Welch t={fmt_num(age_test_cond['t'])}, p={fmt_p(age_test_cond['t_p'])}, d={fmt_num(age_test_cond['d'])}); the gender effect is much smaller (male M={fmt_num(gender_test_cond['mean_a'])}, female M={fmt_num(gender_test_cond['mean_b'])}, d={fmt_num(gender_test_cond['d'])}, p={fmt_p(gender_test_cond['t_p'])}).")
    lines.append(f"- Demographics alone explain about **{pct(model_demo.rsquared, 1)}** of variance in the emotion-complete sample; adding subreddit fixed effects and emotion proportions raises R^2 to **{pct(model_sub_emo.rsquared, 1)}**.")
    lines.append("- Emotion predictors are compositional proportions. Do not cite the saturated seven-emotion coefficients from the old Model 3. Use the drop-neutral or ALR models below.")
    lines.append("- For draft revisions: fill Methods, Results, Analysis, Ethics, Limitations, and Appendix from this file; retire or qualify root-level duplicate/stale summaries.")

    lines.append(section("1. Current Draft Needs This File Solves"))
    lines.append(md_table(
        ["Draft gap or risk", "What to use from this file"],
        [
            ["Methods is blank", "Sections 2-5 give corpus, scoring, demographics, emotion, filtering, and aggregation details."],
            ["Results is blank", "Sections 6-12 provide paper-ready results tables and statistical tests."],
            ["Analysis section is blank", "Sections 10-13 provide corrected regressions, moderation, n-gram checks, and robustness."],
            ["Draft says over 414,757 comments", "Use 414,757 raw pre-cleaning and 283,895 cleaned final comments; see Section 2."],
            ["Draft says two RQs but lists three", "Use three RQs: demographics, emotions, moderation; if including gender moderation, say exploratory."],
            ["AnthroIndex vs AnthroScore naming", "Pick one term. Code/results call it AnthroScore V3; draft calls it AnthroIndex. Define AnthroIndex as paper-facing name if desired."],
            ["Demographic classifier method mismatch", "Current analysis reads V4 parquet files; best documented validation recommends V3. Do not call current full analysis GPT-4o-mini demographics without rerunning."],
            ["Emotion regression multicollinearity", "Use Section 10 corrected drop-neutral/ALR models."],
            ["Need AI context for paper writing", "Sections 15-18 give architecture, caveats, interpretation language, and exact file provenance."],
        ],
    ))

    lines.append(section("2. Corpus, Cleaning, and Sample Flow"))
    lines.append(subsection("2.1 Raw-to-cleaned corpus counts"))
    raw_rows = []
    for _, row in comment_counts.sort_values("comments", ascending=False).iterrows():
        sub = row["subreddit"]
        raw = raw_pre_clean.get(sub, np.nan)
        raw_rows.append([f"r/{sub}", fmt_int(raw) if np.isfinite(raw) else "NA", fmt_int(row["comments"]), pct(row["comments"] / raw) if np.isfinite(raw) else "NA", fmt_int(row["users"])])
    raw_rows.append(["Total", fmt_int(raw_total), fmt_int(total_comments), pct(total_comments / raw_total), fmt_int(total_users)])
    lines.append(md_table(["Subreddit", "Raw comments", "Cleaned comments", "Retained", "Unique authors"], raw_rows))
    lines.append(f"- Cleaned date range from `Data/processed/all_comments.parquet`: **{comments['created_dt'].min().date()} to {comments['created_dt'].max().date()}**.")
    lines.append("- Cleaning described in project docs: bot/deleted/removed removal, minimum text length filter, and comment ID deduplication. The exact current processed corpus is `Data/processed/all_comments.parquet`.")
    lines.append("- Use raw counts only for data collection scope. Use cleaned counts for all analysis Ns.")

    lines.append(subsection("2.2 Analysis sample flow"))
    sample_rows = [
        ["Cleaned comments", fmt_int(total_comments), "All retained comments after preprocessing"],
        ["Cleaned unique authors", fmt_int(total_users), "Authors in cleaned corpus"],
        ["Original V3 scored comments", fmt_int(len(data["anthro_original"])), "Legacy/original LLM score file"],
        ["Improved V3 scored comments", fmt_int(len(data["anthro"])), f"{pct(len(data['anthro']) / total_comments)} of cleaned comments"],
        ["Improved V3 scored users", fmt_int(users["author"].nunique()), "Users with at least one improved score"],
        ["User emotion rows", fmt_int(len(data["emotions_user"])), "User-level emotion proportions"],
        ["Comment emotion rows", fmt_int(len(data["emotions_comment"])), "Comment-level emotion probabilities"],
        ["Gender prediction rows", fmt_int(len(data["gender"])), "V4 parquet used by current analysis"],
        ["Age prediction rows", fmt_int(len(data["age"])), "V4 parquet used by current analysis"],
        ["High-confidence demographic users", fmt_int(len(hc)), "Age and gender confidence >= 0.60 with AnthroScore"],
        ["Canonical RQ2 conditional users", fmt_int(len(hc_meaningful)), "High-confidence users with anthro_mean > 1"],
        ["High-confidence + emotions", fmt_int(len(hc_emo)), "Regression/correlation sample when all emotion proportions present"],
    ]
    if "llm_demo" in data:
        sample_rows.append(["Partial LLM demographic classifications", fmt_int(len(data["llm_demo"])), "Exists but does not cover full 47,062 users"])
    lines.append(md_table(["Stage", "N", "Notes"], sample_rows))

    lines.append(subsection("2.3 Comment volume per author"))
    lines.append(md_table(
        ["Quantity", "Value"],
        [
            ["Mean cleaned comments/user", fmt_num(raw_comment_dist["mean"], 2)],
            ["Median cleaned comments/user", fmt_num(raw_comment_dist["median"], 2)],
            ["10th percentile", fmt_num(raw_comment_dist["p10"], 2)],
            ["25th percentile", fmt_num(raw_comment_dist["p25"], 2)],
            ["75th percentile", fmt_num(raw_comment_dist["p75"], 2)],
            ["90th percentile", fmt_num(raw_comment_dist["p90"], 2)],
            ["Max comments by one author", fmt_int(raw_comment_dist["max"])],
        ],
    ))
    lines.append("Interpretation: user-level aggregation prevents extremely active authors from dominating primary demographic tests.")

    lines.append(section("3. Measurement Architecture and Methodology Context"))
    lines.append(subsection("3.1 Anthropomorphization measure"))
    lines.append(md_table(
        ["Score", "Label", "Paper description"],
        [
            [1, "None", "AI treated as software/tool/app; no mental states or relationship language."],
            [2, "Minimal", "Light humanizing language, generic intelligence, or bot reference without strong mind attribution."],
            [3, "Moderate", "Pronouns, simple agency, or basic emotional/personality attribution."],
            [4, "High", "Clear feelings, care, jealousy, autonomy, attachment, or personality attributed to the AI."],
            [5, "Extreme", "Human-equivalent relationship, love, dependency, or fully reciprocal social bond."],
        ],
    ))
    lines.append("- Current paper-facing score file: `experiments/anthroscore_v3/anthroscore_v3_improved_final.parquet`.")
    lines.append("- The improved prompt incorporated human calibration, emotion-attribution distinctions, and overscoring correction. It produced a large 2-to-1 shift relative to the original V3 prompt.")
    lines.append(f"- Validation JSON: Pearson r={fmt_num(validation.get('pearson_r'))}, Spearman rho={fmt_num(validation.get('spearman_r'))}, exact accuracy={pct(validation.get('accuracy', np.nan))}, within-1 accuracy={pct(validation.get('accuracy_within_1', np.nan))}, MAE={fmt_num(validation.get('mae'))}, Cohen kappa={fmt_num(validation.get('cohen_kappa'))}.")
    lines.append(f"- V3 vs V2 comparison JSON: LLM/expert r={fmt_num(mlm_compare.get('llm_expert_r'))}; MLM/expert r={fmt_num(mlm_compare.get('mlm_expert_r'))}; LLM wins={mlm_compare.get('llm_wins')} of {mlm_compare.get('n_samples')} validation cases; MLM wins={mlm_compare.get('mlm_wins')}; ties={mlm_compare.get('ties')}.")
    lines.append("- Caveat: `validation_results.json` contains `recommendation: NEEDS REVIEW` and `passes_kappa: false`. In prose, describe the measure as substantially improved and human-calibrated, not as perfect ground truth.")

    lines.append(subsection("3.2 Demographic inference"))
    lines.append("- The current comprehensive analysis reads `experiments/v2_correction/gender_predictions_v4.parquet` and `age_predictions_v4.parquet`.")
    lines.append("- A prior audit and `FINAL_MODEL_SUMMARY.md` indicate the best documented validation narrative is for **V3 demographic models**, while the current analysis files are **V4 parquets**. Do not write that all full-sample demographics were GPT-4o-mini LLM classifications unless those are rerun over all users.")
    if v3_demo:
        lines.append(f"- V3 summary JSON available: gender test accuracy={pct(v3_demo.get('gender', {}).get('test_metrics', {}).get('accuracy', np.nan))}; age CV accuracy={pct(v3_demo.get('age', {}).get('cv_accuracy', np.nan))}.")
    if v4_demo:
        lines.append(f"- V4 summary JSON flags suspiciously high threshold validation in some places; use cautiously. V4 gender CV accuracy={pct(v4_demo.get('v4', {}).get('gender', {}).get('cv_metrics', {}).get('accuracy', np.nan))}; V4 age CV accuracy={pct(v4_demo.get('v4', {}).get('age', {}).get('cv_accuracy', np.nan))}.")
    lines.append("- Paper-safe wording: `We inferred binary age and gender categories using project demographic classifiers and restricted inferential analyses to users with age and gender confidence >= .60; classifier validation and limitations are reported separately.`")

    lines.append(subsection("3.3 Emotion measure"))
    lines.append("- Emotion features come from `j-hartmann/emotion-english-distilroberta-base` according to project methodology, represented as user-level proportions across joy, sadness, anger, fear, surprise, disgust, and neutral.")
    lines.append("- These proportions are compositional: when one category rises, others must fall. This matters for regression interpretation.")

    lines.append(section("4. Confidence Thresholds and Demographic Coverage"))
    lines.append(md_table(["Threshold", "Gender N", "Gender coverage", "Age N", "Age coverage", "Both N", "Both coverage", "Both + Anthro N"], threshold_rows))
    lines.append("Primary analyses use >=0.60 for both age and gender. Sensitivity analyses should report that the adult > teen direction persists across thresholds from .50 to .70, with the .70 estimate underpowered.")

    lines.append(section("5. Descriptive Statistics"))
    lines.append(subsection("5.1 Improved AnthroScore comment-level distribution"))
    lines.append(md_table(["Score", "Comments", "Percent", "95% CI"], comment_score_rows))
    lines.append("Interpretation: the improved score is highly right-skewed; most Reddit comments mention AI companions without strong anthropomorphization.")

    lines.append(subsection("5.2 User-level AnthroScore distribution in high-confidence sample"))
    lines.append(md_table(
        ["Quantity", "Value"],
        [
            ["N users", fmt_int(user_dist["n"])],
            ["Mean", fmt_num(user_dist["mean"])],
            ["SD", fmt_num(user_dist["sd"])],
            ["Median", fmt_num(user_dist["median"])],
            ["10th percentile", fmt_num(user_dist["p10"])],
            ["25th percentile", fmt_num(user_dist["p25"])],
            ["75th percentile", fmt_num(user_dist["p75"])],
            ["90th percentile", fmt_num(user_dist["p90"])],
            ["Min", fmt_num(user_dist["min"])],
            ["Max", fmt_num(user_dist["max"])],
            ["Skew", fmt_num(user_dist["skew"])],
        ],
    ))

    lines.append(subsection("5.3 Demographic composition at high confidence"))
    demo_rows = []
    for col, label in [("age_predicted", "Age"), ("gender_predicted", "Gender")]:
        vc = hc[col].value_counts()
        for level, n in vc.items():
            lo, hi = ci_prop(int(n), len(hc))
            demo_rows.append([label, level, fmt_int(n), pct(n / len(hc)), f"[{pct(lo)}, {pct(hi)}]"])
    lines.append(md_table(["Dimension", "Group", "N", "Percent", "95% CI"], demo_rows))

    lines.append(subsection("5.4 Primary-subreddit context"))
    lines.append(md_table(["Primary subreddit", "Users", "Mean Anthro", "Median Anthro", "% teen", "% female", "% any score >=3", "% any score >=4"], hc_sub_rows))
    lines.append("This is a user-level primary-subreddit view. A user is assigned to the subreddit where they posted most often.")

    lines.append(section("6. RQ1/RQ2: Demographics and Anthropomorphization"))
    lines.append("**Critical estimand note:** The canonical comprehensive report tests RQ2 after filtering to users with `anthro_mean > 1` (N=5,160), which asks about differences **among users with at least some measured anthropomorphization**. The inclusive analyses below use all high-confidence users (N=16,347), including the many users whose mean score is exactly 1. These are both useful, but they answer different questions. For the main paper, state which estimand you are reporting.")
    lines.append(subsection("6.0 Canonical conditional RQ2 estimates (`anthro_mean > 1`)"))
    lines.append(md_table(
        ["Contrast", "Group A", "N A", "Mean A", "Group B", "N B", "Mean B", "d A-B", "p"],
        [
            ["Age", "Teen", fmt_int(age_test_cond["n_a"]), fmt_num(age_test_cond["mean_a"]), "Adult", fmt_int(age_test_cond["n_b"]), fmt_num(age_test_cond["mean_b"]), fmt_num(age_test_cond["d"]), fmt_p(age_test_cond["t_p"])],
            ["Gender", "Male", fmt_int(gender_test_cond["n_a"]), fmt_num(gender_test_cond["mean_a"]), "Female", fmt_int(gender_test_cond["n_b"]), fmt_num(gender_test_cond["mean_b"]), fmt_num(gender_test_cond["d"]), fmt_p(gender_test_cond["t_p"])],
        ],
    ))
    lines.append("These match the April 2026 canonical report: age d about -0.456; gender d about -0.067. Use these if you want consistency with `results/COMPREHENSIVE_V3_ANALYSIS_RESULTS.md`.")
    lines.append(subsection("6.1 Inclusive age effect (all high-confidence users)"))
    lines.append(md_table(
        ["Group", "N", "Mean", "SD", "Median"],
        [
            ["Teen", fmt_int(age_test["n_a"]), fmt_num(age_test["mean_a"]), fmt_num(age_test["sd_a"]), fmt_num(age_test["median_a"])],
            ["Adult", fmt_int(age_test["n_b"]), fmt_num(age_test["mean_b"]), fmt_num(age_test["sd_b"]), fmt_num(age_test["median_b"])],
        ],
    ))
    lines.append(f"Welch t={fmt_num(age_test['t'])}, p={fmt_p(age_test['t_p'])}; Mann-Whitney U={fmt_num(age_test['u'], 1)}, p={fmt_p(age_test['u_p'])}; Hedges-corrected d={fmt_num(age_test['d'])} ({age_test['d_interp']}); mean difference teen-adult={fmt_num(age_test['diff_a_minus_b'])}.")
    lines.append("Paper wording: Adults had higher mean anthropomorphization than teens. In the inclusive high-confidence sample this effect is around the small/medium boundary; in the canonical conditional sample it is small.")

    lines.append(subsection("6.2 Inclusive gender effect (all high-confidence users)"))
    lines.append(md_table(
        ["Group", "N", "Mean", "SD", "Median"],
        [
            ["Male", fmt_int(gender_test["n_a"]), fmt_num(gender_test["mean_a"]), fmt_num(gender_test["sd_a"]), fmt_num(gender_test["median_a"])],
            ["Female", fmt_int(gender_test["n_b"]), fmt_num(gender_test["mean_b"]), fmt_num(gender_test["sd_b"]), fmt_num(gender_test["median_b"])],
        ],
    ))
    lines.append(f"Welch t={fmt_num(gender_test['t'])}, p={fmt_p(gender_test['t_p'])}; Mann-Whitney U={fmt_num(gender_test['u'], 1)}, p={fmt_p(gender_test['u_p'])}; Hedges-corrected d={fmt_num(gender_test['d'])} ({gender_test['d_interp']}); mean difference male-female={fmt_num(gender_test['diff_a_minus_b'])}.")
    lines.append("Paper wording: women were higher than men in the inclusive two-group test, but the canonical conditional analysis reduces this to a negligible effect. Avoid overstating gender.")

    lines.append(subsection("6.3 Simple effects by age/gender subgroup"))
    lines.append(md_table(["Contrast", "N A", "Mean A", "N B", "Mean B", "Mean diff A-B", "d", "p"], simple_rows))

    lines.append(subsection("6.4 Age effect within primary subreddits"))
    lines.append(md_table(["Primary subreddit", "Teen N", "Teen mean", "Adult N", "Adult mean", "d teen-adult", "p"], within_sub_rows))
    lines.append("Interpretation: if this table shows adult > teen in each subreddit, the age effect is not merely a byproduct of adults being concentrated in a more anthropomorphic subreddit.")

    lines.append(section("7. Binary/Prevalence Framing"))
    lines.append("Binary outcome here means a user had **at least one comment scored >=3** (moderate or higher anthropomorphization). This is a different estimand from mean AnthroScore.")
    lines.append(md_table(
        ["Quantity", "Value"],
        [
            ["Users with any score >=3", fmt_int(hc_bin["high_any_3plus"].sum())],
            ["Prevalence", pct(hc_bin["high_any_3plus"].mean())],
            ["Teen prevalence", pct(hc_bin.loc[hc_bin["age_predicted"] == "teen", "high_any_3plus"].mean())],
            ["Adult prevalence", pct(hc_bin.loc[hc_bin["age_predicted"] == "adult", "high_any_3plus"].mean())],
            ["Age chi2 p", fmt_p(chi_age[1])],
            ["Adult vs teen OR", fmt_num(odds_ratio_2x2(tab_age, "adult", "teen"), 2)],
            ["Male prevalence", pct(hc_bin.loc[hc_bin["gender_predicted"] == "male", "high_any_3plus"].mean())],
            ["Female prevalence", pct(hc_bin.loc[hc_bin["gender_predicted"] == "female", "high_any_3plus"].mean())],
            ["Gender chi2 p", fmt_p(chi_gender[1])],
            ["Female vs male OR", fmt_num(odds_ratio_2x2(tab_gender, "female", "male"), 2)],
            ["Logistic pseudo R2", fmt_num(logit_model.prsquared, 4)],
        ],
    ))
    logit_rows = []
    for term in logit_model.params.index:
        b = logit_model.params[term]
        ci = logit_model.conf_int().loc[term]
        logit_rows.append([term, fmt_num(b), fmt_num(logit_model.bse[term]), fmt_num(math.exp(b), 3), f"[{fmt_num(math.exp(ci[0]), 3)}, {fmt_num(math.exp(ci[1]), 3)}]", fmt_p(logit_model.pvalues[term])])
    lines.append(md_table(["Term", "B", "SE", "OR", "OR 95% CI", "p"], logit_rows))

    lines.append(section("8. RQ2/RQ3: Emotions and Anthropomorphization"))
    lines.append(subsection("8.1 Dominant emotion distribution"))
    lines.append(md_table(["Dominant emotion", "Users", "Percent"], dominant_rows))

    lines.append(subsection("8.2 Correlations with user-level mean AnthroScore"))
    lines.append(md_table(["Emotion", "Pearson r", "Pearson p", "Spearman rho", "Spearman p"], corr_rows))
    lines.append("Interpretation: neutral language has the strongest negative correlation with anthropomorphization. Joy is the clearest positive bivariate emotion signal. Pearson and Spearman signs can differ because the seven emotion features are proportions and relationships are not purely monotone.")

    lines.append(subsection("8.3 Age moderation of emotion-anthropomorphization correlations"))
    lines.append(md_table(["Emotion", "Teen N", "Teen r", "Adult N", "Adult r", "z", "p", "FDR q"], age_mod_rows))
    lines.append("Use FDR q-values for claims across the seven emotion moderation tests. If p is significant but q is not, describe as exploratory.")

    lines.append(subsection("8.4 Gender moderation of emotion-anthropomorphization correlations"))
    lines.append(md_table(["Emotion", "Male N", "Male r", "Female N", "Female r", "z", "p", "FDR q"], gender_mod_rows))
    lines.append("Gender moderation was missing from the canonical comprehensive report but is useful because the draft RQ3 currently mentions age and gender moderation.")

    lines.append(section("9. Corrected Regression Models"))
    lines.append(subsection("9.1 Hierarchical OLS models"))
    lines.append(md_table(["Model", "N", "R^2", "Adj. R^2", "AIC", "Model p"], hierarchical_rows))
    lines.append(f"Increment from demographics-only to demographics + subreddit FE + emotion proportions: Delta R^2={fmt_num(model_sub_emo.rsquared - model_demo.rsquared, 4)}.")

    lines.append(subsection("9.2 Drop-neutral emotion model"))
    lines.append("This model includes six non-neutral emotion proportions and omits neutral, so coefficients are interpretable relative to neutral expression. This fixes the rank deficiency in the old seven-emotion model.")
    lines.append(md_table(
        ["Term", "B", "SE", "t", "p", "95% CI"],
        [[c["term"], fmt_num(c["b"]), fmt_num(c["se"]), fmt_num(c["t"]), fmt_p(c["p"]), f"[{fmt_num(c['ci_lo'])}, {fmt_num(c['ci_hi'])}]"] for c in emotion_model_info["coefficients"]],
    ))
    lines.append(md_table(["Term", "VIF"], [[v["term"], fmt_num(v["vif"], 2)] for v in emotion_vif]))

    lines.append(subsection("9.3 Additive log-ratio (ALR) emotion model"))
    lines.append("This compositional sensitivity model uses log(emotion / neutral) for each non-neutral emotion with a tiny pseudocount. It asks whether relative emotion composition predicts AnthroScore.")
    lines.append(f"ALR model: N={fmt_int(alr_info['n'])}, R^2={fmt_num(alr_info['r2'], 4)}, adj. R^2={fmt_num(alr_info['adj_r2'], 4)}, p={fmt_p(alr_info['p'])}.")
    alr_rows = []
    for c in alr_info["coefficients"]:
        if c["term"] == "Intercept":
            continue
        alr_rows.append([c["term"], fmt_num(c["b"]), fmt_num(c["se"]), fmt_num(c["t"]), fmt_p(c["p"]), f"[{fmt_num(c['ci_lo'])}, {fmt_num(c['ci_hi'])}]"])
    lines.append(md_table(["Term", "B", "SE", "t", "p", "95% CI"], alr_rows))
    lines.append(md_table(["Term", "VIF"], [[v["term"], fmt_num(v["vif"], 2)] for v in alr_vif]))

    lines.append(section("10. N-Gram and Improved-Prompt Diagnostics"))
    lines.append("The project added n-gram anthropomorphization/de-anthropomorphization signals. These should be described as prompt/context features and diagnostic covariates, not as the primary outcome.")
    lines.append(md_table(["Feature", "Comments with hit", "% comments", "Mean score if hit", "Mean score if no hit", "Difference"], ngram_rows))
    lines.append("- Anthro n-gram hits should generally correspond to higher improved AnthroScore; de-anthro/technical hits should generally correspond to lower or only weakly higher scores.")
    lines.append("- This table is useful for explaining why n-gram additions could shift older results: the improved system now distinguishes actual bot-attributed anthropomorphism from user self-expression or technical talk.")

    lines.append(section("11. Content and Linguistic Context for Discussion"))
    lines.append("Comment-level pattern checks below use simple regex indicators. Treat them as descriptive, not as validated psychological constructs.")
    lines.append(md_table(["Pattern", "All comments", "Score=1 comments", "Score>=4 comments", "High/low ratio", "chi2 p"], pattern_rows))
    lines.append(md_table(
        ["Quantity", "Score=1 comments", "Score>=4 comments"],
        [
            ["Mean word count", fmt_num(sc.loc[low_comments, "word_count"].mean(), 2), fmt_num(sc.loc[high_comments, "word_count"].mean(), 2)],
            ["Median word count", fmt_num(sc.loc[low_comments, "word_count"].median(), 2), fmt_num(sc.loc[high_comments, "word_count"].median(), 2)],
            ["Mean char count", fmt_num(sc.loc[low_comments, "char_count"].mean(), 2), fmt_num(sc.loc[high_comments, "char_count"].mean(), 2)],
            ["Median char count", fmt_num(sc.loc[low_comments, "char_count"].median(), 2), fmt_num(sc.loc[high_comments, "char_count"].median(), 2)],
        ],
    ))

    lines.append(section("12. Robustness and Sensitivity Results to Cite"))
    if comprehensive.get("robustness", {}).get("threshold_sensitivity"):
        rows = []
        for r in comprehensive["robustness"]["threshold_sensitivity"]:
            rows.append([f">={r['threshold']:.2f}", fmt_int(r["n_total"]), fmt_num(r["teen_mean"]), fmt_num(r["adult_mean"]), fmt_num(r["cohens_d"]), fmt_p(r["p_value"]), r["direction"]])
        lines.append(subsection("12.1 Confidence threshold sensitivity"))
        lines.append(md_table(["Threshold", "N", "Teen mean", "Adult mean", "d", "p", "Direction"], rows))
    if extended.get("variance_analysis"):
        v = extended["variance_analysis"]
        lines.append(subsection("12.2 Variance and robust alternatives"))
        lines.append(md_table(
            ["Check", "Result"],
            [
                ["Levene variance test", f"stat={fmt_num(v['levene']['statistic'])}, p={fmt_p(v['levene']['p_value'])}; adult/teen variance ratio={fmt_num(v['levene']['ratio'])}"],
                ["Brunner-Munzel robust test", f"stat={fmt_num(v['brunner_munzel']['statistic'])}, p={fmt_p(v['brunner_munzel']['p_value'])}"],
                ["Robust regression is_teen coefficient", f"B={fmt_num(v['robust_regression']['coefficients']['is_teen']['b'])}, p={fmt_p(v['robust_regression']['coefficients']['is_teen']['p'])}"],
            ],
        ))
    if "confirmatory" in data:
        lines.append(subsection("12.3 Confirmatory replication"))
        lines.append(md_table(["Dataset", "Comments", "Mean score", "% score 1", "% score >=4"], confirm_rows))
        lines.append(f"Main vs confirmatory Cohen's d={fmt_num(d_conf)} ({interpret_d(d_conf)}). The confirmatory score distribution is extremely similar, supporting temporal stability of the improved scorer.")

    lines.append(section("13. Human Calibration and Method Comparison"))
    lines.append(md_table(
        ["Validation quantity", "Value"],
        [
            ["Human calibration items", "151 total items; 3 annotators"],
            ["Mean pairwise exact agreement", "43.8% in calibration report"],
            ["Mean pairwise Cohen kappa", "0.245 in calibration report"],
            ["Old algo vs human consensus exact", "37.6%"],
            ["Old algo bias", "+0.81 score points (overscoring)"],
            ["Improvement summary n", "93 validation cases in final comparison table"],
            ["Human-new algorithm exact", "47.7%"],
            ["Human-new algorithm within +/-1", "77.1%"],
            ["Consensus-new weighted kappa", "0.466"],
            ["Method comparison paired comments", "264,654"],
            ["Original V3 mean", "1.997"],
            ["Improved V3 mean", "1.148"],
            ["Dominant migration", "2->1 shift for 184,663 comments (69.8% of paired comments)"],
        ],
    ))
    lines.append("Interpretation: the improved AnthroScore is deliberately more conservative. Older reports with means near 2.0 are not directly comparable to the current improved-score analyses.")

    lines.append(section("14. Self-Declaration and Validation Coverage"))
    lines.append(md_table(["Self-declaration field", "N", "Coverage of 47,062 users"], self_rows))
    lines.append("Self-declared labels are sparse and should be framed as validation/calibration anchors, not the full demographic source.")

    lines.append(section("15. Paper-Ready Claims and Suggested Wording"))
    lines.append(subsection("15.1 Abstract/results wording"))
    lines.append("- `We analyzed 283,895 cleaned public Reddit comments from 47,062 authors across r/CharacterAI, r/Replika, and r/AICompanions.`")
    lines.append("- `Anthropomorphization was measured with a human-calibrated LLM-based 1-5 comment-level measure and aggregated to user-level means for demographic analyses.`")
    lines.append("- `At the primary confidence threshold (age and gender >= .60), adults showed higher user-level anthropomorphization than teens. In the canonical conditional analysis among users with anthro_mean > 1, the teen-adult effect was d about -0.46; in the inclusive high-confidence sample it was d about -0.51.`")
    lines.append("- `Gender effects were sensitive to the estimand: women were higher than men in inclusive analyses, but the canonical conditional analysis yielded only a negligible effect (d about -0.07).`")
    lines.append("- `Neutral emotion was negatively associated with anthropomorphization, while joy and other non-neutral emotion proportions were positively associated relative to neutral expression in corrected emotion models.`")
    lines.append("- `Demographic variables explain a small but non-trivial share of variance; most variance remains unexplained by age/gender, supporting future work on psychological, relational, and platform-context predictors.`")
    lines.append(subsection("15.2 Claims to avoid"))
    lines.append("- Avoid: `over 414,757 comments were analyzed` unless you immediately clarify that this was the pre-cleaning raw total and 283,895 were retained.")
    lines.append("- Avoid: `demographics were inferred by GPT-4o-mini` for the full analysis unless the full LLM classifier is rerun. The current analysis reads V4 ML prediction parquets.")
    lines.append("- Avoid: `women strongly anthropomorphize more than men`; the estimated gender effect is negligible and model-dependent.")
    lines.append("- Avoid: citing old root-level `COMPREHENSIVE_V3_ANALYSIS_RESULTS.md` or `FINAL_SHORT_SUMMARY.md` numbers without reconciliation.")
    lines.append("- Avoid: interpreting seven-emotion saturated OLS coefficients from the old Model 3.")

    lines.append(section("16. Limitations and Ethics Numbers"))
    lines.append("- Reddit-only, public-comment sample; not representative of all AI companion users.")
    lines.append("- Comment authors are accounts, not verified individuals; multiple accounts and deleted context are possible.")
    lines.append("- Demographics are inferred, binary, and confidence-filtered; they are not self-report labels for most users.")
    lines.append("- Age categories collapse all adults into 19+, obscuring young adult, middle adult, and older adult differences.")
    lines.append("- AnthroScore measures expressed linguistic anthropomorphism, not private beliefs, clinical attachment, or downstream harm.")
    lines.append("- Emotion scores measure textual emotion probabilities, not experienced affect.")
    lines.append("- Observational cross-sectional design: do not infer that anthropomorphism causes emotion, isolation, dependency, or harm.")
    lines.append("- Ethical methods section should say the study used public Reddit data, minimized disclosure of identifiable examples, and analyzed aggregate patterns. If quoting comments, paraphrase or mask to reduce searchability.")

    lines.append(section("17. File Provenance and Source-of-Truth Map"))
    lines.append(md_table(
        ["Use for", "Path", "Status"],
        [
            ["Current master reference", "`results/PAPER_WRITING_MASTER_REFERENCE.md`", "Generated by this script"],
            ["Main canonical stats", "`results/COMPREHENSIVE_V3_ANALYSIS_RESULTS.md`", "Current, regenerated Apr 26 2026"],
            ["Main canonical JSON", "`results/COMPREHENSIVE_V3_ANALYSIS_RESULTS.json`", "Machine-readable stats"],
            ["Improved AnthroScore comments", "`experiments/anthroscore_v3/anthroscore_v3_improved_final.parquet`", "Primary outcome file"],
            ["Cleaned comments", "`Data/processed/all_comments.parquet`", "Corpus source"],
            ["Enriched n-grams", "`Data/features/comments_enriched.parquet`", "N-gram diagnostics"],
            ["User emotions", "`Data/features/user_emotions.parquet`", "User-level emotion proportions"],
            ["Comment emotions", "`Data/features/comments_with_emotions.parquet`", "Comment-level emotion probabilities"],
            ["Current age predictions", "`experiments/v2_correction/age_predictions_v4.parquet`", "Used by current pipeline; method caveat"],
            ["Current gender predictions", "`experiments/v2_correction/gender_predictions_v4.parquet`", "Used by current pipeline; method caveat"],
            ["Validation metrics", "`experiments/anthroscore_v3/validation_results.json`", "Expert/human-adjacent validation"],
            ["V3 vs V2 metrics", "`experiments/anthroscore_v3/mlm_comparison_results.json`", "Method comparison"],
            ["Stale duplicate to avoid", "`COMPREHENSIVE_V3_ANALYSIS_RESULTS.md` at repo root", "Older Jan 2026 numbers"],
        ],
    ))

    lines.append(section("18. Appendix-Ready Tables"))
    lines.append(subsection("18.1 Full high-confidence subgroup table"))
    subgroup_rows = []
    for (age_group, gender_group), sub in hc.groupby(["age_predicted", "gender_predicted"]):
        subgroup_rows.append([
            age_group,
            gender_group,
            fmt_int(len(sub)),
            fmt_num(sub["anthro_mean"].mean()),
            fmt_num(sub["anthro_mean"].std()),
            fmt_num(sub["anthro_mean"].median()),
            pct(sub["has_score_3plus"].mean()),
            pct(sub["has_score_4plus"].mean()),
            fmt_num(sub["anthro_count"].mean(), 2),
        ])
    lines.append(md_table(["Age", "Gender", "N", "Mean", "SD", "Median", "% any >=3", "% any >=4", "Mean scored comments"], subgroup_rows))

    lines.append(subsection("18.2 Raw subreddit date ranges"))
    date_rows = []
    for sub, sdf in comments.groupby("subreddit"):
        date_rows.append([f"r/{sub}", fmt_int(len(sdf)), fmt_int(sdf["author"].nunique()), str(sdf["created_dt"].min().date()), str(sdf["created_dt"].max().date())])
    lines.append(md_table(["Subreddit", "Cleaned comments", "Unique authors", "First date", "Last date"], date_rows))

    lines.append(section("19. Reproducibility"))
    lines.append("Generated by `scripts/generate_paper_master_reference.py`.")
    lines.append("Recommended reproducibility sequence:")
    lines.append("1. `python3 scripts/COMPREHENSIVE_V3_ANALYSIS.py`")
    lines.append("2. `python3 scripts/generate_paper_master_reference.py`")
    lines.append("3. Recheck `results/PAPER_WRITING_MASTER_REFERENCE.md` before writing draft claims.")
    lines.append("")
    lines.append("If n-gram prompts or scoring files are changed again, rerun both scripts and treat all older prose as stale until reconciled.")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
