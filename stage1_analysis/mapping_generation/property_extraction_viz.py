
import os
import re
import ast
import json
import math
import argparse
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import wilcoxon
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ---------- Helpers ----------

def infer_condition_from_name(filename: str) -> str:
    """Infer condition label from a filename string."""
    name = os.path.basename(filename).lower()
    if ("with_desc" in name) or ("with-description" in name) or re.search(r"\bwith\b", name):
        return "with_desc"
    if ("no_desc" in name) or ("without" in name) or ("no-description" in name):
        return "no_desc"
    return "unknown"


def parse_props(cell) -> List[str]:
    """Parse a 'properties' cell tolerant to various formats (JSON, Python list, delimited strings)."""
    if isinstance(cell, list):
        return [str(x).strip() for x in cell if str(x).strip()]
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    # Try JSON
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
    except Exception:
        pass
    # Try Python literal
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
    except Exception:
        pass
    # Fallback: split
    parts = re.split(r"[;,|/]\s*|\]\s*\[\s*|,\s*", s.strip("[](){}"))
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def normalize_tokens(props: List[str]) -> List[str]:
    """Lowercase, trim punctuation/spacing."""
    normed = []
    for p in props:
        t = str(p).lower().strip().strip("\"'")
        t = re.sub(r"[.،؛;:]+$", "", t)
        t = re.sub(r"\s+", " ", t)
        normed.append(t)
    return normed


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_mean(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.mean()) if len(x) else float("nan")


def safe_std(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.std()) if len(x) else float("nan")


# ---------- Loading & Derived Metrics ----------

EXPECTED_METRICS = [
    "overall_accuracy_avg",
    "overall_accuracy_weighted",
    "exact_match_accuracy",
    "fuzzy_match_accuracy",
    "semantic_match_accuracy",
    "avg_fuzzy_score",
    "avg_semantic_score",
]

def load_and_prepare(base_dir: str, file_names: List[str]) -> pd.DataFrame:
    """Load multiple CSVs and add 'condition' inferred from filename."""
    frames = []
    for fn in file_names:
        path = fn if os.path.isabs(fn) else os.path.join(base_dir, fn)
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, encoding="utf-8-sig")
        df = df.copy()
        df["condition"] = infer_condition_from_name(path)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    return data


def add_derived_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Compute #predicted/#gt, over_under, and jaccard for each row when columns exist."""
    pred_col = "predicted_properties" if "predicted_properties" in data.columns else None
    gt_col = "ground_truth_properties" if "ground_truth_properties" in data.columns else None

    pred_counts, gt_counts, j_scores = [], [], []
    for _, row in data.iterrows():
        preds = normalize_tokens(parse_props(row[pred_col])) if pred_col else []
        gts = normalize_tokens(parse_props(row[gt_col])) if gt_col else []
        pred_counts.append(len(set(preds)))
        gt_counts.append(len(set(gts)))
        j_scores.append(jaccard(preds, gts))

    if pred_col:
        data["_pred_count"] = pred_counts
    if gt_col:
        data["_gt_count"] = gt_counts
    if pred_col and gt_col:
        data["_over_under"] = data["_pred_count"] - data["_gt_count"]
        data["_jaccard"] = j_scores
    return data


def save_tidy_summary(data: pd.DataFrame, out_dir: str) -> str:
    keep_cols = [
        "id" if "id" in data.columns else None,
        "model" if "model" in data.columns else None,
        "condition",
        "overall_accuracy_avg",
        "overall_accuracy_weighted",
        "exact_match_accuracy",
        "fuzzy_match_accuracy",
        "semantic_match_accuracy",
        "avg_fuzzy_score",
        "avg_semantic_score",
        "duration_seconds" if "duration_seconds" in data.columns else None,
        "_pred_count" if "_pred_count" in data.columns else None,
        "_gt_count" if "_gt_count" in data.columns else None,
        "_jaccard" if "_jaccard" in data.columns else None,
        "success" if "success" in data.columns else None,
        "error" if "error" in data.columns else None,
    ]
    keep_cols = [c for c in keep_cols if c is not None]
    tidy = data[keep_cols].copy()
    f = os.path.join(out_dir, "tidy_metrics_summary.csv")
    tidy.to_csv(f, index=False)
    return f


# ---------- Visualization Primitives (matplotlib only; no styles/colors set) ----------

def boxplot_by_condition(data: pd.DataFrame, metric: str, out_dir: str):
    if metric not in data.columns:
        return
    plt.figure()
    vals_no = pd.to_numeric(data.loc[data["condition"]=="no_desc", metric], errors="coerce").dropna()
    vals_w  = pd.to_numeric(data.loc[data["condition"]=="with_desc", metric], errors="coerce").dropna()
    plt.boxplot([vals_no.values, vals_w.values], labels=["no_desc", "with_desc"])
    plt.title(f"{metric}: distribution by condition")
    plt.ylabel(metric)
    path = os.path.join(out_dir, f"{metric}_by_condition_boxplot.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def scatter_duration_vs_metric(data: pd.DataFrame, metric: str, out_dir: str):
    if "duration_seconds" not in data.columns or metric not in data.columns:
        return
    plt.figure()
    for cond in data["condition"].dropna().unique().tolist():
        subset = data[data["condition"]==cond]
        x = pd.to_numeric(subset["duration_seconds"], errors="coerce")
        y = pd.to_numeric(subset[metric], errors="coerce")
        plt.scatter(x, y, label=cond, alpha=0.6)
    plt.title(f"Duration vs {metric} (by condition)")
    plt.xlabel("duration_seconds")
    plt.ylabel(metric)
    plt.legend()
    path = os.path.join(out_dir, f"duration_vs_{metric}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def histogram_over_under_by_condition(data: pd.DataFrame, out_dir: str):
    if "_over_under" not in data.columns:
        return
    for cond in data["condition"].dropna().unique().tolist():
        subset = data[data["condition"]==cond]
        plt.figure()
        vals = pd.to_numeric(subset["_over_under"], errors="coerce").dropna().values
        if len(vals) == 0:
            plt.close()
            continue
        plt.hist(vals, bins=20)
        plt.title(f"Over/Under-extraction (pred - gt): {cond}")
        plt.xlabel("pred_count - gt_count")
        plt.ylabel("count")
        path = os.path.join(out_dir, f"over_under_{cond}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()


def barh_top_missed_properties(data: pd.DataFrame, out_dir: str, topk: int = 20):
    pred_col = "predicted_properties" if "predicted_properties" in data.columns else None
    gt_col = "ground_truth_properties" if "ground_truth_properties" in data.columns else None
    if not pred_col or not gt_col:
        return
    miss_counter = defaultdict(Counter)
    for _, row in data.iterrows():
        cond = row.get("condition", "unknown")
        preds = set(normalize_tokens(parse_props(row[pred_col])))
        gts = set(normalize_tokens(parse_props(row[gt_col])))
        for m in (gts - preds):
            miss_counter[cond][m] += 1

    for cond, counter in miss_counter.items():
        common = counter.most_common(topk)
        if not common:
            continue
        labels, counts = zip(*common)
        y = np.arange(len(labels))
        plt.figure(figsize=(10, 6))
        plt.barh(y, counts)
        plt.yticks(y, labels)
        plt.title(f"Top missed ground-truth properties — {cond}")
        plt.xlabel("count of misses")
        plt.gca().invert_yaxis()
        path = os.path.join(out_dir, f"top_missed_properties_{cond}.png")
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()


def paired_delta_hist_and_csv(data: pd.DataFrame, metric: str, out_dir: str, idx_cols: List[str] = ["id","model"]):
    for c in idx_cols:
        if c not in data.columns:
            return None
    if metric not in data.columns:
        return None
    pivot = (data
             .pivot_table(index=idx_cols, columns="condition", values=metric, aggfunc="mean"))
    if not set(["no_desc","with_desc"]).issubset(pivot.columns):
        return None
    pivot = pivot.dropna(subset=["no_desc","with_desc"], how="any")
    pivot["delta"] = pivot["with_desc"] - pivot["no_desc"]

    # Histogram
    plt.figure()
    plt.hist(pivot["delta"].values, bins=20)
    plt.title(f"Lift from adding description (with_desc - no_desc) on {metric}")
    plt.xlabel("delta")
    plt.ylabel("count")
    path_png = os.path.join(out_dir, f"{metric}_delta_hist.png")
    plt.savefig(path_png, bbox_inches="tight")
    plt.close()

    # CSV
    path_csv = os.path.join(out_dir, f"{metric}_deltas.csv")
    pivot.reset_index().to_csv(path_csv, index=False)
    return path_png, path_csv


# ---------- Model-Level Comparisons ----------

def bar_by_model_with_errorbars(data: pd.DataFrame, metric: str, out_dir: str):
    if "model" not in data.columns or metric not in data.columns:
        return
    grp = (data
           .groupby(["model","condition"])[metric]
           .agg(["mean","std","count"])
           .reset_index())
    # Ensure both conditions exist for all models; missing -> NaN
    models = sorted(grp["model"].unique().tolist())
    x = np.arange(len(models))
    width = 0.35

    means_no = []
    stds_no = []
    means_w  = []
    stds_w   = []
    for m in models:
        m_no = grp[(grp["model"]==m) & (grp["condition"]=="no_desc")]
        m_w  = grp[(grp["model"]==m) & (grp["condition"]=="with_desc")]
        means_no.append(float(m_no["mean"].iloc[0]) if len(m_no) else np.nan)
        stds_no.append(float(m_no["std"].iloc[0]) if len(m_no) else np.nan)
        means_w.append(float(m_w["mean"].iloc[0]) if len(m_w) else np.nan)
        stds_w.append(float(m_w["std"].iloc[0]) if len(m_w) else np.nan)

    # Sort by with_desc means desc
    order = np.argsort(np.array(means_w))[::-1]
    models = [models[i] for i in order]
    means_no = [means_no[i] for i in order]
    stds_no = [stds_no[i] for i in order]
    means_w = [means_w[i] for i in order]
    stds_w = [stds_w[i] for i in order]

    plt.figure(figsize=(max(8, len(models)*0.7), 6))
    idx = np.arange(len(models))
    plt.bar(idx - width/2, means_no, width, yerr=stds_no, label="no_desc")
    plt.bar(idx + width/2, means_w,  width, yerr=stds_w,  label="with_desc")
    plt.title(f"{metric}: model comparison")
    plt.xlabel("model")
    plt.ylabel(metric)
    plt.xticks(idx, models, rotation=45, ha="right")
    plt.legend()
    path = os.path.join(out_dir, f"{metric}_by_model.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def barh_model_lift_per_metric(data: pd.DataFrame, metric: str, out_dir: str):
    if "model" not in data.columns or metric not in data.columns:
        return
    # Mean per model per condition
    grp = (data.groupby(["model","condition"])[metric].mean().unstack("condition"))
    if "with_desc" not in grp.columns or "no_desc" not in grp.columns:
        return
    grp["delta"] = grp["with_desc"] - grp["no_desc"]
    grp = grp.sort_values("delta", ascending=True)

    plt.figure(figsize=(10, max(5, len(grp)*0.5)))
    y = np.arange(len(grp.index))
    plt.barh(y, grp["delta"].values)
    plt.yticks(y, grp.index.tolist())
    plt.title(f"{metric}: model lift (with_desc - no_desc)")
    plt.xlabel("delta")
    path = os.path.join(out_dir, f"{metric}_model_lift.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def radar_model_profiles(data: pd.DataFrame, out_dir: str, metrics: Optional[List[str]] = None):
    if metrics is None:
        metrics = [m for m in EXPECTED_METRICS if m in data.columns]
    metrics = [m for m in metrics if m in data.columns]
    if not metrics or "model" not in data.columns:
        return
    # Average per model, per condition
    grp = data.groupby(["model","condition"])[metrics].mean()

    models = sorted(data["model"].dropna().unique().tolist())
    for model in models:
        plt.figure(figsize=(7,7))
        ax = plt.subplot(111, polar=True)
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        def plot_condition(cond_label: str):
            if (model, cond_label) not in grp.index:
                return
            vals = grp.loc[(model, cond_label)].tolist()
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=2, label=cond_label)
            ax.fill(angles, vals, alpha=0.1)

        plot_condition("no_desc")
        plot_condition("with_desc")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title(f"Metric profile — {model}")
        ax.set_rlabel_position(30)
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        path = os.path.join(out_dir, f"model_radar_{re.sub(r'[^a-zA-Z0-9._-]+','_', model)}.png")
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()


# ---------- Significance Tests & Leaderboard ----------

def compute_significance_per_model_metric(data: pd.DataFrame, metric: str, idx_cols: List[str], out_dir: str):
    """Paired Wilcoxon between with_desc and no_desc per model if pairs exist. Saves CSV."""
    if not SCIPY_AVAILABLE:
        return None
    for c in (idx_cols + ["model","condition",metric]):
        if c not in data.columns:
            return None
    rows = []
    for model, dfm in data.groupby("model"):
        pivot = (dfm.pivot_table(index=idx_cols, columns="condition", values=metric, aggfunc="mean")
                      .dropna(subset=["no_desc","with_desc"], how="any"))
        if pivot.shape[0] == 0:
            continue
        try:
            stat, p = wilcoxon(pivot["with_desc"], pivot["no_desc"], zero_method="wilcox", alternative="two-sided")
        except Exception:
            stat, p = (float("nan"), float("nan"))
        rows.append({
            "model": model,
            "metric": metric,
            "n_pairs": int(pivot.shape[0]),
            "mean_no_desc": float(pivot["no_desc"].mean()),
            "mean_with_desc": float(pivot["with_desc"].mean()),
            "delta_mean": float(pivot["with_desc"].mean() - pivot["no_desc"].mean()),
            "wilcoxon_stat": float(stat),
            "p_value": float(p),
            "significant_0.05": (p < 0.05) if not math.isnan(p) else False
        })
    if not rows:
        return None
    out = pd.DataFrame(rows)
    path = os.path.join(out_dir, f"significance_{metric}.csv")
    out.to_csv(path, index=False)
    return path


def build_model_leaderboard(data: pd.DataFrame, out_dir: str) -> str:
    metrics = [m for m in EXPECTED_METRICS if m in data.columns]
    if "model" not in data.columns or not metrics:
        return ""
    grp = data.groupby(["model","condition"])[metrics].mean().reset_index()
    # pivot for overall_accuracy_avg to compute delta and rank
    if "overall_accuracy_avg" in metrics:
        p = grp.pivot(index="model", columns="condition", values="overall_accuracy_avg")
        p["delta"] = p.get("with_desc") - p.get("no_desc")
        p = p.sort_values("with_desc", ascending=False)
        order = p.index.tolist()
        grp["model"] = pd.Categorical(grp["model"], categories=order, ordered=True)
        grp = grp.sort_values(["model","condition"])
    leaderboard_path = os.path.join(out_dir, "model_leaderboard.csv")
    grp.to_csv(leaderboard_path, index=False)
    return leaderboard_path


# ---------- Orchestration ----------

def generate_all_visualizations(base_dir: str, file_names: List[str], out_subfolder: str = "visualizations_property_extraction"):
    """End-to-end pipeline. Saves all charts/tables under base_dir/out_subfolder."""
    out_dir = os.path.join(base_dir, out_subfolder)
    ensure_dir(out_dir)

    # Load & derive
    data = load_and_prepare(base_dir, file_names)
    data = add_derived_columns(data)

    # Save tidy
    tidy_path = save_tidy_summary(data, out_dir)

    # 1) Condition-level distributions for each metric
    for m in [x for x in EXPECTED_METRICS if x in data.columns]:
        boxplot_by_condition(data, m, out_dir)

    # 2) Paired delta hist + CSV for overall_accuracy_avg
    paired_delta_hist_and_csv(data, metric="overall_accuracy_avg", out_dir=out_dir, idx_cols=["id","model"])

    # 3) Success rate by model & condition (clustered bar) if success exists
    if "success" in data.columns and "model" in data.columns:
        grp = (data.groupby(["model","condition"])["success"].mean().reset_index())
        models = grp["model"].unique().tolist()
        x = np.arange(len(models))
        width = 0.35
        means_no = [grp[(grp["model"]==m)&(grp["condition"]=="no_desc")]["success"].mean() for m in models]
        means_w  = [grp[(grp["model"]==m)&(grp["condition"]=="with_desc")]["success"].mean() for m in models]
        plt.figure(figsize=(max(8, len(models)*0.7), 6))
        plt.bar(x - width/2, means_no, width, label="no_desc")
        plt.bar(x + width/2, means_w,  width, label="with_desc")
        plt.title("Success rate by model and condition")
        plt.xlabel("model")
        plt.ylabel("success rate")
        plt.xticks(x, models, rotation=45, ha="right")
        plt.legend()
        fpath = os.path.join(out_dir, "success_rate_by_model_condition.png")
        plt.tight_layout()
        plt.savefig(fpath, bbox_inches="tight")
        plt.close()

    # 4) Duration vs overall accuracy (scatter)
    scatter_duration_vs_metric(data, "overall_accuracy_avg", out_dir)

    # 5) Over/Under histograms by condition
    histogram_over_under_by_condition(data, out_dir)

    # 6) Jaccard by condition
    if "_jaccard" in data.columns:
        boxplot_by_condition(data, "_jaccard", out_dir)

    # 7) Top missed GT properties
    barh_top_missed_properties(data, out_dir)

    # -------- Model-Level Extensions --------

    # 8) Per-metric model bar charts + error bars
    for m in [x for x in EXPECTED_METRICS if x in data.columns]:
        bar_by_model_with_errorbars(data, m, out_dir)

    # 9) Per-metric model lift charts
    for m in [x for x in EXPECTED_METRICS if x in data.columns]:
        barh_model_lift_per_metric(data, m, out_dir)

    # 10) Radar plots per model
    radar_model_profiles(data, out_dir, metrics=[m for m in EXPECTED_METRICS if m in data.columns])

    # 11) Significance tests (if SciPy installed)
    if SCIPY_AVAILABLE:
        for m in [x for x in EXPECTED_METRICS if x in data.columns]:
            compute_significance_per_model_metric(data, m, idx_cols=["id"], out_dir=out_dir)

    # 12) Leaderboard table
    build_model_leaderboard(data, out_dir)

    return {
        "out_dir": out_dir,
        "tidy_csv": tidy_path,
    }


# ---------- CLI ----------

def _parse_args():
    p = argparse.ArgumentParser(description="Property Extraction: Analysis & Visualizations")
    p.add_argument("--base_dir", type=str, required=True, help="Folder where the CSV files live")
    p.add_argument("--files", nargs="+", required=True, help="One or more CSV filenames")
    p.add_argument("--out_subfolder", type=str, default="visualizations_property_extraction", help="Subfolder to save visuals")
    return p.parse_args()


def main():
    args = _parse_args()
    generate_all_visualizations(base_dir=args.base_dir, file_names=args.files, out_subfolder=args.out_subfolder)


if __name__ == "__main__":
    main()
