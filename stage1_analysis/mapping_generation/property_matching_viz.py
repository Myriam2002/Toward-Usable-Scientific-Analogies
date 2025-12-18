
import os
import re
import ast
import json
import math
import argparse
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# Helpers
# =========================================================

def infer_condition_from_name(filename: str) -> str:
    name = os.path.basename(filename).lower()
    if "with_desc" in name or "with-description" in name or re.search(r"\bwith\b", name):
        return "with_desc"
    if "no_desc" in name or "without" in name or "no-description" in name:
        return "no_desc"
    return "unknown"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(float)
    if s.dtype == object:
        mapped = s.map(lambda x: {"true": 1.0, "false": 0.0}.get(str(x).strip().lower(), x))
        return pd.to_numeric(mapped, errors="coerce")
    return pd.to_numeric(s, errors="coerce")

def _num_array(s: pd.Series) -> np.ndarray:
    s = _coerce_numeric_series(s)
    return s.dropna().astype(float).to_numpy()

def normalize_token(t: str) -> str:
    t = str(t).lower().strip().strip("\"'")
    t = re.sub(r"[.،؛;:]+$", "", t)
    t = re.sub(r"\s+", " ", t)
    return t

# ---- Parsing ----

def parse_pair_list(cell) -> Set[Tuple[str, str]]:
    """
    Parse 'matches' into a set of (left,right) pairs.
    Accepts formats:
      - list of pairs: [["b","a"], ("b","a")]
      - dict mapping left->right or left->list[rights]
      - string: "b->a; b2->a3" or "b~a, b2~a3"
      - JSON or Python repr of the above
    """
    pairs: Set[Tuple[str,str]] = set()
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return pairs
    if isinstance(cell, set):
        return set(cell)
    if isinstance(cell, list):
        for it in cell:
            if isinstance(it, (list, tuple)) and len(it) >= 2:
                pairs.add((normalize_token(it[0]), normalize_token(it[1])))
            elif isinstance(it, dict) and "left" in it and "right" in it:
                pairs.add((normalize_token(it["left"]), normalize_token(it["right"])))
        return pairs
    if isinstance(cell, dict):
        for k,v in cell.items():
            k2 = normalize_token(k)
            if isinstance(v, list):
                for t in v:
                    pairs.add((k2, normalize_token(t)))
            else:
                pairs.add((k2, normalize_token(v)))
        return pairs
    s = str(cell).strip()
    # Try JSON
    try:
        obj = json.loads(s)
        return parse_pair_list(obj)
    except Exception:
        pass
    # Try Python literal
    try:
        obj = ast.literal_eval(s)
        return parse_pair_list(obj)
    except Exception:
        pass
    # Fallback tokenization
    sep_pairs = re.split(r"[;|,]\s*", s)
    for p in sep_pairs:
        if "->" in p:
            l, r = p.split("->", 1)
            pairs.add((normalize_token(l), normalize_token(r)))
        elif "~" in p:
            l, r = p.split("~", 1)
            pairs.add((normalize_token(l), normalize_token(r)))
        elif ":" in p:
            l, r = p.split(":", 1)
            pairs.add((normalize_token(l), normalize_token(r)))
    return pairs

def parse_token_list(cell) -> Set[str]:
    """
    Parse 'unmatched/outside' tokens into a set of strings.
    Accepts list, set, comma/semicolon string, JSON or Python repr.
    """
    toks: Set[str] = set()
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return toks
    if isinstance(cell, (list, set, tuple)):
        return set(normalize_token(x) for x in cell)
    s = str(cell).strip()
    try:
        obj = json.loads(s)
        return parse_token_list(obj)
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        return parse_token_list(obj)
    except Exception:
        pass
    parts = re.split(r"[;,|/]\s*|\]\s*\[\s*|,\s*", s.strip("[](){}"))
    for p in parts:
        p = p.strip()
        if p:
            toks.add(normalize_token(p))
    return toks

# ---- Metrics on pairs ----

def pair_precision_recall_f1(pred: Set[Tuple[str,str]], gt: Set[Tuple[str,str]]) -> Tuple[float,float,float,int,int,int]:
    tp = len(pred & gt)
    pp = len(pred)
    gg = len(gt)
    prec = tp/pp if pp else 0.0
    rec = tp/gg if gg else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return prec, rec, f1, tp, pp, gg

def pair_jaccard(pred: Set[Tuple[str,str]], gt: Set[Tuple[str,str]]) -> float:
    if not pred and not gt:
        return 1.0
    return len(pred & gt) / len(pred | gt) if (pred or gt) else 0.0

# =========================================================
# Visualizations
# =========================================================

def boxplot_by_condition(df: pd.DataFrame, metric: str, out_dir: str):
    if metric not in df.columns: 
        return
    vals_no = _num_array(df.loc[df["condition"]=="no_desc", metric])
    vals_w  = _num_array(df.loc[df["condition"]=="with_desc", metric])
    if (len(vals_no) == 0) and (len(vals_w) == 0): return
    X, labels = [], []
    if len(vals_no): X.append(vals_no); labels.append("no_desc")
    if len(vals_w):  X.append(vals_w);  labels.append("with_desc")
    plt.figure(); plt.boxplot(X, labels=labels)
    plt.title(f"{metric}: distribution by condition"); plt.ylabel(metric)
    path = os.path.join(out_dir, f"{metric}_by_condition_boxplot.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()

def paired_delta_hist(df: pd.DataFrame, metric: str, out_dir: str, idx_cols=["id","model"]):
    for c in idx_cols:
        if c not in df.columns: return
    if metric not in df.columns: return
    p = df.pivot_table(index=idx_cols, columns="condition", values=metric, aggfunc="mean")
    for col in ["no_desc","with_desc"]:
        if col in p.columns: p[col] = _coerce_numeric_series(p[col])
    if not {"no_desc","with_desc"}.issubset(p.columns): return
    p = p.dropna(subset=["no_desc","with_desc"], how="any")
    p["delta"] = p["with_desc"] - p["no_desc"]
    if len(p["delta"]) == 0: return
    plt.figure(); plt.hist(p["delta"].values, bins=20)
    plt.title(f"{metric} lift (with_desc - no_desc)"); plt.xlabel("delta"); plt.ylabel("count")
    path = os.path.join(out_dir, f"{metric}_delta_hist.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    csv = os.path.join(out_dir, f"{metric}_deltas.csv"); p.reset_index().to_csv(csv, index=False)

def bar_by_model(df: pd.DataFrame, metric: str, out_dir: str):
    if "model" not in df.columns or metric not in df.columns: return
    g = df.groupby(["model","condition"])[metric].mean().reset_index()
    g[metric] = _coerce_numeric_series(g[metric])
    models = sorted(g["model"].unique())
    means_no = [g[(g.model==m)&(g.condition=="no_desc")][metric].mean() for m in models]
    means_w  = [g[(g.model==m)&(g.condition=="with_desc")][metric].mean() for m in models]
    order = np.argsort(np.array(means_w))[::-1]
    models = [models[i] for i in order]
    means_no = [means_no[i] for i in order]
    means_w  = [means_w[i] for i in order]
    x = np.arange(len(models)); width=0.35
    plt.figure(figsize=(max(8,len(models)*0.7),6))
    plt.bar(x-width/2, means_no, width, label="no_desc")
    plt.bar(x+width/2, means_w,  width, label="with_desc")
    plt.title(f"{metric}: model comparison"); plt.xlabel("model"); plt.ylabel(metric)
    plt.xticks(x, models, rotation=45, ha="right"); plt.legend()
    path = os.path.join(out_dir, f"{metric}_by_model.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()

def radar_model_profiles(df: pd.DataFrame, out_dir: str, metrics: Optional[List[str]] = None):
    if metrics is None:
        metrics = [m for m in [
            "overall_accuracy_avg", "overall_accuracy_weighted",
            "exact_match_accuracy", "fuzzy_match_accuracy",
            "semantic_match_accuracy", "avg_fuzzy_score",
            "avg_semantic_score"
        ] if m in df.columns]
    metrics = [m for m in metrics if m in df.columns]
    if not metrics or "model" not in df.columns or "condition" not in df.columns:
        return
    for m in metrics: df[m] = _coerce_numeric_series(df[m])
    grp = df.groupby(["model","condition"])[metrics].mean()
    models = sorted(df["model"].dropna().unique().tolist())
    for model in models:
        plt.figure(figsize=(7,7))
        ax = plt.subplot(111, polar=True)
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        def plot_cond(label: str):
            if (model, label) not in grp.index: return
            vals = grp.loc[(model, label)].tolist(); vals += vals[:1]
            ax.plot(angles, vals, linewidth=2, label=label)
            ax.fill(angles, vals, alpha=0.1)
        plot_cond("no_desc"); plot_cond("with_desc")
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics)
        ax.set_title(f"Metric profile — {model}"); ax.set_rlabel_position(30)
        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
        safe = re.sub(r"[^a-zA-Z0-9._-]+","_", str(model))
        path = os.path.join(out_dir, f"model_radar_{safe}.png")
        plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()

# =========================================================
# Property Matching specific derived stats
# =========================================================

def compute_pair_metrics(data: pd.DataFrame,
                         gt_col_candidates=("ground_truth_matches","ground_truth_mappings","gt_matches"),
                         pred_col_candidates=("predicted_matches","predicted_mappings","matches")) -> pd.DataFrame:
    gt_col = next((c for c in gt_col_candidates if c in data.columns), None)
    pred_col = next((c for c in pred_col_candidates if c in data.columns), None)
    if gt_col is None or pred_col is None:
        # nothing to compute
        data["_pair_precision"] = np.nan
        data["_pair_recall"] = np.nan
        data["_pair_f1"] = np.nan
        data["_pair_jaccard"] = np.nan
        data["_pair_over_under"] = np.nan
        return data

    precs, recs, f1s, jaccs, diffs = [], [], [], [], []
    for _, row in data.iterrows():
        gt_pairs = parse_pair_list(row.get(gt_col))
        pr_pairs = parse_pair_list(row.get(pred_col))
        p, r, f1, tp, pp, gg = pair_precision_recall_f1(pr_pairs, gt_pairs)
        precs.append(p); recs.append(r); f1s.append(f1)
        jaccs.append(pair_jaccard(pr_pairs, gt_pairs))
        diffs.append(len(pr_pairs) - len(gt_pairs))
    data["_pair_precision"] = precs
    data["_pair_recall"] = recs
    data["_pair_f1"] = f1s
    data["_pair_jaccard"] = jaccs
    data["_pair_over_under"] = diffs
    return data

def compute_unmatched_stats(data: pd.DataFrame,
                            pred_unmatched_cols=("predicted_unmatched","outside_words","predicted_outside"),
                            gt_unmatched_cols=("ground_truth_unmatched","gt_unmatched")) -> pd.DataFrame:
    pred_col = next((c for c in pred_unmatched_cols if c in data.columns), None)
    gt_col = next((c for c in gt_unmatched_cols if c in data.columns), None)
    pred_counts, gt_counts = [], []
    for _, row in data.iterrows():
        pred_un = parse_token_list(row.get(pred_col)) if pred_col else set()
        gt_un = parse_token_list(row.get(gt_col)) if gt_col else set()
        pred_counts.append(len(pred_un))
        gt_counts.append(len(gt_un))
    data["_pred_unmatched_count"] = pred_counts
    data["_gt_unmatched_count"] = gt_counts
    return data

def plot_unmatched_distributions(df: pd.DataFrame, out_dir: str):
    for col in ["_pred_unmatched_count","_gt_unmatched_count"]:
        if col not in df.columns: continue
        plt.figure()
        for cond in df["condition"].unique():
            vals = pd.to_numeric(df.loc[df["condition"]==cond, col], errors="coerce").dropna()
            plt.hist(vals, bins=20, alpha=0.6, label=cond)
        plt.title(f"Distribution of {col}"); plt.xlabel(col); plt.ylabel("count"); plt.legend()
        path = os.path.join(out_dir, f"{col}_hist.png")
        plt.savefig(path, bbox_inches="tight"); plt.close()

def bar_top_outside_tokens(df: pd.DataFrame, out_dir: str, top_k=25):
    # Aggregate predicted unmatched tokens if present
    pred_col = next((c for c in ["predicted_unmatched","outside_words","predicted_outside"] if c in df.columns), None)
    if not pred_col: return
    ctr = Counter()
    for _, row in df.iterrows():
        toks = parse_token_list(row.get(pred_col))
        for t in toks: ctr[t] += 1
    if not ctr: return
    pairs = ctr.most_common(top_k)
    labels, counts = zip(*pairs)
    y = np.arange(len(labels))
    plt.figure(figsize=(10, max(6, len(labels)*0.35)))
    plt.barh(y, counts); plt.yticks(y, labels); plt.gca().invert_yaxis()
    plt.xlabel("count"); plt.title("Top predicted outside/unmatched tokens")
    path = os.path.join(out_dir, "top_predicted_outside_tokens.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()

# =========================================================
# Orchestration
# =========================================================

EXPECTED_METRICS = [
    "system_accuracy",
    "concept_mapping_accuracy",
    "concept_accuracy",
    "overall_accuracy_avg",
    "overall_accuracy_weighted",
    "exact_match_accuracy",
    "fuzzy_match_accuracy",
    "semantic_match_accuracy",
    "avg_fuzzy_score",
    "avg_semantic_score",
]

def generate_property_matching_visuals(base_dir: str, file_names: List[str], out_subfolder: str = "visualizations_property_matching"):
    out_dir = os.path.join(base_dir, out_subfolder); ensure_dir(out_dir)

    # Load
    frames = []
    for fn in file_names:
        path = fn if os.path.isabs(fn) else os.path.join(base_dir, fn)
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, encoding="utf-8-sig")
        df["condition"] = infer_condition_from_name(path)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    # Coerce numeric metrics
    for col in [m for m in EXPECTED_METRICS if m in data.columns]:
        data[col] = _coerce_numeric_series(data[col])

    # Derived: pair metrics and unmatched stats
    data = compute_pair_metrics(data)
    data = compute_unmatched_stats(data)

    # Save tidy
    keep = [c for c in [
        "id" if "id" in data.columns else None,
        "model" if "model" in data.columns else None,
        "condition",
        *[m for m in EXPECTED_METRICS if m in data.columns],
        "_pair_precision" if "_pair_precision" in data.columns else None,
        "_pair_recall" if "_pair_recall" in data.columns else None,
        "_pair_f1" if "_pair_f1" in data.columns else None,
        "_pair_jaccard" if "_pair_jaccard" in data.columns else None,
        "_pair_over_under" if "_pair_over_under" in data.columns else None,
        "_pred_unmatched_count" if "_pred_unmatched_count" in data.columns else None,
        "_gt_unmatched_count" if "_gt_unmatched_count" in data.columns else None,
        "duration_seconds" if "duration_seconds" in data.columns else None,
        "success" if "success" in data.columns else None,
        "error" if "error" in data.columns else None,
    ] if c is not None]
    tidy = data[keep].copy()
    tidy_path = os.path.join(out_dir, "tidy_property_matching_summary.csv")
    tidy.to_csv(tidy_path, index=False)

    # Visuals answering your questions + extras
    # 1) System and concept accuracy
    for m in [x for x in ["system_accuracy","concept_mapping_accuracy","concept_accuracy"] if x in data.columns]:
        boxplot_by_condition(data, m, out_dir)
        paired_delta_hist(data, m, out_dir, idx_cols=["id","model"])
        bar_by_model(data, m, out_dir)

    # 2) Outside words / unmatched
    plot_unmatched_distributions(data, out_dir)
    bar_top_outside_tokens(data, out_dir)
    boxplot_by_condition(data, "_pred_unmatched_count", out_dir)
    paired_delta_hist(data, "_pred_unmatched_count", out_dir, idx_cols=["id","model"])

    # 3) Which models are best at matching? (use pair F1 & Jaccard)
    for m in ["_pair_f1","_pair_jaccard","_pair_precision","_pair_recall"]:
        boxplot_by_condition(data, m, out_dir)
        paired_delta_hist(data, m, out_dir, idx_cols=["id","model"])
        bar_by_model(data, m, out_dir)

    # Duration tradeoff
    if "duration_seconds" in data.columns:
        for m in ["system_accuracy","_pair_f1"]:
            if m not in data.columns: continue
            plt.figure()
            for cond in data["condition"].unique():
                sub = data[data["condition"]==cond]
                x = _coerce_numeric_series(sub["duration_seconds"])
                y = _coerce_numeric_series(sub[m])
                plt.scatter(x, y, alpha=0.6, label=cond)
            plt.title(f"Duration vs {m}"); plt.xlabel("duration_seconds"); plt.ylabel(m); plt.legend()
            path = os.path.join(out_dir, f"duration_vs_{m}.png")
            plt.savefig(path, bbox_inches="tight"); plt.close()

    # Radar profiles per model (if those metrics exist)
    radar_metrics = [m for m in [
        "overall_accuracy_avg","overall_accuracy_weighted",
        "exact_match_accuracy","fuzzy_match_accuracy",
        "semantic_match_accuracy","avg_fuzzy_score","avg_semantic_score"
    ] if m in data.columns]
    if radar_metrics:
        radar_model_profiles(data, out_dir, metrics=radar_metrics)

    return {"out_dir": out_dir, "tidy_csv": tidy_path}

# =========================================================
# CLI
# =========================================================

def _parse_args():
    p = argparse.ArgumentParser(description="Property Matching Visualizations")
    p.add_argument("--base_dir", type=str, required=True)
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--out_subfolder", type=str, default="visualizations_property_matching")
    return p.parse_args()

def main():
    args = _parse_args()
    generate_property_matching_visuals(args.base_dir, args.files, args.out_subfolder)

if __name__ == "__main__":
    main()
