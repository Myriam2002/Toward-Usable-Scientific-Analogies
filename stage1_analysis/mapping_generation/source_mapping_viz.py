
import os
import re
import ast
import json
import math
import argparse
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Helpers ----------------

def infer_condition_from_name(filename: str) -> str:
    name = os.path.basename(filename).lower()
    if "with_desc" in name or "with-description" in name or re.search(r"\bwith\b", name):
        return "with_desc"
    if "no_desc" in name or "without" in name or "no-description" in name:
        return "no_desc"
    return "unknown"

def _to_list(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    if isinstance(x, list):
        return x
    s = str(x).strip()
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return v
    except Exception:
        pass
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return v
    except Exception:
        pass
    parts = re.split(r"[;,|/]\s*|\]\s*\[\s*|,\s*", s.strip("[](){}"))
    return [p.strip() for p in parts if p.strip()]

def normalize_token(t: str) -> str:
    t = str(t).lower().strip().strip("\"'")
    t = re.sub(r"[.،؛;:]+$", "", t)
    t = re.sub(r"\s+", " ", t)
    return t

def parse_mapping_cell(cell) -> Dict[str, List[str]]:
    """
    Robustly parse mapping formats. Returns dict: source_prop -> list of target_props.
    Accepts:
      - JSON or Python dict
      - list of pairs like [["b","a"], ...] or [("b","a"), ...]
      - list of dicts like [{"source":"b","target":"a"}, ...]
      - string with '->' per mapping, delimited by ';' or ','
    """
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return {}
    if isinstance(cell, dict):
        out = {}
        for k,v in cell.items():
            ks = normalize_token(k)
            if isinstance(v, list):
                out[ks] = [normalize_token(x) for x in v]
            else:
                out[ks] = [normalize_token(v)]
        return out
    s = str(cell).strip()
    # try JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return {normalize_token(k): [normalize_token(x) for x in (v if isinstance(v,list) else [v])] for k,v in obj.items()}
        if isinstance(obj, list):
            out = defaultdict(list)
            for item in obj:
                if isinstance(item, (list,tuple)) and len(item)>=2:
                    out[normalize_token(item[0])].append(normalize_token(item[1]))
                elif isinstance(item, dict) and "source" in item and "target" in item:
                    out[normalize_token(item["source"])].append(normalize_token(item["target"]))
            return dict(out)
    except Exception:
        pass
    # try Python literal
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return {normalize_token(k): [normalize_token(x) for x in (v if isinstance(v,list) else [v])] for k,v in obj.items()}
        if isinstance(obj, list):
            out = defaultdict(list)
            for item in obj:
                if isinstance(item, (list,tuple)) and len(item)>=2:
                    out[normalize_token(item[0])].append(normalize_token(item[1]))
                elif isinstance(item, dict) and "source" in item and "target" in item:
                    out[normalize_token(item["source"])].append(normalize_token(item["target"]))
            return dict(out)
    except Exception:
        pass
    # fallback: split "b->a" tokens
    out = defaultdict(list)
    parts = re.split(r"[;|]\s*", s)
    for p in parts:
        if "->" in p:
            b,a = p.split("->",1)
            out[normalize_token(b)].append(normalize_token(a))
    return dict(out)

def mapping_to_edges(mapping: Dict[str, List[str]]) -> set:
    edges = set()
    for src, tgts in mapping.items():
        for t in tgts:
            edges.add((normalize_token(src), normalize_token(t)))
    return edges

def precision_recall_f1(pred_edges: set, gt_edges: set) -> Tuple[float,float,float]:
    tp = len(pred_edges & gt_edges)
    pp = len(pred_edges)
    gg = len(gt_edges)
    prec = tp/pp if pp else 0.0
    rec  = tp/gg if gg else 0.0
    if prec+rec==0: 
        f1=0.0
    else:
        f1 = 2*prec*rec/(prec+rec)
    return prec, rec, f1

def ensure_dir(p): os.makedirs(p, exist_ok=True)

# ---- NEW: robust numeric coercion ----

def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce booleans and 'true'/'false' strings to numeric floats; leave others numeric."""
    if s.dtype == bool:
        return s.astype(float)
    if s.dtype == object:
        mapped = s.map(lambda x: {"true": 1.0, "false": 0.0}.get(str(x).strip().lower(), x))
        return pd.to_numeric(mapped, errors="coerce")
    return pd.to_numeric(s, errors="coerce")

def _num_array(s: pd.Series) -> np.ndarray:
    s = _coerce_numeric_series(s)
    return s.dropna().astype(float).to_numpy()

# ---------------- Visualizations ----------------

def boxplot_by_condition(df: pd.DataFrame, metric: str, out_dir: str):
    if metric not in df.columns: 
        return
    import matplotlib.pyplot as plt
    vals_no = _num_array(df.loc[df["condition"]=="no_desc", metric])
    vals_w  = _num_array(df.loc[df["condition"]=="with_desc", metric])

    if (len(vals_no) == 0) and (len(vals_w) == 0):
        return

    X, labels = [], []
    if len(vals_no) > 0:
        X.append(vals_no); labels.append("no_desc")
    if len(vals_w) > 0:
        X.append(vals_w); labels.append("with_desc")

    plt.figure()
    plt.boxplot(X, labels=labels)
    plt.title(f"{metric}: distribution by condition")
    plt.ylabel(metric)
    path = os.path.join(out_dir, f"{metric}_by_condition_boxplot.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def paired_delta_hist(df: pd.DataFrame, metric: str, out_dir: str, idx_cols=["id","model"]):
    for c in idx_cols:
        if c not in df.columns: return
    if metric not in df.columns: return
    p = df.pivot_table(index=idx_cols, columns="condition", values=metric, aggfunc="mean")
    # coerce to numeric
    for col in ["no_desc","with_desc"]:
        if col in p.columns:
            p[col] = _coerce_numeric_series(p[col])
    if not {"no_desc","with_desc"}.issubset(p.columns): return
    p = p.dropna(subset=["no_desc","with_desc"], how="any")
    p["delta"] = p["with_desc"] - p["no_desc"]
    import matplotlib.pyplot as plt
    if len(p["delta"]) == 0:
        return
    plt.figure(); plt.hist(p["delta"].values, bins=20)
    plt.title(f"{metric} lift (with_desc - no_desc)"); plt.xlabel("delta"); plt.ylabel("count")
    path = os.path.join(out_dir, f"{metric}_delta_hist.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    csv = os.path.join(out_dir, f"{metric}_deltas.csv"); p.reset_index().to_csv(csv, index=False)
    return path, csv

def bar_by_model(df: pd.DataFrame, metric: str, out_dir: str):
    if "model" not in df.columns or metric not in df.columns: return
    import numpy as np, matplotlib.pyplot as plt
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

def heatmap_confusions(df: pd.DataFrame, out_dir: str, top_k: int = 20):
    if "predicted_mappings" not in df.columns: return
    import numpy as np, matplotlib.pyplot as plt
    cnt_src = Counter()
    cnt_tgt = Counter()
    pairs = Counter()
    for _, row in df.iterrows():
        pred = parse_mapping_cell(row["predicted_mappings"])
        for s, tlist in pred.items():
            cnt_src[s] += len(tlist)
            for t in tlist:
                cnt_tgt[t] += 1
                pairs[(s,t)] += 1
    top_src = [s for s,_ in cnt_src.most_common(top_k)]
    top_tgt = [t for t,_ in cnt_tgt.most_common(top_k)]
    if not top_src or not top_tgt: return
    M = np.zeros((len(top_src), len(top_tgt)), dtype=float)
    for i,s in enumerate(top_src):
        for j,t in enumerate(top_tgt):
            M[i,j] = pairs.get((s,t), 0)
    plt.figure(figsize=(max(8,len(top_tgt)*0.5), max(6,len(top_src)*0.4)))
    plt.imshow(M, aspect="auto")
    plt.colorbar(label="count")
    plt.yticks(range(len(top_src)), top_src)
    plt.xticks(range(len(top_tgt)), top_tgt, rotation=45, ha="right")
    plt.title("Confusion heatmap: source_B → predicted target_A (counts)")
    path = os.path.join(out_dir, "confusion_heatmap_counts.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()

def bar_top_missed_sources(df: pd.DataFrame, out_dir: str, top_k: int = 20):
    if "ground_truth_mappings" not in df.columns or "predicted_mappings" not in df.columns: return
    import numpy as np, matplotlib.pyplot as plt
    miss_counter = Counter()
    for _, row in df.iterrows():
        gt = parse_mapping_cell(row["ground_truth_mappings"])
        pred = parse_mapping_cell(row["predicted_mappings"])
        for s, gt_targets in gt.items():
            gt_set = set([normalize_token(x) for x in gt_targets])
            pred_set = set([normalize_token(x) for x in pred.get(s, [])])
            if not gt_set.issubset(pred_set):  # either missing or wrong
                miss_counter[normalize_token(s)] += 1
    common = miss_counter.most_common(top_k)
    if not common: return
    labels, counts = zip(*common)
    y = np.arange(len(labels))
    plt.figure(figsize=(10, max(6, len(labels)*0.35)))
    plt.barh(y, counts); plt.yticks(y, labels); plt.gca().invert_yaxis()
    plt.xlabel("count"); plt.title("Top source_B properties with misses (any example)")
    path = os.path.join(out_dir, "top_missed_sources.png")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()

def radar_model_profiles(df: pd.DataFrame, out_dir: str, metrics: Optional[List[str]] = None):
    """Generate radar charts showing model performance profiles across different metrics."""
    if metrics is None:
        metrics = [m for m in [
            "source_prop_overall_accuracy_avg", "source_prop_overall_accuracy_weighted",
            "source_prop_exact_match_accuracy", "source_prop_fuzzy_match_accuracy",
            "source_prop_semantic_match_accuracy", "source_prop_avg_fuzzy_score", "source_prop_avg_semantic_score"
        ] if m in df.columns]
    if not metrics or "model" not in df.columns or "condition" not in df.columns:
        return
    for m in metrics:
        df[m] = _coerce_numeric_series(df[m])
    grp = df.groupby(["model", "condition"])[metrics].mean()
    models = sorted(df["model"].dropna().unique())
    for model in models:
        plt.figure(figsize=(7, 7))
        ax = plt.subplot(111, polar=True)
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        def plot_cond(label):
            if (model, label) not in grp.index:
                return
            vals = grp.loc[(model, label)].tolist()
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=2, label=label)
            ax.fill(angles, vals, alpha=0.1)
        plot_cond("no_desc")
        plot_cond("with_desc")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title(f"Metric profile — {model}")
        ax.set_rlabel_position(30)
        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
        path = os.path.join(out_dir, f"model_radar_{re.sub(r'[^a-zA-Z0-9._-]+', '_', model)}.png")
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()

# ---------------- Orchestration ----------------

EXPECTED_METRICS = [
    "system_accuracy",
    "concept_mapping_accuracy",
    "source_prop_overall_accuracy_avg",
    "source_prop_overall_accuracy_weighted",
    "source_prop_exact_match_accuracy",
    "source_prop_fuzzy_match_accuracy",
    "source_prop_semantic_match_accuracy",
    "source_prop_avg_fuzzy_score",
    "source_prop_avg_semantic_score",
]

def generate_source_mapping_visuals(base_dir: str, file_names: List[str], out_subfolder: str = "visualizations"):
    out_dir = os.path.join(base_dir, out_subfolder); os.makedirs(out_dir, exist_ok=True)

    # load
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

    # ---- NEW: force key metrics numeric (handles bool/strings) ----
    numeric_cols = [
        "system_accuracy",
        "concept_mapping_accuracy",
        "source_prop_overall_accuracy_avg",
        "source_prop_overall_accuracy_weighted",
        "source_prop_exact_match_accuracy",
        "source_prop_fuzzy_match_accuracy",
        "source_prop_semantic_match_accuracy",
        "source_prop_avg_fuzzy_score",
        "source_prop_avg_semantic_score",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = _coerce_numeric_series(data[col])

    # Derived per-row edge metrics
    jaccards = []
    precs, recs, f1s = [], [], []
    over_under_edges = []
    for _, row in data.iterrows():
        gt_map = parse_mapping_cell(row.get("ground_truth_mappings"))
        pr_map = parse_mapping_cell(row.get("predicted_mappings"))
        gt_edges = mapping_to_edges(gt_map)
        pr_edges = mapping_to_edges(pr_map)
        # edge Jaccard
        inter = len(gt_edges & pr_edges); union = len(gt_edges | pr_edges) if (gt_edges or pr_edges) else 1
        jaccards.append(inter/union)
        p,r,f1 = precision_recall_f1(pr_edges, gt_edges)
        precs.append(p); recs.append(r); f1s.append(f1)
        over_under_edges.append(len(pr_edges) - len(gt_edges))
    data["_edge_jaccard"] = jaccards
    data["_edge_precision"] = precs
    data["_edge_recall"] = recs
    data["_edge_f1"] = f1s
    data["_edge_over_under"] = over_under_edges

    # Save tidy
    keep = [c for c in [
        "id" if "id" in data.columns else None,
        "model" if "model" in data.columns else None,
        "condition",
        "system_accuracy" if "system_accuracy" in data.columns else None,
        "concept_mapping_accuracy" if "concept_mapping_accuracy" in data.columns else None,
        "source_prop_overall_accuracy_avg" if "source_prop_overall_accuracy_avg" in data.columns else None,
        "source_prop_overall_accuracy_weighted" if "source_prop_overall_accuracy_weighted" in data.columns else None,
        "source_prop_exact_match_accuracy" if "source_prop_exact_match_accuracy" in data.columns else None,
        "source_prop_fuzzy_match_accuracy" if "source_prop_fuzzy_match_accuracy" in data.columns else None,
        "source_prop_semantic_match_accuracy" if "source_prop_semantic_match_accuracy" in data.columns else None,
        "source_prop_avg_fuzzy_score" if "source_prop_avg_fuzzy_score" in data.columns else None,
        "source_prop_avg_semantic_score" if "source_prop_avg_semantic_score" in data.columns else None,
        "duration_seconds" if "duration_seconds" in data.columns else None,
        "_edge_jaccard", "_edge_precision", "_edge_recall", "_edge_f1", "_edge_over_under",
        "success" if "success" in data.columns else None,
        "error" if "error" in data.columns else None,
    ] if c is not None]
    tidy = data[keep].copy()
    tidy_path = os.path.join(out_dir, "tidy_source_mapping_summary.csv")
    tidy.to_csv(tidy_path, index=False)

    # Visuals: condition-level
    for m in [x for x in EXPECTED_METRICS if x in data.columns]:
        boxplot_by_condition(data, m, out_dir)
        paired_delta_hist(data, m, out_dir, idx_cols=["id","model"])
        bar_by_model(data, m, out_dir)

    # Edge-level condition visuals
    for m in ["_edge_jaccard","_edge_precision","_edge_recall","_edge_f1"]:
        boxplot_by_condition(data, m, out_dir)
        paired_delta_hist(data, m, out_dir, idx_cols=["id","model"])
        bar_by_model(data, m, out_dir)

    # Error structure
    heatmap_confusions(data, out_dir, top_k=20)
    bar_top_missed_sources(data, out_dir, top_k=20)
    
    # Radar model profiles
    radar_model_profiles(data, out_dir)

    # Duration tradeoff
    if "duration_seconds" in data.columns:
        import matplotlib.pyplot as plt
        for m in ["system_accuracy", "_edge_f1"]:
            if m not in data.columns: 
                continue
            plt.figure()
            for cond in data["condition"].unique():
                sub = data[data["condition"]==cond]
                x = _coerce_numeric_series(sub["duration_seconds"])
                y = _coerce_numeric_series(sub[m])
                plt.scatter(x, y, alpha=0.6, label=cond)
            plt.title(f"Duration vs {m}"); plt.xlabel("duration_seconds"); plt.ylabel(m); plt.legend()
            path = os.path.join(out_dir, f"duration_vs_{m}.png")
            plt.savefig(path, bbox_inches="tight"); plt.close()

    return {"out_dir": out_dir, "tidy_csv": tidy_path}

# --------------- CLI ---------------

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Source→System Mapping Visualizations")
    p.add_argument("--base_dir", type=str, required=True)
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--out_subfolder", type=str, default="visualizations")
    return p.parse_args()

def main():
    args = _parse_args()
    generate_source_mapping_visuals(args.base_dir, args.files, args.out_subfolder)

if __name__ == "__main__":
    main()
