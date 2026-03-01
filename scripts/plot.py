#!/usr/bin/env python3
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

ROOT   = Path("/Users/abhinavbachu/Leaf_Cutter")
COUNTS = ROOT / "lc2/test_demo_wrapped.junction_counts.gz"
SHEET  = ROOT / "out/samples_TEST.tsv"
ANN    = ROOT / "tools/leafcutter2/clustering/test_demo_wrapped_junction_classifications.txt"  # optional

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PNG = SCRIPT_DIR / "two_tissue_bar.png"
OUT_TXT = SCRIPT_DIR / "two_tissue_bar.txt"

def read_ws(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+")
    first = df.columns[0]
    if first != "junction_id":
        df = df.rename(columns={first: "junction_id"})
    df.columns = ["junction_id"] + [str(c) for c in df.columns[1:]]
    return df

def load_samples(sheet: Path) -> pd.DataFrame:
    meta = pd.read_csv(sheet, sep="\t", dtype=str)
    need = {"sample","condition"}
    if not need.issubset(meta.columns):
        raise SystemExit(f"{sheet} must have columns: sample, condition")
    meta["sample"] = meta["sample"].astype(str)
    meta["condition"] = meta["condition"].astype(str)
    meta["stem"] = meta["sample"].str.replace(r"\.juncs\.bed$", "", regex=True)
    meta["base"] = meta["sample"].apply(lambda s: Path(s).name)
    return meta

def find_unproductive_mask(ann_path: Path):
    if not ann_path.exists():
        return None
    ann = pd.read_csv(ann_path, sep="\t", dtype=str)
    key = ann.columns[0]
    ann = ann.rename(columns={key: "junction_id"})
    for c in ann.columns:
        if c == "junction_id": continue
        col = ann[c].astype(str)
        if col.str.contains("unprod", case=False, na=False).any():
            return ann.set_index("junction_id")[c].str.contains("unprod", case=False, na=False)
    return None

CLRE = re.compile(r"(clu_\d+_[+-])")
def clu_id(j):
    m = CLRE.search(j); return m.group(1) if m else j

def pct_non_dominant(counts: pd.DataFrame) -> pd.DataFrame:
    samples = [c for c in counts.columns if c != "junction_id"]
    tmp = counts.copy()
    for s in samples:
        tmp[s] = pd.to_numeric(tmp[s], errors="coerce").fillna(0.0)
    tmp["cluster_id"] = tmp["junction_id"].map(clu_id)
    rows = []
    for _, sub in tmp.groupby("cluster_id"):
        M = sub[samples].to_numpy(float)
        tot = M.sum(axis=0); mx = M.max(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(tot == 0, np.nan, (tot - mx) / tot)
        rows.append(pd.DataFrame({"sample": samples, "frac": frac, "w": tot}))
    per = pd.concat(rows, ignore_index=True)
    outs = []
    for samp, g in per.groupby("sample"):
        mask = (~np.isnan(g["frac"])) & (g["w"] > 0)
        val = float(np.average(g.loc[mask,"frac"], weights=g.loc[mask,"w"]) * 100.0) if np.any(mask) else np.nan
        outs.append({"sample": samp, "metric_pct": val})
    return pd.DataFrame(outs)

def pct_unproductive(counts: pd.DataFrame, unp_mask: pd.Series) -> pd.DataFrame:
    samples = [c for c in counts.columns if c != "junction_id"]
    tmp = counts.copy()
    for s in samples:
        tmp[s] = pd.to_numeric(tmp[s], errors="coerce").fillna(0.0)
    is_unp = tmp["junction_id"].map(unp_mask).fillna(False).to_numpy()
    M = tmp[samples].to_numpy(float)
    tot = M.sum(axis=0)
    up  = M[is_unp, :].sum(axis=0) if is_unp.any() else np.zeros_like(tot)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(tot == 0, np.nan, up / tot * 100.0)
    return pd.DataFrame({"sample": samples, "metric_pct": pct})

def robust_map_samples(metric_df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    # Build multiple mappings
    exact = dict(zip(meta["sample"], meta["condition"]))
    by_stem = dict(zip(meta["stem"], meta["condition"]))
    by_base = dict(zip(meta["base"], meta["condition"]))

    df = metric_df.copy()
    # try exact
    df["condition"] = df["sample"].map(exact)
    # try stems
    miss = df["condition"].isna()
    if miss.any():
        df.loc[miss, "condition"] = df.loc[miss, "sample"].str.replace(r"\.juncs\.bed$", "", regex=True).map(by_stem)
    # try basenames
    miss = df["condition"].isna()
    if miss.any():
        df.loc[miss, "condition"] = df.loc[miss, "sample"].apply(lambda s: Path(s).name).map(by_base)
    return df

def summarize(vals):
    arr = pd.to_numeric(vals, errors="coerce").dropna().to_numpy()
    n = arr.size
    return {
        "mean": float(np.mean(arr)) if n else np.nan,
        "sd": float(np.std(arr, ddof=1)) if n>1 else np.nan,
        "median": float(np.median(arr)) if n else np.nan,
        "p25": float(np.percentile(arr,25)) if n else np.nan,
        "p75": float(np.percentile(arr,75)) if n else np.nan,
        "n": int(n),
        "vals": arr
    }

def main():
    counts = read_ws(COUNTS)
    meta   = load_samples(SHEET)
    print(f"[INFO] counts columns (samples): {counts.columns[1:].tolist()}")
    print(f"[INFO] meta samples: {meta['sample'].tolist()}")

    unp = find_unproductive_mask(ANN)
    if unp is not None:
        metric = pct_unproductive(counts, unp)
        metric_name = "% unproductive junction reads (LC2)"
    else:
        metric = pct_non_dominant(counts)
        metric_name = "% non-dominant usage (proxy)"

    df = robust_map_samples(metric, meta)
    # keep only the two tissues
    df = df[df["condition"].isin(["BRAIN","SPLEEN"])].copy()

    n_brain = df.loc[df["condition"]=="BRAIN","metric_pct"].notna().sum()
    n_spleen = df.loc[df["condition"]=="SPLEEN","metric_pct"].notna().sum()
    print(f"[MAP] BRAIN samples mapped: {n_brain}; SPLEEN samples mapped: {n_spleen}")
    if n_brain==0 and n_spleen==0:
        raise SystemExit("[ERR] No samples mapped to BRAIN/SPLEEN. Check names in samples_TEST.tsv vs counts header.")

    s_b = summarize(df.loc[df["condition"]=="BRAIN","metric_pct"])
    s_s = summarize(df.loc[df["condition"]=="SPLEEN","metric_pct"])

    # Welch t-test if we have >=2 per group
    t_stat, p_val = np.nan, np.nan
    if s_b["n"]>=2 and s_s["n"]>=2:
        t_stat, p_val = ttest_ind(s_b["vals"], s_s["vals"], equal_var=False, nan_policy="omit")

    # Plot
    labels = ["BRAIN","SPLEEN"]; means=[s_b["mean"], s_s["mean"]]; sds=[s_b["sd"], s_s["sd"]]
    x = np.arange(2)
    plt.figure(figsize=(5.2,3.8), dpi=180)
    plt.bar(x, means, yerr=sds, capsize=5)
    plt.xticks(x, labels)
    plt.ylabel(metric_name)
    plt.title("Two-tissue comparison (test run)")
    plt.tight_layout()
    plt.savefig(OUT_PNG); plt.close()

    # Report
    def fmt(s, name):
        iqr = s["p75"] - s["p25"] if np.isfinite(s["p75"]) and np.isfinite(s["p25"]) else np.nan
        return f"{name}: mean={s['mean']:.3f}, sd={s['sd']:.3f}, median={s['median']:.3f}, IQR={iqr:.3f}, n={s['n']}"
    lines = []
    lines.append("Two-tissue comparison — TEST run")
    lines.append(f"Metric: {metric_name}")
    lines.append(fmt(s_b,"BRAIN"))
    lines.append(fmt(s_s,"SPLEEN"))
    lines.append(f"Welch t-test: t={t_stat:.3f}, p={p_val:.3g}")
    lines.append("Per-sample values (%):")
    for nm, s in [("BRAIN", s_b), ("SPLEEN", s_s)]:
        vals_str = ", ".join(f"{v:.3f}" for v in s["vals"])
        lines.append(f"  {nm}: {vals_str}")
    (SCRIPT_DIR / "two_tissue_bar.txt").write_text("\n".join(lines))

    print(f"[DONE] {OUT_PNG}")
    print(f"[DONE] {OUT_TXT}")

if __name__ == "__main__":
    main()
