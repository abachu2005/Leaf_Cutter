#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

ROOT   = Path("/Users/abhinavbachu/Leaf_Cutter")
COUNTS = ROOT / "lc2/test_demo_wrapped.junction_counts.gz"
EXON   = ROOT / "clustering/test_demo_wrapped_exon_stats.txt"
LONG   = ROOT / "clustering/test_demo_wrapped_long_exon_distances.txt"
NUC    = ROOT / "clustering/test_demo_wrapped_nuc_rule_distances.txt"
SHEET  = ROOT / "out/samples_TEST.tsv"

HERE   = Path(__file__).resolve().parent
TXT    = HERE / "unproductive_by_tissue_DEBUG.txt"
PNG    = HERE / "unproductive_by_tissue_DEBUG.png"

def read_ws(path):
    return pd.read_csv(path, sep=r"\s+", engine="c")

def id_to_range(junc_id: str) -> str:
    s = str(junc_id)
    parts = s.split(":")
    if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
        return f"{parts[0]}:{parts[1]}-{parts[2]}"
    m = re.search(r"^([^:]+):(\d+)-(\d+)$", s)
    return m.group(0) if m else np.nan

def stem(name: str) -> str:
    for suf in (".juncs.bed", ".juncs", ".bed", ".sorted.gz"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name

def map_samples(count_cols, meta_samples):
    exact = {c: c for c in count_cols if c in meta_samples}
    remaining_c = [c for c in count_cols if c not in exact]
    remaining_m = [m for m in meta_samples if m not in exact]
    by_stem = {}
    for m in remaining_m:
        by_stem.setdefault(stem(m), []).append(m)
    out = dict(exact)
    for c in remaining_c:
        st = stem(c)
        if st in by_stem and by_stem[st]:
            out[c] = by_stem[st][0]
    return out

def parse_min_distance(row):
    def nums(x):
        if pd.isna(x): return []
        return [float(t) for t in re.findall(r"[-+]?\d*\.?\d+", str(x))]
    pts = nums(row["PTC_position"])
    lens = nums(row["Exon_length"])
    if not pts or not lens: return np.nan
    return float(np.nanmin([L - p for L in lens for p in pts]))

def summarize_and_plot(pct_df, out_txt, out_png, title_note):
    conds, means, sds = [], [], []
    lines = []
    for cond, sub in pct_df.groupby("condition"):
        v = sub["pct_unproductive"].dropna().values
        if len(v) == 0: continue
        conds.append(cond)
        means.append(np.nanmean(v))
        sds.append(np.nanstd(v, ddof=1) if len(v)>1 else 0.0)
        lines.append(f"[{cond}] n={len(v)} mean={np.nanmean(v):.4f} sd={np.nanstd(v, ddof=1) if len(v)>1 else np.nan:.4f}")
    with open(out_txt, "a") as fh:
        fh.write("\n=== Tissue summary ===\n" + "\n".join(lines) + "\n")

    x = np.arange(len(conds))
    fig, ax = plt.subplots(figsize=(7,4), dpi=150)
    ax.bar(x, means, yerr=sds, capsize=6)
    ax.set_xticks(x); ax.set_xticklabels(conds)
    ax.set_ylabel("% unproductive junction reads")
    ax.set_title(f"Unproductive splicing by tissue (test run) — {title_note}")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png)

def main():
    with open(TXT, "w") as fh:
        fh.write("[SETUP]\n")

    # Load counts
    counts = read_ws(COUNTS)
    jcol = counts.columns[0]
    sample_cols = [c for c in counts.columns if c != jcol]
    counts["intron_key"] = counts[jcol].map(id_to_range)
    with open(TXT, "a") as fh:
        fh.write(f"Counts rows={len(counts)} samples={len(sample_cols)}\n")
        fh.write(f"First headers: {sample_cols[:5]}\n")

    # Sample sheet
    meta = pd.read_csv(SHEET, sep="\t")
    meta["sample"] = meta["sample"].astype(str)
    meta["condition"] = meta["condition"].astype(str)
    name_map = map_samples(sample_cols, meta["sample"])
    smap = pd.DataFrame({"counts_col": list(name_map.keys()),
                         "meta_sample": list(name_map.values())})
    with open(TXT, "a") as fh:
        fh.write(f"Matched {smap['meta_sample'].notna().sum()} / {len(sample_cols)} columns to sample sheet\n")

    # exon + long
    exon = read_ws(EXON)
    long = read_ws(LONG)
    if "Intron_coord" not in exon or "Exons_after" not in exon:
        raise ValueError("exon_stats is missing required columns.")
    if not {"Intron_coord","PTC_position","Exon_length"}.issubset(long.columns):
        raise ValueError("long_exon_distances missing required columns.")
    long["dist_to_last_junc"] = long.apply(parse_min_distance, axis=1)

    merged = counts.merge(exon[["Intron_coord","Exons_after"]],
                          left_on="intron_key", right_on="Intron_coord", how="left")
    merged = merged.merge(long[["Intron_coord","dist_to_last_junc"]],
                          on="Intron_coord", how="left")

    n_with_exon = merged["Exons_after"].notna().sum()
    n_with_long = merged["dist_to_last_junc"].notna().sum()
    with open(TXT, "a") as fh:
        fh.write(f"Matched exon_stats: {n_with_exon} rows; long_exon_distances: {n_with_long} rows\n")

    ex_after = pd.to_numeric(merged["Exons_after"], errors="coerce")
    last_exon = (ex_after <= 0.5)
    dist = pd.to_numeric(merged["dist_to_last_junc"], errors="coerce")
    nmd_geom = (dist >= 55) & (~last_exon)
    n_flag = int(nmd_geom.sum())
    with open(TXT, "a") as fh:
        fh.write(f"Rows passing (dist>=55 and not last_exon): {n_flag}\n")

    def per_sample_percent(df_mask):
        sum_unprod = merged.loc[df_mask, sample_cols].sum(axis=0)
        sum_total  = merged.loc[:, sample_cols].sum(axis=0).replace(0, np.nan)
        pct = (sum_unprod / sum_total) * 100.0
        out = pd.DataFrame({"sample": pct.index, "pct_unproductive": pct.values})
        # map to condition
        out["sample_join"] = out["sample"].map(name_map).fillna(out["sample"])
        out = out.merge(meta[["sample","condition"]],
                        left_on="sample_join", right_on="sample",
                        how="left", suffixes=("", "_meta"))
        out["condition"] = out["condition"].fillna("NA")
        # NOTE: after merge we have columns: 'sample' (left), 'sample_meta' (right)
        return out[["sample","condition","pct_unproductive"]]

    used_rule = "exon+long"
    pct_df = per_sample_percent(nmd_geom)

    if n_flag == 0:
        used_rule = "nuc_rule_distances + exon_stats"
        nuc = read_ws(NUC)
        if not {"Intron_coord","ejc_distance"}.issubset(nuc.columns):
            raise ValueError("nuc_rule_distances missing required columns.")
        m2 = counts.merge(exon[["Intron_coord","Exons_after"]],
                          left_on="intron_key", right_on="Intron_coord", how="left")
        m2 = m2.merge(nuc[["Intron_coord","ejc_distance"]], on="Intron_coord", how="left")
        ex_after2 = pd.to_numeric(m2["Exons_after"], errors="coerce")
        last_exon2 = (ex_after2 <= 0.5)
        ejc = pd.to_numeric(m2["ejc_distance"], errors="coerce")
        mask2 = (ejc >= 55) & (~last_exon2)
        with open(TXT, "a") as fh:
            fh.write(f"[Fallback] rows passing ejc>=55 and not last_exon: {int(mask2.sum())}\n")
        # reuse merged for contributor listing
        global merged  # so we can reuse below consistently
        merged = m2
        pct_df = per_sample_percent(mask2)
        nmd_geom = mask2

    # Top contributors
    contrib = merged.loc[nmd_geom, [jcol, "intron_key"] + sample_cols].copy()
    contrib["sum_counts"] = contrib[sample_cols].sum(axis=1)
    contrib = contrib.sort_values("sum_counts", ascending=False).head(20)
    with open(TXT, "a") as fh:
        fh.write(f"\n=== Top contributing junctions (rule={used_rule}) ===\n")
        for _, r in contrib.iterrows():
            fh.write(f"{r['intron_key']}  total={int(r['sum_counts'])}\n")

    summarize_and_plot(pct_df, TXT, PNG, title_note=used_rule)
    print(f"[DONE] {PNG}\n[LOG]  {TXT}")

if __name__ == "__main__":
    main()
