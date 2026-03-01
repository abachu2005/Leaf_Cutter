#!/usr/bin/env python3

"""
Compute NMD-like splicing using nearest-annotated intron matching.

This version DOES NOT require strict intron coordinate matching.
Instead, each intron in the perind file is mapped to the *nearest*
annotated intron from long_exon + nuc_rule annotation tables.

NMD-like intron = has_PTC (from long_exon) AND ejc_distance > 0.

This works for recount3-derived synthetic intron coordinates.
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser(
        description="Compute NMD-like splicing using nearest-intron annotation matching."
    )
    ap.add_argument("--perind", required=True)
    ap.add_argument("--long_exon", required=True)
    ap.add_argument("--nuc_rule", required=True)
    ap.add_argument("--sample_table", required=True)
    ap.add_argument("--output_prefix", required=True)
    return ap.parse_args()


def intron_coord_from_index(idx: str) -> tuple:
    """
    Convert LC2 intron ID like:
        chr1:939460:941143:clu_000001
    into numeric tuple:
        ("chr1", start, end)
    """
    p = idx.split(":")
    return p[0], int(p[1]), int(p[2])


def nearest_match(chr_, start, end, annot):
    """
    Given an intron (chr,start,end) and an annotation table (subset per chromosome),
    return annotation row of nearest annotated intron by Manhattan distance.
    """
    df = annot.get(chr_)
    if df is None or df.empty:
        return None

    d = np.abs(df["start"] - start) + np.abs(df["end"] - end)
    idx = d.idxmin()
    return df.loc[idx]


def main():
    args = parse_args()

    # --------------------------------------------------------------------------
    # Load annotation tables (long_exon: PTC; nuc_rule: EJ distance)
    # --------------------------------------------------------------------------
    print("[LOAD] long_exon + nuc_rule annotations")
    lex = pd.read_csv(args.long_exon, sep="\t")
    ncr = pd.read_csv(args.nuc_rule, sep="\t")

    # Normalize gene name columns
    if "Gene_name" in lex.columns:
        lex = lex.rename(columns={"Gene_name": "gene_name"})
    if "Gene_name" in ncr.columns:
        ncr = ncr.rename(columns={"Gene_name": "gene_name"})

    # Extract coordinates
    def split_coord(df):
        df["chr"] = df["Intron_coord"].str.split(":").str[0]
        coords = df["Intron_coord"].str.split(":").str[1]
        df["start"] = coords.str.split("-").str[0].astype(int)
        df["end"] = coords.str.split("-").str[1].astype(int)
        return df

    lex = split_coord(lex)
    ncr = split_coord(ncr)

    # Mark PTC introns
    lex["has_PTC"] = lex["PTC_position"].notna()

    # Merge PTC + EJ distance
    anno = lex.merge(
        ncr[["gene_name", "Intron_coord", "ejc_distance"]],
        on=["gene_name", "Intron_coord"],
        how="left"
    )
    anno["ejc_distance"] = anno["ejc_distance"].fillna(-1)
    anno["is_NMD"] = anno["has_PTC"] & (anno["ejc_distance"] > 0)

    # Pre-split annotation by chromosome for faster nearest lookup
    annot_by_chr = {c: df for c, df in anno.groupby("chr")}

    print(f"[INFO] Annotated introns: {len(anno)}")
    print(f"[INFO] NMD-like annotated introns: {anno['is_NMD'].sum()}")

    # --------------------------------------------------------------------------
    # Load perind file
    # --------------------------------------------------------------------------
    print("[LOAD] perind:", args.perind)

    header_df = pd.read_csv(args.perind, sep=r"\s+", header=None, nrows=1, engine="python")
    header = header_df.iloc[0, :].tolist()

    body = pd.read_csv(args.perind, sep=r"\s+", header=None, skiprows=1, engine="python")
    col_names = ["intron_id"] + [str(s) for s in header]
    body.columns = col_names

    sample_cols = col_names[1:]

    # Convert intron_id → chr,start,end tuple
    print("[PARSE] intron coordinates")
    intron_coords = body["intron_id"].apply(intron_coord_from_index)
    body["chr"] = intron_coords.apply(lambda x: x[0])
    body["start"] = intron_coords.apply(lambda x: x[1])
    body["end"] = intron_coords.apply(lambda x: x[2])

    # --------------------------------------------------------------------------
    # NEAREST annotation matching (Option B)
    # --------------------------------------------------------------------------
    print("[MATCH] nearest annotated introns for all perind introns...")

    nearest_PTClist = []
    nearest_ejc_list = []
    nearest_isNMD = []

    for i, r in body.iterrows():
        annot_row = nearest_match(r["chr"], r["start"], r["end"], annot_by_chr)
        if annot_row is None:
            nearest_PTClist.append(False)
            nearest_ejc_list.append(-1)
            nearest_isNMD.append(False)
            continue

        nearest_PTClist.append(bool(annot_row["has_PTC"]))
        nearest_ejc_list.append(int(annot_row["ejc_distance"]))
        nearest_isNMD.append(bool(annot_row["is_NMD"]))

    body["nearest_has_PTC"] = nearest_PTClist
    body["nearest_ejc"] = nearest_ejc_list
    body["nearest_is_NMD"] = nearest_isNMD

    print("[INFO] Nearest-mapped NMD-like introns:", body["nearest_is_NMD"].sum())

    # --------------------------------------------------------------------------
    # Compute per-sample % NMD
    # --------------------------------------------------------------------------
    print("[COMPUTE] NMD fractions per sample")
    total_reads = body[sample_cols].sum()

    nmd_reads = body.loc[body["nearest_is_NMD"], sample_cols].sum()

    frac_nmd = (nmd_reads / total_reads).fillna(0)
    percent_nmd = frac_nmd * 100
    percent_nmd.name = "percent_NMD"

    out_sample = args.output_prefix + "_percent_NMD_by_sample.tsv"
    percent_nmd.to_frame().to_csv(out_sample, sep="\t")
    print("[WRITE]", out_sample)

    # --------------------------------------------------------------------------
    # Aggregate by condition
    # --------------------------------------------------------------------------
    samp = pd.read_csv(args.sample_table, sep=r"\s+")

    def clean_sample(s):
        for suf in [".bed", ".juncs", ".juncs.bed"]:
            if s.endswith(suf):
                return s[: -len(suf)]
        return s

    samp["sample_clean"] = samp["sample"].apply(clean_sample)

    df = percent_nmd.to_frame().reset_index().rename(columns={"index": "sample"})
    df = df.merge(samp[["sample_clean", "condition"]], left_on="sample", right_on="sample_clean", how="left")

    cond_summary = df.groupby("condition")["percent_NMD"].agg(["mean", "std", "count"])

    out_cond = args.output_prefix + "_percent_NMD_by_condition.tsv"
    cond_summary.to_csv(out_cond, sep="\t")
    print("[WRITE]", out_cond)

    # --------------------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------------------
    print("[PLOT] Saving barplot")
    plt.figure(figsize=(6,4))
    plt.bar(cond_summary.index, cond_summary["mean"], yerr=cond_summary["std"], capsize=5)
    plt.ylabel("% junction reads in NMD-like introns (nearest-match)")
    plt.title("Nearest-annotation NMD-like splicing by condition")
    plt.tight_layout()

    fig_path = args.output_prefix + "_percent_NMD_by_condition.png"
    plt.savefig(fig_path, dpi=300)
    print("[WRITE FIG]", fig_path)

    print("[DONE]")


if __name__ == "__main__":
    main()
