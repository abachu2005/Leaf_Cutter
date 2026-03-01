#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


def parse_args():
    ap = argparse.ArgumentParser(
        description="Annotation-only NMD plot using nuc_rule + long_exon tables."
    )
    ap.add_argument("--long_exon", required=True,
                    help="clusters_long_exon_distances.txt")
    ap.add_argument("--nuc_rule", required=True,
                    help="clusters_nuc_rule_distances.txt")
    ap.add_argument("--sample_table", required=True,
                    help="TSV with columns: sample, condition")
    ap.add_argument("--output_prefix", required=True,
                    help="Prefix for output TSV + PNG")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    # -------------------------------
    # Load annotation tables
    # -------------------------------
    long_exon = pd.read_csv(args.long_exon, sep="\t")
    nuc_rule = pd.read_csv(args.nuc_rule, sep="\t")

    # Normalize gene_name column
    if "Gene_name" in long_exon.columns:
        long_exon = long_exon.rename(columns={"Gene_name": "gene_name"})
    if "Gene_name" in nuc_rule.columns:
        nuc_rule = nuc_rule.rename(columns={"Gene_name": "gene_name"})

    # Merge PTC info with EJC distance
    anno = long_exon.merge(
        nuc_rule[["gene_name", "Intron_coord", "ejc_distance"]],
        on=["gene_name", "Intron_coord"],
        how="left"
    )

    # Mark PTC introns
    anno["has_PTC"] = anno["PTC_position"].notna()

    # Fill missing EJC distance with -1 (non-NMD)
    anno["ejc_distance"] = anno["ejc_distance"].fillna(-1)

    # NMD definition (strict LeafCutter2):
    #   intron has PTC and EJC distance > 0
    anno["is_NMD"] = anno["has_PTC"] & (anno["ejc_distance"] > 0)

    # -------------------------------
    # Load sample table
    # -------------------------------
    st = pd.read_csv(args.sample_table, sep=r"\s+", engine="python")
    if not {"sample", "condition"}.issubset(st.columns):
        raise ValueError("sample_table must have sample and condition columns")

    # Count NMD introns (same for every sample)
    n_NMD = int(anno["is_NMD"].sum())
    n_nonNMD = int((~anno["is_NMD"]).sum())

    summary = []
    for cond in sorted(st["condition"].unique()):
        summary.append({
            "condition": cond,
            "NMD_introns": n_NMD,
            "nonNMD_introns": n_nonNMD,
            "fraction_NMD": n_NMD / (n_NMD + n_nonNMD)
        })

    summary_df = pd.DataFrame(summary)
    summary_path = args.output_prefix + "_annotation_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print("Wrote:", summary_path)

    # -------------------------------
    # Plot
    # -------------------------------
    plt.figure(figsize=(6, 4))

    plt.bar(
        summary_df["condition"],
        summary_df["fraction_NMD"] * 100
    )

    plt.ylabel("% of annotated introns that are NMD-triggering")
    plt.title("NMD Potential (Annotation Only)")
    plt.xticks(rotation=30)
    plt.tight_layout()

    fig_path = args.output_prefix + "_annotation_plot.png"
    plt.savefig(fig_path, dpi=300)
    print("Wrote:", fig_path)

    print("Done.")


if __name__ == "__main__":
    main()
