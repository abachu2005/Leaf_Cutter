#!/usr/bin/env python3
"""
LeafCutter2 end-to-end pipeline (Python-only launcher)
- Works on macOS / Linux / Windows (Anaconda Prompt)
- No bash scripting required
"""

import argparse, gzip, os, sys, shutil, subprocess, json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------
# Helpers
# ---------------------------

def check_exe(name: str):
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Required executable '{name}' not found on PATH. Install it or activate the right conda env.")
    return path

def run(cmd: List[str], workdir: Path = None):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(workdir) if workdir else None, check=True)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_junction_filelist(paths: List[Path], out_path: Path) -> Path:
    with open(out_path, "w") as fh:
        for p in paths:
            fh.write(str(p) + "\n")
    return out_path

# ---------------------------
# Junction extraction
# ---------------------------

def extract_junctions_from_bams(bams: List[Path], out_dir: Path) -> List[Path]:
    """regtools junctions extract for each BAM -> BED"""
    check_exe("regtools")
    ensure_dir(out_dir)
    out_beds = []
    for bam in tqdm(bams, desc="Extracting junctions with regtools"):
        if not bam.exists():
            raise FileNotFoundError(bam)
        bed = out_dir / (bam.stem + ".juncs.bed")
        cmd = ["regtools", "junctions", "extract", "-o", str(bed), str(bam)]
        run(cmd)
        out_beds.append(bed)
    return out_beds

def convert_star_sj_to_bed(sj_tabs: List[Path], out_dir: Path) -> List[Path]:
    """
    Minimal converter STAR SJ.out.tab -> BED-like with counts in score field.
    Columns in SJ.out.tab (STAR manual):
      1: chr, 2: intronStart, 3: intronEnd, 4: strand(0/1/2), 7: uniquely mapping reads crossing junction
      We output BED6: chr, start(exon end), end(exon start), name, score=count, strand
    """
    ensure_dir(out_dir)
    beds = []
    for sj in tqdm(sj_tabs, desc="Converting STAR SJ.out.tab to BED"):
        if not sj.exists():
            raise FileNotFoundError(sj)
        df = pd.read_csv(sj, sep="\t", header=None, comment="#")
        if df.shape[1] < 7:
            raise ValueError(f"{sj} appears malformed; expected >=7 columns.")
        chr_, start, end, strand_code, uniq = df[0], df[1], df[2], df[3], df[6]
        strand = strand_code.map({0:"." ,1:"+", 2:"-"}).fillna(".")
        # BED is 0-based start; STAR intronStart is 1-based intron start
        bed = pd.DataFrame({
            "chrom": chr_,
            "start": start.astype(int) - 1,
            "end": end.astype(int),
            "name": "JUNC",
            "score": uniq.astype(int),
            "strand": strand
        })
        out = out_dir / (sj.parent.name + "_" + sj.stem + ".bed")
        bed.to_csv(out, sep="\t", header=False, index=False)
        beds.append(out)
    return beds

# ---------------------------
# LeafCutter clustering
# ---------------------------

def leafcutter_cluster(leafcutter_repo: Path, bed_paths: List[Path], out_prefix: Path, min_reads=50, max_intron_len=500000):
    """Call LeafCutter clustering script (Python), using regtools BEDs list"""
    cluster_script = leafcutter_repo / "clustering" / "leafcutter_cluster_regtools.py"
    if not cluster_script.exists():
        raise FileNotFoundError(f"LeafCutter clustering script not found at {cluster_script}")
    filelist = write_junction_filelist(bed_paths, out_prefix.parent / (out_prefix.name + "_junction_files.txt"))
    cmd = [
        sys.executable, str(cluster_script),
        "-j", str(filelist),
        "-m", str(min_reads),
        "-o", str(out_prefix),
        "-l", str(max_intron_len)
    ]
    run(cmd)

    # Expected outputs
    perind = out_prefix.with_name(out_prefix.name + "_perind.counts.gz")
    perind_num = out_prefix.with_name(out_prefix.name + "_perind_numers.counts.gz")
    if not perind.exists() or not perind_num.exists():
        raise RuntimeError("LeafCutter clustering did not produce expected perind files.")
    return perind, perind_num, filelist

# ---------------------------
# LeafCutter2 classification
# ---------------------------

def leafcutter2_classify(
    leafcutter2_repo: Path,
    fasta: Path,
    gtf: Path,
    junction_filelist: Path,
    run_dir: Path,
    out_prefix: str,
    min_cluster_reads: int = 30,
    max_intron_len: int = 100000,
) -> Dict[str, str]:
    lc2 = leafcutter2_repo / "leafcutter2.py"
    if not lc2.exists():
        # sometimes the script is in scripts/
        lc2 = leafcutter2_repo / "scripts" / "leafcutter2.py"
    if not lc2.exists():
        raise FileNotFoundError("leafcutter2.py not found in the provided repo path.")
    cmd = [
        sys.executable, str(lc2),
        "-j", str(junction_filelist),
        "-r", str(run_dir),
        "-o", out_prefix,
        "-A", str(gtf),
        "-G", str(fasta),
        "-m", str(min_cluster_reads),
        "-l", str(max_intron_len),
    ]
    run(cmd)
    expected = [
        run_dir / f"{out_prefix}.cluster_ratios.gz",
        run_dir / f"{out_prefix}.junction_counts.gz",
        run_dir / "clustering" / f"{out_prefix}_long_exon_distances.txt",
        run_dir / "clustering" / f"{out_prefix}_nuc_rule_distances.txt",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise RuntimeError("LeafCutter2 completed but expected outputs are missing:\n  - " + "\n  - ".join(missing))
    return {
        "cluster_ratios": str(expected[0]),
        "junction_counts": str(expected[1]),
        "long_exon_distances": str(expected[2]),
        "nuc_rule_distances": str(expected[3]),
        "run_dir": str(run_dir),
    }

# ---------------------------
# Defaults + parser (Option 1)
# ---------------------------

VALID_CLASSIFICATIONS = {"UP", "PR", "NE", "IN"}


def _bed_path_to_col_name(bed_path: str) -> str:
    """Replicate the column-naming logic from leafcutter_cluster_regtools.py."""
    return Path(bed_path).name.split(".junc")[0]


def _resolve_tissue_labels(
    sample_cols: List[str],
    junction_filelist_path: Path,
    samples_tsv_path: Optional[Path] = None,
) -> List[str]:
    """Map junction_counts column names to tissue/group labels.

    Strategy (in priority order):
      1. BED parent directory names from the junction filelist (GTEx-style).
      2. ``condition`` column from samples_tsv (non-GTEx with metadata).
      3. Use each sample name as its own group (fallback).
    """
    col_to_tissue: Dict[str, str] = {}

    if junction_filelist_path.exists():
        bed_paths = [
            l.strip()
            for l in junction_filelist_path.read_text().splitlines()
            if l.strip()
        ]
        for bp in bed_paths:
            p = Path(bp)
            tissue = p.parent.name
            col_name = _bed_path_to_col_name(bp)
            col_to_tissue[col_name] = tissue
            col_to_tissue[p.name] = tissue
            col_to_tissue[p.stem] = tissue

    tissues = [col_to_tissue.get(c, col_to_tissue.get(Path(c).name, None)) for c in sample_cols]

    unique = set(t for t in tissues if t is not None)
    all_resolved = all(t is not None for t in tissues)
    single_group = len(unique) <= 1

    if (not all_resolved or single_group) and samples_tsv_path and samples_tsv_path.exists():
        meta = pd.read_csv(samples_tsv_path, sep="\t", dtype=str)
        if {"sample", "condition"}.issubset(meta.columns):
            cond_map: Dict[str, str] = {}
            for _, row in meta.iterrows():
                s = str(row["sample"])
                c = str(row["condition"])
                cond_map[s] = c
                cond_map[Path(s).name] = c
                cond_map[Path(s).stem] = c
                for suf in [".juncs.bed", ".juncs", ".bed", ".sorted.gz"]:
                    if s.endswith(suf):
                        cond_map[s[: -len(suf)]] = c
            tissues = [
                cond_map.get(c, cond_map.get(Path(c).name, cond_map.get(Path(c).stem, c)))
                for c in sample_cols
            ]
            return tissues

    return [t if t is not None else c for t, c in zip(tissues, sample_cols)]


def compute_unproductive_by_tissue(
    junction_counts_path: Path,
    junction_filelist_path: Path,
    outdir: Path,
    samples_tsv_path: Optional[Path] = None,
) -> Optional[Dict]:
    """Compute per-tissue unproductive junction read percentages and generate
    a bar-chart PNG, a TSV, and a JSON summary.

    Uses the LC2 classification labels (UP/PR/NE/IN) embedded in
    ``junction_counts.gz`` junction IDs.  Returns metadata dict for
    inclusion in ``summary.json``, or ``None`` on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping unproductive-by-tissue plot")
        return None

    if not junction_counts_path.exists():
        print(f"[WARN] junction_counts not found at {junction_counts_path}")
        return None

    # ---- 1. Parse junction_counts.gz ----
    with gzip.open(str(junction_counts_path), "rt") as f:
        header_tokens = f.readline().strip().split()
        sample_cols = header_tokens[1:]
        n = len(sample_cols)
        if n == 0:
            return None

        total_reads = np.zeros(n, dtype=np.float64)
        up_reads = np.zeros(n, dtype=np.float64)
        has_classification = False

        for line in f:
            fields = line.strip().split()
            if len(fields) < 2:
                continue
            junc_id = fields[0]
            reads = np.array([float(x) for x in fields[1 : n + 1]])

            label = junc_id.rsplit(":", 1)[-1]
            if label in VALID_CLASSIFICATIONS:
                has_classification = True
            total_reads += reads
            if label == "UP":
                up_reads += reads

    if not has_classification:
        print("[WARN] junction_counts has no UP/PR/NE/IN labels — classification may not have run")
        return None

    with np.errstate(divide="ignore", invalid="ignore"):
        pct_per_sample = np.where(
            total_reads > 0, up_reads / total_reads * 100.0, np.nan
        )

    # ---- 2. Resolve tissue labels ----
    tissues = _resolve_tissue_labels(
        sample_cols, junction_filelist_path, samples_tsv_path
    )

    # ---- 3. Aggregate by tissue ----
    tissue_samples: Dict[str, List[float]] = defaultdict(list)
    for i, tissue in enumerate(tissues):
        if not np.isnan(pct_per_sample[i]):
            tissue_samples[tissue].append(pct_per_sample[i])

    if not tissue_samples:
        return None

    rows = []
    for tissue in sorted(tissue_samples):
        vals = np.array(tissue_samples[tissue])
        idxs = [i for i, t in enumerate(tissues) if t == tissue]
        rows.append(
            {
                "tissue": tissue,
                "n_samples": len(vals),
                "mean_pct": round(float(np.mean(vals)), 4),
                "median_pct": round(float(np.median(vals)), 4),
                "std_pct": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0,
                "total_reads": int(sum(total_reads[i] for i in idxs)),
                "unproductive_reads": int(sum(up_reads[i] for i in idxs)),
            }
        )

    result_df = pd.DataFrame(rows).sort_values("mean_pct").reset_index(drop=True)

    # Global statistics
    all_pcts = np.array([v for vs in tissue_samples.values() for v in vs])
    global_weighted_mean = float(np.sum(up_reads[~np.isnan(pct_per_sample)]) /
                                 np.sum(total_reads[~np.isnan(pct_per_sample)]) * 100.0)
    global_median = float(np.median(all_pcts))
    n_tissues = len(tissue_samples)

    # ---- 4. Write TSV ----
    tsv_path = outdir / "unproductive_by_tissue.tsv"
    result_df.to_csv(tsv_path, sep="\t", index=False)

    # ---- 5. Write JSON ----
    json_path = outdir / "unproductive_by_tissue.json"
    json_payload = {
        "n_tissues": n_tissues,
        "n_samples": int(np.sum(~np.isnan(pct_per_sample))),
        "global_weighted_mean_pct": round(global_weighted_mean, 4),
        "global_median_pct": round(global_median, 4),
        "tissues": rows,
    }
    with open(json_path, "w") as fh:
        json.dump(json_payload, fh, indent=2)

    # ---- 6. Plot ----
    png_path = outdir / "unproductive_by_tissue.png"
    _plot_unproductive_chart(result_df, tissue_samples, global_weighted_mean, png_path)

    print(f"[TISSUE] {n_tissues} tissues, weighted mean = {global_weighted_mean:.3f}%")
    print(f"[TISSUE] TSV:  {tsv_path}")
    print(f"[TISSUE] PNG:  {png_path}")
    print(f"[TISSUE] JSON: {json_path}")

    return {
        "png": "unproductive_by_tissue.png",
        "tsv": "unproductive_by_tissue.tsv",
        "json": "unproductive_by_tissue.json",
        "n_tissues": n_tissues,
        "n_samples": int(np.sum(~np.isnan(pct_per_sample))),
        "global_weighted_mean_pct": round(global_weighted_mean, 4),
        "global_median_pct": round(global_median, 4),
    }


def _plot_unproductive_chart(
    result_df: pd.DataFrame,
    tissue_samples: Dict[str, List[float]],
    global_mean: float,
    out_path: Path,
) -> None:
    """Render a sorted bar chart of per-tissue unproductive junction read %."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(result_df)
    fig_width = max(8, n * 0.45 + 2)
    fig_height = max(5, 4.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)

    x = np.arange(n)
    cmap = plt.get_cmap("tab20" if n <= 20 else "gist_rainbow")
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    means = result_df["mean_pct"].values
    stds = result_df["std_pct"].values
    labels = result_df["tissue"].values

    ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="white",
           linewidth=0.5, alpha=0.85, zorder=2)

    for i, tissue in enumerate(labels):
        vals = tissue_samples.get(tissue, [])
        if vals and len(vals) > 1:
            jitter = np.random.default_rng(42).uniform(-0.25, 0.25, size=len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                s=10, alpha=0.5, color="black", zorder=3, linewidths=0,
            )

    ax.axhline(global_mean, color="grey", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)
    ax.text(
        n - 0.5, global_mean, f"  mean = {global_mean:.2f}%",
        va="bottom", ha="right", fontsize=7, color="grey",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("% unproductive junction reads")
    ax.set_title(
        "Unproductive splicing by tissue",
        fontsize=11, fontweight="bold",
    )
    ax.text(
        0.5, 1.01,
        f"{n} tissues  |  weighted mean = {global_mean:.2f}%  |  median = {np.median(means):.2f}%",
        transform=ax.transAxes, ha="center", fontsize=7, color="grey",
    )
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _expand_star_files(patterns: List[str]) -> List[str]:
    """Expand globs and deduplicate while preserving order."""
    paths: List[str] = []
    for pat in patterns:
        expanded = sorted(glob(pat))
        if not expanded and ("*" not in pat and "?" not in pat and "[" not in pat):
            expanded = [pat]
        paths.extend(expanded)
    seen = set()
    uniq = []
    for p in paths:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Python-only LeafCutter + LeafCutter2 pipeline (with sane defaults).")

    # Repos / refs
    p.add_argument("--leafcutter_repo", default="tools/leafcutter",
                   help="Path to LeafCutter repo (default: tools/leafcutter)")
    p.add_argument("--leafcutter2_repo", default="tools/leafcutter2",
                   help="Path to LeafCutter2 repo (default: tools/leafcutter2)")
    p.add_argument("--genome_fasta", default="refs/GRCh38.fa",
                   help="Reference genome FASTA (default: refs/GRCh38.fa)")
    p.add_argument("--gencode_gtf", default="refs/gencode.v46.annotation.gtf",
                   help="GENCODE GTF with CDS/start/stop (default: refs/gencode.v46.annotation.gtf)")

    # Samples
    p.add_argument("--samples_tsv", default="out/samples.tsv",
                   help="Optional sample sheet for downstream analysis; not required for core LC2 run.")

    # Junction inputs (STAR SJ files) — default to Brain_Cortex + Liver from recount3 conversion
    default_sj = (
        sorted(glob("star_sj/Brain_Cortex/*.SJ.out.tab")) +
        sorted(glob("star_sj/Liver/*.SJ.out.tab"))
    )
    p.add_argument("--star_sj", nargs="+", default=default_sj,
                   help="One or more STAR SJ.out.tab files (supports globs)")

    # Alternative BAM route (kept optional)
    p.add_argument("--bams", nargs="*", type=Path, default=None,
                   help="Alternative: BAMs to extract junctions via regtools (if not providing --star_sj)")

    # Pre-made BED files (e.g. from gtex_gct_to_bed.py or recount3_to_bed.py)
    p.add_argument("--junction_beds", nargs="+", default=None,
                   help="Pre-made junction BED files — skip SJ/BAM conversion and go straight to clustering")

    # Run options
    p.add_argument("--workdir", default=".",
                   help="Working/output directory (default: .)")
    p.add_argument("--prefix", default="gtex_demo",
                   help="Run prefix used for output file names (default: gtex_demo)")

    # LeafCutter / LC2 knobs
    p.add_argument("--min_reads", type=int, default=50,
                   help="Minimum reads per intron for clustering (default: 50)")
    p.add_argument("--max_intron_len", type=int, default=500000,
                   help="Maximum intron length for clustering (default: 500000)")
    p.add_argument("--nmd_distance", type=int, default=55,
                   help="Reserved for downstream analysis extensions; unused in core LC2 run.")

    return p

# ---------------------------
# Main
# ---------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Expand any globs passed to --star_sj (or used in defaults)
    sj_files = _expand_star_files(args.star_sj)

    # --- sanity checks ---
    problems = []
    for path in [args.leafcutter_repo, args.leafcutter2_repo, args.genome_fasta, args.gencode_gtf]:
        if not os.path.exists(path):
            problems.append(f"Missing: {path}")
    if not sj_files and not args.bams and not args.junction_beds:
        problems.append("No junction inputs found. Provide --star_sj, --bams, or --junction_beds.")
    if problems:
        msg = "Input validation failed:\n  - " + "\n  - ".join(problems)
        raise SystemExit(msg)

    # Prepare dirs
    wd = ensure_dir(Path(args.workdir))
    jdir = ensure_dir(wd / "junctions")
    cdir = ensure_dir(wd / "clusters")
    lc2dir = ensure_dir(wd / "lc2")
    outdir = ensure_dir(wd / "out")

    # Step 1: Junction files
    if args.junction_beds:
        bed_paths = [Path(p) for p in args.junction_beds]
        missing_beds = [str(p) for p in bed_paths if not p.exists()]
        if missing_beds:
            raise SystemExit(f"Missing BED files:\n  " + "\n  ".join(missing_beds))
        print(f"Using {len(bed_paths)} pre-made junction BED files")
    elif sj_files:
        bed_paths = convert_star_sj_to_bed([Path(p) for p in sj_files], jdir)
    else:
        check_exe("regtools")
        bed_paths = extract_junctions_from_bams([Path(p) for p in args.bams], jdir)

    # Step 2: Cluster (LeafCutter)
    prefix_path = cdir / args.prefix
    perind, perind_num, junction_filelist = leafcutter_cluster(
        Path(args.leafcutter_repo),
        bed_paths,
        prefix_path,
        min_reads=args.min_reads,
        max_intron_len=args.max_intron_len,
    )

    # Step 3: Classify (LeafCutter2) with the real CLI contract:
    #   -j junction_list -r run_dir -o prefix -A gtf -G fasta
    lc2_prefix = args.prefix + "_lc2"
    lc2_outputs = leafcutter2_classify(
        Path(args.leafcutter2_repo),
        Path(args.genome_fasta),
        Path(args.gencode_gtf),
        junction_filelist,
        lc2dir,
        lc2_prefix,
        min_cluster_reads=args.min_reads,
        max_intron_len=args.max_intron_len,
    )

    # Step 4: Per-tissue unproductive junction read analysis
    samples_tsv = Path(args.samples_tsv) if args.samples_tsv else None
    tissue_metrics = None
    try:
        tissue_metrics = compute_unproductive_by_tissue(
            junction_counts_path=Path(lc2_outputs["junction_counts"]),
            junction_filelist_path=junction_filelist,
            outdir=outdir,
            samples_tsv_path=samples_tsv,
        )
    except Exception as exc:
        print(f"[WARN] Tissue unproductive analysis failed: {exc}")

    # Save JSON summary
    summary = {
        "n_junction_inputs": len(bed_paths),
        "junction_file_list": str(junction_filelist),
        "lc2_outputs": lc2_outputs,
        "cluster_files": {
            "perind_counts": str(perind),
            "perind_numers": str(perind_num),
        },
        "metrics": {},
        "notes": "Core pipeline run complete. Use long_exon/nuc_rule outputs for NMD-focused downstream analysis.",
    }
    if tissue_metrics:
        summary["metrics"]["unproductive_by_tissue"] = tissue_metrics

    with open(outdir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nDone.")
    print(f"- LC2 cluster ratios: {lc2_outputs['cluster_ratios']}")
    print(f"- LC2 junction counts: {lc2_outputs['junction_counts']}")
    print(f"- Summary: {outdir / 'summary.json'}")

if __name__ == "__main__":
    main()
