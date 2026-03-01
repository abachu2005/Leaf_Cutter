#!/usr/bin/env python3
"""
LeafCutter2 end-to-end pipeline (Python-only launcher)
- Works on macOS / Linux / Windows (Anaconda Prompt)
- No bash scripting required
"""

import argparse, os, sys, shutil, subprocess, json
from pathlib import Path
from typing import List, Dict
from glob import glob

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

    # Save a tiny JSON summary
    summary = {
        "n_junction_inputs": len(bed_paths),
        "junction_file_list": str(junction_filelist),
        "lc2_outputs": lc2_outputs,
        "cluster_files": {
            "perind_counts": str(perind),
            "perind_numers": str(perind_num)
        },
        "notes": "Core pipeline run complete. Use long_exon/nuc_rule outputs for NMD-focused downstream analysis."
    }
    with open(outdir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nDone.")
    print(f"- LC2 cluster ratios: {lc2_outputs['cluster_ratios']}")
    print(f"- LC2 junction counts: {lc2_outputs['junction_counts']}")
    print(f"- Summary: {outdir / 'summary.json'}")

if __name__ == "__main__":
    main()
