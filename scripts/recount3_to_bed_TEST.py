#!/usr/bin/env python3
"""
recount3_to_bed_TEST.py  — SMOKE TEST (writes BED12)
- Auto-discovers recount3 triplets (MM/RR/ID) in /Users/abhinavbachu/Leaf_Cutter/junctions
- Writes tiny per-sample BED12 to /Users/abhinavbachu/Leaf_Cutter/junctions_bed_test/<TISSUE>/
- Produces file lists in /Users/abhinavbachu/Leaf_Cutter/out/junction_files_TEST_<TISSUE>.txt
- Progress bars shown even in non-TTY IDE consoles.

Caps (edit below):
  TEST_MAX_SAMPLES   : first N samples only
  TEST_MAX_ROWS      : only junction row indices < this
  TEST_MAX_NONZEROS  : stop after this many SEEN MM nonzeros (pre-filter)
"""

from pathlib import Path
from glob import glob
from collections import defaultdict
import gzip, re, sys
import pandas as pd
from tqdm.auto import tqdm

# ---------- Progress bar config ----------
TQDM_CFG = dict(disable=False, ascii=True, dynamic_ncols=True, mininterval=0.1, file=sys.stdout)
TQDM_INNER = dict(TQDM_CFG)
TQDM_INNER.update(dict(mininterval=0, miniters=1))

# ---------- Project paths ----------
ROOT            = Path("/Users/abhinavbachu/Leaf_Cutter").expanduser()
JUNCTIONS_DIR   = ROOT / "junctions"
BED_ROOT        = ROOT / "junctions_bed_test"
OUT_DIR         = ROOT / "out"
TISSUES         = ("BRAIN", "SPLEEN")

# ---------- Test caps / filters ----------
MIN_COUNT          = 1
TEST_MAX_SAMPLES   = 5_000
TEST_MAX_ROWS      = 250_000
TEST_MAX_NONZEROS  = 1_000_000_000
ANCHOR_BP          = 50

# ---------- Helpers ----------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def open_maybe_gzip(p: Path):
    return gzip.open(p, "rt") if str(p).endswith(".gz") else open(p, "rt")

def to_chr(x: str) -> str:
    x = str(x).strip()
    if not x.lower().startswith("chr"):
        if x and (x[0].isdigit() or x in ("X","Y","M")):
            return "chr" + x
    return x

def strand_clean(s: str) -> str:
    s = str(s).strip()
    return s if s in {"+","-"} else "."

# ---------- Loaders ----------
def load_sample_ids(id_path: Path):
    print(f"[IDs] {id_path.name}")
    ids = []
    with open_maybe_gzip(id_path) as fh:
        for line in fh:
            t = line.strip()
            if t:
                ids.append(t)
    if not ids:
        raise RuntimeError(f"No sample IDs in {id_path}")
    ids = [sid for sid in ids if sid != "rail_id"]
    ids = ids[:TEST_MAX_SAMPLES]
    print(f"[IDs] loaded {len(ids)} samples (capped)")
    return ids

def load_row_ranges(rr_path: Path) -> pd.DataFrame:
    print(f"[RR] {rr_path.name}")

    def _try_headered(skiprows=None):
        return pd.read_csv(rr_path, sep="\t", dtype=str, low_memory=False, comment=None, skiprows=skiprows)

    df = None
    for sk in (None, 1):
        try:
            df = _try_headered(skiprows=sk); break
        except Exception:
            df = None
    if df is None:
        raise RuntimeError(f"[RR] Could not read {rr_path}")

    cols = {c.lower(): c for c in df.columns}
    names = list(cols.keys())

    def pick(regexes, exclude=()):
        import re as _re
        for key in names:
            if any(_re.search(rx, key) for rx in regexes) and not any(_re.search(erx, key) for erx in exclude):
                return cols[key]
        return None

    chr_col   = pick([r"^seqnames?$", r"^seqname$", r"^chrom$", r"^chr$", r"seqname", r"chrom"])
    start_col = pick([r"(^|[\.\:_])start($|[\.\:_])", r"^ranges\.start$"])
    end_col   = pick([r"(^|[\.\:_])end($|[\.\:_])",   r"^ranges\.end$"], exclude=[r"friend"])
    strand_col= pick([r"^strand$"])

    if not all([chr_col, start_col, end_col, strand_col]):
        print("[RR] Non-standard header → heuristic...")
        df2 = pd.read_csv(rr_path, sep="\t", header=None, dtype=str, low_memory=False, comment=None)
        strand_j = None
        for j in range(min(12, df2.shape[1])):
            vals = set(df2[j].dropna().astype(str).str.strip().unique())
            if vals.issubset({"+","-",".","*"}):
                strand_j = j; break
        if strand_j is None or df2.shape[1] < 4:
            print("[RR] Preview (first 5 non-empty lines):")
            with open_maybe_gzip(rr_path) as fh:
                shown = 0
                for line in fh:
                    t = line.strip()
                    if not t: continue
                    print("   ", t[:200]); shown += 1
                    if shown >= 5: break
            raise RuntimeError(f"[RR] Could not parse {rr_path}")
        chr_j, start_j, end_j = max(0, strand_j-3), max(0, strand_j-2), max(0, strand_j-1)
        df = df2[[chr_j, start_j, end_j, strand_j]].copy()
        df.columns = ["seqnames","start","end","strand"]
    else:
        df = df[[chr_col, start_col, end_col, strand_col]].copy()
        df.columns = ["seqnames","start","end","strand"]

    before = len(df)
    df["seqnames"] = df["seqnames"].astype(str).map(to_chr)
    df["start"]    = pd.to_numeric(df["start"], errors="coerce").astype("Int64")
    df["end"]      = pd.to_numeric(df["end"],   errors="coerce").astype("Int64")
    df["strand"]   = df["strand"].astype(str).str.strip().map(strand_clean)
    df = df.dropna(subset=["start","end"])
    df = df[(df["start"] > 0) & (df["end"] > 0)]
    dropped = before - len(df)
    if dropped: print(f"[RR] Dropped {dropped} malformed rows")

    df["bed_start"] = df["start"].astype(int) - 1
    df["bed_end"]   = df["end"].astype(int)
    return df.reset_index(drop=True)

def stream_mm(mm_path: Path):
    with open_maybe_gzip(mm_path) as fh:
        header = fh.readline()
        if not header.startswith("%%MatrixMarket"):
            raise RuntimeError(f"{mm_path} not Matrix Market")
        line = fh.readline()
        while line.startswith("%") or not line.strip():
            line = fh.readline()
        nrow, ncol, nnz = map(int, line.strip().split())

    def it():
        with open_maybe_gzip(mm_path) as f2:
            _ = f2.readline()
            line = f2.readline()
            while line.startswith("%") or not line.strip():
                line = f2.readline()
            for line in f2:
                line = line.strip()
                if not line: continue
                i_str, j_str, x_str = line.split()
                i = int(i_str) - 1
                j = int(j_str) - 1
                x = int(float(x_str))
                if x > 0:
                    yield i, j, x
    return nrow, ncol, nnz, it()

# ---------- Discovery ----------
def find_triplet(jdir: Path, tissue: str):
    pat = re.compile(rf"\.{tissue}\.", re.IGNORECASE)
    def pick(glob_pat):
        cands = [Path(p) for p in glob(str(jdir / glob_pat))]
        cands = [p for p in cands if pat.search(p.name)]
        if not cands:
            raise FileNotFoundError(f"No {glob_pat} for {tissue} in {jdir}")
        cands.sort()
        return cands[-1]
    mm  = pick("*UNIQUE.MM.gz")
    rr  = pick("*UNIQUE.RR.gz")
    idf = pick("*UNIQUE.ID.gz")
    print(f"[DISCOVER] {tissue}\n  MM: {mm.name}\n  RR: {rr.name}\n  ID: {idf.name}")
    return mm, rr, idf

# ---------- Conversion ----------
def write_beds_for_tissue(tissue: str):
    mm, rr, idf = find_triplet(JUNCTIONS_DIR, tissue)
    rr_df = load_row_ranges(rr)
    sample_ids = load_sample_ids(idf)

    nrow, ncol, nnz, nz_iter = stream_mm(mm)
    print(f"[MM] rows={nrow} cols={ncol} nnz={nnz} (caps: samples≤{TEST_MAX_SAMPLES}, rows<{TEST_MAX_ROWS}, nz≤{TEST_MAX_NONZEROS})")

    buckets = defaultdict(lambda: defaultdict(int))

    seen = 0
    kept = 0
    skipped_low = 0
    skipped_col = 0
    skipped_row = 0
    cap_total = min(nnz, TEST_MAX_NONZEROS)

    for i, j, x in tqdm(nz_iter, total=cap_total, unit="nz",
                        desc=f"[MM] {tissue} (test)", leave=True, **TQDM_CFG):
        if seen >= TEST_MAX_NONZEROS:
            break
        seen += 1

        if i >= TEST_MAX_ROWS:
            skipped_row += 1
            continue
        if j >= TEST_MAX_SAMPLES:
            skipped_col += 1
            continue
        if x < MIN_COUNT:
            skipped_low += 1
            continue

        buckets[j][i] += x
        kept += 1

    print(f"[MM] done. seen={seen} kept={kept} skip_row={skipped_row} skip_col={skipped_col} skip_low={skipped_low}")

    tissue_dir = ensure_dir(BED_ROOT / tissue.upper())
    bed_list = []

    # ---------- per-sample write (NO INNER TQDM) ----------
    for sidx, sname in tqdm(list(enumerate(sample_ids)),
                            unit="smp", desc=f"[BED] {tissue} samples (test)",
                            leave=True, **TQDM_CFG):

        rows = buckets.get(sidx, {})
        outp = tissue_dir / f"{sname}.juncs.bed"
        bed_list.append(outp)

        row_items = list(rows.items())
        with open(outp, "w") as fw:
            for ridx, cnt in row_items:   # << inner bar removed >>
                if ridx >= len(rr_df):
                    continue
                chrom  = rr_df.at[ridx, "seqnames"]
                start  = int(rr_df.at[ridx, "bed_start"])
                end    = int(rr_df.at[ridx, "bed_end"])
                strand = rr_df.at[ridx, "strand"]
                name   = f"{chrom}:{start}-{end}:{strand}"

                intron_len = max(1, end - start)
                left  = min(ANCHOR_BP, max(1, intron_len // 2 - 1))
                right = min(ANCHOR_BP, max(1, intron_len - left))

                thickStart = start
                thickEnd   = end
                itemRgb    = "0,0,0"
                blockCount = 2
                blockSizes = f"{left},{right}"
                blockStarts= f"0,{intron_len - right}"

                fw.write(
                    f"{chrom}\t{start}\t{end}\t{name}\t{int(cnt)}\t{strand}\t"
                    f"{thickStart}\t{thickEnd}\t{itemRgb}\t{blockCount}\t{blockSizes}\t{blockStarts}\n"
                )

    ensure_dir(OUT_DIR)
    list_path = OUT_DIR / f"junction_files_TEST_{tissue.upper()}.txt"
    with open(list_path, "w") as fh:
        for p in bed_list:
            fh.write(str(p) + "\n")
    print(f"[BED] wrote {len(bed_list)} mini BED12s  →  {tissue_dir}")
    print(f"[LIST] {list_path}")

# ---------- Main ----------
def main():
    print("[SETUP] ROOT:", ROOT)
    if not JUNCTIONS_DIR.exists():
        raise SystemExit(f"junctions dir not found: {JUNCTIONS_DIR}")
    ensure_dir(BED_ROOT); ensure_dir(OUT_DIR)

    for tissue in TISSUES:
        print(f"\n=== {tissue} (TEST) ===")
        write_beds_for_tissue(tissue)

    print("\n[DONE] Mini per-sample BED12 files are under:", BED_ROOT)
    print("[NEXT] Cluster with leafcutter_cluster_regtools.py using the TEST file lists.")

if __name__ == "__main__":
    main()
