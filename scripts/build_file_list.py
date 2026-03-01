#!/usr/bin/env python3
"""
make_test_lists_and_samples.py
Scan junctions_bed_test/{BRAIN,SPLEEN} for *.juncs.bed (BED12),
write:
  - out/junction_files_TEST_BRAIN.txt
  - out/junction_files_TEST_SPLEEN.txt
  - out/junction_files_TEST_ALL.txt
  - out/samples_TEST.tsv   (columns: sample<TAB>condition)
No args; paths are fixed to your Leaf_Cutter project.
"""

from pathlib import Path

ROOT = Path("/Users/abhinavbachu/Leaf_Cutter").expanduser()
BED_ROOT = ROOT / "junctions_bed_test"
OUT = ROOT / "out"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = []  # (path, sample, tissue)

    for tissue in ("BRAIN", "SPLEEN"):
        tdir = BED_ROOT / tissue
        if not tdir.exists():
            print(f"[WARN] Missing folder: {tdir}")
            continue
        for p in sorted(tdir.glob("*.juncs.bed")):
            sample = p.name
            pairs.append((p, sample, tissue))

    # Per-tissue lists
    for tissue in ("BRAIN", "SPLEEN"):
        with open(OUT / f"junction_files_TEST_{tissue}.txt", "w") as fh:
            for p, s, t in pairs:
                if t == tissue:
                    fh.write(str(p) + "\n")

    # Combined list
    with open(OUT / "junction_files_TEST_ALL.txt", "w") as fh:
        for p, s, t in pairs:
            fh.write(str(p) + "\n")

    # samples.tsv
    with open(OUT / "samples_TEST.tsv", "w") as fh:
        fh.write("sample\tcondition\n")
        for _, s, t in pairs:
            fh.write(f"{s}\t{t}\n")

    print("[OK] Wrote:")
    print(" -", OUT / "junction_files_TEST_BRAIN.txt")
    print(" -", OUT / "junction_files_TEST_SPLEEN.txt")
    print(" -", OUT / "junction_files_TEST_ALL.txt")
    print(" -", OUT / "samples_TEST.tsv")

if __name__ == "__main__":
    main()
