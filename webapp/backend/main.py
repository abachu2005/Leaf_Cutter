#!/usr/bin/env python3
"""
LeafCutter2 Web API — FastAPI backend for local and Quest/Slurm pipeline execution.
"""
from __future__ import annotations

import json
import os
import re
import signal
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

APP_ROOT = Path(__file__).resolve().parents[2]  # repo root
DATA_ROOT = APP_ROOT / "webapp" / "data"
JOBS_ROOT = DATA_ROOT / "jobs"
DB_PATH = DATA_ROOT / "jobs.db"
FRONTEND_DIR = APP_ROOT / "webapp" / "frontend"

GTEX_CACHE = DATA_ROOT / "gtex_cache"
GTEX_GCT_URL = (
    "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/"
    "GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct.gz"
)
GTEX_GCT_FILENAME = "GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct.gz"
GTEX_ANNOT_URL = (
    "https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/"
    "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
)
GTEX_ANNOT_FILENAME = "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"

# ---------------------------------------------------------------------------
# Thread-safe state
# ---------------------------------------------------------------------------

DB_LOCK = threading.Lock()
ACTIVE_WORKERS: Dict[str, threading.Thread] = {}
ACTIVE_PROCS: Dict[str, subprocess.Popen] = {}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_paths() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    GTEX_CACHE.mkdir(parents=True, exist_ok=True)


def download_if_missing(url: str, dest: Path) -> Path:
    """Download a file via curl/wget if it does not already exist locally."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("curl"):
        subprocess.run(
            ["curl", "-fSL", "--progress-bar", "-o", str(dest), url],
            check=True,
        )
    elif shutil.which("wget"):
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(dest), url],
            check=True,
        )
    else:
        import urllib.request
        urllib.request.urlretrieve(url, str(dest))
    return dest


def ensure_gtex_annotations() -> Path:
    """Download and cache the GTEx sample annotations; return local path."""
    return download_if_missing(GTEX_ANNOT_URL, GTEX_CACHE / GTEX_ANNOT_FILENAME)


def parse_gtex_tissues(annot_path: Path) -> List[Dict[str, Any]]:
    """Parse annotation TSV and return sorted list of {tissue, sample_count}."""
    from collections import Counter

    counts: Counter = Counter()
    with open(annot_path, "r") as fh:
        header = fh.readline().strip().split("\t")
        try:
            tis_col = header.index("SMTSD")
        except ValueError:
            return []
        for line in fh:
            fields = line.strip().split("\t")
            if len(fields) > tis_col and fields[tis_col]:
                counts[fields[tis_col]] += 1

    return sorted(
        [{"tissue": t, "sample_count": c} for t, c in counts.items()],
        key=lambda x: x["tissue"],
    )


# ---------------------------------------------------------------------------
# Database helpers (SQLite)
# ---------------------------------------------------------------------------


def init_db() -> None:
    ensure_paths()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id              TEXT PRIMARY KEY,
                mode            TEXT NOT NULL,
                status          TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                work_dir        TEXT NOT NULL,
                input_payload   TEXT NOT NULL,
                config_payload  TEXT NOT NULL,
                quest_job_id    TEXT,
                quest_account   TEXT,
                quest_partition TEXT,
                error           TEXT,
                summary_path    TEXT,
                artifacts_zip   TEXT,
                runner_ref      TEXT
            )
            """
        )
        for col, typedef in [
            ("progress_pct", "INTEGER DEFAULT 0"),
            ("progress_label", "TEXT DEFAULT ''"),
        ]:
            try:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {typedef}")
            except sqlite3.OperationalError:
                pass
        conn.commit()


def db_execute(query: str, params: tuple = ()) -> None:
    with DB_LOCK:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(query, params)
            conn.commit()


def db_fetch_one(query: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    with DB_LOCK:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(query, params).fetchone()


def db_fetch_all(query: str, params: tuple = ()) -> List[sqlite3.Row]:
    with DB_LOCK:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(query, params).fetchall()


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = now_iso()
    keys = list(fields.keys())
    values = [fields[k] for k in keys]
    set_clause = ", ".join(f"{k}=?" for k in keys)
    db_execute(f"UPDATE jobs SET {set_clause} WHERE id=?", tuple(values + [job_id]))


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    for k in ("input_payload", "config_payload"):
        if d.get(k):
            try:
                d[k] = json.loads(d[k])
            except (json.JSONDecodeError, TypeError):
                pass
    return d


def get_job_or_404(job_id: str) -> Dict[str, Any]:
    row = db_fetch_one("SELECT * FROM jobs WHERE id=?", (job_id,))
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _row_to_dict(row)


# ---------------------------------------------------------------------------
# Shell / SSH helpers
# ---------------------------------------------------------------------------


def run_local_cmd(
    cmd: List[str], cwd: Optional[Path] = None
) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, text=True, capture_output=True)


def _ssh_base_opts() -> List[str]:
    """Common SSH/SCP options: batch mode, auto-accept host keys, timeout, explicit key."""
    ssh_key = Path.home() / ".ssh" / "id_ed25519"
    opts = [
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=30",
    ]
    if ssh_key.exists():
        opts += ["-i", str(ssh_key)]
    return opts


def run_ssh_cmd(host: str, user: str, remote_cmd: str) -> subprocess.CompletedProcess:
    cmd = ["ssh"] + _ssh_base_opts() + [f"{user}@{host}", remote_cmd]
    try:
        return run_local_cmd(cmd, cwd=APP_ROOT)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() if exc.stderr else f"exit status {exc.returncode}"
        raise subprocess.CalledProcessError(
            exc.returncode, exc.cmd, output=exc.stdout, stderr=exc.stderr
        ) from RuntimeError(f"SSH failed: {detail}")


def run_scp_cmd(srcs: List[str], dest: str, recursive: bool = False) -> subprocess.CompletedProcess:
    cmd = ["scp"] + _ssh_base_opts()
    if recursive:
        cmd.append("-r")
    cmd += srcs + [dest]
    try:
        return run_local_cmd(cmd, cwd=APP_ROOT)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() if exc.stderr else f"exit status {exc.returncode}"
        raise subprocess.CalledProcessError(
            exc.returncode, exc.cmd, output=exc.stdout, stderr=exc.stderr
        ) from RuntimeError(f"SCP failed: {detail}")


# ---------------------------------------------------------------------------
# Pipeline command builder
# ---------------------------------------------------------------------------


def build_pipeline_cmd(
    run_dir: Path,
    star_sj_paths: List[Path],
    prefix: str,
    leafcutter_repo: str,
    leafcutter2_repo: str,
    genome_fasta: str,
    gencode_gtf: str,
    min_reads: int,
    max_intron_len: int,
    samples_tsv: Optional[Path] = None,
) -> List[str]:
    cmd = [
        "python3",
        str(APP_ROOT / "scripts" / "lc2_pipeline.py"),
        "--workdir", str(run_dir),
        "--prefix", prefix,
        "--leafcutter_repo", leafcutter_repo,
        "--leafcutter2_repo", leafcutter2_repo,
        "--genome_fasta", genome_fasta,
        "--gencode_gtf", gencode_gtf,
        "--min_reads", str(min_reads),
        "--max_intron_len", str(max_intron_len),
        "--star_sj",
    ]
    cmd.extend(str(p) for p in star_sj_paths)
    if samples_tsv is not None:
        cmd.extend(["--samples_tsv", str(samples_tsv)])
    return cmd


# ---------------------------------------------------------------------------
# Artifact packaging
# ---------------------------------------------------------------------------


def package_outputs(job_id: str, work_dir: Path) -> Optional[Path]:
    out_dir = work_dir / "out"
    if not out_dir.exists():
        return None
    zip_base = work_dir / f"{job_id}_artifacts"
    return Path(shutil.make_archive(str(zip_base), "zip", root_dir=str(out_dir)))


# ---------------------------------------------------------------------------
# Slurm state mapping
# ---------------------------------------------------------------------------

SLURM_MAP_SUBMITTED = {"PENDING", "CONFIGURING", "REQUEUED"}
SLURM_MAP_RUNNING = {"RUNNING", "COMPLETING", "SUSPENDED"}
SLURM_MAP_SUCCEEDED = {"COMPLETED"}
SLURM_MAP_CANCELLED = {"CANCELLED", "CANCELLED+"}
SLURM_MAP_FAILED = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}


def map_slurm_state(raw: str) -> str:
    state = (raw or "").upper().strip()
    if state in SLURM_MAP_SUBMITTED:
        return "submitted"
    if state in SLURM_MAP_RUNNING:
        return "running"
    if state in SLURM_MAP_SUCCEEDED:
        return "succeeded"
    if state in SLURM_MAP_CANCELLED or state.startswith("CANCELLED"):
        return "cancelled"
    if state in SLURM_MAP_FAILED:
        return "failed"
    return "submitted"


# ---------------------------------------------------------------------------
# Progress tracking helpers
# ---------------------------------------------------------------------------

_RE_GCT_ROWS = re.compile(r"\.\.\.\s+([\d,]+)/([\d,]+)\s+rows")
_RE_PIPELINE_CLUSTER = re.compile(r">>\s+.*leafcutter_cluster")
_RE_PIPELINE_LC2 = re.compile(r">>\s+.*leafcutter2")
_RE_PIPELINE_DONE = re.compile(r"^Done\.\s*$", re.MULTILINE)


def _read_log_tail(log_path: Path, tail_bytes: int = 8192) -> str:
    """Read the last tail_bytes of a log file."""
    if not log_path.exists():
        return ""
    try:
        size = log_path.stat().st_size
        with open(log_path, "rb") as f:
            f.seek(max(0, size - tail_bytes))
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _gct_row_progress(log_tail: str, pct_min: int, pct_max: int) -> int:
    """Extract GCT conversion progress from row-count markers in log tail."""
    matches = list(_RE_GCT_ROWS.finditer(log_tail))
    if not matches:
        return pct_min
    last = matches[-1]
    current = int(last.group(1).replace(",", ""))
    total = int(last.group(2).replace(",", ""))
    if total <= 0:
        return pct_min
    frac = min(1.0, current / total)
    return pct_min + int(frac * (pct_max - pct_min))


def _upload_pipeline_progress(log_tail: str) -> tuple:
    """Progress for upload-source local pipeline (full 10-95 range)."""
    if _RE_PIPELINE_DONE.search(log_tail):
        return 95, "Pipeline complete"
    if _RE_PIPELINE_LC2.search(log_tail):
        return 70, "Classifying splicing events..."
    if _RE_PIPELINE_CLUSTER.search(log_tail):
        return 40, "Clustering junctions..."
    return 10, "Running pipeline..."


def _gtex_pipeline_progress(log_tail: str) -> tuple:
    """Progress for GTEx-source pipeline phase (60-92 range)."""
    if _RE_PIPELINE_DONE.search(log_tail):
        return 92, "Pipeline complete"
    if _RE_PIPELINE_LC2.search(log_tail):
        return 80, "Classifying splicing events..."
    if _RE_PIPELINE_CLUSTER.search(log_tail):
        return 65, "Clustering junctions..."
    return 60, "Running pipeline..."


def _poll_proc(
    job_id: str,
    proc: subprocess.Popen,
    log_path: Path,
    default_pct: int,
    default_label: str,
    progress_fn=None,
    interval: float = 3.0,
) -> int:
    """Poll subprocess until it exits, updating progress periodically.

    progress_fn(log_tail: str) -> (pct: int, label: str)
    Returns the subprocess exit code.
    """
    update_job(job_id, progress_pct=default_pct, progress_label=default_label)
    while proc.poll() is None:
        time.sleep(interval)
        if progress_fn:
            try:
                tail = _read_log_tail(log_path)
                pct, label = progress_fn(tail)
                update_job(job_id, progress_pct=pct, progress_label=label)
            except Exception:
                pass
    return proc.returncode


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------


def local_worker(job_id: str, cmd: List[str], work_dir: Path) -> None:
    log_file = work_dir / "pipeline.log"
    update_job(job_id, status="running", progress_pct=5, progress_label="Starting pipeline...")
    try:
        with open(log_file, "w") as log:
            proc = subprocess.Popen(
                cmd, cwd=str(APP_ROOT), stdout=log, stderr=subprocess.STDOUT, text=True,
            )
            ACTIVE_PROCS[job_id] = proc
            update_job(job_id, runner_ref=str(proc.pid))
            rc = _poll_proc(
                job_id, proc, log_file, 10, "Running pipeline...",
                _upload_pipeline_progress,
            )

        if rc < 0:
            update_job(job_id, status="cancelled", progress_label="Terminated", error=f"Terminated by signal {-rc}")
            return
        if rc != 0:
            tail = ""
            try:
                lines = log_file.read_text().splitlines()
                tail = "\n".join(lines[-20:])
            except Exception:
                pass
            update_job(job_id, status="failed", progress_label="Failed", error=f"Exit code {rc}\n{tail}")
            return

        summary_path = work_dir / "out" / "summary.json"
        zip_path = package_outputs(job_id, work_dir)
        update_job(
            job_id,
            status="succeeded",
            progress_pct=100,
            progress_label="Complete",
            summary_path=str(summary_path) if summary_path.exists() else None,
            artifacts_zip=str(zip_path) if zip_path else None,
        )
    except Exception as exc:
        update_job(job_id, status="failed", progress_label="Error", error=f"{type(exc).__name__}: {exc}")
    finally:
        ACTIVE_PROCS.pop(job_id, None)
        ACTIVE_WORKERS.pop(job_id, None)


def slurm_worker(
    job_id: str,
    host: str,
    user: str,
    remote_run_dir: str,
    remote_out_dir: str,
) -> None:
    try:
        job = get_job_or_404(job_id)
        slurm_id = job["quest_job_id"]
        poll_interval = 15
        update_job(job_id, progress_pct=5, progress_label="Submitted to cluster")

        while True:
            try:
                sq = run_ssh_cmd(host, user, f"squeue -h -j {slurm_id} -o %T")
                sq_state = sq.stdout.strip()
            except subprocess.CalledProcessError:
                sq_state = ""

            if sq_state:
                mapped = map_slurm_state(sq_state)
                if mapped == "running":
                    update_job(job_id, status="running", progress_pct=30, progress_label="Running on cluster")
                else:
                    update_job(job_id, status=mapped, progress_pct=10, progress_label="Queued on cluster")
                time.sleep(poll_interval)
                continue

            # Job left the queue — get final state from sacct
            try:
                sa = run_ssh_cmd(host, user, f"sacct -n -P -j {slurm_id} --format=State,ExitCode")
                first_line = sa.stdout.strip().splitlines()[0] if sa.stdout.strip() else ""
                final_state = first_line.split("|")[0].strip() if first_line else "UNKNOWN"
            except (subprocess.CalledProcessError, IndexError):
                final_state = "UNKNOWN"

            mapped = map_slurm_state(final_state)

            if mapped == "cancelled":
                update_job(job_id, status="cancelled", progress_label="Cancelled", error=f"Slurm: {final_state}")
                return
            if mapped != "succeeded":
                update_job(job_id, status="failed", progress_label="Failed", error=f"Slurm: {final_state}")
                return

            # Succeeded — copy outputs back
            update_job(job_id, progress_pct=85, progress_label="Retrieving outputs...")
            local_work = Path(get_job_or_404(job_id)["work_dir"])
            local_out = local_work / "out"
            local_out.mkdir(parents=True, exist_ok=True)
            try:
                run_scp_cmd([f"{user}@{host}:{remote_out_dir}/."], str(local_out), recursive=True)
            except subprocess.CalledProcessError as e:
                update_job(job_id, status="failed", progress_label="Output retrieval failed", error=f"Output retrieval failed: {e}")
                return

            summary_path = local_out / "summary.json"
            zip_path = package_outputs(job_id, local_work)
            update_job(
                job_id,
                status="succeeded",
                progress_pct=100,
                progress_label="Complete",
                summary_path=str(summary_path) if summary_path.exists() else None,
                artifacts_zip=str(zip_path) if zip_path else None,
            )
            return

    except Exception as exc:
        update_job(job_id, status="failed", progress_label="Error", error=f"{type(exc).__name__}: {exc}")
    finally:
        ACTIVE_WORKERS.pop(job_id, None)


def gtex_local_worker(job_id: str, tissues: str, work_dir: Path, config: Dict[str, Any]) -> None:
    """Background worker: download GTEx GCT, convert to BED, run pipeline."""
    log_file = work_dir / "pipeline.log"
    update_job(job_id, status="running", progress_pct=2, progress_label="Preparing...")
    try:
        with open(log_file, "w") as log:
            # Phase 1: ensure annotations + GCT are cached
            log.write("[GTEx] Ensuring annotations are cached ...\n")
            log.flush()
            update_job(job_id, progress_pct=5, progress_label="Downloading annotations...")

            def _dl(url: str, dest: Path, label: str) -> bool:
                if dest.exists():
                    return True
                log.write(f"[GTEx] Downloading {label} from {url}\n")
                log.flush()
                if shutil.which("curl"):
                    cmd = ["curl", "-fSL", "--progress-bar", "-o", str(dest), url]
                elif shutil.which("wget"):
                    cmd = ["wget", "-q", "--show-progress", "-O", str(dest), url]
                else:
                    log.write("[GTEx] ERROR: neither curl nor wget found\n")
                    return False
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True).returncode
                return rc == 0

            annot_path = GTEX_CACHE / GTEX_ANNOT_FILENAME
            if not _dl(GTEX_ANNOT_URL, annot_path, "annotations"):
                update_job(job_id, status="failed", progress_label="Download failed", error="Annotations download failed")
                return

            update_job(job_id, progress_pct=10, progress_label="Downloading junction data...")
            gct_path = GTEX_CACHE / GTEX_GCT_FILENAME
            if not gct_path.exists():
                log.write("[GTEx] This will take a while on the first run (~4 GB) ...\n")
                log.flush()
            if not _dl(GTEX_GCT_URL, gct_path, "junction GCT"):
                gct_path.unlink(missing_ok=True)
                update_job(job_id, status="failed", progress_label="Download failed", error="GCT download failed")
                return
            else:
                log.write("[GTEx] Using cached GCT\n")
                log.flush()

            update_job(job_id, progress_pct=20, progress_label="Data ready")

            # Phase 2: convert GCT -> per-sample BEDs
            bed_dir = work_dir / "junctions_bed"
            log.write(f"[GTEx] Converting GCT for tissues: {tissues}\n")
            log.flush()
            convert_cmd = [
                "python3",
                str(APP_ROOT / "scripts" / "gtex_gct_to_bed.py"),
                "--gct", str(gct_path),
                "--annotations", str(annot_path),
                "--tissues", tissues,
                "--outdir", str(bed_dir),
                "--max_samples_per_tissue", str(config.get("max_samples_per_tissue", 0)),
            ]
            proc = subprocess.Popen(
                convert_cmd, cwd=str(APP_ROOT), stdout=log, stderr=subprocess.STDOUT, text=True,
            )
            ACTIVE_PROCS[job_id] = proc
            rc = _poll_proc(
                job_id, proc, log_file, 25, "Converting GCT to BED...",
                lambda tail: (_gct_row_progress(tail, 25, 55), "Converting GCT to BED..."),
            )
            if rc != 0:
                update_job(job_id, status="failed", progress_label="Conversion failed", error=f"GCT-to-BED conversion failed (exit {rc})")
                return

            filelist = bed_dir / "junction_files.txt"
            if not filelist.exists():
                update_job(job_id, status="failed", progress_label="Conversion failed", error="Conversion produced no junction_files.txt")
                return

            bed_files = [l.strip() for l in filelist.read_text().splitlines() if l.strip()]
            log.write(f"[GTEx] Conversion done: {len(bed_files)} BED files\n")
            log.flush()
            update_job(job_id, progress_pct=55, progress_label="Conversion complete")

            # Phase 3: run the LeafCutter2 pipeline using --junction_beds
            pipeline_cmd = [
                "python3",
                str(APP_ROOT / "scripts" / "lc2_pipeline.py"),
                "--workdir", str(work_dir),
                "--prefix", config.get("prefix", "gtex_run"),
                "--leafcutter_repo", config.get("leafcutter_repo", "tools/leafcutter"),
                "--leafcutter2_repo", config.get("leafcutter2_repo", "tools/leafcutter2"),
                "--genome_fasta", config.get("genome_fasta", "refs/GRCh38.fa"),
                "--gencode_gtf", config.get("gencode_gtf", "refs/gencode.v46.annotation.gtf"),
                "--min_reads", str(config.get("min_reads", 50)),
                "--max_intron_len", str(config.get("max_intron_len", 500000)),
                "--junction_beds",
            ] + bed_files

            log.write("[Pipeline] Starting LeafCutter2 pipeline ...\n")
            log.flush()
            proc = subprocess.Popen(
                pipeline_cmd, cwd=str(APP_ROOT), stdout=log, stderr=subprocess.STDOUT, text=True,
            )
            ACTIVE_PROCS[job_id] = proc
            update_job(job_id, runner_ref=str(proc.pid))
            rc = _poll_proc(
                job_id, proc, log_file, 60, "Starting pipeline...",
                _gtex_pipeline_progress,
            )

        if rc != 0:
            tail = ""
            try:
                lines = log_file.read_text().splitlines()
                tail = "\n".join(lines[-20:])
            except Exception:
                pass
            update_job(job_id, status="failed", progress_label="Pipeline failed", error=f"Pipeline failed (exit {rc})\n{tail}")
            return

        update_job(job_id, progress_pct=95, progress_label="Packaging outputs...")
        summary_path = work_dir / "out" / "summary.json"
        zip_path = package_outputs(job_id, work_dir)
        update_job(
            job_id,
            status="succeeded",
            progress_pct=100,
            progress_label="Complete",
            summary_path=str(summary_path) if summary_path.exists() else None,
            artifacts_zip=str(zip_path) if zip_path else None,
        )
    except Exception as exc:
        update_job(job_id, status="failed", progress_label="Error", error=f"{type(exc).__name__}: {exc}")
    finally:
        ACTIVE_PROCS.pop(job_id, None)
        ACTIVE_WORKERS.pop(job_id, None)


# ---------------------------------------------------------------------------
# File upload helper
# ---------------------------------------------------------------------------


def write_upload(upload: UploadFile, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as fh:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    return target


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="LeafCutter2 Pipeline", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()


# Serve frontend static files
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def root() -> FileResponse:
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(str(index), media_type="text/html")


# ---------------------------------------------------------------------------
# GET /gtex/tissues — list available GTEx tissues
# ---------------------------------------------------------------------------


@app.get("/gtex/tissues")
def gtex_tissues() -> JSONResponse:
    try:
        annot = ensure_gtex_annotations()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch GTEx annotations: {exc}")
    return JSONResponse(parse_gtex_tissues(annot))


# ---------------------------------------------------------------------------
# POST /jobs — create and dispatch a pipeline job
# ---------------------------------------------------------------------------


@app.post("/jobs")
async def create_job(
    mode: str = Form("local"),
    source: str = Form("upload"),
    gtex_tissues_csv: str = Form(""),
    prefix: str = Form("web_run"),
    min_reads: int = Form(50),
    max_intron_len: int = Form(500000),
    leafcutter_repo: str = Form("tools/leafcutter"),
    leafcutter2_repo: str = Form("tools/leafcutter2"),
    genome_fasta: str = Form("refs/GRCh38.fa"),
    gencode_gtf: str = Form("refs/gencode.v46.annotation.gtf"),
    files: List[UploadFile] = File(default=[]),
    samples_tsv: Optional[UploadFile] = File(default=None),
    # Quest/Slurm fields
    quest_host: str = Form("login.quest.northwestern.edu"),
    quest_user: str = Form(""),
    quest_account: str = Form(""),
    quest_partition: str = Form(""),
    remote_repo_root: str = Form(""),
    remote_work_root: str = Form("~/leafcutter_jobs"),
    remote_juncfiles: str = Form(""),
    max_samples_per_tissue: int = Form(0),
) -> JSONResponse:
    mode = mode.strip().lower()
    if mode not in {"local", "slurm"}:
        raise HTTPException(status_code=400, detail="mode must be 'local' or 'slurm'")

    job_id = str(uuid.uuid4())
    work_dir = JOBS_ROOT / job_id
    inputs_dir = work_dir / "inputs"
    sj_dir = inputs_dir / "star_sj"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded files
    local_sj_files: List[Path] = []
    for upload in files:
        if not upload.filename:
            continue
        local_sj_files.append(write_upload(upload, sj_dir / Path(upload.filename).name))

    local_samples: Optional[Path] = None
    if samples_tsv and samples_tsv.filename:
        local_samples = write_upload(samples_tsv, inputs_dir / "samples.tsv")

    config = {
        "prefix": prefix,
        "min_reads": min_reads,
        "max_intron_len": max_intron_len,
        "leafcutter_repo": leafcutter_repo,
        "leafcutter2_repo": leafcutter2_repo,
        "genome_fasta": genome_fasta,
        "gencode_gtf": gencode_gtf,
        "quest_host": quest_host,
        "quest_user": quest_user,
        "quest_account": quest_account,
        "quest_partition": quest_partition,
        "remote_repo_root": remote_repo_root,
        "remote_work_root": remote_work_root,
        "max_samples_per_tissue": max_samples_per_tissue,
    }
    source = source.strip().lower()
    gtex_tissues_list = [t.strip() for t in gtex_tissues_csv.split(",") if t.strip()]

    input_payload = {
        "source": source,
        "star_sj_files": [str(p) for p in local_sj_files],
        "samples_tsv": str(local_samples) if local_samples else None,
        "remote_juncfiles": remote_juncfiles or None,
        "gtex_tissues": gtex_tissues_list if source == "gtex" else None,
    }

    db_execute(
        """
        INSERT INTO jobs
        (id, mode, status, created_at, updated_at, work_dir, input_payload, config_payload,
         quest_job_id, quest_account, quest_partition, error, summary_path, artifacts_zip, runner_ref)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, NULL, NULL, NULL, NULL)
        """,
        (
            job_id, mode, "queued", now_iso(), now_iso(), str(work_dir),
            json.dumps(input_payload), json.dumps(config),
            quest_account or None, quest_partition or None,
        ),
    )

    # ---- LOCAL MODE ----
    if mode == "local":
        if source == "gtex":
            if not gtex_tissues_list:
                update_job(job_id, status="failed", error="No GTEx tissues selected.")
                return JSONResponse(status_code=400, content={"job_id": job_id, "error": "No GTEx tissues selected."})
            t = threading.Thread(
                target=gtex_local_worker,
                args=(job_id, ",".join(gtex_tissues_list), work_dir, config),
                daemon=True,
            )
            ACTIVE_WORKERS[job_id] = t
            t.start()
            return JSONResponse({"job_id": job_id, "status": "queued", "mode": "local", "source": "gtex"})

        # source == "upload" (default)
        if not local_sj_files:
            update_job(job_id, status="failed", error="No STAR SJ files uploaded.")
            return JSONResponse(status_code=400, content={"job_id": job_id, "error": "No STAR SJ files uploaded."})
        cmd = build_pipeline_cmd(
            run_dir=work_dir,
            star_sj_paths=local_sj_files,
            prefix=prefix,
            leafcutter_repo=leafcutter_repo,
            leafcutter2_repo=leafcutter2_repo,
            genome_fasta=genome_fasta,
            gencode_gtf=gencode_gtf,
            min_reads=min_reads,
            max_intron_len=max_intron_len,
            samples_tsv=local_samples,
        )
        t = threading.Thread(target=local_worker, args=(job_id, cmd, work_dir), daemon=True)
        ACTIVE_WORKERS[job_id] = t
        t.start()
        return JSONResponse({"job_id": job_id, "status": "queued", "mode": "local"})

    # ---- SLURM MODE ----
    missing = []
    if not quest_user:
        missing.append("quest_user")
    if not quest_account:
        missing.append("quest_account")
    if not quest_partition:
        missing.append("quest_partition")
    if not remote_repo_root:
        missing.append("remote_repo_root")
    if missing:
        msg = f"Slurm mode requires: {', '.join(missing)}"
        update_job(job_id, status="failed", error=msg)
        raise HTTPException(status_code=400, detail=msg)

    try:
        # Resolve ~ to absolute home path on the remote host so that
        # SBATCH directives (which don't expand ~) and double-quoted
        # shell strings all work correctly.
        if remote_work_root.startswith("~"):
            home_result = run_ssh_cmd(quest_host, quest_user, "echo $HOME")
            remote_home = home_result.stdout.strip()
            remote_work_root = remote_work_root.replace("~", remote_home, 1)

        remote_run_dir = f"{remote_work_root.rstrip('/')}/{job_id}"
        run_ssh_cmd(quest_host, quest_user, f"mkdir -p {remote_run_dir}/inputs")

        rr = remote_repo_root.rstrip("/")
        sbatch_preamble = [
            "#!/bin/bash",
            "set -eo pipefail",
            f"#SBATCH --account={quest_account}",
            f"#SBATCH --partition={quest_partition}",
            "#SBATCH --time=48:00:00",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --mem=32G",
            f"#SBATCH --job-name=lc2_{job_id[:8]}",
            f"#SBATCH --output={remote_run_dir}/slurm_%j.out",
            f"#SBATCH --error={remote_run_dir}/slurm_%j.err",
            "",
            f"cd {rr}",
            "module purge",
            "module load anaconda3/2022.05",
            "",
        ]

        if source == "gtex":
            if not gtex_tissues_list:
                raise HTTPException(status_code=400, detail="No GTEx tissues selected.")
            tissues_escaped = ",".join(gtex_tissues_list)
            gtex_cache_remote = f"{remote_run_dir}/gtex_cache"
            bed_dir_remote = f"{remote_run_dir}/junctions_bed"
            max_spt = config.get("max_samples_per_tissue", 0)
            sbatch_body = "\n".join(sbatch_preamble + [
                f"mkdir -p {gtex_cache_remote}",
                "",
                f'ANNOT="{gtex_cache_remote}/{GTEX_ANNOT_FILENAME}"',
                f'GCT="{gtex_cache_remote}/{GTEX_GCT_FILENAME}"',
                "",
                f'[ -f "$ANNOT" ] || wget -q -O "$ANNOT" "{GTEX_ANNOT_URL}"',
                f'[ -f "$GCT" ] || wget -q --show-progress -O "$GCT" "{GTEX_GCT_URL}"',
                "",
                (
                    f'python3 {rr}/scripts/gtex_gct_to_bed.py'
                    f' --gct "$GCT"'
                    f' --annotations "$ANNOT"'
                    f' --tissues "{tissues_escaped}"'
                    f' --outdir {bed_dir_remote}'
                    f' --max_samples_per_tissue {max_spt}'
                ),
                "",
                (
                    f"python3 {rr}/scripts/lc2_pipeline.py"
                    f" --workdir {remote_run_dir}"
                    f" --prefix {prefix}"
                    f" --leafcutter_repo {rr}/{leafcutter_repo}"
                    f" --leafcutter2_repo {rr}/{leafcutter2_repo}"
                    f" --genome_fasta {rr}/{genome_fasta}"
                    f" --gencode_gtf {rr}/{gencode_gtf}"
                    f" --min_reads {min_reads}"
                    f" --max_intron_len {max_intron_len}"
                    f" --junction_beds $(cat {bed_dir_remote}/junction_files.txt)"
                ),
            ]) + "\n"
        else:
            # Upload source — resolve junction file list on Quest
            remote_junc_list = remote_juncfiles.strip()
            if not remote_junc_list:
                if not local_sj_files:
                    raise HTTPException(status_code=400, detail="Provide remote_juncfiles or upload STAR SJ files.")
                remote_lines: List[str] = []
                for p in local_sj_files:
                    rpath = f"{remote_run_dir}/inputs/{p.name}"
                    run_scp_cmd([str(p)], f"{quest_user}@{quest_host}:{rpath}")
                    remote_lines.append(rpath)
                local_list = work_dir / "remote_junction_files.txt"
                local_list.write_text("\n".join(remote_lines) + "\n")
                remote_junc_list = f"{remote_run_dir}/inputs/junction_files.txt"
                run_scp_cmd([str(local_list)], f"{quest_user}@{quest_host}:{remote_junc_list}")

            sbatch_body = "\n".join(sbatch_preamble + [
                (
                    f"python3 scripts/lc2_pipeline.py"
                    f" --workdir {remote_run_dir}"
                    f" --prefix {prefix}"
                    f" --leafcutter_repo {rr}/{leafcutter_repo}"
                    f" --leafcutter2_repo {rr}/{leafcutter2_repo}"
                    f" --genome_fasta {rr}/{genome_fasta}"
                    f" --gencode_gtf {rr}/{gencode_gtf}"
                    f" --min_reads {min_reads}"
                    f" --max_intron_len {max_intron_len}"
                    f" --star_sj $(cat {remote_junc_list})"
                ),
            ]) + "\n"

        slurm_script = work_dir / "job.sbatch"
        slurm_script.write_text(sbatch_body)
        remote_script = f"{remote_run_dir}/job.sbatch"
        run_scp_cmd([str(slurm_script)], f"{quest_user}@{quest_host}:{remote_script}")

        submit = run_ssh_cmd(quest_host, quest_user, f"sbatch --parsable {remote_script}")
        quest_job_id = submit.stdout.strip().split(";")[0]
        if not quest_job_id:
            raise RuntimeError("sbatch returned no job ID")

        update_job(job_id, status="submitted", quest_job_id=quest_job_id)

        t = threading.Thread(
            target=slurm_worker,
            args=(job_id, quest_host, quest_user, remote_run_dir, f"{remote_run_dir}/out"),
            daemon=True,
        )
        ACTIVE_WORKERS[job_id] = t
        t.start()
        return JSONResponse({
            "job_id": job_id, "status": "submitted", "mode": "slurm", "quest_job_id": quest_job_id,
        })

    except HTTPException:
        raise
    except subprocess.CalledProcessError as exc:
        ssh_err = exc.stderr.strip() if exc.stderr else str(exc)
        update_job(job_id, status="failed", error=f"SSH/SCP error: {ssh_err}")
        raise HTTPException(status_code=500, detail=f"Slurm submission failed: {ssh_err}")
    except Exception as exc:
        update_job(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail=f"Slurm submission failed: {exc}")


# ---------------------------------------------------------------------------
# GET /jobs — list all jobs
# ---------------------------------------------------------------------------


@app.get("/jobs")
def list_jobs() -> JSONResponse:
    rows = db_fetch_all("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 100")
    return JSONResponse([_row_to_dict(r) for r in rows])


# ---------------------------------------------------------------------------
# GET /jobs/{id}
# ---------------------------------------------------------------------------


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str) -> JSONResponse:
    return JSONResponse(get_job_or_404(job_id))


# ---------------------------------------------------------------------------
# GET /jobs/{id}/results
# ---------------------------------------------------------------------------


@app.get("/jobs/{job_id}/results")
def get_job_results(job_id: str) -> JSONResponse:
    job = get_job_or_404(job_id)
    summary = None
    sp = job.get("summary_path")
    if sp and Path(sp).exists():
        try:
            summary = json.loads(Path(sp).read_text())
        except Exception:
            pass

    log_tail = None
    log_path = Path(job["work_dir"]) / "pipeline.log"
    if log_path.exists():
        try:
            lines = log_path.read_text().splitlines()
            log_tail = lines[-50:]
        except Exception:
            pass

    return JSONResponse({
        "job_id": job_id,
        "status": job["status"],
        "error": job.get("error"),
        "summary": summary,
        "log_tail": log_tail,
    })


# ---------------------------------------------------------------------------
# GET /jobs/{id}/logs — incremental log streaming
# ---------------------------------------------------------------------------


@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str, offset: int = 0, limit: int = 65536) -> JSONResponse:
    job = get_job_or_404(job_id)
    log_path = Path(job["work_dir"]) / "pipeline.log"
    terminal = job["status"] in {"succeeded", "failed", "cancelled"}

    if not log_path.exists():
        return JSONResponse({"lines": [], "next_offset": 0, "eof": terminal})

    try:
        size = log_path.stat().st_size
    except OSError:
        return JSONResponse({"lines": [], "next_offset": offset, "eof": True})

    if offset >= size:
        return JSONResponse({"lines": [], "next_offset": offset, "eof": terminal})

    with open(log_path, "rb") as f:
        f.seek(offset)
        raw = f.read(limit)

    new_offset = offset + len(raw)
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    eof = new_offset >= size and terminal

    return JSONResponse({"lines": lines, "next_offset": new_offset, "eof": eof})


# ---------------------------------------------------------------------------
# GET /jobs/{id}/download
# ---------------------------------------------------------------------------


@app.get("/jobs/{job_id}/download")
def download_artifacts(job_id: str) -> FileResponse:
    job = get_job_or_404(job_id)
    zp = job.get("artifacts_zip")
    if not zp or not Path(zp).exists():
        raise HTTPException(status_code=404, detail="Artifacts not available yet.")
    return FileResponse(path=zp, filename=f"{job_id}_artifacts.zip", media_type="application/zip")


# ---------------------------------------------------------------------------
# GET /jobs/{id}/artifacts/{filename} — single artifact download
# ---------------------------------------------------------------------------

_ARTIFACT_MIME = {
    ".png": "image/png",
    ".tsv": "text/tab-separated-values",
    ".json": "application/json",
    ".txt": "text/plain",
    ".gz": "application/gzip",
}


@app.get("/jobs/{job_id}/artifacts/{filename}")
def download_single_artifact(job_id: str, filename: str) -> FileResponse:
    job = get_job_or_404(job_id)
    safe_name = Path(filename).name
    if safe_name != filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    artifact = Path(job["work_dir"]) / "out" / safe_name
    if not artifact.exists():
        raise HTTPException(status_code=404, detail=f"Artifact '{safe_name}' not found")

    mime = _ARTIFACT_MIME.get(artifact.suffix, "application/octet-stream")
    return FileResponse(path=str(artifact), filename=safe_name, media_type=mime)


# ---------------------------------------------------------------------------
# POST /jobs/{id}/cancel
# ---------------------------------------------------------------------------


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> JSONResponse:
    job = get_job_or_404(job_id)

    if job["status"] in {"succeeded", "failed", "cancelled"}:
        return JSONResponse({"job_id": job_id, "status": job["status"], "message": "Job already terminal."})

    # Local cancel
    if job["mode"] == "local":
        proc = ACTIVE_PROCS.get(job_id)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        else:
            pid_str = job.get("runner_ref")
            if pid_str:
                try:
                    os.kill(int(pid_str), signal.SIGTERM)
                except (ProcessLookupError, ValueError, PermissionError):
                    pass
        update_job(job_id, status="cancelled", error="Cancelled by user")
        ACTIVE_PROCS.pop(job_id, None)
        return JSONResponse({"job_id": job_id, "status": "cancelled"})

    # Slurm cancel
    cfg = job.get("config_payload", {})
    slurm_id = job.get("quest_job_id")
    host = cfg.get("quest_host", "login.quest.northwestern.edu")
    user = cfg.get("quest_user")
    if slurm_id and user:
        try:
            run_ssh_cmd(host, user, f"scancel {slurm_id}")
        except Exception:
            pass
    update_job(job_id, status="cancelled", error="Cancelled by user")
    return JSONResponse({"job_id": job_id, "status": "cancelled", "quest_job_id": slurm_id})


# ---------------------------------------------------------------------------
# DELETE /jobs/{id} — delete a single job (files + DB row)
# ---------------------------------------------------------------------------


def _delete_job_impl(job: Dict[str, Any]) -> Dict[str, Any]:
    """Shared logic: cancel if active, remove local/remote files, delete DB row."""
    job_id = job["id"]
    result: Dict[str, Any] = {"job_id": job_id, "local_cleaned": False, "remote_cleaned": False}

    # Cancel if still active
    if job["status"] in ACTIVE:
        if job["mode"] == "local":
            proc = ACTIVE_PROCS.get(job_id)
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            ACTIVE_PROCS.pop(job_id, None)
        else:
            cfg = job.get("config_payload", {})
            slurm_id = job.get("quest_job_id")
            host = cfg.get("quest_host", "login.quest.northwestern.edu")
            user = cfg.get("quest_user")
            if slurm_id and user:
                try:
                    run_ssh_cmd(host, user, f"scancel {slurm_id}")
                except Exception:
                    pass
        ACTIVE_WORKERS.pop(job_id, None)

    # Remove local work directory
    work_dir = Path(job["work_dir"])
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
        result["local_cleaned"] = True

    # Remove remote job directory (best-effort)
    if job["mode"] == "slurm":
        cfg = job.get("config_payload", {})
        host = cfg.get("quest_host", "login.quest.northwestern.edu")
        user = cfg.get("quest_user")
        remote_work_root = cfg.get("remote_work_root", "")
        if user and remote_work_root:
            remote_dir = f"{remote_work_root.rstrip('/')}/{job_id}"
            try:
                run_ssh_cmd(host, user, f"rm -rf {remote_dir}")
                result["remote_cleaned"] = True
            except Exception:
                pass

    # Delete DB row
    db_execute("DELETE FROM jobs WHERE id=?", (job_id,))
    return result


ACTIVE = {"queued", "submitted", "running"}


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> JSONResponse:
    job = get_job_or_404(job_id)
    result = _delete_job_impl(job)
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# DELETE /jobs — delete all jobs
# ---------------------------------------------------------------------------


@app.delete("/jobs")
def delete_all_jobs() -> JSONResponse:
    rows = db_fetch_all("SELECT * FROM jobs")
    deleted = 0
    for row in rows:
        try:
            _delete_job_impl(_row_to_dict(row))
            deleted += 1
        except Exception:
            pass
    return JSONResponse({"deleted": deleted, "total": len(rows)})
