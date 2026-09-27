"""JSONL job-queue helpers for --jobs_file mode in scripts/run_pipeline.py.

File-locking (filelock) allows multiple GPU worker processes to pull from the
same jobs file concurrently.
"""
import json
import os
from pathlib import Path

from filelock import FileLock, Timeout


def _read_jobs_jsonl(path: Path):
    if not path.exists():
        return []
    jobs = []
    with path.open("r") as f:
        for line in f:
            s = line.strip()
            try:
                rec = json.loads(s)
                jobs.append(rec)
            except json.JSONDecodeError:
                # allow bare names from older files
                print(f"WARNING: ignoring invalid line in jobs file: {s}")
    return jobs


def _write_jobs_jsonl_atomic(path: Path, jobs) -> None:
    # Per-PID tmp file so concurrent workers don't clobber each other's rename source
    tmp = path.parent / f"{path.name}.tmp.{os.getpid()}"
    with tmp.open("w") as f:
        for rec in jobs:
            # keep only minimal fields (take/status) to avoid bloat
            take = rec["take"]
            status = rec.get("status", "not_done")
            error_msg = rec.get("error", None)
            f.write(json.dumps({"take": take, "status": status, "error": error_msg}) + "\n")
    tmp.replace(path)


def _claim_one_job_with_lock(jobs_path: Path, lock: FileLock):
    try:
        with lock:
            jobs = _read_jobs_jsonl(jobs_path)
            for i, rec in enumerate(jobs):
                if rec["status"] == "not_done":
                    jobs[i]["status"] = "running"
                    _write_jobs_jsonl_atomic(jobs_path, jobs)
                    return rec["take"]
            return None
    except Timeout:
        print("ERROR: Timeout acquiring lock to claim job")
        return None
    except Exception as e:
        print(f"ERROR: Failed to claim job: {e}")
        return None


def _finalize_job_with_lock(jobs_path: Path, lock: FileLock, take_name: str, ok: bool, err_msg: str = None):
    """Under lock: set running/not_done -> done/failed for the given take."""
    try:
        with lock:
            jobs = _read_jobs_jsonl(jobs_path)
            for i, rec in enumerate(jobs):
                if rec["take"] == take_name and rec["status"] in ("running", "not_done"):
                    jobs[i]["status"] = "done" if ok else "failed"
                    if not ok and err_msg:
                        jobs[i]["error"] = err_msg
                    elif ok:
                        jobs[i]["error"] = None  # Clear any previous error
                    _write_jobs_jsonl_atomic(jobs_path, jobs)
                    break
    except Exception as e:
        print(f"ERROR: Failed to finalize job {take_name}: {e}")
        raise
