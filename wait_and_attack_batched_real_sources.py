#!/usr/bin/env python3
"""Wait for the batched fake-audio manifest, then launch real-source attacks."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def resolve(value, base):
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def completion_state(plan, manifest_path, status_path):
    if not manifest_path.is_file():
        return 0, len(plan["jobs"]), None, False
    stat = manifest_path.stat()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0, len(plan["jobs"]), (stat.st_size, stat.st_mtime_ns), False
    complete = 0
    for job in plan["jobs"]:
        record = manifest.get(f"job_{job['job_id']:05d}")
        if record and len(record.get("levels", [])) == len(job["planned_trajectory"]):
            complete += 1
    pipeline_complete = False
    if status_path.is_file():
        try:
            pipeline_complete = (
                json.loads(status_path.read_text(encoding="utf-8")).get("status")
                == "complete"
            )
        except (json.JSONDecodeError, OSError):
            pass
    return complete, len(plan["jobs"]), (stat.st_size, stat.st_mtime_ns), pipeline_complete


def main(args):
    config_path = Path(args.config).resolve()
    base = config_path.parent
    pipeline = json.loads(config_path.read_text(encoding="utf-8"))
    plan_path = args.plan or resolve(pipeline["generation_batch_plan_path"], base)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    fake_root = Path(plan["metadata"]["output_root"])
    manifest_path = args.generated_manifest or fake_root / "batched_generation.json"
    status_path = fake_root / "batched_generation_status.json"
    lock_path = Path(f"/tmp/batched-real-source-watcher-{os.getuid()}.lock")
    lock = lock_path.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Another batched real-source watcher is running") from error

    previous_signature = None
    stable_since = None
    while True:
        complete, expected, signature, pipeline_complete = completion_state(
            plan, manifest_path, status_path
        )
        now = time.monotonic()
        if signature != previous_signature:
            previous_signature = signature
            stable_since = now
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(
            f"[{stamp}] batched generation={complete}/{expected}, "
            f"pipeline_complete={pipeline_complete}",
            flush=True,
        )
        stable_for = 0 if stable_since is None else now - stable_since
        if pipeline_complete and complete == expected and stable_for >= args.settle_seconds:
            break
        time.sleep(args.poll_seconds)

    command = [
        sys.executable, "-u",
        str(Path(__file__).with_name("attack_batched_real_sources.py")),
        "--config", str(config_path), "--plan", str(plan_path),
        "--gpu", str(args.gpu),
    ]
    if args.output_root:
        command.extend(["--output-root", str(args.output_root)])
    print("Batched generation complete; starting real-source attacks.", flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pipeline.config")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--generated-manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--settle-seconds", type=float, default=300.0)
    main(parser.parse_args())
