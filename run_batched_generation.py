#!/usr/bin/env python3
"""Execute a saved generation plan with persistent generators and detectors."""

import argparse
import gc
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from apply_attack import AudioAttackClass
from evaluate_generated_audio import detector_variants, load_detector, score_file
from quality_check import QualityCheckClass


ROOT = Path(__file__).resolve().parent


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def cuda_library_dirs(python_executable):
    root = Path(python_executable).expanduser().resolve().parent.parent
    return sorted({
        str(path)
        for nvidia in root.glob("lib/python*/site-packages/nvidia")
        for path in nvidia.glob("*/lib") if path.is_dir()
    })


def configs_by_mode(plan):
    result = {}
    for job in plan["jobs"]:
        mode = job["mode"]
        if mode not in result:
            path = Path(job["config_path"])
            result[mode] = json.loads(path.read_text(encoding="utf-8"))
    return result


def generate_batches(plan, plan_path, output_root, gpu, instruct_path):
    configs = configs_by_mode(plan)
    jobs_by_id = {job["job_id"]: job for job in plan["jobs"]}
    batch_root = plan_path.parent / ".generation_batches"
    batch_root.mkdir(parents=True, exist_ok=True)
    for batch in plan["model_batches"]:
        mode = batch["mode"]
        config = configs[mode]
        jobs = []
        for job_id in batch["job_ids"]:
            job = dict(jobs_by_id[job_id])
            job["output_path"] = str(output_root / job["output_relative_path"])
            jobs.append(job)
        jobs_path = batch_root / f"{mode}.jobs.json"
        state_path = batch_root / f"{mode}.state.json"
        atomic_json(jobs_path, jobs)
        python = config.get("python") or sys.executable
        command = [
            python, "-u", str(ROOT / "persistent_batch_generator.py"),
            "--model", mode, "--jobs", str(jobs_path), "--state", str(state_path),
        ]
        if instruct_path:
            command.extend(["--instruct-path", str(instruct_path)])
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        if mode == "maskgct":
            libraries = cuda_library_dirs(python)
            existing = environment.get("LD_LIBRARY_PATH", "")
            environment["LD_LIBRARY_PATH"] = ":".join(
                libraries + ([existing] if existing else [])
            )
        print(f"\n=== Persistent generator: {mode} ({len(jobs)} jobs) ===", flush=True)
        subprocess.run(command, check=True, env=environment)


def apply_trajectories(plan, output_root, manifest_path, pipeline):
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    attack = AudioAttackClass(
        pipeline["cafe_audio_path"],
        parameter_profile=pipeline.get("attack_parameter_profile", "all"),
        music_audio_path=pipeline["music_audio_path"],
    )
    quality = QualityCheckClass()
    for position, job in enumerate(plan["jobs"], start=1):
        key = f"job_{job['job_id']:05d}"
        clean = output_root / job["output_relative_path"]
        existing = manifest.get(key)
        if (
            existing
            and len(existing.get("levels", [])) == len(job["planned_trajectory"])
            and all(Path(level["attacked_audio"]).is_file() for level in existing["levels"])
        ):
            continue
        if not clean.is_file():
            raise FileNotFoundError(f"Missing generated clean audio: {clean}")
        record = {
            "job_id": job["job_id"], "mode": job["mode"],
            "original_audio": str(clean), "detector_scores": {},
            "identity": {
                field: job.get(field) for field in (
                    "source_pair_id", "text_source_subset", "dataset_split",
                    "speaker_id", "speaker_number", "reference_audio", "source_text",
                )
            },
            "planned_trajectory": job["planned_trajectory"], "levels": [],
        }
        current = clean
        print(f"Trajectory {position}/{len(plan['jobs'])}: {key}", flush=True)
        for spec in job["planned_trajectory"]:
            seed = int(spec["randomization_seed"])
            random.seed(seed)
            np.random.seed(seed)
            attacked, attack_type, attack_parameter = attack.apply_named_attack(
                spec["attack_name"], str(current), str(clean.parent),
                int(spec["level"]), spec["parameters"],
            )
            if int(spec["level"]) == 0:
                stoi = pesq = utmos = None
            else:
                stoi = quality.compute_stoi(str(clean), str(attacked))
                pesq = quality.compute_pesq(str(clean), str(attacked))
                utmos = quality.compute_utmos(str(attacked))
            level = {
                "level": spec["level"], "attack_type": attack_type,
                "attack_parameter": attack_parameter,
                "attacked_audio": str(attacked), "passed_condition": True,
                "randomization_seed": seed, "detector_scores": {},
            }
            if stoi is not None:
                level.update({"stoi": stoi, "pesq": pesq, "utmos": utmos})
            record["levels"].append(level)
            current = Path(attacked)
        manifest[key] = record
        atomic_json(manifest_path, manifest)
    del quality
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest


def detector_specs(requested):
    for requested_name in requested:
        names = ("sonar-full",) if requested_name == "sonar" else (requested_name,)
        for model_name in names:
            for variant, checkpoint in detector_variants(model_name):
                yield f"{model_name}_{variant}", model_name, checkpoint


def detector_scores_complete(manifest, scorer_name):
    """Return whether every original and attack level has this detector score."""
    if not manifest:
        return False
    for record in manifest.values():
        if scorer_name not in record.get("detector_scores", {}):
            return False
        levels = record.get("levels", [])
        expected_levels = len(record.get("planned_trajectory", []))
        if len(levels) != expected_levels:
            return False
        if any(
            scorer_name not in level.get("detector_scores", {})
            for level in levels
        ):
            return False
    return True


def score_detectors(manifest, manifest_path, requested, gpu):
    if not requested:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for scorer_name, model_name, checkpoint in detector_specs(requested):
        if detector_scores_complete(manifest, scorer_name):
            print(
                f"\n=== Persistent detector: {scorer_name} "
                "(already complete; skipping) ===",
                flush=True,
            )
            continue
        print(f"\n=== Persistent detector: {scorer_name} ===", flush=True)
        scorer = load_detector(model_name, device, checkpoint)
        dirty = False
        try:
            for index, record in enumerate(manifest.values(), start=1):
                if scorer_name not in record["detector_scores"]:
                    record["detector_scores"][scorer_name] = score_file(
                        record["original_audio"], scorer, device
                    )
                    dirty = True
                for level in record["levels"]:
                    if scorer_name not in level["detector_scores"]:
                        level["detector_scores"][scorer_name] = score_file(
                            level["attacked_audio"], scorer, device
                        )
                        dirty = True
                if index % 25 == 0 and dirty:
                    atomic_json(manifest_path, manifest)
                    dirty = False
            if dirty:
                atomic_json(manifest_path, manifest)
        finally:
            close = getattr(scorer, "close", None)
            if callable(close):
                close()
            del scorer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main(args):
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    pipeline_path = Path(args.config).resolve()
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    base = pipeline_path.parent
    for key in ("cafe_audio_path", "music_audio_path", "instruct_path"):
        value = Path(pipeline[key]).expanduser()
        pipeline[key] = str(value if value.is_absolute() else base / value)
    output_root = Path(plan["metadata"]["output_root"])
    manifest_path = args.manifest or output_root / "batched_generation.json"
    status_path = output_root / "batched_generation_status.json"
    atomic_json(status_path, {
        "status": "running",
        "plan": str(plan_path),
        "requested_detectors": args.evaluate_model,
    })
    generate_batches(
        plan, plan_path, output_root, args.gpu, Path(pipeline["instruct_path"])
    )
    manifest = apply_trajectories(plan, output_root, manifest_path, pipeline)
    score_detectors(manifest, manifest_path, args.evaluate_model, args.gpu)
    atomic_json(status_path, {
        "status": "complete",
        "plan": str(plan_path),
        "manifest": str(manifest_path),
        "requested_detectors": args.evaluate_model,
        "completed_jobs": len(manifest),
    })
    print(f"Completed batch pipeline: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--config", default="pipeline.config")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--evaluate-model", action="append", default=[])
    main(parser.parse_args())
