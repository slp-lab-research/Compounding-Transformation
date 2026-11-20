#!/usr/bin/env python3
"""Apply a saved batch plan to matched and novel real LibriTTS audio."""

import argparse
import gc
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from apply_attack import AudioAttackClass
from quality_check import QualityCheckClass
from run_batched_generation import score_detectors


ALL_DETECTORS = (
    "aasist",
    "rawnet",
    "whisper",
    "rawbmamba",
    "hyperpotter",
    "slsforasvspoof",
    "prosdd",
    "sonar",
)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve(value, base):
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def select_novel_sources(plan, speaker_pool, paired_paths, seed):
    available = defaultdict(list)
    metadata = {}
    for speaker in speaker_pool["speakers"]:
        for raw_path in speaker["audio_files"]:
            path = Path(raw_path)
            key = str(path.resolve())
            if key in paired_paths:
                continue
            available[speaker["split"]].append(path)
            metadata[key] = speaker
    rng = random.Random(seed + 9000)
    for paths in available.values():
        rng.shuffle(paths)
    selected = {}
    random_jobs = sorted(
        (job for job in plan["jobs"] if job["text_source_subset"] == "random_mcv_text"),
        key=lambda job: job["source_pair_id"],
    )
    for job in random_jobs:
        split = job["dataset_split"]
        if not available[split]:
            raise RuntimeError(f"No unused LibriTTS audio remains for split {split}")
        path = available[split].pop()
        selected[job["job_id"]] = (path, metadata[str(path.resolve())])
    return selected


def completed(record, expected_levels):
    return (
        record is not None
        and len(record.get("levels", [])) == expected_levels
        and all(Path(level["attacked_audio"]).is_file() for level in record["levels"])
    )


def apply_job(job, source, output, cohort, speaker, attack, quality):
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    record = {
        "job_id": job["job_id"],
        "mode": job["mode"],
        "original_audio": str(output),
        "detector_scores": {},
        "identity": {
            "type": "real_source_audio",
            "cohort": cohort,
            "source_pair_id": job["source_pair_id"],
            "source_audio": str(source),
            "speaker_id": speaker.get("speaker_id"),
            "speaker_number": speaker.get("speaker_number"),
            "dataset_split": speaker.get("split", job["dataset_split"]),
            "planned_from_generated_job": job["job_id"],
        },
        "planned_trajectory": job["planned_trajectory"],
        "levels": [],
    }
    current = output
    for spec in job["planned_trajectory"]:
        seed = int(spec["randomization_seed"])
        random.seed(seed)
        np.random.seed(seed)
        attacked, attack_type, attack_parameter = attack.apply_named_attack(
            spec["attack_name"], str(current), str(output.parent),
            int(spec["level"]), spec["parameters"],
        )
        level = {
            "level": spec["level"],
            "attack_type": attack_type,
            "attack_parameter": attack_parameter,
            "attacked_audio": str(attacked),
            "randomization_seed": seed,
            "passed_condition": True,
            "detector_scores": {},
        }
        if int(spec["level"]) > 0:
            level.update({
                "stoi": quality.compute_stoi(str(output), str(attacked)),
                "pesq": quality.compute_pesq(str(output), str(attacked)),
                "utmos": quality.compute_utmos(str(attacked)),
            })
        record["levels"].append(level)
        current = Path(attacked)
    return record


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config_path = Path(args.config).resolve()
    base = config_path.parent
    pipeline = json.loads(config_path.read_text(encoding="utf-8"))
    plan_path = args.plan or resolve(pipeline["generation_batch_plan_path"], base)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    pool_path = Path(plan["metadata"]["speaker_pool_file"])
    speaker_pool = json.loads(pool_path.read_text(encoding="utf-8"))
    speaker_by_number = {
        speaker["speaker_number"]: speaker for speaker in speaker_pool["speakers"]
    }
    paired_jobs = [
        job for job in plan["jobs"] if job["text_source_subset"] == "source_transcript"
    ]
    paired_paths = {str(Path(job["reference_audio"]).resolve()) for job in paired_jobs}
    novel = select_novel_sources(
        plan, speaker_pool, paired_paths, int(plan["metadata"]["speaker_identity_seed"])
    )
    output_root = args.output_root or resolve(
        pipeline.get(
            "batched_real_source_audio_output_folder",
            "/mnt/synology_nas_00/nathan/datasets/batched_real_source_attack_outputs",
        ),
        base,
    )
    manifest_path = output_root / "batched_real_generation.json"
    status_path = output_root / "batched_real_generation_status.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    requested_detectors = [] if args.skip_detectors else (
        args.evaluate_model or list(ALL_DETECTORS)
    )
    if args.skip_detectors and args.evaluate_model:
        raise ValueError("--skip-detectors cannot be combined with --evaluate-model")
    atomic_json(status_path, {
        "status": "transforming",
        "plan": str(plan_path),
        "manifest": str(manifest_path),
        "requested_detectors": requested_detectors,
    })
    attack = AudioAttackClass(
        resolve(pipeline["cafe_audio_path"], base),
        parameter_profile=plan["metadata"].get("attack_parameter_profile", "all"),
        music_audio_path=resolve(pipeline["music_audio_path"], base),
    )
    quality = QualityCheckClass()
    cohorts = [
        (
            "paired_source", paired_jobs,
            lambda job: (
                Path(job["reference_audio"]),
                speaker_by_number[job["speaker_number"]],
            ),
        ),
        (
            "novel_source",
            [job for job in plan["jobs"] if job["text_source_subset"] == "random_mcv_text"],
            lambda job: novel[job["job_id"]],
        ),
    ]
    total = sum(len(jobs) for _, jobs, _ in cohorts)
    position = 0
    for cohort, jobs, source_for in cohorts:
        for job in jobs:
            position += 1
            key = f"{cohort}_job_{job['job_id']:05d}"
            if completed(manifest.get(key), len(job["planned_trajectory"])):
                continue
            source, speaker = source_for(job)
            output = output_root / cohort / f"{key}{source.suffix.lower()}"
            print(f"Real source {position}/{total}: {key}", flush=True)
            manifest[key] = apply_job(
                job, source, output, cohort, speaker, attack, quality
            )
            atomic_json(manifest_path, manifest)

    # Older/resumed manifests predate detector fields. Normalize them before
    # handing the records to the shared persistent detector pass.
    for record in manifest.values():
        record.setdefault("detector_scores", {})
        for level in record.get("levels", []):
            level.setdefault("detector_scores", {})
    atomic_json(manifest_path, manifest)

    # Detection starts only after every real-source trajectory is complete.
    # Release the quality model first so each detector gets as much VRAM as
    # possible; score_detectors then loads and closes one detector at a time.
    del quality
    del attack
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if requested_detectors:
        atomic_json(status_path, {
            "status": "detecting",
            "plan": str(plan_path),
            "manifest": str(manifest_path),
            "requested_detectors": requested_detectors,
        })
        score_detectors(
            manifest, manifest_path, requested_detectors, args.gpu
        )

    summary = {
        "planned": total,
        "completed": sum(
            completed(record, len(record.get("planned_trajectory", [])))
            for record in manifest.values()
        ),
        "plan": str(plan_path),
        "manifest": str(manifest_path),
        "requested_detectors": requested_detectors,
    }
    atomic_json(output_root / "summary.json", summary)
    atomic_json(status_path, {
        "status": "complete",
        "plan": str(plan_path),
        "manifest": str(manifest_path),
        "requested_detectors": requested_detectors,
        "completed_jobs": summary["completed"],
    })
    print(f"Completed batched real-source dataset: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pipeline.config")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--evaluate-model",
        action="append",
        choices=ALL_DETECTORS,
        help=(
            "Detector family to run after transformation; repeat to select a "
            "subset. If omitted, every detector family is evaluated."
        ),
    )
    parser.add_argument(
        "--skip-detectors",
        action="store_true",
        help="Apply transformations and quality metrics without detector scoring.",
    )
    main(parser.parse_args())
