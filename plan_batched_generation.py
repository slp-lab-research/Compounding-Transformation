#!/usr/bin/env python3
"""Create an immutable, model-batched generation plan before inference.

The plan freezes every assignment that would otherwise be randomized at run
time: model, LibriTTS source, text, output label, and attack recipe.  Jobs are
also indexed by model so a later persistent worker can load one model once and
generate its entire batch without changing the dataset composition.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from main import (
    _atomic_write_json,
    _make_generation_schedule,
    _random_mcv_english_text,
    _read_reference_transcript,
    _resolve_path,
)
from speaker_identities import write_audio_speaker_pool


ATTACK_NAMES = (
    "reverb",
    "gaussian_noise",
    "white_noise",
    "bgm",
    "cafe_background",
    "resampling",
    "compression_aac",
    "compression_mp3",
    "compression_opus",
)
OPUS_SAMPLE_RATES = frozenset({8000, 12000, 16000, 24000, 48000})


def _profile_values(profile):
    if profile == "all":
        return {
            "snr": (5, 10, 15, 20, 25, 30),
            "bitrate": ("16k", "32k", "64k", "128k", "192k", "256k"),
            "room_size": (0.1, 0.3, 0.5, 0.7),
            "sample_rate": (8000, 16000, 22050, 48000),
        }
    if profile == "moderate":
        return {
            "snr": (10, 15, 20, 25),
            "bitrate": ("32k", "64k", "128k", "192k", "256k"),
            "room_size": (0.3, 0.5),
            "sample_rate": (16000, 22050, 48000),
        }
    raise ValueError(f"Unknown attack parameter profile: {profile}")


def _parameters_for_attack(attack_name, values, rng, output_suffix=None):
    if attack_name.startswith("compression_"):
        return {"bitrate": rng.choice(values["bitrate"])}
    if attack_name in {"gaussian_noise", "white_noise", "bgm", "cafe_background"}:
        return {"snr_db": rng.choice(values["snr"])}
    if attack_name == "reverb":
        return {"room_size": rng.choice(values["room_size"])}
    if attack_name == "resampling":
        sample_rates = values["sample_rate"]
        if output_suffix == ".opus":
            sample_rates = tuple(
                rate for rate in sample_rates
                if rate in OPUS_SAMPLE_RATES
            )
        return {"sample_rate": rng.choice(sample_rates)}
    raise ValueError(f"Unknown attack: {attack_name}")


def _suffix_after_attack(current_suffix, attack_name):
    return {
        "compression_aac": ".m4a",
        "compression_mp3": ".mp3",
        "compression_opus": ".opus",
    }.get(attack_name, current_suffix)


def _planned_trajectory(job_id, maximum_attack, seed, parameter_profile):
    """Create the level-0 compression plus planned compound attacks."""
    values = _profile_values(parameter_profile)
    rng = random.Random(seed + job_id * 1_000_003)
    compression = rng.choice(("compression_aac", "compression_mp3", "compression_opus"))
    levels = [
        {
            "level": 0,
            "attack_name": compression,
            "parameters": _parameters_for_attack(compression, values, rng),
            "randomization_seed": rng.randrange(2**32),
        }
    ]
    output_suffix = _suffix_after_attack(".wav", compression)
    for level in range(1, maximum_attack + 1):
        attack_name = rng.choice(ATTACK_NAMES)
        parameters = _parameters_for_attack(
            attack_name, values, rng, output_suffix=output_suffix
        )
        levels.append(
            {
                "level": level,
                "attack_name": attack_name,
                "parameters": parameters,
                "randomization_seed": rng.randrange(2**32),
            }
        )
        output_suffix = _suffix_after_attack(output_suffix, attack_name)
    return levels


def _load_enabled_configs(config_folder, config_base_dir):
    configs = {}
    for config_path in sorted(config_folder.glob("*.config")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("enabled", True) is False:
            continue
        if "clone" not in ([config.get("generation_type", "clone")] if isinstance(config.get("generation_type", "clone"), str) else config["generation_type"]):
            continue
        configs[str(config_path)] = config
    if not configs:
        raise ValueError("No enabled clone model configurations were found")
    return configs


def main(args):
    config_path = Path(args.config).resolve()
    config_base_dir = config_path.parent
    pipeline = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = _resolve_path(pipeline["audio_output_folder"], config_base_dir)
    plan_path = args.plan_path or _resolve_path(
        pipeline.get("generation_batch_plan_path", output_root / "generation_batch_plan.json"),
        config_base_dir,
    )
    if plan_path.exists() and not args.force:
        raise FileExistsError(f"Plan already exists: {plan_path}; use --force to replace it")

    data_number = int(pipeline["data_number"])
    maximum_attack = int(pipeline.get("maximum_attack", 5))
    seed = int(pipeline.get("speaker_identity_seed", 1))
    subsets = pipeline.get("text_source_subsets", ["random_mcv_text"])
    if isinstance(subsets, str):
        subsets = [subsets]
    if data_number % len(subsets):
        raise ValueError("data_number must be divisible by text_source_subsets")

    config_folder = _resolve_path(pipeline["config_folder_path"], config_base_dir)
    configs = _load_enabled_configs(config_folder, config_base_dir)
    first_config = next(iter(configs.values()))
    source_root = _resolve_path(
        pipeline.get("source_audio_path", first_config["reference_audio_path"]),
        config_base_dir,
    )
    speaker_pool_path = _resolve_path(
        pipeline["libritts_speaker_file"], config_base_dir
    )
    minimum_duration = float(pipeline.get("minimum_reference_duration_seconds", 1.5))
    write_audio_speaker_pool(
        source_root,
        speaker_pool_path,
        seed,
        pipeline.get("generation_split_proportions"),
        minimum_duration_seconds=minimum_duration,
    )
    speaker_pool = json.loads(speaker_pool_path.read_text(encoding="utf-8"))
    speakers = speaker_pool["speakers"]

    config_specs = [
        {
            "path": Path(path),
            "generation_types": ["clone"],
            "minimum_reference_duration_seconds": float(
                config.get("minimum_reference_duration_seconds", 0.0)
            ),
        }
        for path, config in configs.items()
    ]
    scheduled_jobs = _make_generation_schedule(
        data_number,
        speakers,
        config_specs,
        pipeline.get("generation_split_proportions"),
        seed,
        subsets,
    )

    parameter_profile = pipeline.get("attack_parameter_profile", "all")
    jobs = []
    batches = defaultdict(list)
    for job_id, scheduled in enumerate(scheduled_jobs, start=1):
        config = configs[str(scheduled["config_path"])]
        mode = config["mode"]
        reference_audio = Path(scheduled["reference_audio_path"])
        if scheduled["text_source_subset"] == "source_transcript":
            source_text, transcript_path = _read_reference_transcript(reference_audio)
        else:
            source_text = _random_mcv_english_text(
                _resolve_path(config["script_path"], config_base_dir),
                scheduled["text_seed"],
            )
            transcript_path = None
        speaker = speakers[scheduled["speaker_index"]]
        record = {
            "job_id": job_id,
            "mode": mode,
            "config_path": str(scheduled["config_path"]),
            "generation_type": "clone",
            "text_source_subset": scheduled["text_source_subset"],
            "source_pair_id": scheduled["source_pair_id"],
            "dataset_split": scheduled["dataset_split"],
            "speaker_index": scheduled["speaker_index"],
            "speaker_number": speaker["speaker_number"],
            "speaker_id": speaker["speaker_id"],
            "reference_audio": str(reference_audio),
            "reference_transcript": str(transcript_path) if transcript_path else None,
            "source_text": source_text,
            "output_relative_path": str(
                Path(scheduled["text_source_subset"]) / f"{mode}_{job_id:05d}.wav"
            ),
            "planned_trajectory": _planned_trajectory(
                job_id, maximum_attack, seed, parameter_profile
            ),
        }
        jobs.append(record)
        batches[mode].append(job_id)

    payload = {
        "version": 1,
        "created_from": str(config_path),
        "metadata": {
            "data_number": data_number,
            "maximum_attack": maximum_attack,
            "speaker_identity_seed": seed,
            "source_audio_path": str(source_root),
            "speaker_pool_file": str(speaker_pool_path),
            "minimum_reference_duration_seconds": minimum_duration,
            "attack_parameter_profile": parameter_profile,
            "output_root": str(output_root),
        },
        "jobs": jobs,
        "model_batches": [
            {"mode": mode, "job_ids": job_ids}
            for mode, job_ids in sorted(batches.items())
        ],
    }
    _atomic_write_json(plan_path, payload)
    print(f"Saved {len(jobs)} planned jobs to {plan_path}")
    for batch in payload["model_batches"]:
        print(f"  {batch['mode']}: {len(batch['job_ids'])} jobs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pipeline.config")
    parser.add_argument("--plan-path", type=Path)
    parser.add_argument("--force", action="store_true")
    main(parser.parse_args())
