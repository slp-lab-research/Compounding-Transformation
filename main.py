import argparse
import csv
import gc
import json
import os
import random
import subprocess
import sys
import tempfile
import torch
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Librosa imports Numba. Keep its compiled cache writable and outside the
# repository so a fresh checkout does not fail during module import.
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(Path(tempfile.gettempdir()) / f"numba-cache-{os.getuid()}"),
)

from data_storing import MetaDataStorage
from evaluate_generated_audio import detector_variants
from evaluate_generated_audio import load_detector, score_file
from speaker_identities import (
    _audio_duration_seconds,
    write_audio_speaker_pool,
)


TEXT_SOURCE_SUBSETS = {"source_transcript", "random_mcv_text"}


def _choose_reference_audio(audio_files, minimum_duration_seconds, rng):
    """Choose one deterministic reference WAV meeting a model-specific minimum."""
    candidates = list(audio_files)
    rng.shuffle(candidates)
    if minimum_duration_seconds <= 0:
        return candidates[0]
    for audio_path in candidates:
        duration = _audio_duration_seconds(Path(audio_path))
        if duration is not None and duration >= minimum_duration_seconds:
            return audio_path
    return None


def _choose_model_reference(
    source_speakers, speaker_index, split, minimum_duration_seconds, rng
):
    """Select a model-compatible reference, falling back within its split."""
    chosen = _choose_reference_audio(
        source_speakers[speaker_index]["audio_files"],
        minimum_duration_seconds,
        rng,
    )
    if chosen is not None:
        return speaker_index, chosen

    fallback_indices = [
        index for index, speaker in enumerate(source_speakers)
        if speaker.get("split") == split and index != speaker_index
    ]
    rng.shuffle(fallback_indices)
    for fallback_index in fallback_indices:
        chosen = _choose_reference_audio(
            source_speakers[fallback_index]["audio_files"],
            minimum_duration_seconds,
            rng,
        )
        if chosen is not None:
            return fallback_index, chosen
    raise ValueError(
        f"No {split} LibriTTS speaker has a reference WAV of at least "
        f"{minimum_duration_seconds:.1f}s"
    )

def _resolve_path(value, base_dir):
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def _atomic_write_json(path, value):
    """Write a checkpoint without leaving a truncated JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def _checkpoint_timestamp():
    return datetime.now(timezone.utc).isoformat()


def _serialize_job(job):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in job.items()
    }


def _deserialize_job(job):
    job = dict(job)
    job["config_path"] = Path(job["config_path"])
    return job


def _record_is_complete(record, maximum_attack, quality_gate_enabled):
    """Return whether an existing generation log entry is safe to skip."""
    if not record or not record.get("original_audio"):
        return False
    levels = record.get("levels", [])
    if not levels:
        return False

    last_level = levels[-1]
    if (
        last_level.get("rejection_reason") == "quality"
        and not quality_gate_enabled
    ):
        # This record was stopped by a gate that is disabled in the current run.
        return False
    if not last_level.get("passed_condition", True):
        # Detector-consensus rejection is an intentional terminal state.
        return True
    return len(levels) >= maximum_attack + 1


def _record_matches_job(record, job, mode, text_source_subset, reference_audio):
    """Prevent an old log from another schedule/dataset being skipped."""
    identity = record.get("identity", {}) if record else {}
    logged_reference = identity.get("reference_audio")
    if not logged_reference:
        return False
    try:
        same_reference = (
            Path(logged_reference).resolve() == Path(reference_audio).resolve()
        )
    except OSError:
        same_reference = str(logged_reference) == str(reference_audio)
    return (
        same_reference
        and record.get("mode") == mode
        and identity.get("source_pair_id") == job.get("source_pair_id")
        and identity.get("text_source_subset") == text_source_subset
    )


def _worker_python(mode, config):
    configured = config.get("python") or os.environ.get(
        f"IEMOCAP_{mode.upper()}_PYTHON"
    )
    return configured or sys.executable


def _maskgct_cuda_library_dirs(python_executable):
    """Return CUDA shared-library directories bundled with the MaskGCT env."""
    environment_root = Path(python_executable).expanduser().resolve().parent.parent
    library_dirs = []
    for nvidia_root in environment_root.glob("lib/python*/site-packages/nvidia"):
        library_dirs.extend(
            path for path in nvidia_root.glob("*/lib") if path.is_dir()
        )
    return sorted({str(path) for path in library_dirs})


def _render_worker_command(config, values, config_base_dir):
    """Render a config-defined command for a model-specific worker."""
    worker = config.get("worker", {})
    command_template = worker.get("command")
    if not command_template:
        raise ValueError(
            f"Model {config.get('mode')!r} needs worker.command in its config"
        )

    values = dict(values)
    values.update({
        "python": _worker_python(config["mode"], config),
        "worker_script": str(
            _resolve_path(worker.get("script", ""), config_base_dir)
        ),
        "model": config["mode"],
    })
    try:
        return [str(part).format_map(values) for part in command_template]
    except KeyError as error:
        raise ValueError(
            f"Unknown worker command placeholder {error.args[0]!r} in "
            f"{config['mode']!r} config"
        ) from error


def _read_reference_transcript(reference_audio_path):
    """Return the paired LibriTTS text and the sidecar that supplied it."""
    reference_audio_path = Path(reference_audio_path)
    for suffix in (".normalized.txt", ".original.txt"):
        transcript_path = reference_audio_path.with_suffix(suffix)
        if transcript_path.is_file():
            transcript = transcript_path.read_text(encoding="utf-8").strip()
            if transcript:
                return transcript, transcript_path
    raise FileNotFoundError(
        "No non-empty LibriTTS transcript sidecar found for "
        f"{reference_audio_path}; expected .normalized.txt or .original.txt"
    )


def _random_mcv_english_text(script_path, seed, maximum_attempts=1000):
    """Sample a deterministic English MCV sentence without loading the 2+ GB TSV."""
    script_path = Path(script_path)
    with script_path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle, delimiter="\t"), None)
    if not header or "lang" not in header or "sentence" not in header:
        raise ValueError(f"Expected lang and sentence columns in {script_path}")

    language_index = header.index("lang")
    sentence_index = header.index("sentence")
    random_source = random.Random(seed)
    file_size = script_path.stat().st_size
    with script_path.open("rb") as handle:
        for _ in range(maximum_attempts):
            offset = random_source.randrange(file_size)
            handle.seek(offset)
            if offset:
                handle.readline()  # Discard a partial TSV row.
            row_bytes = handle.readline()
            if not row_bytes:
                continue
            try:
                row = next(
                    csv.reader([row_bytes.decode("utf-8")], delimiter="\t"), []
                )
            except UnicodeDecodeError:
                continue
            if (
                len(row) > max(language_index, sentence_index)
                and row[language_index] == "en"
                and row[sentence_index].strip()
            ):
                return row[sentence_index].strip()

    raise RuntimeError(
        f"Could not sample an English MCV sentence from {script_path} after "
        f"{maximum_attempts} attempts"
    )


def _make_split_schedule(data_number, seed, configured_proportions=None):
    '''
    Create a randomize schedule to split the identity from the cloning source and generated identity
    based on the configured proportions for train, validation, and test splits. The schedule is generated using a random seed to ensure reproducibility.
    The function returns a list of split names corresponding to the number of identities to be generated.

    Data_number: The total number of identities to be generated.
    Seed: The random seed for reproducibility.
    Configured_proportions: A dictionary specifying the proportions for train, validation, and test
    '''

    proportions = configured_proportions or {
        "train": 0.7,
        "validation": 0.1,
        "test": 0.2,
    }
    if set(proportions) != {"train", "validation", "test"}:
        raise ValueError(
            "generation_split_proportions must define train, validation, and test"
        )
    if any(proportion < 0 for proportion in proportions.values()):
        raise ValueError("generation split proportions cannot be negative")
    if abs(sum(proportions.values()) - 1.0) > 1e-8:
        raise ValueError("generation split proportions must sum to 1")
    counts = {
        split: int(data_number * proportion)
        for split, proportion in proportions.items()
    }
    remaining = data_number - sum(counts.values())
    fractional = sorted(
        proportions,
        key=lambda split: data_number * proportions[split] - counts[split],
        reverse=True,
    )
    for split in fractional[:remaining]:
        counts[split] += 1

    schedule = [split for split, count in counts.items() for _ in range(count)]
    random.Random(seed).shuffle(schedule)
    return schedule


def _balanced_pool_indices(pool, split, count, seed, number_key):
    '''
    Select a balanced set of indices from a pool of items based on a specific split and count.
    '''

    candidates = [
        index for index, item in enumerate(pool)
        if item.get("split") == split
    ]

    # If the count is greater than the number of candidates, raise an error
    if count and not candidates:
        raise ValueError("No {} identities available for {} split".format(number_key, split))

    random.Random(seed).shuffle(candidates)
    selected = [candidates[index % len(candidates)] for index in range(count)]
    random.Random(seed + 1).shuffle(selected)
    return selected


def _make_generation_schedule(
    data_number,
    source_speakers,
    config_specs,
    split_proportions,
    seed,
    text_source_subsets,
):
    '''Create paired clone jobs from the LibriTTS speaker pool.'''

    clone_configs = [
        spec for spec in config_specs if "clone" in spec["generation_types"]
    ]
    if not clone_configs:
        raise ValueError("No configuration supports voice cloning")

    if not text_source_subsets:
        raise ValueError("At least one text source subset is required")
    unknown_subsets = set(text_source_subsets) - TEXT_SOURCE_SUBSETS
    if unknown_subsets:
        raise ValueError(f"Unknown text source subsets: {sorted(unknown_subsets)}")
    if data_number % len(text_source_subsets) != 0:
        raise ValueError(
            "data_number must be divisible by the number of text source subsets "
            f"({len(text_source_subsets)})"
        )

    source_pair_count = data_number // len(text_source_subsets)
    method_splits = _make_split_schedule(
        source_pair_count, seed, split_proportions
    )
    candidates_by_split = {}
    for split in ("train", "validation", "test"):
        split_count = method_splits.count(split)
        candidates_by_split[split] = _balanced_pool_indices(
            source_speakers, split, split_count, seed + len(split), "speaker_number")

    # Allocate each speaker's jobs across clone models as evenly as possible.
    config_order = [None] * source_pair_count
    config_random = random.Random(seed + 100)
    model_cursor = 0
    for split in ("train", "validation", "test"):
        split_indices = [
            index for index, split_name in enumerate(method_splits)
            if split_name == split
        ]
        speaker_sequence = candidates_by_split[split]
        speaker_counts = Counter(speaker_sequence)
        speaker_order = list(speaker_counts)
        config_random.shuffle(speaker_order)
        assignments = {}
        for speaker_index in speaker_order:
            speaker_configs = []
            for _ in range(speaker_counts[speaker_index]):
                speaker_configs.append(clone_configs[model_cursor % len(clone_configs)])
                model_cursor += 1
            config_random.shuffle(speaker_configs)
            assignments[speaker_index] = speaker_configs

        for index, speaker_index in zip(split_indices, speaker_sequence):
            config_order[index] = assignments[speaker_index].pop()

    source_jobs = []
    split_positions = {split: 0 for split in candidates_by_split}
    reference_random = random.Random(seed + 150)
    for index, split in enumerate(method_splits):
        speaker_index = candidates_by_split[split][split_positions[split]]
        split_positions[split] += 1
        minimum_duration = config_order[index]["minimum_reference_duration_seconds"]
        speaker_index, reference_audio_path = _choose_model_reference(
            source_speakers,
            speaker_index,
            split,
            minimum_duration,
            reference_random,
        )
        source_jobs.append({
            "config_path": config_order[index]["path"],
            "generation_type": "clone",
            "dataset_split": split,
            "speaker_index": speaker_index,
            "reference_audio_path": reference_audio_path,
            "text_seed": seed + 300 + index,
        })

    random.Random(seed + 200).shuffle(source_jobs)
    jobs = []
    for source_pair_id, source_job in enumerate(source_jobs, start=1):
        for text_source_subset in text_source_subsets:
            jobs.append({
                **source_job,
                "source_pair_id": source_pair_id,
                "text_source_subset": text_source_subset,
            })
    return jobs


def _detector_specs(requested_models):
    """Resolve requested detector checkpoints without loading them onto the GPU."""
    specs = []
    for requested_model_name in requested_models or []:
        model_names = (
            ("sonar-full",)
            if requested_model_name == "sonar"
            else (requested_model_name,)
        )
        for model_name in model_names:
            for variant_name, checkpoint_path in detector_variants(model_name):
                specs.append((f"{model_name}_{variant_name}", model_name, checkpoint_path))
    return specs


def _score_with_detectors(audio_path, detector_specs, detector_device):
    """Score one audio file while keeping at most one detector on the GPU."""
    scores = {}
    for scorer_name, model_name, checkpoint_path in detector_specs:
        print(f"Loading {scorer_name} detector on {detector_device} ...")
        scorer = None
        try:
            scorer = load_detector(model_name, detector_device, checkpoint_path)
            scores[scorer_name] = score_file(audio_path, scorer, detector_device)
        finally:
            if scorer is not None:
                close = getattr(scorer, "close", None)
                if callable(close):
                    close()
                del scorer
            gc.collect()
            if detector_device is not None and detector_device.type == "cuda":
                torch.cuda.empty_cache()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    from apply_attack import AudioAttackClass
    from quality_check import QualityCheckClass

    detector_specs = []
    detector_device = None

    if args.evaluate_model:
        # Resolve checkpoints now, but load each detector only after the TTS
        # worker exits.  Keeping every detector loaded alongside the TTS model
        # exhausts GPU memory for larger combinations of models.
        detector_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        detector_specs = _detector_specs(args.evaluate_model)

    with open(args.config, "r") as file:
        pipeline = json.load(file)
    
    """
    Extract parameters from the pipeline configuration file
    1. The number of data to be generated form this pipeline
    2. Maximum number of attacks to apply to each generated audio
    3. The folder containing the configuration files for the models to be used in this pipeline
    6. The path to the output folder for the generated audio
    7. The path to the log file for the generation process
    8. The paths to the cafe and street audio datasets, which will be used for applying realistic audio attacks
    9. The paths to the model configuration files, which will be randomly selected
    10. The paths to the model configuration files for the detector models, which will be used for quality checking
    """

    data_number = pipeline.get("data_number", 0)
    max_attack_number = pipeline.get("maximum_attack", 0)
    remove_if_all_baselines_fake = pipeline.get("remove_if_all_baselines_fake", False)
    quality_gate_enabled = bool(pipeline.get("quality_gate_enabled", False))
    resume_enabled = bool(pipeline.get("resume_enabled", True)) and not args.no_resume
    speaker_identity_seed = pipeline.get("speaker_identity_seed", 1)
    minimum_reference_duration_seconds = float(
        pipeline.get("minimum_reference_duration_seconds", 1.5)
    )
    if minimum_reference_duration_seconds < 0:
        raise ValueError("minimum_reference_duration_seconds must be non-negative")
    text_source_subsets = pipeline.get(
        "text_source_subsets", ["random_mcv_text"]
    )
    if isinstance(text_source_subsets, str):
        text_source_subsets = [text_source_subsets]
    config_base_dir = Path(args.config).resolve().parent
    config_folder = _resolve_path(pipeline["config_folder_path"], config_base_dir)
    
    # Load all configuration files in the specified folder for random selection during data generation
    config_files = [
        path for path in config_folder.glob("*.config")
        if path.name != Path(args.config).name
    ]

    if not config_files:
        raise FileNotFoundError(f"No model configs found in {config_folder}")
    
    # Create output folder and generation log file if they don't exist
    audio_output_folder = _resolve_path(
        pipeline["audio_output_folder"], config_base_dir)
    generation_log_path = _resolve_path(
        pipeline["generation_log_path"], config_base_dir)
    schedule_path = _resolve_path(
        pipeline.get(
            "generation_schedule_path",
            generation_log_path.parent / "generation_schedule.json",
        ),
        config_base_dir,
    )
    progress_path = _resolve_path(
        pipeline.get(
            "generation_progress_path",
            generation_log_path.parent / "generation_progress.json",
        ),
        config_base_dir,
    )
    libritts_speaker_file = _resolve_path(
        pipeline.get(
            "libritts_speaker_file",
            config_base_dir / "libritts_speakers.json",
        ),
        config_base_dir,
    )

    split_proportions = pipeline.get("generation_split_proportions")

    instruct_path = _resolve_path(pipeline["instruct_path"], config_base_dir)

    reference_config_paths = []
    for path in config_files:
        config = json.loads(path.read_text(encoding="utf-8"))
        if config.get("reference_audio_path"):
            reference_config_paths.append(path)
    
    source_audio_path = None
    if reference_config_paths:
        with reference_config_paths[0].open("r", encoding="utf-8") as file:
            reference_config = json.load(file)
        source_audio_path = _resolve_path(
            pipeline.get("source_audio_path", reference_config["reference_audio_path"]),
            config_base_dir,
        )
        use_cached_pool = resume_enabled and schedule_path.is_file() and libritts_speaker_file.is_file()
        if use_cached_pool:
            with libritts_speaker_file.open("r", encoding="utf-8") as file:
                cached_pool = json.load(file)
            cached_dataset = Path(cached_pool.get("dataset_path", "")).resolve()
            cached_minimum = float(cached_pool.get("minimum_duration_seconds", -1))
            if (
                cached_dataset != source_audio_path.resolve()
                or cached_minimum != minimum_reference_duration_seconds
            ):
                raise RuntimeError(
                    "Cached speaker pool does not match the current source dataset "
                    "or minimum reference duration; use --no-resume to rebuild it"
                )
            print(f"Reusing cached speaker pool: {libritts_speaker_file}")
        else:
            write_audio_speaker_pool(
                source_audio_path,
                libritts_speaker_file,
                speaker_identity_seed,
                split_proportions,
                minimum_duration_seconds=minimum_reference_duration_seconds,
            )

    with libritts_speaker_file.open("r", encoding="utf-8") as file:
        source_speakers = json.load(file)["speakers"]
    config_specs = []
    for config_path in config_files:
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
        if config.get("enabled", True) is False:
            print(f"Skipping disabled model config: {config_path}")
            continue
        mode = config["mode"]
        generation_types = config.get("generation_type", "clone")
        generation_types = [generation_types] if isinstance(generation_types, str) else generation_types
        if "clone" not in generation_types:
            print(f"Skipping non-clone model config: {config_path}")
            continue
        config_specs.append({
            "path": config_path,
            "generation_types": ["clone"],
            "minimum_reference_duration_seconds": float(
                config.get("minimum_reference_duration_seconds", 0.0)
            ),
        })

    schedule_metadata = {
        "data_number": data_number,
        "text_source_subsets": text_source_subsets,
        "speaker_identity_seed": speaker_identity_seed,
        "source_audio_path": str(source_audio_path) if source_audio_path else None,
        "minimum_reference_duration_seconds": minimum_reference_duration_seconds,
        "config_paths": sorted(str(spec["path"]) for spec in config_specs),
    }
    if resume_enabled and schedule_path.is_file():
        schedule_payload = json.loads(schedule_path.read_text(encoding="utf-8"))
        if schedule_payload.get("metadata") != schedule_metadata:
            raise RuntimeError(
                f"Existing schedule does not match the current configuration: {schedule_path}. "
                "Use --no-resume with a fresh output directory to start over."
            )
        jobs = [_deserialize_job(job) for job in schedule_payload["jobs"]]
        if len(jobs) != data_number:
            raise RuntimeError(
                f"Existing schedule has {len(jobs)} jobs; expected {data_number}"
            )
        print(f"Resuming from saved generation schedule: {schedule_path}")
    else:
        # Persist the randomized schedule before any model is launched. This
        # makes reference/audio/text assignments stable across restarts.
        jobs = _make_generation_schedule(
            data_number,
            source_speakers,
            config_specs,
            split_proportions,
            speaker_identity_seed,
            text_source_subsets,
        )
        _atomic_write_json(
            schedule_path,
            {
                "created_at": _checkpoint_timestamp(),
                "metadata": schedule_metadata,
                "jobs": [_serialize_job(job) for job in jobs],
            },
        )
        print(f"Saved generation schedule: {schedule_path}")

    if resume_enabled and progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("metadata") != schedule_metadata:
            raise RuntimeError(
                f"Existing progress log does not match the current schedule: {progress_path}. "
                "Use --no-resume with a fresh output directory to start over."
            )
    else:
        progress = {
            "created_at": _checkpoint_timestamp(),
            "metadata": schedule_metadata,
            "jobs": {},
        }
        _atomic_write_json(progress_path, progress)
    print(f"Generation progress log: {progress_path}")

    # Initialize the path for the WHAM background audio dataset
    cafe_audio_path = _resolve_path(pipeline["cafe_audio_path"], config_base_dir)
    music_audio_path = _resolve_path(
        pipeline.get(
            "music_audio_path",
            "/mnt/synology_nas_00/nathan/datasets/fma_small",
        ),
        config_base_dir,
    )
    tts_home = _resolve_path(
        pipeline.get(
            "tts_home",
            "/home/nathan/.local/share/tts",
        ),
        config_base_dir,
    )

    """
    Initialize the attack, quality check, and data storing classes
    For applying random audio attacks
    For checking the quality of the attacked audio
    For storing metadata about the generated audio and attacks
    """

    attack = AudioAttackClass(cafe_audio_path, music_audio_path=music_audio_path)
    quality_check = QualityCheckClass()
    # Keep the two paired text subsets physically and log-wise separate.  The
    # paired jobs still share one reference WAV and source_pair_id, but every
    # generated file and every compounded attack stays in its own subset tree.
    subset_output_folders = {
        subset: audio_output_folder / subset
        for subset in text_source_subsets
    }
    data_storing_by_subset = {
        subset: MetaDataStorage(
            subset_output_folders[subset],
            generation_log_path.parent / subset / generation_log_path.name,
        )
        for subset in text_source_subsets
    }
    # Loop through the number of data to be generated based on the configuration, randomly selecting a model config for each iteration
    for i, job in enumerate(jobs):
        print(f"Data Generation {i + 1}/{data_number} on GPU {args.gpu} ...")
        selected_config_path = job["config_path"]
        dataset_split = job["dataset_split"]

        with selected_config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        mode = config["mode"]
        script_path = _resolve_path(config["script_path"], config_base_dir)
        reference_audio_override = Path(job["reference_audio_path"])
        if not reference_audio_override.is_file():
            raise FileNotFoundError(
                f"Scheduled reference audio is missing: {reference_audio_override}"
            )

        text_source_subset = job["text_source_subset"]
        transcript_path = None
        if text_source_subset == "source_transcript":
            source_text, transcript_path = _read_reference_transcript(
                reference_audio_override
            )
        else:
            source_text = _random_mcv_english_text(
                script_path, job["text_seed"]
            )

        # The original output and all later attack levels remain inside the
        # folder for this text subset.
        output_path = (
            subset_output_folders[text_source_subset] / f"{mode}_{i + 1:05d}.wav"
        )
        current_audio_path = output_path

        existing_record = data_storing_by_subset[text_source_subset].generation_log.get(
            output_path.name
        )
        if (
            resume_enabled
            and output_path.is_file()
            and _record_matches_job(
                existing_record,
                job,
                mode,
                text_source_subset,
                reference_audio_override,
            )
            and _record_is_complete(
                existing_record,
                max_attack_number,
                quality_gate_enabled,
            )
        ):
            progress["jobs"][str(i)] = {
                "status": "complete",
                "output_path": str(output_path),
                "updated_at": _checkpoint_timestamp(),
            }
            print(
                f"Skipping completed job {i + 1}/{data_number}: {output_path.name}"
            )
            continue

        progress["jobs"][str(i)] = {
            "status": "running",
            "output_path": str(output_path),
            "started_at": _checkpoint_timestamp(),
        }
        _atomic_write_json(progress_path, progress)

        # Use a configured generation type when provided; otherwise choose randomly.
        item = job["generation_type"]
        identity_metadata = {
            "selection": item,
            "dataset_split": dataset_split,
            "source_pair_id": job["source_pair_id"],
            "text_source_subset": text_source_subset,
            "output_subset": text_source_subset,
            "source_text": source_text,
            "reference_audio": str(reference_audio_override),
        }
        if transcript_path is not None:
            identity_metadata["reference_transcript"] = str(transcript_path)

        # Every enabled model is a clone worker.  Its config owns the command
        # shape; this driver supplies the common job values to its placeholders.
        if item != "clone":
            raise ValueError(f"Model {mode!r} supports clone mode only")

        speaker_index = job["speaker_index"]
        selected_speaker = source_speakers[speaker_index]
        identity_metadata.update({
            "type": "libritts_speaker",
            "pool_file": str(libritts_speaker_file),
            "speaker_number": selected_speaker["speaker_number"],
            "speaker_id": selected_speaker["speaker_id"],
            "split": selected_speaker["split"],
        })
        command = _render_worker_command(
            config,
            {
                "script_path": str(script_path),
                "output_path": str(output_path),
                "reference_audio_path": str(
                    source_audio_path
                    or _resolve_path(config["reference_audio_path"], config_base_dir)
                ),
                "item": item,
                "speaker_file": str(libritts_speaker_file),
                "speaker_index": str(speaker_index),
                "instruct_path": str(instruct_path),
                "transcript_text": source_text,
                "reference_audio_override": str(reference_audio_override),
            },
            config_base_dir,
        )

        # Since we are using subprocesses to run the TTS models, we enforce the use of the correct Python environment
        # and the correct GPU for the subprocesses by setting the environment variables accordingly
        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        # Librosa uses Numba during the Coqui import.  The environments live
        # behind NAS symlinks, so Numba cannot infer a cache location there.
        # Give every worker a stable local cache unless the caller supplied one.
        numba_cache_dir = Path(
            worker_env.get(
                "NUMBA_CACHE_DIR",
                str(Path(tempfile.gettempdir()) / f"numba-cache-{os.getuid()}"),
            )
        )
        numba_cache_dir.mkdir(parents=True, exist_ok=True)
        worker_env["NUMBA_CACHE_DIR"] = str(numba_cache_dir)

        # Keep reusable Hugging Face downloads on the local NVMe cache. Set
        # each path explicitly so worker environments do not inherit a stale
        # external cache setting from the launcher shell.
        home_cache_dir = Path.home() / ".cache"
        hf_cache_dir = home_cache_dir / "huggingface"
        torch_cache_dir = home_cache_dir / "torch"
        nltk_cache_dir = Path.home() / ".local" / "share" / "nltk_data"
        for cache_dir in (hf_cache_dir, torch_cache_dir, nltk_cache_dir):
            cache_dir.mkdir(parents=True, exist_ok=True)
        worker_env["HF_HOME"] = str(hf_cache_dir)
        worker_env["HUGGINGFACE_HUB_CACHE"] = str(hf_cache_dir / "hub")
        worker_env["TRANSFORMERS_CACHE"] = str(hf_cache_dir / "hub")
        worker_env["TORCH_HOME"] = str(torch_cache_dir)
        worker_env["NLTK_DATA"] = str(nltk_cache_dir)

        # Keep Coqui XTTS-v2/YourTTS checkpoints in the user's home cache.
        if mode in {"xtts_v2", "yourtts"}:
            coqui_cache_dir = Path(worker_env.get("TTS_HOME", str(tts_home)))
            coqui_cache_dir.mkdir(parents=True, exist_ok=True)
            worker_env["TTS_HOME"] = str(coqui_cache_dir)

        if mode == "chatterbox":
            worker_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

        if mode == "maskgct":
            # onnxruntime-gpu loads cuDNN/CUDA through the dynamic linker. The
            # pip-provided CUDA libraries live below site-packages/nvidia, so
            # expose them before the worker process starts.
            cuda_library_dirs = _maskgct_cuda_library_dirs(
                _worker_python(mode, config)
            )
            if not cuda_library_dirs:
                raise FileNotFoundError(
                    "No CUDA library directories found in the MaskGCT environment"
                )
            existing_library_path = worker_env.get("LD_LIBRARY_PATH", "")
            worker_env["LD_LIBRARY_PATH"] = ":".join(
                cuda_library_dirs
                + ([existing_library_path] if existing_library_path else [])
            )

        # Execute the subprocess command to generate audio using the selected model
        subprocess.run(command, check=True, env=worker_env)

        # Store the first metadata about the generated audio file, including its model and generation type
        original_scores = _score_with_detectors(
            output_path, detector_specs, detector_device
        )

        data_storing = data_storing_by_subset[text_source_subset]

        # Store the first metadata about the generated audio file, including its model and generation type
        data_storing.store_metadata(output_path.name, mode, item, None, None, None, None, None, None, None, None, first=True, detector_scores=original_scores, identity=identity_metadata)

        # Preserve the generated WAV as level 0, then use a compressed copy as the
        # starting point for the randomized attack chain.
        compression_method = attack.choose_compression()
        print(f"Initial compression: {compression_method.__name__}")
        current_audio_path, _, _ = attack.apply_compression(
            str(output_path),
            str(output_path.parent),
            0,
            compression_method,
        )

        compressed_scores = _score_with_detectors(
            current_audio_path, detector_specs, detector_device
        )
        data_storing.store_metadata(
            output_path.name,
            None,
            None,
            None,
            None,
            None,
            current_audio_path,
            "compression",
            compression_method.__name__,
            0,
            passed_condition=True,
            first=False,
            detector_scores=compressed_scores,
        )

        # Apply only randomized attacks at levels 1 through maximum_attack.
        for attack_index in range(max_attack_number):
            print(f"Applying attack {attack_index + 1} from maximum {max_attack_number} ...")

            attacked_path, attack_type, attack_parameter = attack.random_attack(
                str(current_audio_path),
                str(output_path.parent),
                attack_index + 1,
            )

            # Compute the quality metrics for the attacked audio, including STOI, PESQ, and UTMOS scores
            stoi_score = quality_check.compute_stoi(str(output_path), str(attacked_path))
            pesq_score = quality_check.compute_pesq(str(output_path), str(attacked_path))
            utmos_score = quality_check.compute_utmos(str(attacked_path))

            # Quality failures are recorded, then discarded from the audio dataset.
            bad_metrics = sum([utmos_score < 2.0, stoi_score < 0.55, pesq_score < 1.5])
            if quality_gate_enabled and bad_metrics >= 2:
                data_storing.store_metadata(
                    output_path.name, None, None, stoi_score, pesq_score,
                    utmos_score, attacked_path, attack_type, attack_parameter,
                    attack_index + 1, passed_condition=False, first=False,
                    rejection_reason="quality",
                )
                print("Removing rejected attack: {} (quality thresholds failed)".format(
                    attacked_path))
                Path(attacked_path).unlink(missing_ok=True)
                break

            attack_scores = _score_with_detectors(
                attacked_path, detector_specs, detector_device
            )
            all_baselines_fake = bool(attack_scores) and all(
                score["prediction"] == "fake"
                for score in attack_scores.values()
            )

            baseline_rejected = (
                remove_if_all_baselines_fake and all_baselines_fake)

            # Consensus-fake files are logged for auditability before removal.
            data_storing.store_metadata(output_path.name, None, None, stoi_score, pesq_score, utmos_score, 
                    attacked_path, attack_type, attack_parameter, attack_index + 1,
                    passed_condition=not baseline_rejected, first=False,
                    detector_scores=attack_scores,
                    rejection_reason=("baseline_consensus" if baseline_rejected else None))

            if baseline_rejected:
                print("Removing rejected attack: {} (all baseline models predicted fake)".format(
                    attacked_path))
                Path(attacked_path).unlink(missing_ok=True)
                break

            current_audio_path = attacked_path

        progress["jobs"][str(i)] = {
            "status": "complete",
            "output_path": str(output_path),
            "updated_at": _checkpoint_timestamp(),
        }
        _atomic_write_json(progress_path, progress)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Physical GPU ID to use for inference",
    )

    parser.add_argument(
        "--config",
        default="pipeline.config",
        help="Path to the pipeline configuration file",
    )

    parser.add_argument(
        "--evaluate-model",
        action="append",
        choices=[
            "rawnet", "aasist", "sonar", "sonar-finetune", "hyperpotter",
            "slsforasvspoof", "rawbmamba", "whisper", "prosdd",
        ],
        help="Score original and accepted attack audio in generation.json; repeat for multiple models",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing schedule/progress checkpoints and start the configured jobs again",
    )

    main(parser.parse_args())
