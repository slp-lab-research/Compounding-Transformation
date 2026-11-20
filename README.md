# Batched Fake/Real Audio Generation

This repository contains the two-stage batch workflow used to create matched
synthetic and genuine-speech transformation datasets.

1. `plan_batched_generation.py` creates a deterministic generation plan.
2. `run_batched_generation.py` generates cloned synthetic audio, applies its
   planned transformations, scores the selected detectors, and writes
   `batched_generation.json`.
3. `attack_batched_real_sources.py` applies the matching transformation plans
   to real source audio and writes `batched_real_generation.json`.
4. `wait_and_attack_batched_real_sources.py` optionally waits for the synthetic
   run to finish before launching step 3.

The repository contains code and configuration only. Datasets, model weights,
audio outputs, manifests, checkpoints, and caches are intentionally ignored by
Git.

## Core files

- `pipeline.config`: local batch configuration; copy and adapt its absolute
  paths before running on another machine.
- `plan_batched_generation.py`: deterministic batch-plan builder.
- `run_batched_generation.py`: persistent model-batch generation, attack, and
  detector-scoring runner.
- `attack_batched_real_sources.py`: matched real-source transformation runner.
- `wait_and_attack_batched_real_sources.py`: optional fake-to-real-stage
  coordinator.
- `persistent_batch_generator.py`: loads one configured TTS model at a time
  and processes its complete batch.
- `main.py`: shared planning utilities retained for
  `plan_batched_generation.py`; it is not the batch entry point.

## Required local assets

- A LibriTTS-compatible source corpus with transcript sidecars.
- An English transcript TSV with `lang` and `sentence` columns.
- WHAM noise audio and, when BGM attacks are enabled, a music collection.
- The checkpoints for enabled detector models.
- The Python environments and model weights for every enabled TTS model in
  `config/models/`.

## Run

Copy `pipeline.config` to a local file and replace every development-server
path with an absolute local path. Choose a local plan path, then run:

```bash
PLAN=/path/to/generation_batch_plan.json

python plan_batched_generation.py \
  --config /path/to/local-batch.config \
  --plan "$PLAN"

python run_batched_generation.py \
  --config /path/to/local-batch.config \
  --plan "$PLAN" \
  --gpu 0 \
  --evaluate-model rawnet \
  --evaluate-model aasist

python attack_batched_real_sources.py \
  --config /path/to/local-batch.config \
  --plan "$PLAN" \
  --gpu 0 \
  --evaluate-model rawnet \
  --evaluate-model aasist
```

Use the same detector list for the synthetic and real manifests when they will
be compared. Add or remove model configurations before plan creation; the plan
freezes the chosen models, sources, text, attack parameters, and random seeds.

## GitHub preparation

Before pushing, inspect the staged files:

```bash
git init
git add .
git status --short
git commit -m "Add batched fake and real audio generation workflow"
```

Do not force-add generated manifests or audio. `batched_generation.json` and
`batched_real_generation.json` can be hundreds of megabytes and must remain
outside Git.
