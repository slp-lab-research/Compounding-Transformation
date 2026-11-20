#!/usr/bin/env python3
"""Load one voice-cloning model once and execute a JSON batch of jobs."""

import argparse
import json
import os
import random
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def reference_text(reference):
    for suffix in (".normalized.txt", ".original.txt"):
        candidate = reference.with_suffix(suffix)
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8").strip()
            if text:
                return text
    raise FileNotFoundError(f"Missing transcript sidecar for {reference}")


@contextmanager
def cosyvoice_reference_prompt(reference, torchaudio, max_seconds=29.5):
    """Yield a CosyVoice-compatible prompt without modifying the source WAV."""
    waveform, sample_rate = torchaudio.load(str(reference))
    maximum_frames = int(float(max_seconds) * sample_rate)
    if waveform.shape[1] <= maximum_frames:
        yield str(reference)
        return

    original_seconds = waveform.shape[1] / sample_rate
    with tempfile.TemporaryDirectory(prefix="cosyvoice-prompt-") as directory:
        trimmed_reference = Path(directory) / "reference.wav"
        torchaudio.save(
            str(trimmed_reference),
            waveform[:, :maximum_frames],
            sample_rate,
        )
        print(
            f"Trimmed CosyVoice reference from {original_seconds:.2f}s "
            f"to {max_seconds:.2f}s: {reference}",
            flush=True,
        )
        yield str(trimmed_reference)


class Generator:
    def __init__(self, mode, instruct_path=None):
        import torch

        self.mode = mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.instruct_path = Path(instruct_path) if instruct_path else None
        self.runtime = self._load()

    def _load(self):
        mode = self.mode
        if mode == "chatterbox":
            from chatterbox.tts import ChatterboxTTS
            return ChatterboxTTS.from_pretrained(device="cuda:0")
        if mode == "qwen":
            import torch
            from qwen_tts import Qwen3TTSModel
            return Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                attn_implementation="sdpa",
            )
        if mode == "cosyvoice":
            model_root = Path(os.environ.get(
                "IEMOCAP_MODEL_ROOT", "/mnt/synology_nas_00/nathan/all_model"
            ))
            source = model_root / "CosyVoice"
            sys.path[:0] = [str(source), str(source / "third_party/Matcha-TTS")]
            from cosyvoice.cli.cosyvoice import AutoModel
            model_dir = model_root / "pretrained_models/Fun-CosyVoice3-0.5B"
            return AutoModel(model_dir=str(model_dir))
        if mode in {"xtts_v2", "yourtts"}:
            from TTS.api import TTS
            model_id = {
                "xtts_v2": "tts_models/multilingual/multi-dataset/xtts_v2",
                "yourtts": "tts_models/multilingual/multi-dataset/your_tts",
            }[mode]
            return TTS(model_id).to(self.device)
        if mode == "styletts2":
            from styletts2 import tts
            return tts.StyleTTS2()
        if mode == "voxcpm2":
            from voxcpm import VoxCPM
            return VoxCPM.from_pretrained(
                "openbmb/VoxCPM2", device="auto", optimize=True
            )
        if mode == "f5tts":
            from f5_tts.api import F5TTS
            return F5TTS(model="F5TTS_v1_Base", device=self.device)
        if mode == "openvoice":
            import torch
            from melo.api import TTS as MeloTTS
            from openvoice.api import ToneColorConverter
            root = Path(
                "/mnt/synology_nas_00/nathan/huggingface_model/OpenVoice/checkpoints_v2"
            )
            converter = ToneColorConverter(
                str(root / "converter/config.json"), device=self.device
            )
            converter.load_ckpt(str(root / "converter/checkpoint.pth"))
            melo = MeloTTS(language="EN_NEWEST", device=self.device)
            source_se = torch.load(
                str(root / "base_speakers/ses/en-newest.pth"),
                map_location=self.device,
            )
            return converter, melo, source_se
        if mode == "maskgct":
            return self._load_maskgct()
        raise ValueError(f"Unsupported model mode: {mode}")

    def _load_maskgct(self):
        import safetensors.torch
        import torch
        repository = Path("/mnt/synology_nas_00/nathan/huggingface_model/Amphion")
        checkpoints = Path("/mnt/synology_nas_00/nathan/huggingface_model/MaskGCT")
        sys.path.insert(0, str(repository))
        previous = Path.cwd()
        os.chdir(repository)
        try:
            from models.tts.maskgct.maskgct_utils import (
                MaskGCT_Inference_Pipeline, build_acoustic_codec,
                build_s2a_model, build_semantic_codec, build_semantic_model,
                build_t2s_model, load_config,
            )
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            config = load_config("models/tts/maskgct/config/maskgct.json")
            semantic_model, mean, std = build_semantic_model(device)
            semantic_codec = build_semantic_codec(config.model.semantic_codec, device)
            encoder, decoder = build_acoustic_codec(config.model.acoustic_codec, device)
            t2s = build_t2s_model(config.model.t2s_model, device)
            s2a1 = build_s2a_model(config.model.s2a_model.s2a_1layer, device)
            s2af = build_s2a_model(config.model.s2a_model.s2a_full, device)
            pairs = (
                (semantic_codec, "semantic_codec/model.safetensors"),
                (encoder, "acoustic_codec/model.safetensors"),
                (decoder, "acoustic_codec/model_1.safetensors"),
                (t2s, "t2s_model/model.safetensors"),
                (s2a1, "s2a_model/s2a_model_1layer/model.safetensors"),
                (s2af, "s2a_model/s2a_model_full/model.safetensors"),
            )
            for model, relative in pairs:
                safetensors.torch.load_model(model, str(checkpoints / relative))
            return MaskGCT_Inference_Pipeline(
                semantic_model, semantic_codec, encoder, decoder, t2s,
                s2a1, s2af, mean, std, device,
            )
        finally:
            os.chdir(previous)

    def generate(self, job):
        import soundfile as sf
        import torch
        output = Path(job["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        reference = Path(job["reference_audio"])
        text = job["source_text"]
        mode = self.mode
        if mode == "chatterbox":
            import torchaudio
            wav = self.runtime.generate(text, audio_prompt_path=str(reference))
            torchaudio.save(str(output), wav, sample_rate=24000)
        elif mode == "qwen":
            wavs, rate = self.runtime.generate_voice_clone(
                text=text, language="English", ref_audio=str(reference),
                x_vector_only_mode=True,
            )
            sf.write(str(output), wavs[0], rate)
        elif mode == "cosyvoice":
            import torchaudio
            instruction = self._instruction(job["job_id"])
            with cosyvoice_reference_prompt(reference, torchaudio) as prompt:
                result = next(self.runtime.inference_instruct2(
                    text, instruction, prompt, stream=False
                ))
            torchaudio.save(str(output), result["tts_speech"], self.runtime.sample_rate)
        elif mode in {"xtts_v2", "yourtts"}:
            self.runtime.tts_to_file(
                text=text, speaker_wav=str(reference), language="en",
                file_path=str(output),
            )
        elif mode == "styletts2":
            # Gruut spells long all-uppercase passages letter by letter, which
            # can push otherwise short text beyond PLBERT's 512-token limit.
            # Case does not alter the intended spoken wording, and the original
            # source_text remains unchanged in the plan and metadata.
            styletts_text = text.lower()
            self.runtime.inference(
                styletts_text,
                target_voice_path=str(reference),
                output_wav_file=str(output),
            )
        elif mode == "voxcpm2":
            audio = self.runtime.generate(text=text, reference_wav_path=str(reference))
            rate = getattr(getattr(self.runtime, "tts_model", None), "sample_rate", 16000)
            sf.write(str(output), audio, rate)
        elif mode == "f5tts":
            self.runtime.infer(
                ref_file=reference, ref_text=reference_text(reference),
                gen_text=text, file_wave=str(output), seed=42,
            )
        elif mode == "openvoice":
            from openvoice import se_extractor
            converter, melo, source_se = self.runtime
            try:
                target_se, _ = se_extractor.get_se(str(reference), converter, vad=True)
            except AssertionError:
                target_se, _ = se_extractor.get_se(str(reference), converter, vad=False)
            temporary = output.with_name(f".{output.stem}.melo.wav")
            speaker_id = melo.hps.data.spk2id["EN-Newest"]
            melo.tts_to_file(text, speaker_id, str(temporary), speed=1.0)
            converter.convert(
                audio_src_path=str(temporary), src_se=source_se, tgt_se=target_se,
                output_path=str(output), message="@MyShell",
            )
            temporary.unlink(missing_ok=True)
        elif mode == "maskgct":
            repository = Path("/mnt/synology_nas_00/nathan/huggingface_model/Amphion")
            previous = Path.cwd()
            os.chdir(repository)
            try:
                audio = self.runtime.maskgct_inference(
                    str(reference), reference_text(reference), text, "en", "en"
                )
            finally:
                os.chdir(previous)
            sf.write(str(output), audio, 24000)
        else:
            raise ValueError(mode)

    def _instruction(self, job_id):
        data = json.loads(self.instruct_path.read_text(encoding="utf-8"))
        rng = random.Random(job_id)
        values = {
            key: rng.choice(options)
            for key, options in data["speaking_traits"].items()
        }
        result = data["prompt_template"]
        for key, value in values.items():
            result = result.replace(f"[{key}]", value)
        return result


def main(args):
    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    state_path = Path(args.state)
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    pending_jobs = [
        job for job in jobs
        if not (
            state.get(str(job["job_id"]), {}).get("status") == "complete"
            and Path(job["output_path"]).is_file()
        )
    ]
    if not pending_jobs:
        print(f"All {args.model} generation jobs are already complete.", flush=True)
        return
    generator = Generator(args.model, args.instruct_path)
    for position, job in enumerate(pending_jobs, start=1):
        key = str(job["job_id"])
        output = Path(job["output_path"])
        print(
            f"{args.model} {position}/{len(pending_jobs)}: job {key}",
            flush=True,
        )
        state[key] = {"status": "running", "output_path": str(output)}
        atomic_json(state_path, state)
        generator.generate(job)
        state[key] = {"status": "complete", "output_path": str(output)}
        atomic_json(state_path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--instruct-path", type=Path)
    main(parser.parse_args())
