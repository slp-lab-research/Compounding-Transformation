import argparse
import atexit
import csv
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(Path(tempfile.gettempdir()) / f"numba-cache-{os.getuid()}"),
)

import librosa
import numpy as np
import torch
import torch.nn.functional as F

# The bundled Fairseq version still references aliases removed in NumPy 2.x.
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

SAMPLE_RATE = 16000
TARGET_SAMPLES = 64600
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".opus", ".flac", ".aiff"}
ROOT = Path(__file__).resolve().parent
MODEL_ROOT = Path(
    os.environ.get("IEMOCAP_MODEL_ROOT", "/mnt/synology_nas_00/nathan/all_model")
).expanduser()
ENV_ROOT = Path(
    os.environ.get("IEMOCAP_CONDA_ENV_ROOT", "/mnt/synology_nas_00/nathan/conda_envs")
).expanduser()
ISOLATED_WORKER = ROOT / "isolated_detector_worker.py"
ISOLATED_ENVIRONMENTS = {
    "rawbmamba": "mamba-ssm",
    "whisper": "whisper",
    "prosdd": "prosdd",
}


def _model_path(*parts):
    """Return a checked path below the shared detector/model asset root."""
    path = MODEL_ROOT.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(f"Missing model asset: {path}")
    return path


def _load_state_dict(checkpoint_path, device):
    """Load the state dictionary formats used by the supplied detector repos."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing detector checkpoint: {checkpoint_path}")
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        # PyTorch 1.x does not support weights_only.
        state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload in {checkpoint_path}")
    return {
        key.removeprefix("module."): value
        for key, value in state.items()
    }


class IsolatedDetector:
    """A persistent detector process in a model-specific conda environment."""

    def __init__(self, model_name, device, checkpoint_path):
        environment_name = ISOLATED_ENVIRONMENTS[model_name]
        python_path = Path(
            os.environ.get(
                f"IEMOCAP_{model_name.upper()}_PYTHON",
                ENV_ROOT / environment_name / "bin/python",
            )
        )
        if not python_path.is_file():
            raise FileNotFoundError(
                f"Missing Python for {model_name} ({environment_name}): {python_path}. "
                f"Set IEMOCAP_{model_name.upper()}_PYTHON to its interpreter."
            )
        if not ISOLATED_WORKER.is_file():
            raise FileNotFoundError(f"Missing isolated detector worker: {ISOLATED_WORKER}")

        self.model_name = model_name
        self.device = torch.device(device)
        self.command = [
            str(python_path),
            str(ISOLATED_WORKER),
            "--model",
            model_name,
            "--model-root",
            str(MODEL_ROOT),
            "--checkpoint",
            str(checkpoint_path),
            "--device",
            str(device),
        ]
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        response = self._read_response()
        if response.get("status") != "ready":
            self.close()
            detail = response.get("error", response)
            raise RuntimeError(f"{model_name} worker failed to start:\n{detail}")
        atexit.register(self.close)

    def _read_response(self):
        if self.process.stdout is None:
            raise RuntimeError(f"{self.model_name} worker has no stdout pipe")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(
                f"{self.model_name} worker exited unexpectedly with code "
                f"{self.process.poll()}"
            )
        try:
            return json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"{self.model_name} worker produced a malformed protocol response: "
                f"{line.rstrip()}"
            ) from error

    def score_path(self, path):
        if self.process.poll() is not None:
            raise RuntimeError(
                f"{self.model_name} worker exited with code {self.process.returncode}"
            )
        if self.process.stdin is None:
            raise RuntimeError(f"{self.model_name} worker has no stdin pipe")
        self.process.stdin.write(json.dumps({"path": str(path)}) + "\n")
        self.process.stdin.flush()
        response = self._read_response()
        if response.get("status") != "ok":
            raise RuntimeError(
                f"{self.model_name} worker failed to score {path}:\n"
                f"{response.get('error', response)}"
            )
        return torch.tensor(response["logits"], dtype=torch.float32, device=self.device).unsqueeze(0)

    def close(self):
        if getattr(self, "process", None) is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def __del__(self):
        self.close()


def load_waveform(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(audio) >= TARGET_SAMPLES:
        audio = audio[:TARGET_SAMPLES]
    else:
        audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)))
    return torch.from_numpy(audio.astype(np.float32))


def score_logits(logits):
    probs = F.softmax(logits, dim=1)[0]
    spoof = float(probs[0].item())
    bona_fide = float(probs[1].item())
    return spoof, bona_fide, "fake" if spoof >= 0.5 else "real"


def output_rows(files, scorer, device):
    with torch.no_grad():
        for path in files:
            if isinstance(scorer, IsolatedDetector):
                logits = scorer.score_path(path)
            else:
                waveform = load_waveform(str(path)).unsqueeze(0).to(device)
                logits = scorer(waveform)
            spoof, bona_fide, prediction = score_logits(logits)
            yield {
                "file": str(path),
                "name": path.name,
                "spoof_probability": f"{spoof:.6f}",
                "bonafide_probability": f"{bona_fide:.6f}",
                "prediction": prediction,
            }


def write_scores(files, scorer, output_path, device):
    rows = list(output_rows(files, scorer, device))
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])) if rows else None
        if writer:
            writer.writeheader()
            writer.writerows(rows)
    fake_rate = sum(row["prediction"] == "fake" for row in rows) / max(1, len(rows))
    print(f"Scored {len(rows)} files; fake detection rate: {fake_rate:.4f}")
    print(f"Saved: {output_path}")


def load_aasist(device, checkpoint_path=None):
    aasist_dir = _model_path("aasist")
    if str(aasist_dir) not in sys.path:
        sys.path.insert(0, str(aasist_dir))
    config = json.loads((aasist_dir / "config/AASIST_in_the_wild.conf").read_text())
    module = importlib.import_module("models.AASIST")
    model = module.Model(config["model_config"]).to(device)
    checkpoint_path = checkpoint_path or aasist_dir / "models/weights/AASIST.pth"
    model.load_state_dict(_load_state_dict(checkpoint_path, device))
    model.eval()
    return lambda x: model(x)[1]


def load_rawnet(device, checkpoint_path=None):
    import yaml
    rawnet_dir = _model_path("2021", "DF", "Baseline-RawNet2")
    module = _load_module_from_path(
        "rawnet2_detector_model", rawnet_dir / "model.py"
    )
    config = yaml.safe_load((rawnet_dir / "model_config_RawNet.yaml").read_text())
    model = module.RawNet(config["model"], str(device)).to(device)
    checkpoint_path = checkpoint_path or _model_path("2021", "pre_trained_DF_RawNet2.pth")
    model.load_state_dict(_load_state_dict(checkpoint_path, device))
    model.eval()
    return model


def load_sonar_full(device, checkpoint_path=None):
    sonar_dir = _model_path("SONAR")
    if str(sonar_dir) not in sys.path:
        sys.path.insert(0, str(sonar_dir))
    from sonar.inference import SONARDetector
    detector = SONARDetector(
        ckpt=str(checkpoint_path or sonar_dir / "checkpoints/sonar_full_xlsr_aasist_eer6.pth"),
        device=str(device),
        xlsr_ckpt=str(sonar_dir / "checkpoints/xlsr2_300m.pt"),
    )

    def scorer(waveform):
        result = detector.score_audio(waveform[0].cpu().numpy())
        return torch.tensor([[result.spoof_prob, result.bonafide_prob]], device=device)

    return scorer

def load_sonar_finetune(device, checkpoint_path=None):
    sonar_dir = _model_path("SONAR")
    if str(sonar_dir) not in sys.path:
        sys.path.insert(0, str(sonar_dir))
    from sonar.inference import SONARDetector
    detector = SONARDetector(
        ckpt=str(checkpoint_path or sonar_dir / "checkpoints/sonar_finetune_xlsr_mamba_eer5p5.pth"),
        device=str(device),
        xlsr_ckpt=str(sonar_dir / "checkpoints/xlsr2_300m.pt"),
    )

    def scorer(waveform):
        result = detector.score_audio(waveform[0].cpu().numpy())
        return torch.tensor([[result.spoof_prob, result.bonafide_prob]], device=device)

    return scorer


def load_hyperpotter(device, checkpoint_path=None):
    hyperpotter_dir = _model_path("HyperPotter")
    xlsr_checkpoint = _model_path("SONAR", "checkpoints", "xlsr2_300m.pt")
    if not xlsr_checkpoint.is_file():
        raise FileNotFoundError(
            f"HyperPotter requires the XLS-R checkpoint at {xlsr_checkpoint}."
        )
    os.environ["HYPERPOTTER_XLSR_CKPT"] = str(xlsr_checkpoint)
    if str(hyperpotter_dir) not in sys.path:
        sys.path.insert(0, str(hyperpotter_dir))
    module = _load_module_from_path(
        "hyperpotter_detector_model", hyperpotter_dir / "model.py"
    )
    model = module.Model(argparse.Namespace(), device).to(device)
    checkpoint_path = checkpoint_path or hyperpotter_dir / "HyperPotter.pth"
    model.load_state_dict(_load_state_dict(checkpoint_path, device))
    model.eval()
    return model


def load_slsforasvspoof(device, checkpoint_path=None):
    sls_dir = _model_path("SLSforASVspoof-2021-DF")
    xlsr_checkpoint = _model_path("SONAR", "checkpoints", "xlsr2_300m.pt")
    if not xlsr_checkpoint.is_file():
        raise FileNotFoundError(
            f"SLS requires the XLS-R checkpoint at {xlsr_checkpoint}. "
            "Download it before using --evaluate-model slsforasvspoof."
        )
    if str(sls_dir) not in sys.path:
        sys.path.insert(0, str(sls_dir))
    fairseq_dir = sls_dir / "fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1"
    if str(fairseq_dir) not in sys.path:
        sys.path.insert(0, str(fairseq_dir))
    os.environ["SLS_XLSR_CKPT"] = str(xlsr_checkpoint)
    module = _load_module_from_path("sls_detector_model", sls_dir / "model.py")
    model = module.Model(argparse.Namespace(), device).to(device)
    checkpoint_path = checkpoint_path or sls_dir / "MMpaper_model.pth"
    model.load_state_dict(_load_state_dict(checkpoint_path, device))
    model.eval()

    def scorer(waveform):
        audio = waveform[0].detach().cpu().numpy()
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(sls_dir)) as temp_handle:
            feature_path = Path(temp_handle.name)
        sf.write(feature_path, audio, SAMPLE_RATE)
        try:
            import librosa
            audio_features = librosa.load(str(feature_path), sr=16000)[0]
            if audio_features.size >= 64600:
                audio_features = audio_features[:64600]
            else:
                repeats = 64600 // max(1, audio_features.size) + 1
                audio_features = np.tile(audio_features, repeats)[:64600]
            logits = model(torch.from_numpy(audio_features).float().unsqueeze(0).to(device))
            return logits
        finally:
            feature_path.unlink(missing_ok=True)

    return scorer

def load_rawbmamba(device, checkpoint_path=None):
    checkpoint_path = checkpoint_path or _model_path(
        "RawBMamba", "model", "rawbmamba_bset.pt"
    )
    return IsolatedDetector("rawbmamba", device, checkpoint_path)


def load_whisper(device, checkpoint_path=None):
    checkpoint_path = checkpoint_path or _model_path(
        "deepfake-whisper-features",
        "models",
        "whisper_mfcc_mesonet_finetuned",
        "weights.pth",
    )
    return IsolatedDetector("whisper", device, checkpoint_path)


def load_prosdd(device, checkpoint_path=None):
    checkpoint_path = checkpoint_path or _model_path(
        "codes", "models", "ProSDD", "ProSDD_Trained2019"
    )
    return IsolatedDetector("prosdd", device, checkpoint_path)


def _load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load detector module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_detector(model_name, device, checkpoint_path=None):
    loaders = {
        "rawnet": load_rawnet,
        "aasist": load_aasist,
        "sonar-full": load_sonar_full,
        "sonar-finetune": load_sonar_finetune,
        "hyperpotter": load_hyperpotter,
        "slsforasvspoof": load_slsforasvspoof,
        "rawbmamba": load_rawbmamba,
        "whisper": load_whisper,
        "prosdd": load_prosdd,
    }
    try:
        return loaders[model_name](device, checkpoint_path)
    except KeyError as error:
        raise ValueError(f"Unsupported detector: {model_name}") from error


def detector_variants(model_name):
    checkpoints = {
        "rawnet": [
            ("pretrained", _model_path("2021", "pre_trained_DF_RawNet2.pth")),
            ("asvspoof2024", _model_path("models", "Baselines_Trained2024", "RawNet2_Trained2024.pth")),
        ],
        "aasist": [
            ("pretrained", _model_path("aasist", "models", "weights", "AASIST.pth")),
            ("asvspoof2024", _model_path("models", "Baselines_Trained2024", "AASIST_Trained2024.pth")),
        ],
        "sonar-full": [
            ("pretrained", _model_path("SONAR", "checkpoints", "sonar_full_xlsr_aasist_eer6.pth")),
        ],
        "sonar-finetune": [
            ("pretrained", _model_path("SONAR", "checkpoints", "sonar_finetune_xlsr_mamba_eer5p5.pth")),
        ],
        "hyperpotter": [
            ("pretrained", _model_path("HyperPotter", "HyperPotter.pth")),
        ],
        "slsforasvspoof": [
            ("pretrained", _model_path("SLSforASVspoof-2021-DF", "MMpaper_model.pth")),
        ],
        "rawbmamba": [
            ("pretrained", _model_path("RawBMamba", "model", "rawbmamba_bset.pt")),
        ],
        "whisper": [
            ("whisper_frontend_mesonet", _model_path(
                "deepfake-whisper-features",
                "models",
                "whisper_mfcc_mesonet_finetuned",
                "weights.pth",
            )),
        ],
        "prosdd": [
            ("asvspoof2019", _model_path("codes", "models", "ProSDD", "ProSDD_Trained2019")),
            ("asvspoof2024", _model_path("codes", "models", "ProSDD", "ProSDD_Trained2024")),
        ],
    }
    try:
        return [variant for variant in checkpoints[model_name] if variant[1].is_file()]
    except KeyError as error:
        raise ValueError(f"Unsupported detector: {model_name}") from error


def score_file(path, scorer, device):
    with torch.no_grad():
        if isinstance(scorer, IsolatedDetector):
            logits = scorer.score_path(path)
        else:
            waveform = load_waveform(str(path)).unsqueeze(0).to(device)
            logits = scorer(waveform)
    spoof, bona_fide, prediction = score_logits(logits)
    return {
        "spoof_probability": spoof,
        "bonafide_probability": bona_fide,
        "prediction": prediction,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation_results")
    parser.add_argument(
        "--model",
        choices=[
            "rawnet", "aasist", "sonar-full", "sonar-finetune", "hyperpotter",
            "slsforasvspoof", "rawbmamba", "whisper", "prosdd",
        ],
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional checkpoint override (useful for one ProSDD or 2024-baseline variant)",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    files = sorted(path for path in args.audio_dir.rglob("*") if path.suffix.lower() in AUDIO_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No audio files found in {args.audio_dir}")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scorer = load_detector(args.model, device, args.checkpoint)
    try:
        output_path = args.output_dir / f"{args.model}_scores.csv"
        write_scores(files, scorer, output_path, device)
    finally:
        if isinstance(scorer, IsolatedDetector):
            scorer.close()


if __name__ == "__main__":
    main()
