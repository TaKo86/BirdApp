"""Audio and dataset metadata helpers for the New Zealand bird-sound data."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as audio_functional
from torchaudio.transforms import MelSpectrogram

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
SAMPLE_RATE = 22_050
CLIP_SECONDS = 4.0
N_MELS = 128


def iter_audio_files(root: Path):
    """Yield supported audio files in a stable order."""
    yield from sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        ),
        key=lambda path: str(path).casefold(),
    )


def source_recording_id(relative_path: str | Path) -> str:
    """Strip a trailing numbered crop suffix and retain its dataset collection."""
    path = Path(str(relative_path).replace("/", "\\"))
    collection = path.parts[0] if len(path.parts) > 1 else "dataset"
    stem = re.sub(r"_\d+$", "", path.stem)
    return f"{collection}/{stem}".casefold()


def load_species_names(root: Path) -> dict[str, dict[str, str]]:
    """Read included bird naming maps keyed by eBird species code."""
    names: dict[str, dict[str, str]] = {}
    for path in root.glob("DOC_*/*/bird_naming_map.csv"):
        frame = pd.read_csv(path, dtype=str).fillna("")
        code_column = "eBird" if "eBird" in frame else None
        if code_column is None:
            continue
        for row in frame.to_dict(orient="records"):
            code = str(row.get(code_column, "")).strip().casefold()
            if code:
                names[code] = {
                    "common_name": str(row.get("CommonName", "")).strip(),
                    "scientific_name": str(row.get("ScientificName", "")).strip(),
                }
    return names


def load_metadata_labels(root: Path) -> dict[str, str]:
    """Map dataset-relative audio paths to their primary labels in metadata CSVs."""
    labels: dict[str, str] = {}
    for metadata_path in root.glob("DOC_*/*/*_metadata.csv"):
        metadata = pd.read_csv(
            metadata_path, usecols=["filename", "primary_label"]
        )
        collection = metadata_path.relative_to(root).parts[0]
        audio_root = metadata_path.parent / "train_audio"
        for row in metadata.itertuples(index=False):
            rel = (Path(collection) / "train_audio" / Path(str(row.filename))).as_posix()
            labels[rel.casefold()] = str(row.primary_label).strip().casefold()
        if not audio_root.is_dir():
            raise FileNotFoundError(f"Metadata audio directory does not exist: {audio_root}")
    return labels


def decode_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Read a FLAC/audio clip, convert to mono and resample to the model rate."""
    audio, original_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1, keepdims=True)
    waveform = torch.from_numpy(audio[:, 0].copy())
    if original_rate != sample_rate:
        waveform = audio_functional.resample(waveform, original_rate, sample_rate)
    result = waveform.numpy()
    if result.size == 0:
        raise ValueError(f"No audio samples found in {path}")
    return result


def fixed_length_crop(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    clip_seconds: float = CLIP_SECONDS,
) -> np.ndarray:
    """Center crop to a fixed number of samples and pad short recordings."""
    target_samples = round(sample_rate * clip_seconds)
    if audio.size >= target_samples:
        start = (audio.size - target_samples) // 2
        return audio[start : start + target_samples].astype(np.float32, copy=False)
    crop = np.zeros(target_samples, dtype=np.float32)
    start = (target_samples - audio.size) // 2
    crop[start : start + audio.size] = audio
    return crop


@lru_cache(maxsize=4)
def _mel_transform(sample_rate: int, n_mels: int) -> MelSpectrogram:
    return MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=n_mels,
        f_min=50,
        f_max=sample_rate / 2,
        power=2.0,
    )


def log_mel_spectrogram(
    audio: np.ndarray, sample_rate: int = SAMPLE_RATE, n_mels: int = N_MELS
) -> np.ndarray:
    """Return a normalized, channel-first log-mel spectrogram."""
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32).copy())
    mel_power = _mel_transform(sample_rate, n_mels)(waveform)
    log_mel = (10.0 * torch.log10(mel_power.clamp_min(1e-10))).clamp_min(-80.0)
    log_mel = log_mel - log_mel.mean()
    log_mel = log_mel / log_mel.std(unbiased=False).clamp_min(1e-6)
    return log_mel.unsqueeze(0).numpy().astype(np.float32, copy=False)
