"""Shared audio and annotation helpers for the CACOPHONY bird dataset."""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import imageio_ffmpeg
import numpy as np
import torch
from torchaudio.transforms import MelSpectrogram

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
SAMPLE_RATE = 22_050
CLIP_SECONDS = 4.0
N_MELS = 128
GENERIC_BIRD_LABELS = {"bird", "unidentified", "other"}


def iter_audio_files(root: Path) -> Iterator[Path]:
    """Yield supported audio files in a stable order."""
    yield from sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS),
        key=lambda path: str(path).casefold(),
    )


def read_annotation(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read one JSON sidecar, returning an explicit error for malformed files."""
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(value, dict):
        return None, "annotation root must be a JSON object"
    return value, None


def iter_manual_bird_segments(
    metadata: dict[str, Any], audio_path: Path
) -> Iterator[dict[str, Any]]:
    """Yield manually tagged bird segments; shard directory names are not labels."""
    recording_id = str(metadata.get("id", audio_path.stem))
    for track_index, track in enumerate(metadata.get("Tracks") or []):
        if not isinstance(track, dict):
            continue
        try:
            start = float(track["start"])
            end = float(track["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
            continue

        seen_labels: set[str] = set()
        for tag in track.get("tags") or []:
            if not isinstance(tag, dict) or tag.get("automatic") is True:
                continue
            label = str(tag.get("what") or "").strip().casefold()
            tag_path = str(tag.get("path") or "")
            if (
                not label
                or label in GENERIC_BIRD_LABELS
                or not tag_path.startswith("all.bird.")
                or label in seen_labels
            ):
                continue
            seen_labels.add(label)
            yield {
                "recording_id": recording_id,
                "audio_path": audio_path,
                "track_index": track_index,
                "start": start,
                "end": end,
                "label": label,
                "tag_path": tag_path,
            }


def decode_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode a complete audio file to mono float32 PCM using bundled FFmpeg."""
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_f32le",
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"FFmpeg produced no audio samples for {path}")
    return audio


def crop_audio(
    audio: np.ndarray,
    start: float,
    end: float,
    sample_rate: int = SAMPLE_RATE,
    clip_seconds: float = CLIP_SECONDS,
) -> np.ndarray:
    """Return a fixed-length window centered on an annotated segment."""
    target_samples = round(sample_rate * clip_seconds)
    center_sample = round(((start + end) / 2) * sample_rate)
    first_sample = center_sample - target_samples // 2
    last_sample = first_sample + target_samples
    crop = np.zeros(target_samples, dtype=np.float32)
    source_start = max(0, first_sample)
    source_end = min(audio.size, last_sample)
    if source_end > source_start:
        destination_start = source_start - first_sample
        destination_end = destination_start + source_end - source_start
        crop[destination_start:destination_end] = audio[source_start:source_end]
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
    """Create a per-clip normalized log-mel feature with a channel dimension."""
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32).copy())
    mel_power = _mel_transform(sample_rate, n_mels)(waveform)
    log_mel = (10.0 * torch.log10(mel_power.clamp_min(1e-10))).clamp_min(-80.0)
    log_mel = log_mel - log_mel.mean()
    log_mel = log_mel / log_mel.std(unbiased=False).clamp_min(1e-6)
    return log_mel.unsqueeze(0).numpy().astype(np.float32, copy=False)
