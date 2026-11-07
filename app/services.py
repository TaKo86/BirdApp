"""audio helpers and demo results; swap out classify() when the model is ready."""

import hashlib
from copy import deepcopy
from datetime import datetime
from io import BytesIO
from typing import TypedDict
from zoneinfo import ZoneInfo

import av
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import streamlit as st
from matplotlib.figure import Figure
from numpy.typing import NDArray

from mock_data import SPECIES, example_clip

# same settings as make_spectrograms.py, without running the dataset scripts
SAMPLE_RATE, CLIP_SECONDS = 32000, 5
N_FFT, HOP_LENGTH, N_MELS, FMIN, FMAX = 2048, 512, 128, 50, 15000
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
HISTORY_KEY = "birdcall_demo_history"


class AudioError(ValueError):
    """used when an audio file is empty or can't be read."""


class Prediction(TypedDict):
    label: str
    display_name: str
    confidence: float
    probabilities: dict[str, float]
    model: str
    mock: bool


@st.cache_data(show_spinner=False, max_entries=4, ttl=600)
def _decode_audio(audio_bytes: bytes) -> tuple[NDArray[np.float32], int]:
    """read the audio as mono, using FFmpeg if SoundFile can't open it."""
    if not audio_bytes:
        raise AudioError("This file is empty. Choose an audio recording or use the example clip.")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise AudioError("This recording is too large. Please choose a file under 25 MB.")
    try:
        try:
            data, rate = sf.read(BytesIO(audio_bytes), dtype="float32", always_2d=True)
            y = data.mean(axis=1)
        except (sf.LibsndfileError, ValueError):
            pieces = []
            resampler = None
            rate = 0
            with av.open(BytesIO(audio_bytes)) as container:
                if not container.streams.audio:
                    raise AudioError("This file has no audio track. Please choose another recording.")
                for frame in container.decode(audio=0):
                    if resampler is None:
                        rate = frame.sample_rate
                        resampler = av.AudioResampler(format="fltp", layout=frame.layout, rate=rate)
                    for decoded in resampler.resample(frame):
                        pieces.append(decoded.to_ndarray().mean(axis=0))
                if resampler is not None:
                    for decoded in resampler.resample(None):
                        pieces.append(decoded.to_ndarray().mean(axis=0))
            y = np.concatenate(pieces) if pieces else np.array([], dtype=np.float32)
        if not len(y):
            raise AudioError("This recording contains no audio samples. Please choose another file.")
        if rate <= 0 or not np.isfinite(y).all():
            raise AudioError("This recording contains invalid audio samples. Please choose another file.")
        return np.ascontiguousarray(y, dtype=np.float32), int(rate)
    except AudioError:
        raise
    except Exception as exc:
        raise AudioError("We couldn't read this recording. Try a valid WAV, MP3, M4A, OGG or FLAC file.") from exc


def get_audio_details(audio_bytes: bytes) -> dict[str, float | int]:
    """get the recording's length and original sample rate."""
    y, rate = _decode_audio(audio_bytes)
    return {"duration": len(y) / rate, "sample_rate": rate}


def get_playback_audio(audio_bytes: bytes) -> bytes:
    """convert to WAV so the browser can play it, without saving a file."""
    y, rate = _decode_audio(audio_bytes)
    buffer = BytesIO()
    sf.write(buffer, y, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def classify(audio_bytes: bytes, filename: str) -> Prediction:
    """give the same fake result for the same audio, even if it's renamed."""
    _decode_audio(audio_bytes)  # check the file works before making a demo result
    # todo: replace with a call to the real model / Flask API
    digest = hashlib.sha256(audio_bytes).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "big"))
    winner = int(rng.integers(len(SPECIES)))
    confidence = float(rng.uniform(0.35, 0.96))
    weights = rng.uniform(0.4, 1.0, len(SPECIES))
    weights[winner] = 0
    weights *= (1 - confidence) / weights.sum()
    weights[winner] = confidence
    probabilities = {SPECIES[i]["display_name"]: float(weights[i])
                     for i in np.argsort(-weights)[:5]}
    species = SPECIES[winner]
    return {"label": species["label"], "display_name": species["display_name"],
            "confidence": confidence, "probabilities": probabilities,
            "model": "Demo model (placeholder)", "mock": True}


def get_spectrogram_figure(audio_bytes: bytes) -> Figure:
    """plot the first five seconds as log-Mel, adding silence if the clip is shorter."""
    y, rate = _decode_audio(audio_bytes)
    y = librosa.resample(y[:rate * CLIP_SECONDS], orig_sr=rate, target_sr=SAMPLE_RATE)
    y = librosa.util.fix_length(y, size=SAMPLE_RATE * CLIP_SECONDS)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    db = librosa.power_to_db(mel, ref=np.max)
    fig = Figure(figsize=(10, 3.5), layout="constrained", facecolor="#fffefa")
    ax = fig.subplots()
    plot = librosa.display.specshow(
        db, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis="time", y_axis="mel",
        fmin=FMIN, fmax=FMAX, cmap="magma", vmin=-80, vmax=0, ax=ax,
    )
    ax.set(xlabel="Time (seconds)", ylabel="Frequency (Hz)", xlim=(0, CLIP_SECONDS))
    ax.tick_params(colors="#4d6058", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(plot, ax=ax, label="dB relative to peak", pad=0.02)
    return fig


def get_example_clip() -> bytes:
    """get the example tone so the demo works without an upload."""
    return example_clip()


def get_demo_species() -> list[dict[str, str]]:
    """get a copy of the demo bird list."""
    return deepcopy(SPECIES)


def get_species_details(label: str) -> dict[str, str]:
    """find the bird's details, with a fallback for labels added later."""
    return next((dict(item) for item in SPECIES if item["label"] == label),
                {"maori_name": "", "fact": "Species information will be added here.", "source": ""})


def save_to_history(result: Prediction, filename: str) -> None:
    """add the latest result to the top of this session's history."""
    rows = list(st.session_state.get(HISTORY_KEY, []))
    rows.insert(0, {"time": datetime.now(ZoneInfo("Pacific/Auckland")).strftime("%Y-%m-%d %H:%M:%S %Z"),
                    "filename": filename, "species": result["display_name"],
                    "confidence": result["confidence"]})
    st.session_state[HISTORY_KEY] = rows


def load_history() -> list[dict[str, str | float]]:
    """get this session's results, newest first."""
    return deepcopy(st.session_state.get(HISTORY_KEY, []))


def clear_history() -> None:
    """clear the results from this session."""
    st.session_state[HISTORY_KEY] = []
