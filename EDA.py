"""Explore ollypowell/new-zealand-bird-sound and write dataset reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import kagglehub
import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from audio_utils import (
    AUDIO_EXTENSIONS,
    load_metadata_labels,
    load_species_names,
    source_recording_id,
)

DATASET = "ollypowell/new-zealand-bird-sound"
OUT_DIR = Path(__file__).resolve().parent / "dataset_report"
ENDANGERED_LABELS: set[str] = set()


def count_endangered_crops(audio: pd.DataFrame, endangered_labels: set[str]) -> int:
    return int(audio["label"].isin(endangered_labels).sum())


def download() -> Path:
    """Download the Kaggle dataset or return its local KaggleHub cache path."""
    path = Path(kagglehub.dataset_download(DATASET))
    print(f"Dataset at: {path}")
    return path


def audio_info(path: Path) -> dict[str, Any]:
    """Read audio header information without decoding the full recording."""
    try:
        info = sf.info(str(path))
        return {
            "sr": info.samplerate,
            "channels": info.channels,
            "duration_s": info.duration,
            "ok": True,
            "err": "",
        }
    except (OSError, RuntimeError, sf.SoundFileRuntimeError) as exc:
        return {
            "sr": None,
            "channels": None,
            "duration_s": None,
            "ok": False,
            "err": str(exc)[:160],
        }


def build_inventory(
    root: Path, cache_path: Path, refresh: bool = False
) -> pd.DataFrame:
    metadata_labels = load_metadata_labels(root)
    species_names = load_species_names(root)
    if cache_path.is_file() and not refresh:
        cached = pd.read_csv(cache_path)
        if {"rel", "ext", "parent", "size_kb", "sr", "channels", "duration_s", "ok"}.issubset(cached.columns) and len(cached) == len(metadata_labels):
            cached["collection"] = cached["rel"].astype(str).map(
                lambda rel: Path(rel).parts[0] if Path(rel).parts else ""
            )
            cached["path"] = cached["rel"].astype(str).map(
                lambda rel: str(root / Path(rel))
            )
            cached["source_id"] = cached["rel"].map(source_recording_id)
            cached["err"] = ""
            labels: list[str] = []
            common_names: list[str] = []
            scientific_names: list[str] = []
            for row in cached.itertuples(index=False):
                relative = str(row.rel).replace("\\", "/")
                label = metadata_labels.get(
                    relative.casefold(), str(row.parent).casefold()
                )
                name_data = species_names.get(label, {})
                labels.append(label)
                common_names.append(name_data.get("common_name", ""))
                scientific_names.append(name_data.get("scientific_name", ""))
            cached["label"] = labels
            cached["common_name"] = common_names
            cached["scientific_name"] = scientific_names
            print(f"Reused cached audio headers for {len(cached):,} files")
            return cached
        print("Cached audio index does not match the dataset; rebuilding it")

    rows: list[dict[str, Any]] = []
    audio_paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        ),
        key=lambda path: str(path).casefold(),
    )
    for index, path in enumerate(audio_paths, start=1):
        relative = path.relative_to(root).as_posix()
        parts = Path(relative).parts
        fallback_label = path.parent.name.casefold()
        label = metadata_labels.get(relative.casefold(), fallback_label)
        name_data = species_names.get(label, {})
        info = audio_info(path)
        rows.append(
            {
                "path": str(path),
                "rel": relative,
                "collection": parts[0] if parts else "",
                "source_id": source_recording_id(relative),
                "ext": path.suffix.lower(),
                "parent": path.parent.name.casefold(),
                "size_kb": path.stat().st_size / 1024,
                "sr": info["sr"],
                "channels": info["channels"],
                "duration_s": info["duration_s"],
                "ok": info["ok"],
                "err": info["err"],
                "label": label,
                "common_name": name_data.get("common_name", ""),
                "scientific_name": name_data.get("scientific_name", ""),
            }
        )
        if index % 50_000 == 0:
            print(f"Indexed {index:,}/{len(audio_paths):,} audio files")
    return pd.DataFrame(rows)


def save_class_balance(audio: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    good = audio[audio["ok"]].copy()
    class_counts = (
        good.groupby(["label", "common_name", "scientific_name"], dropna=False)
        .agg(
            clip_count=("label", "size"),
            recording_count=("source_id", "nunique"),
            median_duration_s=("duration_s", "median"),
        )
        .reset_index()
        .sort_values("clip_count", ascending=False)
    )
    class_counts.to_csv(output_dir / "species_class_counts.csv", index=False)
    if not class_counts.empty:
        chart_labels = [
            f"{row.common_name} ({row.label})" if row.common_name else row.label
            for row in class_counts.itertuples()
        ]
        ordered = class_counts.assign(chart_label=chart_labels).sort_values("clip_count")
        height = max(6, len(ordered) * 0.18)
        _, axis = plt.subplots(figsize=(12, height))
        axis.barh(ordered["chart_label"], ordered["clip_count"])
        axis.set_xlabel("Readable audio clips")
        axis.set_title("Audio class balance")
        plt.tight_layout()
        plt.savefig(output_dir / "class_balance.png", dpi=130)
        plt.close()
    return class_counts


def save_spectrogram_samples(audio: pd.DataFrame, output_dir: Path) -> None:
    good = audio[audio["ok"]]
    samples = good.groupby("label", sort=True).head(1).head(12)
    if samples.empty:
        return
    columns = 3
    rows = (len(samples) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(5 * columns, 3 * rows), squeeze=False
    )
    for axis, (_, row) in zip(axes.flat, samples.iterrows()):
        waveform, sample_rate = librosa.load(
            row["path"], sr=None, mono=True, duration=8.0
        )
        mel = librosa.feature.melspectrogram(
            y=waveform, sr=sample_rate, n_mels=128
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(
            mel_db, sr=sample_rate, x_axis="time", y_axis="mel", ax=axis
        )
        title = row["common_name"] or row["label"]
        axis.set_title(f"{title} ({row['label']})", fontsize=9)
    for axis in axes.flat[len(samples) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "spectrogram_samples.png", dpi=120)
    plt.close(figure)


def build_report(
    data_dir: Path, output_dir: Path, refresh: bool = False
) -> dict[str, Any]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio = build_inventory(
        data_dir, output_dir / "audio_index.csv", refresh=refresh
    )
    if audio.empty:
        raise ValueError(f"No supported audio files found under {data_dir}")
    audio[
        [
            "rel",
            "ext",
            "parent",
            "size_kb",
            "sr",
            "channels",
            "duration_s",
            "ok",
            "label",
        ]
    ].to_csv(output_dir / "audio_index.csv", index=False)

    class_counts = save_class_balance(audio, output_dir)
    save_spectrogram_samples(audio, output_dir)
    good = audio[audio["ok"]]
    bad = audio[~audio["ok"]]
    extension_counts = audio["ext"].value_counts().to_dict()
    duration_stats = (
        good["duration_s"].describe().to_dict() if not good.empty else {}
    )
    source_sizes = good.groupby("collection")["source_id"].nunique().to_dict()
    label_counts = good["label"].value_counts()
    report: dict[str, Any] = {
        "dataset": DATASET,
        "dataset_dir": str(data_dir),
        "audio_files": int(len(audio)),
        "readable_audio_files": int(len(good)),
        "unreadable_audio_files": int(len(bad)),
        "audio_extensions": extension_counts,
        "collections": audio["collection"].value_counts().to_dict(),
        "unique_source_recordings": int(good["source_id"].nunique()),
        "unique_source_recordings_by_collection": source_sizes,
        "class_count": int(good["label"].nunique()),
        "duration_seconds": {key: float(value) for key, value in duration_stats.items()},
        "sample_rates": {
            str(key): int(value)
            for key, value in good["sr"].value_counts().to_dict().items()
        },
        "channels": {
            str(key): int(value)
            for key, value in good["channels"].value_counts().to_dict().items()
        },
        "class_imbalance_max_to_min": (
            float(label_counts.max() / label_counts.min()) if len(label_counts) else None
        ),
        "top_classes": class_counts.head(20).to_dict(orient="records"),
        "unreadable_examples": bad[["rel", "err"]].head(20).to_dict(orient="records"),
        "source_grouping": (
            "Dataset collection plus filename stem after removing a trailing "
            "numeric crop index; all crops from a source stay in one split."
        ),
    }
    if ENDANGERED_LABELS:
        report["endangered_label_clip_count"] = count_endangered_crops(
            good, ENDANGERED_LABELS
        )
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Audio files: {len(audio):,}; readable: {len(good):,}; unreadable: {len(bad):,}")
    print(f"Collections: {report['collections']}")
    print(f"Species-code classes: {report['class_count']}")
    print(f"Unique source groups: {report['unique_source_recordings']:,}")
    print(f"Duration summary: {duration_stats}")
    print("\nMost frequent labels:")
    print(class_counts[["label", "common_name", "clip_count", "recording_count"]].head(20).to_string(index=False))
    print(f"\nEDA outputs: {output_dir.resolve()}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Dataset root; defaults to KaggleHub's cached/downloaded dataset.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-read every audio header instead of reusing an existing EDA index.",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve() if args.data_dir else download()
    build_report(
        data_dir, args.output_dir.expanduser().resolve(), refresh=args.refresh
    )


if __name__ == "__main__":
    main()
