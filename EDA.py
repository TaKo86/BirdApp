"""Explore CACOPHONY recordings and their manually tagged bird segments."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audio_utils import (
    SAMPLE_RATE,
    crop_audio,
    decode_audio,
    iter_audio_files,
    iter_manual_bird_segments,
    log_mel_spectrogram,
    read_annotation,
)

DEFAULT_DATA_DIR = Path(r"E:\CACOPHONY")
DEFAULT_OUTPUT_DIRNAME = "eda_output"


def build_report(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory: list[dict[str, Any]] = []
    bird_segments: list[dict[str, Any]] = []
    all_tag_counts: Counter[str] = Counter()
    annotation_errors: list[dict[str, str]] = []
    durations: list[float] = []

    audio_files = list(iter_audio_files(data_dir))
    for audio_path in audio_files:
        relative_path = audio_path.relative_to(data_dir).as_posix()
        sidecar = audio_path.with_suffix(".txt")
        metadata, error = read_annotation(sidecar) if sidecar.is_file() else (None, None)
        status = "missing" if not sidecar.is_file() else "valid" if error is None else "invalid"
        row: dict[str, Any] = {
            "relative_path": relative_path,
            "extension": audio_path.suffix.lower(),
            "size_mb": audio_path.stat().st_size / (1024 * 1024),
            "annotation_status": status,
            "recording_id": None,
            "duration_s": None,
        }
        inventory.append(row)

        if status == "missing":
            annotation_errors.append({"file": relative_path, "error": "missing .txt sidecar"})
            continue
        if error is not None or metadata is None:
            annotation_errors.append({"file": relative_path, "error": error or "invalid annotation"})
            continue

        row["recording_id"] = metadata.get("id")
        try:
            duration_s = float(metadata["duration"])
        except (KeyError, TypeError, ValueError):
            duration_s = None
        row["duration_s"] = duration_s
        if duration_s is not None and np.isfinite(duration_s) and duration_s > 0:
            durations.append(duration_s)

        for track in metadata.get("Tracks") or []:
            if not isinstance(track, dict):
                continue
            for tag in track.get("tags") or []:
                if (
                    isinstance(tag, dict)
                    and tag.get("automatic") is not True
                    and str(tag.get("what") or "").strip()
                ):
                    all_tag_counts[str(tag["what"]).strip().casefold()] += 1

        for segment in iter_manual_bird_segments(metadata, audio_path):
            bird_segments.append(
                {
                    "relative_path": relative_path,
                    "recording_id": segment["recording_id"],
                    "track_index": segment["track_index"],
                    "start_s": segment["start"],
                    "end_s": segment["end"],
                    "duration_s": segment["end"] - segment["start"],
                    "label": segment["label"],
                    "tag_path": segment["tag_path"],
                }
            )

    inventory_frame = pd.DataFrame(inventory)
    segments_frame = pd.DataFrame(
        bird_segments,
        columns=[
            "relative_path",
            "recording_id",
            "track_index",
            "start_s",
            "end_s",
            "duration_s",
            "label",
            "tag_path",
        ],
    )
    inventory_frame.to_csv(output_dir / "audio_inventory.csv", index=False)
    segments_frame.to_csv(output_dir / "bird_segments.csv", index=False)
    pd.DataFrame(
        all_tag_counts.most_common(), columns=["label", "manual_tag_count"]
    ).to_csv(output_dir / "manual_tag_counts.csv", index=False)

    if not segments_frame.empty:
        class_counts = (
            segments_frame.groupby("label")
            .agg(
                segment_count=("label", "size"),
                recording_count=("recording_id", "nunique"),
                median_segment_s=("duration_s", "median"),
            )
            .sort_values("segment_count", ascending=False)
            .reset_index()
        )
    else:
        class_counts = pd.DataFrame(
            columns=["label", "segment_count", "recording_count", "median_segment_s"]
        )
    class_counts.to_csv(output_dir / "bird_class_counts.csv", index=False)

    status_counts = inventory_frame["annotation_status"].value_counts().to_dict()
    if durations:
        duration_stats = {
            "min_s": float(np.min(durations)),
            "median_s": float(np.median(durations)),
            "mean_s": float(np.mean(durations)),
            "max_s": float(np.max(durations)),
        }
    else:
        duration_stats = {}

    if class_counts.empty:
        imbalance = None
        classes = 0
    else:
        counts = class_counts["segment_count"]
        imbalance = float(counts.max() / counts.min())
        classes = len(class_counts)

    report: dict[str, Any] = {
        "data_dir": str(data_dir),
        "audio_files": len(audio_files),
        "audio_extensions": inventory_frame["extension"].value_counts().to_dict(),
        "annotation_status": status_counts,
        "valid_recording_duration_s": duration_stats,
        "manual_bird_species_classes": classes,
        "manual_bird_segments": len(segments_frame),
        "species_segment_imbalance_max_to_min": imbalance,
        "top_bird_labels": class_counts.head(15).to_dict(orient="records"),
        "annotation_error_examples": annotation_errors[:20],
        "split_recommendation": (
            "Split by recording_id, not segment, to prevent adjacent segments from "
            "one recording leaking between train and evaluation sets."
        ),
        "labeling_note": (
            "00-ff are storage-shard directories. Labels come from non-automatic "
            "all.bird.* track tags in the JSON sidecars."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not class_counts.empty:
        ordered_counts = class_counts.sort_values("segment_count")
        figure_height = max(4.0, len(ordered_counts) * 0.24)
        _, axis = plt.subplots(figsize=(10, figure_height))
        axis.barh(ordered_counts["label"], ordered_counts["segment_count"])
        axis.set_xlabel("Manually tagged bird segments")
        axis.set_title("Bird label distribution")
        plt.tight_layout()
        plt.savefig(output_dir / "bird_class_balance.png", dpi=130)
        plt.close()

        top_labels = class_counts["label"].head(9).tolist()
        by_label: dict[str, dict[str, Any]] = {}
        for segment in bird_segments:
            label = segment["label"]
            if label in top_labels and label not in by_label:
                by_label[label] = segment
        if by_label:
            columns = 3
            rows = (len(by_label) + columns - 1) // columns
            figure, axes = plt.subplots(
                rows, columns, figsize=(5 * columns, 3 * rows), squeeze=False
            )
            for axis, (label, segment) in zip(axes.flat, by_label.items()):
                path = data_dir / Path(segment["relative_path"])
                audio = decode_audio(path, SAMPLE_RATE)
                clip = crop_audio(audio, segment["start_s"], segment["end_s"])
                feature = log_mel_spectrogram(clip)[0]
                axis.imshow(
                    feature,
                    aspect="auto",
                    origin="lower",
                    extent=(0, clip.size / SAMPLE_RATE, 0, feature.shape[0]),
                    cmap="magma",
                )
                axis.set_title(label[:35], fontsize=9)
                axis.set_xlabel("Time (s)")
                axis.set_ylabel("Mel bin")
            for axis in axes.flat[len(by_label) :]:
                axis.axis("off")
            figure.suptitle("Log-mel samples from manually tagged segments")
            figure.tight_layout()
            figure.savefig(output_dir / "bird_spectrogram_samples.png", dpi=120)
            plt.close(figure)

    print(f"Dataset: {data_dir}")
    print(f"Audio files: {len(audio_files)} ({report['audio_extensions']})")
    print(f"Sidecars: {status_counts}")
    print(
        "Valid recording duration (s): "
        + (", ".join(f"{key}={value:.1f}" for key, value in duration_stats.items()) or "unavailable")
    )
    print(f"Manually tagged bird segments: {len(segments_frame)}")
    print(f"Bird labels: {classes}; max/min segment imbalance: {imbalance}")
    if not class_counts.empty:
        print("\nMost frequent bird labels:")
        print(class_counts[["label", "segment_count", "recording_count"]].head(15).to_string(index=False))
    print(f"\nEDA outputs: {output_dir}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("BIRD_DATA_DIR", str(DEFAULT_DATA_DIR))),
        help="Dataset root containing audio and JSON .txt sidecars.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Report directory (default: <data-dir>/{DEFAULT_OUTPUT_DIRNAME}).",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else data_dir / DEFAULT_OUTPUT_DIRNAME
    )
    build_report(data_dir, output_dir)


if __name__ == "__main__":
    main()
