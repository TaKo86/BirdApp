"""prepare all eight-species audio once for the SVM, RF and CNN comparison."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from training_common import CLASSES, SETTINGS, assign_splits, file_hash, versions, write_json


def collect_files(root: Path) -> pd.DataFrame:
    """read primary labels from all three collections without sampling or capping."""
    frames = []
    for collection, metadata in (("DOC_001_Tier1", "001_metadata.csv"),
                                 ("DOC_002_DuncanBayParinga", "002_metadata.csv"),
                                 ("DOC_003_XenoCanto", "003_metadata.csv")):
        base = root / collection / collection
        frame = pd.read_csv(base / metadata, usecols=["filename", "primary_label"])
        frame = frame[frame.primary_label.isin(CLASSES)].copy()
        frame["label"] = frame.primary_label
        relative = [Path(collection) / collection / "train_audio" / str(name) for name in frame.filename]
        if any(path.is_absolute() or ".." in path.parts for path in relative):
            raise ValueError("Unsafe filename in dataset metadata.")
        frame["relative_path"] = [path.as_posix() for path in relative]
        frame["audio_path"] = [str(root / path) for path in relative]
        stems = [re.sub(r"_\d+$", "", Path(str(name)).stem) for name in frame.filename]
        frame["group"] = [f"{collection}/{stem}".casefold() for stem in stems]
        frames.append(frame[["relative_path", "audio_path", "label", "group"]])
    result = pd.concat(frames, ignore_index=True).sort_values("relative_path").reset_index(drop=True)
    if result.relative_path.duplicated().any():
        raise ValueError("Duplicate file paths in metadata; resolve these before preparation.")
    if set(result.label) != set(CLASSES):
        raise ValueError("The metadata does not contain all eight species.")
    result["cache_row"] = np.arange(len(result))
    return result


def fixed_length(audio: np.ndarray) -> np.ndarray:
    """centre crop long clips and centre pad short ones, as in Warner's experiment."""
    target = round(SETTINGS["sample_rate"] * SETTINGS["clip_seconds"])
    if len(audio) >= target:
        start = (len(audio) - target) // 2
        return audio[start:start + target].astype(np.float32)
    left = (target - len(audio)) // 2
    return np.pad(audio, (left, target - len(audio) - left)).astype(np.float32)


def extract_audio(path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """derive both feature types from exactly the same mono audio segment."""
    audio, original_rate = sf.read(path, dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio is empty or contains non-finite samples.")
    duration = len(audio) / original_rate
    mono = audio.mean(axis=1)
    mono = librosa.resample(mono, orig_sr=original_rate, target_sr=SETTINGS["sample_rate"], res_type="soxr_hq")
    segment = fixed_length(mono)
    power = librosa.feature.melspectrogram(y=segment, sr=SETTINGS["sample_rate"],
        n_fft=1024, hop_length=256, n_mels=128, fmin=50, fmax=11025,
        htk=True, norm=None, power=2.0, window="hann", center=True, pad_mode="reflect")
    log_mel = np.maximum(10 * np.log10(np.maximum(power, 1e-10)), -80)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=20, dct_type=2, norm="ortho", lifter=0)
    vector = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)
    normalized = ((log_mel - log_mel.mean()) / max(float(log_mel.std()), 1e-6)).astype(np.float16)
    if not np.isfinite(vector).all() or not np.isfinite(normalized).all():
        raise ValueError("Non-finite features after extraction.")
    return vector, normalized, {"duration_seconds": duration, "original_sample_rate": original_rate,
        "start_seconds": max(0.0, (len(mono) - len(segment)) // 2 / SETTINGS["sample_rate"]),
        "segment_seconds": SETTINGS["clip_seconds"]}


def process_row(item: tuple[int, str]) -> tuple:
    """return failures to the parent so all models exclude the same files."""
    row_id, path = item
    try:
        vector, mel, details = extract_audio(path)
        return row_id, vector, mel, details, None
    except (OSError, RuntimeError, ValueError) as exc:
        return row_id, None, None, {}, str(exc)


def prepare(root: Path, out: Path, workers: int, resume: bool = False) -> None:
    """cache features on disk and write a shared manifest only when complete."""
    candidates = collect_files(root.resolve())
    settings = {"settings": SETTINGS, "classes": CLASSES, "versions": versions()}
    frame_count = 1 + round(SETTINGS["sample_rate"] * SETTINGS["clip_seconds"]) // 256
    if out.exists():
        if not resume:
            raise FileExistsError(f"{out} exists. Use --resume or choose a new --out directory.")
        if (out / "dataset.json").exists():
            raise ValueError("This dataset is already complete; point the model scripts at it.")
        previous = json.loads((out / "preparation.json").read_text(encoding="utf-8"))
        if previous != settings:
            raise ValueError("Settings or library versions changed. Use a new output folder.")
        try:
            pd.testing.assert_frame_equal(candidates, pd.read_csv(out / "candidates.csv"), check_dtype=False)
        except AssertionError as exc:
            raise ValueError("The input file list changed. Use a new output folder.") from exc
        mfcc = np.load(out / "mfcc.npy", mmap_mode="r+")
        mel = np.load(out / "mel.npy", mmap_mode="r+")
        state = np.load(out / "state.npy", mmap_mode="r+")
        details = json.loads((out / "progress.json").read_text(encoding="utf-8"))
    else:
        required_bytes = len(candidates) * (128 * frame_count * 2 + 40 * 4 + 1)
        out.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(out.parent).free < required_bytes + 512 * 1024**2:
            raise OSError(f"Need roughly {required_bytes / 1024**3:.1f} GiB plus 0.5 GiB free for the feature cache.")
        out.mkdir()
        candidates.to_csv(out / "candidates.csv", index=False)
        write_json(out / "preparation.json", settings)
        mfcc = np.lib.format.open_memmap(out / "mfcc.npy", mode="w+", dtype="float32", shape=(len(candidates), 40))
        mel = np.lib.format.open_memmap(out / "mel.npy", mode="w+", dtype="float16", shape=(len(candidates), 128, frame_count))
        state = np.lib.format.open_memmap(out / "state.npy", mode="w+", dtype="uint8", shape=(len(candidates),))
        state[:] = 0
        state.flush()
        details = {}
        write_json(out / "progress.json", details)

    pending = candidates.loc[state == 0]
    print(f"{len(candidates):,} files; {len(pending):,} left to process. No class/source cap.", flush=True)
    jobs = [(int(row.cache_row), row.audio_path) for row in pending.itertuples()]
    completed = []

    def checkpoint() -> None:
        # features and metadata reach disk before their rows are marked complete
        mfcc.flush()
        mel.flush()
        write_json(out / "progress.json", details)
        for row_id, status in completed:
            state[row_id] = status
        state.flush()
        completed.clear()

    with Pool(workers) as pool:
        for row_id, vector, spectrum, info, error in tqdm(pool.imap(process_row, jobs, chunksize=8), total=len(jobs), desc="audio features"):
            details[str(row_id)] = {**info, "error": error}
            if error is None:
                mfcc[row_id], mel[row_id] = vector, spectrum
            completed.append((row_id, 2 if error else 1))
            if len(completed) >= 512:
                checkpoint()
    checkpoint()
    records = pd.DataFrame([{"cache_row": int(key), **value} for key, value in details.items()])
    failures = records[records.error.notna()].merge(candidates, on="cache_row")
    failures.to_csv(out / "decode_errors.csv", index=False)
    good = candidates.loc[state == 1].merge(records.drop(columns="error"), on="cache_row", validate="one_to_one")
    print("Assigning source-grouped splits...", flush=True)
    good = assign_splits(good)
    good.to_csv(out / "manifest.csv", index=False)
    counts = good.groupby(["label", "split"]).agg(clips=("cache_row", "size"), source_groups=("group", "nunique"))
    counts.to_csv(out / "class_counts.csv")
    write_json(out / "dataset.json", {**settings, "candidate_count": len(candidates),
        "usable_count": len(good), "failed_count": len(failures),
        "manifest_sha256": file_hash(out / "manifest.csv"),
        "preparation_code_sha256": file_hash(Path(__file__)),
        "mfcc_feature_order": [f"mfcc_{stat}_{i}" for stat in ("mean", "std") for i in range(20)]})
    print(counts.to_string())
    print(f"Ready: {out}. Skipped {len(failures)} files; see decode_errors.csv.")


def main() -> None:
    """prepare the shared dataset once before running any of the models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("output/eight_species_data"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Workers must be positive.")
    prepare(args.data_root, args.out, args.workers, args.resume)


if __name__ == "__main__":
    main()
