"""shared data checks and reports for the three bird model baselines."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = {
    "gryger1": "Grey Warbler", "kea1": "Kea", "morepo2": "Morepork",
    "nezbel1": "Bellbird", "nezfan1": "Fantail", "riflem1": "Rifleman",
    "tomtit1": "Tomtit", "tui1": "Tui",
}
SETTINGS = {
    "schema_version": 1, "sample_rate": 22050, "clip_seconds": 4.0,
    "n_fft": 1024, "hop_length": 256, "n_mels": 128, "n_mfcc": 20,
    "fmin": 50, "fmax": 11025, "seed": 42,
    "mel_scale": "htk", "mel_norm": None, "window": "hann",
    "center": True, "pad_mode": "reflect", "power": 2.0,
    "resample_type": "soxr_hq", "mfcc_dct_type": 2, "mfcc_norm": "ortho",
    "log_mel": "10*log10(max(power,1e-10)), floor -80 dB",
    "cnn_normalization": "per-clip mean and population standard deviation",
    "audio_selection": "all readable non-empty primary-labelled files; no per-class/source cap",
    "segment": "one centre crop per file; short files padded equally on both sides",
    "split": "StratifiedGroupKFold: fold 0 test, fold 1 val, folds 2-4 train",
}
FEATURES = [f"mfcc_{stat}_{i}" for stat in ("mean", "std") for i in range(20)]


def write_json(path: Path, value: Any) -> None:
    """write metadata without leaving half a JSON file after interruption."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    """hash a file without loading the whole thing into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def versions() -> dict[str, str]:
    """record the libraries that affect feature extraction and training."""
    result = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scikit-learn", "librosa", "soundfile", "torch"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def assign_splits(frame: pd.DataFrame) -> pd.DataFrame:
    """split by source recording once, then save the assignments for every model."""
    frame = frame.copy()
    if set(frame.label) != set(CLASSES):
        raise ValueError("All eight species must have readable audio.")
    if frame.groupby("label").group.nunique().min() < 5:
        raise ValueError("Each species needs at least five source groups for five folds.")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SETTINGS["seed"])
    targets = frame.label.map({label: i for i, label in enumerate(CLASSES)})
    frame["split"] = "train"
    for fold, (_, positions) in enumerate(splitter.split(np.zeros(len(frame)), targets, frame.group)):
        frame.loc[frame.index[positions], "split"] = "test" if fold == 0 else "val" if fold == 1 else "train"
    validate_manifest(frame)
    return frame


def validate_manifest(frame: pd.DataFrame) -> None:
    """reject missing classes, duplicate examples and source leakage."""
    required = ["relative_path", "label", "group", "split", "cache_row"]
    if not set(required).issubset(frame) or frame.empty or frame[required].isna().any().any():
        raise ValueError("The manifest is empty or missing required values.")
    if frame.relative_path.duplicated().any() or frame.cache_row.duplicated().any():
        raise ValueError("Duplicate audio paths or cache rows in the manifest.")
    if set(frame.split) != {"train", "val", "test"}:
        raise ValueError("Expected train, val and test splits.")
    if frame.groupby("group").split.nunique().gt(1).any():
        raise ValueError("Source recordings overlap between splits.")
    for split in ("train", "val", "test"):
        if set(frame.loc[frame.split == split, "label"]) != set(CLASSES):
            raise ValueError(f"The {split} split does not contain all eight classes.")


def load_dataset(folder: Path) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray]:
    """open the prepared caches; model scripts never create their own splits."""
    config = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    if config["settings"] != SETTINGS or config["classes"] != CLASSES:
        raise ValueError("Dataset settings differ from this code. Use its matching preparation version.")
    if file_hash(folder / "manifest.csv") != config["manifest_sha256"]:
        raise ValueError("The shared manifest has changed since preparation.")
    frame = pd.read_csv(folder / "manifest.csv")
    validate_manifest(frame)
    mfcc = np.load(folder / "mfcc.npy", mmap_mode="r", allow_pickle=False)
    mel = np.load(folder / "mel.npy", mmap_mode="r", allow_pickle=False)
    expected_frames = 1 + round(SETTINGS["sample_rate"] * SETTINGS["clip_seconds"]) // SETTINGS["hop_length"]
    if mfcc.shape != (config["candidate_count"], 40) or mel.shape != (len(mfcc), 128, expected_frames):
        raise ValueError("Feature cache dimensions do not match the dataset metadata.")
    ids = frame.cache_row.to_numpy()
    if not np.issubdtype(ids.dtype, np.integer) or ids.min() < 0 or ids.max() >= len(mfcc):
        raise ValueError("Invalid cache row in the manifest.")
    if not np.isfinite(mfcc[ids]).all():
        raise ValueError("MFCC features contain non-finite values.")
    return frame, config, mfcc, mel


def training_parser(model: str) -> argparse.ArgumentParser:
    """give each trainer the same dataset and evaluation options."""
    parser = argparse.ArgumentParser(description=f"Train the {model} eight-species baseline.")
    parser.add_argument("--data", type=Path, default=Path("output/eight_species_data"))
    parser.add_argument("--out", type=Path, default=Path(f"output/{model}_full"))
    parser.add_argument("--evaluate-test", action="store_true", help="score test only after settings are fixed")
    return parser


def start_run(args: argparse.Namespace, config: dict, model: str) -> None:
    """keep each experiment in its own folder with its dataset fingerprint."""
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "run_info.json", {
        "model": model, "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "dataset": config, "versions": versions(), "seed": SETTINGS["seed"],
        "evaluation_unit": "one centre-cropped or padded four-second segment per file",
        "training_code_sha256": {name: file_hash(Path(__file__).parent / name)
                                 for name in ("training_common.py", f"train_{model}.py")},
    })


def save_scores(out: Path, split: str, rows: pd.DataFrame, predicted: np.ndarray,
                scores: np.ndarray, majority_label: str, score_kind: str) -> dict:
    """save identical metrics and per-file predictions for every model."""
    labels = list(CLASSES)
    actual = rows.label.to_numpy()
    if scores.shape != (len(rows), len(labels)) or not np.isfinite(scores).all():
        raise ValueError("Invalid model scores; expected one column per class in the saved order.")
    report = classification_report(actual, predicted, labels=labels, output_dict=True, zero_division=0)
    top_three = np.asarray(labels)[np.argsort(-scores, axis=1, kind="stable")[:, :3]]
    matrix = confusion_matrix(actual, predicted, labels=labels)
    baseline = np.full(len(rows), majority_label)
    result = {
        "samples": len(rows), "source_groups": int(rows.group.nunique()),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": report["macro avg"]["recall"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "top_3_accuracy": float((top_three == actual[:, None]).any(axis=1).mean()),
        "majority_baseline": {"label": majority_label,
            "accuracy": float(accuracy_score(actual, baseline)),
            "macro_f1": float(f1_score(actual, baseline, labels=labels, average="macro", zero_division=0))},
        "uniform_guess_accuracy_reference": 1 / len(labels),
        "classification_report": report, "class_order": labels,
        "confusion_matrix": matrix.tolist(), "score_kind": score_kind,
    }
    pd.DataFrame(matrix, index=labels, columns=labels).rename_axis("actual_label").to_csv(out / f"{split}_confusion_matrix.csv")
    per_class = pd.DataFrame.from_dict({label: report[label] for label in labels}, orient="index")
    per_class.insert(0, "species", [CLASSES[label] for label in labels])
    per_class.rename_axis("label").to_csv(out / f"{split}_per_class.csv")
    predictions = rows[["relative_path", "label", "group", "split", "cache_row"]].copy()
    predictions["predicted_label"] = predicted
    for i, label in enumerate(labels):
        predictions[f"score_{label}"] = scores[:, i]
    predictions.to_csv(out / f"{split}_predictions.csv", index=False)
    print(f"{split}: accuracy={result['accuracy']:.4f}, macro F1={result['macro_f1']:.4f}, top-3={result['top_3_accuracy']:.4f}", flush=True)
    return result


def run_sklearn(args: argparse.Namespace, estimator: Any, model_name: str) -> None:
    """fit on shared training MFCCs and evaluate the requested held-out splits."""
    frame, config, mfcc, _ = load_dataset(args.data)
    start_run(args, config, model_name)
    train = frame[frame.split == "train"]
    started = time.perf_counter()
    estimator.fit(mfcc[train.cache_row], train.label)
    elapsed = time.perf_counter() - started
    joblib.dump({"model": estimator, "classes": list(CLASSES), "feature_columns": FEATURES,
                 "dataset": config}, args.out / "model.joblib")
    majority = train.label.value_counts().idxmax()
    metrics = {"fit_seconds": elapsed, "test_evaluated": args.evaluate_test}
    for split in (["val", "test"] if args.evaluate_test else ["val"]):
        rows = frame[frame.split == split]
        x = mfcc[rows.cache_row]
        if hasattr(estimator, "predict_proba"):
            scores, kind = estimator.predict_proba(x), "probability"
        else:
            scores, kind = estimator.decision_function(x), "decision_function (not probability)"
        scores = scores[:, [list(estimator.classes_).index(label) for label in CLASSES]]
        metrics[split] = save_scores(args.out, split, rows, estimator.predict(x), scores, majority, kind)
    write_json(args.out / "metrics.json", metrics)
    print(f"Saved {model_name} to {args.out}. Training took {elapsed:.1f} seconds.")
