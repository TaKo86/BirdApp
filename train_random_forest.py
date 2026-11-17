"""train a Random Forest from the MFCC features in spectrogram_index.csv."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def read_index(path: Path) -> pd.DataFrame:
    """read the features and check that recordings stay in one split."""
    frame = pd.read_csv(path, dtype={"label": str, "group": str, "split": str, "spec_path": str})
    required = {"label", "group", "split", "spec_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing index columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("The spectrogram index is empty.")
    for column in required:
        if frame[column].isna().any() or frame[column].str.strip().eq("").any():
            raise ValueError(f"The {column} column contains empty values.")
    if not set(frame["split"]).issubset({"train", "val", "test"}):
        raise ValueError("Splits must be train, val or test.")
    if frame["spec_path"].duplicated().any():
        raise ValueError("Duplicate spectrogram paths found; check the preprocessing output.")
    if frame.groupby("group")["split"].nunique().gt(1).any():
        raise ValueError("Recording groups overlap between splits. Fix this before training.")
    return frame


def class_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """show window counts and independent recording counts for each class."""
    return frame.groupby(["label", "split"]).agg(
        windows=("spec_path", "size"), recordings=("group", "nunique")
    ).unstack("split", fill_value=0).sort_index()


def select_classes(frame: pd.DataFrame, labels: list[str] | None) -> pd.DataFrame:
    """keep the agreed classes without changing any split assignments."""
    if labels:
        unknown = set(labels) - set(frame["label"])
        if unknown:
            raise ValueError(f"Unknown classes: {', '.join(sorted(unknown))}. Run --inspect to see available labels.")
        frame = frame[frame["label"].isin(labels)].copy()
    classes = set(frame["label"])
    if len(classes) < 2:
        raise ValueError("Select at least two classes for classification.")
    for split in ("train", "val", "test"):
        missing = classes - set(frame.loc[frame["split"] == split, "label"])
        if missing:
            raise ValueError(f"Classes missing from {split}: {', '.join(sorted(missing))}")
    return frame


def feature_columns(frame: pd.DataFrame, n_mfcc: int) -> list[str]:
    """use the same numeric feature order as make_spectrograms.py."""
    if n_mfcc < 1:
        raise ValueError("N_MFCC in the preprocessing config must be positive.")
    # keep all means first, then all standard deviations; don't sort names alphabetically
    columns = [f"mfcc_mean_{i}" for i in range(n_mfcc)] + [f"mfcc_std_{i}" for i in range(n_mfcc)]
    actual = {column for column in frame if column.startswith("mfcc_")}
    if actual != set(columns):
        raise ValueError("MFCC columns do not match N_MFCC in the preprocessing config.")
    try:
        values = frame[columns].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("MFCC features must be numeric.") from exc
    if not np.isfinite(values).all():
        raise ValueError("MFCC features contain missing or infinite values.")
    return columns


def evaluate(model: Any, frame: pd.DataFrame, columns: list[str], labels: list[str]) -> dict[str, Any]:
    """measure performance on one split using the saved feature order."""
    predicted = model.predict(frame[columns])
    return {
        "windows": len(frame),
        "recordings": int(frame["group"].nunique()),
        "accuracy": float(accuracy_score(frame["label"], predicted)),
        "macro_f1": float(f1_score(frame["label"], predicted, labels=labels, average="macro", zero_division=0)),
        "classification_report": classification_report(
            frame["label"], predicted, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(frame["label"], predicted, labels=labels).tolist(),
    }


def train(
    frame: pd.DataFrame, columns: list[str], trees: int = 200, jobs: int = 4
) -> tuple[RandomForestClassifier, DummyClassifier, float]:
    """fit the forest and a majority-class baseline on training rows only."""
    training = frame[frame["split"] == "train"]
    model = RandomForestClassifier(
        n_estimators=trees, class_weight="balanced", random_state=42, n_jobs=jobs
    )
    started = time.perf_counter()
    model.fit(training[columns], training["label"])
    elapsed = time.perf_counter() - started
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(training[columns], training["label"])
    return model, baseline, elapsed


def main() -> None:
    """inspect available classes or train and save one baseline run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("output/spectrogram_index.csv"))
    parser.add_argument("--config", type=Path, help="defaults to spectrogram_config.json beside the index")
    parser.add_argument("--inspect", action="store_true", help="print class/split counts without training")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--classes", nargs="+", help="eBird labels agreed with the group")
    scope.add_argument("--all-classes", action="store_true", help="explicitly train on every class in the index")
    parser.add_argument("--out", type=Path, default=Path("output/random_forest"), help="a new directory for this run")
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--evaluate-test", action="store_true", help="also evaluate test data; use only after choices are frozen")
    args = parser.parse_args()
    try:
        frame = read_index(args.index)
        if args.inspect:
            print(class_summary(frame).to_string())
            print("\nChoose classes with enough independent recordings in every split.")
            return
        if not args.classes and not args.all_classes:
            raise ValueError("Choose --classes LABEL ... or --all-classes. Use --inspect first.")
        if args.trees < 1 or args.jobs < 1:
            raise ValueError("--trees and --jobs must both be positive.")
        if args.out.exists():
            raise ValueError(f"{args.out} already exists. Choose a new --out directory to keep earlier runs.")
        config_path = args.config or args.index.parent / "spectrogram_config.json"
        preprocessing = json.loads(config_path.read_text(encoding="utf-8"))
        if "N_MFCC" not in preprocessing:
            raise ValueError("The preprocessing config is missing N_MFCC.")
        frame = select_classes(frame, args.classes)
        columns = feature_columns(frame, int(preprocessing["N_MFCC"]))
        print(class_summary(frame).to_string(), flush=True)
        print(f"\nTraining {args.trees} trees using {len(columns)} MFCC features...", flush=True)
        model, baseline, elapsed = train(frame, columns, args.trees, args.jobs)
        labels = model.classes_.tolist()
        # test rows are kept out of model fitting and aren't scored unless requested
        scores = {}
        for split in (["val", "test"] if args.evaluate_test else ["val"]):
            subset = frame[frame["split"] == split]
            scores[split] = {
                "random_forest": evaluate(model, subset, columns, labels),
                "majority_baseline": evaluate(baseline, subset, columns, labels),
            }
        args.out.mkdir(parents=True, exist_ok=False)
        joblib.dump({"model": model, "feature_columns": columns, "classes": labels,
                     "preprocessing": preprocessing}, args.out / "model.joblib")
        metadata = {
            "classes": labels, "feature_columns": columns, "preprocessing": preprocessing,
            "model_parameters": model.get_params(), "fit_seconds": elapsed,
            "index_sha256": hashlib.sha256(args.index.read_bytes()).hexdigest(),
            "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__, "test_evaluated": args.evaluate_test,
            "evaluation_unit": "five-second window; splits grouped by original recording",
        }
        (args.out / "run_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (args.out / "metrics.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
        frame.to_csv(args.out / "selected_index.csv", index=False)
        pd.DataFrame({"feature": columns, "importance": model.feature_importances_}).sort_values(
            "importance", ascending=False
        ).to_csv(args.out / "feature_importances.csv", index=False)
        for split, results in scores.items():
            result = results["random_forest"]
            pd.DataFrame(result["confusion_matrix"], index=labels, columns=labels).rename_axis(
                "actual_label"
            ).to_csv(args.out / f"{split}_confusion_matrix.csv")
            print(f"{split}: accuracy={result['accuracy']:.3f}, macro F1={result['macro_f1']:.3f}")
        print(f"Saved model and results to {args.out}. Training took {elapsed:.1f} seconds.")
        if not args.evaluate_test:
            print("Test scores were not calculated. Freeze model choices before using --evaluate-test.")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
