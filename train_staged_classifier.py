"""Train and evaluate an acoustic-trait-routed bird classifier."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import joblib
import kagglehub
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from audio_utils import (
    SAMPLE_RATE,
    decode_audio,
    fixed_length_crop,
    load_species_names,
    source_recording_id,
)
from train_cnn import DATASET, DEFAULT_INDEX, NON_BIRD_LABELS, SEED, SPLITS

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "staged_model_outputs"
FEATURE_NAMES = (
    "centroid_median_hz",
    "centroid_std_hz",
    "bandwidth_median_hz",
    "bandwidth_std_hz",
    "rolloff50_median_hz",
    "rolloff85_median_hz",
    "flatness_median",
    "flatness_std",
    "peak_frequency_median_hz",
    "peak_frequency_std_hz",
    "rms_median",
    "rms_std",
    "zero_crossing_rate_median",
    "spectral_entropy_median",
    "spectral_flux_median",
)
N_ESTIMATORS = 100
N_JOBS = 1


def extract_acoustic_traits(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Summarize objective spectral and temporal traits from one audio clip."""
    clip = fixed_length_crop(audio, sample_rate=sample_rate)
    n_fft = 1024
    hop_length = 256
    padded = np.pad(clip, (n_fft // 2, n_fft // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop_length]
    windowed = frames * np.hanning(n_fft).astype(np.float32)
    magnitude = np.abs(np.fft.rfft(windowed, axis=1)).astype(np.float32)
    power = np.square(magnitude)
    frequencies = np.fft.rfftfreq(n_fft, d=1 / sample_rate).astype(np.float32)
    power_sum = np.maximum(power.sum(axis=1), 1e-12)
    magnitude_sum = np.maximum(magnitude.sum(axis=1), 1e-12)
    magnitude_probabilities = magnitude / magnitude_sum[:, None]
    power_probabilities = power / power_sum[:, None]

    centroid = (magnitude_probabilities * frequencies).sum(axis=1)
    bandwidth = np.sqrt(
        (magnitude_probabilities * np.square(frequencies - centroid[:, None])).sum(axis=1)
    )
    cumulative_power = np.cumsum(power, axis=1)
    rolloff_50 = frequencies[
        np.argmax(cumulative_power >= power_sum[:, None] * 0.50, axis=1)
    ]
    rolloff_85 = frequencies[
        np.argmax(cumulative_power >= power_sum[:, None] * 0.85, axis=1)
    ]
    flatness = np.exp(np.mean(np.log(np.maximum(power, 1e-12)), axis=1)) / np.maximum(
        power.mean(axis=1), 1e-12
    )
    peak_frequency = frequencies[np.argmax(power, axis=1)]
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    zero_crossings = np.mean(np.diff(np.signbit(frames), axis=1), axis=1)
    entropy = -np.sum(
        power_probabilities * np.log(np.maximum(power_probabilities, 1e-12)), axis=1
    ) / np.log(power.shape[1])
    normalized_magnitude = magnitude_probabilities
    spectral_flux = np.sqrt(
        np.sum(np.square(np.diff(normalized_magnitude, axis=0)), axis=1)
    )

    values = np.asarray(
        [
            np.median(centroid),
            np.std(centroid),
            np.median(bandwidth),
            np.std(bandwidth),
            np.median(rolloff_50),
            np.median(rolloff_85),
            np.median(flatness),
            np.std(flatness),
            np.median(peak_frequency),
            np.std(peak_frequency),
            np.median(rms),
            np.std(rms),
            np.median(zero_crossings),
            np.median(entropy),
            np.median(spectral_flux),
        ],
        dtype=np.float32,
    )
    if not np.isfinite(values).all():
        raise ValueError("Acoustic trait extraction produced non-finite values.")
    return values


def select_clips(
    index_path: Path,
    species_names: dict[str, dict[str, str]],
    min_clips: int,
    min_recordings: int,
    max_recordings_per_class: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[str], dict[str, dict[str, str]]]:
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing audio index {index_path}. Run EDA.py first.")
    frame = pd.read_csv(index_path)
    required = {"rel", "ok", "label"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{index_path} is missing columns {sorted(missing_columns)}.")
    frame = frame[frame["ok"].astype(bool)].copy()
    frame["label"] = frame["label"].astype(str).str.casefold()
    frame["source_id"] = frame["rel"].map(source_recording_id)
    support = frame.groupby("label").agg(
        clips=("label", "size"), recordings=("source_id", "nunique")
    )
    labels = [
        label
        for label in sorted(support.index)
        if support.at[label, "clips"] >= min_clips
        and support.at[label, "recordings"] >= min_recordings
        and label not in NON_BIRD_LABELS
        and species_names.get(label, {}).get("common_name")
        and species_names.get(label, {}).get("scientific_name")
    ]
    if len(labels) < 2:
        raise ValueError(
            "Fewer than two named bird species meet the support criteria "
            f"(min_clips={min_clips}, min_recordings={min_recordings})."
        )

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    class_names: dict[str, dict[str, str]] = {}
    eligible = frame[frame["label"].isin(labels)]
    for label in labels:
        class_names[label] = species_names[label]
        one_per_recording: list[dict[str, Any]] = []
        for _, rows in eligible[eligible["label"] == label].groupby(
            "source_id", sort=True
        ):
            one_per_recording.append(
                rng.choice(rows.to_dict(orient="records"))
            )
        rng.shuffle(one_per_recording)
        selected.extend(one_per_recording[:max_recordings_per_class])
    return selected, labels, class_names


def extract_dataset_features(
    clips: list[dict[str, Any]], dataset_root: Path, output_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    errors: list[dict[str, str]] = []
    for index, row in enumerate(clips, start=1):
        relative = Path(str(row["rel"]).replace("\\", "/"))
        audio_path = dataset_root / relative
        try:
            audio = decode_audio(audio_path)
            feature = extract_acoustic_traits(audio)
        except (
            OSError,
            RuntimeError,
            ValueError,
            sf.SoundFileRuntimeError,
            subprocess.SubprocessError,
        ) as exc:
            errors.append({"file": str(relative), "error": str(exc)[:500]})
            continue
        features.append(feature)
        labels.append(str(row["label"]))
        groups.append(source_recording_id(str(row["rel"])))
        if index % 500 == 0 or index == len(clips):
            print(f"Extracted acoustic traits for {index:,}/{len(clips):,} clips")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(errors, columns=["file", "error"]).to_csv(
        output_dir / "decode_errors.csv", index=False
    )
    if errors:
        print(f"Skipped {len(errors)} clips; see decode_errors.csv.")
    if not features:
        raise RuntimeError("No audio features were extracted.")
    return (
        np.stack(features),
        np.asarray(labels, dtype=str),
        np.asarray(groups, dtype=str),
    )


def make_group_splits(
    targets: np.ndarray, groups: np.ndarray, labels: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=SPLITS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(np.zeros(len(targets)), targets, groups))
    test_indices = folds[0][1]
    validation_indices = folds[1][1]
    train_indices = np.concatenate([fold[1] for fold in folds[2:]])
    for name, indices in (
        ("train", train_indices),
        ("validation", validation_indices),
        ("test", test_indices),
    ):
        missing = set(labels) - set(targets[indices])
        if missing:
            raise ValueError(f"The {name} split is missing species: {sorted(missing)}")
    partitions = [set(groups[indices]) for indices in (train_indices, validation_indices, test_indices)]
    if (
        partitions[0] & partitions[1]
        or partitions[0] & partitions[2]
        or partitions[1] & partitions[2]
    ):
        raise RuntimeError("Source recording leakage detected between splits.")
    return train_indices, validation_indices, test_indices


def fit_stage(
    features: np.ndarray,
    labels: np.ndarray,
    species: list[str],
    number_of_groups: int,
) -> dict[str, Any]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    class_centers = np.stack(
        [scaled[labels == label].mean(axis=0) for label in species]
    )
    clustering = KMeans(
        n_clusters=number_of_groups, random_state=SEED, n_init=20
    ).fit(class_centers)
    group_ids = sorted(np.unique(clustering.labels_).tolist())
    species_to_group = {
        label: int(group)
        for label, group in zip(species, clustering.labels_, strict=True)
    }
    group_targets = np.asarray([species_to_group[label] for label in labels])
    router = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=N_JOBS,
    ).fit(scaled, group_targets)
    experts: dict[int, RandomForestClassifier] = {}
    for group in group_ids:
        group_species = {
            label for label, species_group in species_to_group.items()
            if species_group == group
        }
        group_mask = np.isin(labels, list(group_species))
        experts[group] = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=N_JOBS,
        ).fit(scaled[group_mask], labels[group_mask])
    return {
        "scaler": scaler,
        "clustering": clustering,
        "species_to_group": species_to_group,
        "router": router,
        "experts": experts,
        "species": species,
        "number_of_groups": len(group_ids),
    }


def predict_staged_proba(
    model: dict[str, Any], features: np.ndarray, top_groups: int
) -> tuple[np.ndarray, list[str]]:
    scaled = model["scaler"].transform(features)
    router = model["router"]
    group_probabilities = router.predict_proba(scaled)
    group_labels = router.classes_
    species = model["species"]
    species_indices = {label: index for index, label in enumerate(species)}
    probabilities = np.zeros((len(features), len(species)), dtype=np.float64)
    selected_groups = min(top_groups, len(group_labels))
    for row_index, group_scores in enumerate(group_probabilities):
        candidate_indices = np.argsort(group_scores)[-selected_groups:]
        for group_index in candidate_indices:
            group = int(group_labels[group_index])
            expert = model["experts"][group]
            local_probabilities = expert.predict_proba(scaled[row_index : row_index + 1])[0]
            group_weight = group_scores[group_index]
            for label, local_probability in zip(
                expert.classes_, local_probabilities, strict=True
            ):
                probabilities[row_index, species_indices[str(label)]] += (
                    group_weight * local_probability
                )
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise RuntimeError("Staged classifier produced an empty species distribution.")
    probabilities /= row_sums
    return probabilities, species


def metric_summary(
    targets: np.ndarray, probabilities: np.ndarray, labels: list[str]
) -> dict[str, float]:
    predictions = np.asarray(labels)[probabilities.argmax(axis=1)]
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "precision_macro": float(
            precision_score(targets, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(targets, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "top_3_accuracy": float(
            top_k_accuracy_score(
                targets, probabilities, k=min(3, len(labels)), labels=labels
            )
        ),
    }


def write_split_support(
    labels: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    species: list[str],
    class_names: dict[str, dict[str, str]],
    output_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for label in species:
        row: dict[str, Any] = {
            "Label": label,
            "Species": class_names[label]["common_name"],
        }
        for split_name, indices in (
            ("Training", train_indices),
            ("Validation", validation_indices),
            ("Test", test_indices),
        ):
            mask = labels[indices] == label
            row[f"{split_name} recordings"] = int(len(np.unique(groups[indices][mask])))
            row[f"{split_name} windows"] = int(mask.sum())
        rows.append(row)
    columns = [
        "Label",
        "Species",
        "Training recordings",
        "Validation recordings",
        "Test recordings",
        "Training windows",
        "Validation windows",
        "Test windows",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(
        output_dir / "eligible_species.csv", index=False
    )


def train_and_evaluate(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    species: list[str],
    class_names: dict[str, dict[str, str]],
    number_of_groups: list[int],
    output_dir: Path,
) -> dict[str, Any]:
    label_to_index = {label: index for index, label in enumerate(species)}
    targets = np.asarray([label_to_index[label] for label in labels], dtype=np.int64)
    train_indices, validation_indices, test_indices = make_group_splits(
        targets, groups, list(range(len(species)))
    )
    x_train, y_train = features[train_indices], labels[train_indices]
    x_validation, y_validation = features[validation_indices], labels[validation_indices]
    x_test, y_test = features[test_indices], labels[test_indices]

    validation_options: list[tuple[float, int, int, dict[str, Any]]] = []
    for group_count in number_of_groups:
        if group_count < 2 or group_count >= len(species):
            continue
        candidate = fit_stage(x_train, y_train, species, group_count)
        for top_groups in (1, 2):
            probabilities, _ = predict_staged_proba(
                candidate, x_validation, top_groups
            )
            score = f1_score(
                y_validation,
                np.asarray(species)[probabilities.argmax(axis=1)],
                labels=species,
                average="macro",
                zero_division=0,
            )
            validation_options.append(
                (float(score), group_count, top_groups, candidate)
            )
        print(f"Validated {group_count} acoustic groups")
    if not validation_options:
        raise ValueError("No valid group counts to evaluate.")
    best_validation_f1, best_group_count, best_top_groups, _ = max(
        validation_options, key=lambda result: (result[0], -result[1], -result[2])
    )

    train_validation_indices = np.concatenate((train_indices, validation_indices))
    x_train_validation = features[train_validation_indices]
    y_train_validation = labels[train_validation_indices]
    staged_model = fit_stage(
        x_train_validation, y_train_validation, species, best_group_count
    )
    staged_probabilities, _ = predict_staged_proba(
        staged_model, x_test, best_top_groups
    )
    flat_scaler = StandardScaler()
    flat_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=N_JOBS,
    ).fit(flat_scaler.fit_transform(x_train_validation), y_train_validation)
    flat_probabilities = flat_model.predict_proba(flat_scaler.transform(x_test))
    flat_probability_matrix = np.zeros(
        (len(x_test), len(species)), dtype=np.float64
    )
    for column, label in enumerate(flat_model.classes_):
        flat_probability_matrix[:, label_to_index[str(label)]] = flat_probabilities[:, column]

    staged_predictions = np.asarray(species)[staged_probabilities.argmax(axis=1)]
    flat_predictions = np.asarray(species)[flat_probability_matrix.argmax(axis=1)]
    metrics: dict[str, Any] = {
        "dataset": DATASET,
        "eligible_species_count": len(species),
        "eligible_species_rule": (
            "At least 100 readable clips and 50 source recordings before sampling; "
            "named birds only; non-bird labels excluded."
        ),
        "sampling": "At most one clip per source recording and 150 recordings per species.",
        "feature_names": list(FEATURE_NAMES),
        "selected_acoustic_groups": best_group_count,
        "selected_routing_groups": best_top_groups,
        "validation_macro_f1": best_validation_f1,
        "validation_candidates": [
            {
                "groups": groups_count,
                "top_groups_routed": top_count,
                "macro_f1": score,
            }
            for score, groups_count, top_count, _ in validation_options
        ],
        "test_samples": int(len(test_indices)),
        "test_source_recordings": int(len(np.unique(groups[test_indices]))),
        "species": [
            {"code": label, **class_names[label]} for label in species
        ],
        "test_metrics": {
            "staged": metric_summary(y_test, staged_probabilities, species),
            "flat_baseline": metric_summary(
                y_test, flat_probability_matrix, species
            ),
        },
        "test_per_class_report": classification_report(
            y_test,
            staged_predictions,
            labels=species,
            target_names=[
                f"{class_names[label]['common_name']} ({label})"
                for label in species
            ],
            output_dict=True,
            zero_division=0,
        ),
        "split_method": (
            "5-fold StratifiedGroupKFold grouped by collection and source filename; "
            "fold 0 test, fold 1 validation, folds 2-4 train."
        ),
        "seed": SEED,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "Label": label,
                "Species": class_names[label]["common_name"],
                "Scientific name": class_names[label]["scientific_name"],
                "Acoustic group": staged_model["species_to_group"][label],
            }
            for label in species
        ]
    ).to_csv(output_dir / "acoustic_groups.csv", index=False)
    pd.DataFrame(
        {
            "species": [
                f"{class_names[label]['common_name']} ({label})"
                for label in y_test
            ],
            "true_label": y_test,
            "staged_prediction": staged_predictions,
            "flat_prediction": flat_predictions,
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)
    report_rows = []
    for label in species:
        report = metrics["test_per_class_report"][
            f"{class_names[label]['common_name']} ({label})"
        ]
        report_rows.append({"Label": label, "Species": class_names[label]["common_name"], **report})
    pd.DataFrame(report_rows).to_csv(output_dir / "staged_per_class_metrics.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    joblib.dump(
        {
            "staged_model": staged_model,
            "species": species,
            "class_names": class_names,
            "feature_names": FEATURE_NAMES,
            "sample_rate": SAMPLE_RATE,
            "clip_seconds": 4.0,
            "selected_routing_groups": best_top_groups,
        },
        output_dir / "staged_bird_model.joblib",
    )
    write_split_support(
        labels,
        groups,
        train_indices,
        validation_indices,
        test_indices,
        species,
        class_names,
        output_dir,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--index-csv", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-clips", type=int, default=100)
    parser.add_argument("--min-recordings", type=int, default=50)
    parser.add_argument("--max-recordings-per-class", type=int, default=150)
    parser.add_argument("--group-counts", default="3,5,7")
    args = parser.parse_args()
    if min(args.min_clips, args.min_recordings, args.max_recordings_per_class) < 1:
        parser.error("Support limits must be positive.")
    try:
        group_counts = sorted({int(value) for value in args.group_counts.split(",")})
    except ValueError as exc:
        parser.error(f"--group-counts must be comma-separated integers: {exc}")
    if not group_counts or min(group_counts) < 2:
        parser.error("--group-counts must contain values of at least two.")
    dataset_root = (
        args.data_dir.expanduser().resolve()
        if args.data_dir
        else Path(kagglehub.dataset_download(DATASET))
    )
    output_dir = args.output_dir.expanduser().resolve()
    clips, species, class_names = select_clips(
        args.index_csv.expanduser().resolve(),
        load_species_names(dataset_root),
        args.min_clips,
        args.min_recordings,
        args.max_recordings_per_class,
        SEED,
    )
    print(
        f"Selected {len(clips):,} clips across {len(species)} named bird species."
    )
    features, labels, groups = extract_dataset_features(
        clips, dataset_root, output_dir
    )
    retained = sorted(set(labels.tolist()))
    if retained != species:
        raise ValueError(
            "At least one eligible species has no decodable selected clips; "
            "regenerate the index or adjust the support criteria."
        )
    print(
        f"Extracted {len(features):,} clips across {len(species)} species "
        f"and {len(np.unique(groups)):,} source recordings."
    )
    metrics = train_and_evaluate(
        features,
        labels,
        groups,
        species,
        class_names,
        group_counts,
        output_dir,
    )
    print(f"Test metrics: {metrics['test_metrics']}")
    print(f"Model and reports saved to: {output_dir}")


if __name__ == "__main__":
    main()
