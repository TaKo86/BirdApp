"""Train a grouped-split CNN on manually annotated CACOPHONY bird segments."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset, WeightedRandomSampler

from audio_utils import (
    CLIP_SECONDS,
    N_MELS,
    SAMPLE_RATE,
    crop_audio,
    decode_audio,
    log_mel_spectrogram,
)

DEFAULT_DATA_DIR = Path(r"E:\CACOPHONY")
DEFAULT_OUTPUT_DIRNAME = "cnn_output"
SEED = 42
SPLITS = 5


class BirdAudioCNN(nn.Module):
    def __init__(self, number_of_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.40),
            nn.Linear(128 * 4 * 8, number_of_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def select_segments(
    segments_path: Path,
    data_dir: Path,
    min_segments: int,
    min_recordings: int,
    max_per_recording: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not segments_path.is_file():
        raise FileNotFoundError(
            f"Missing EDA segment index: {segments_path}. Run EDA.py first."
        )

    frame = pd.read_csv(segments_path)
    if frame.empty:
        raise ValueError(f"No bird segments found in {segments_path}")

    frame["recording_id"] = frame["recording_id"].astype(str)
    support = frame.groupby("label").agg(
        segments=("label", "size"),
        recordings=("recording_id", "nunique"),
    )
    eligible_labels = sorted(
        support.index[
            (support["segments"] >= min_segments)
            & (support["recordings"] >= min_recordings)
        ].tolist()
    )
    if len(eligible_labels) < 2:
        raise ValueError(
            "Fewer than two labels meet the support thresholds "
            f"(min_segments={min_segments}, min_recordings={min_recordings})."
        )

    eligible = frame[frame["label"].isin(eligible_labels)].to_dict(orient="records")
    by_label_recording: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_label_recording[(str(row["label"]), str(row["recording_id"]))].append(row)

    rng = random.Random(SEED)
    selected: list[dict[str, Any]] = []
    for rows in by_label_recording.values():
        rows.sort(key=lambda row: float(row["start_s"]))
        if len(rows) > max_per_recording:
            rows = sorted(rng.sample(rows, max_per_recording), key=lambda row: float(row["start_s"]))
        selected.extend(rows)

    for row in selected:
        audio_path = (data_dir / Path(str(row["relative_path"]))).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file listed in EDA index is missing: {audio_path}")
        row["audio_path"] = audio_path
        row["recording_id"] = str(row["recording_id"])
    return selected, eligible_labels


def extract_features(
    segments: list[dict[str, Any]], output_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[dict[str, str]]]:
    labels = sorted({str(segment["label"]) for segment in segments})
    label_to_index = {label: index for index, label in enumerate(labels)}
    grouped: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        grouped[Path(segment["audio_path"])].append(segment)

    features: list[np.ndarray] = []
    targets: list[int] = []
    recording_ids: list[str] = []
    decode_errors: list[dict[str, str]] = []
    for file_index, (audio_path, file_segments) in enumerate(grouped.items(), start=1):
        try:
            audio = decode_audio(audio_path)
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            detail = (
                exc.stderr.decode("utf-8", errors="replace")[-500:]
                if isinstance(exc, subprocess.CalledProcessError) and exc.stderr
                else str(exc)
            )
            decode_errors.append({"file": str(audio_path), "error": detail})
            continue

        for segment in file_segments:
            clip = crop_audio(audio, float(segment["start_s"]), float(segment["end_s"]))
            features.append(log_mel_spectrogram(clip))
            targets.append(label_to_index[str(segment["label"])])
            recording_ids.append(str(segment["recording_id"]))

        if file_index % 100 == 0 or file_index == len(grouped):
            print(f"Decoded {file_index}/{len(grouped)} recordings")

    if not features:
        raise RuntimeError("No audio features were extracted; see decode_errors.csv.")

    feature_array = np.stack(features).astype(np.float32, copy=False)
    target_array = np.asarray(targets, dtype=np.int64)
    group_array = np.asarray(recording_ids, dtype=str)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(decode_errors, columns=["file", "error"]).to_csv(
        output_dir / "decode_errors.csv", index=False
    )
    if decode_errors:
        print(f"Audio decode errors: {len(decode_errors)} (details in decode_errors.csv)")
    return feature_array, target_array, group_array, labels, decode_errors


def make_group_splits(
    targets: np.ndarray, groups: np.ndarray
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
        class_groups: dict[int, set[str]] = defaultdict(set)
        for index in indices:
            class_groups[int(targets[index])].add(str(groups[index]))
        if len(class_groups) < 2:
            raise ValueError(f"The {name} split contains fewer than two labels.")
    if set(groups[train_indices]) & set(groups[validation_indices]):
        raise RuntimeError("Recording leakage detected between train and validation.")
    if set(groups[train_indices]) & set(groups[test_indices]):
        raise RuntimeError("Recording leakage detected between train and test.")
    if set(groups[validation_indices]) & set(groups[test_indices]):
        raise RuntimeError("Recording leakage detected between validation and test.")
    return train_indices, validation_indices, test_indices


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    sample_count = 0
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1)
            total_loss += criterion(logits, targets).item() * targets.size(0)
            sample_count += targets.size(0)
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(logits.argmax(dim=1).cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
    return (
        total_loss / sample_count,
        np.concatenate(all_targets),
        np.concatenate(all_predictions),
        np.concatenate(all_probabilities),
    )


def calculate_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    number_of_classes: int,
) -> dict[str, float]:
    top_k = min(3, number_of_classes)
    top_k_predictions = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "precision_macro": float(
            precision_score(
                targets, predictions, average="macro", zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(targets, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(targets, predictions, average="weighted", zero_division=0)
        ),
        "top_3_accuracy": float(
            np.any(top_k_predictions == targets[:, np.newaxis], axis=1).mean()
        ),
        "log_loss": float(
            log_loss(targets, probabilities, labels=list(range(number_of_classes)))
        ),
        "matthews_correlation": float(matthews_corrcoef(targets, predictions)),
        "cohen_kappa": float(cohen_kappa_score(targets, predictions)),
    }


def train(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    labels: list[str],
    output_dir: Path,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    train_indices, validation_indices, test_indices = make_group_splits(targets, groups)
    feature_tensor = torch.from_numpy(features)
    target_tensor = torch.from_numpy(targets)
    dataset = TensorDataset(feature_tensor, target_tensor)
    train_labels = targets[train_indices]
    train_counts = np.bincount(train_labels, minlength=len(labels))
    sample_weights = torch.as_tensor(
        1.0 / train_counts[train_labels], dtype=torch.double
    )
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_indices),
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    train_evaluation_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = BirdAudioCNN(len(labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience = 6
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, batch_targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), batch_targets)
            loss.backward()
            optimizer.step()

        train_evaluation_loss, train_targets, train_predictions, train_probabilities = evaluate(
            model, train_evaluation_loader, criterion, device
        )
        validation_loss, validation_targets, validation_predictions, validation_probabilities = evaluate(
            model, validation_loader, criterion, device
        )
        scheduler.step(validation_loss)
        train_metrics = calculate_metrics(
            train_targets, train_predictions, train_probabilities, len(labels)
        )
        validation_metrics = calculate_metrics(
            validation_targets,
            validation_predictions,
            validation_probabilities,
            len(labels),
        )
        epoch_row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_evaluation_loss,
            "validation_loss": validation_loss,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_row)
        print(
            f"Epoch {epoch:02d}/{epochs}: "
            f"train_loss={epoch_row['train_loss']:.4f}, "
            f"val_loss={validation_loss:.4f}, "
            f"val_accuracy={validation_metrics['accuracy']:.4f}, "
            f"val_macro_f1={validation_metrics['f1_macro']:.4f}, "
            f"lr={epoch_row['learning_rate']:.2g}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    if best_state is None:
        raise RuntimeError("Training completed without a valid checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    test_loss, test_targets, test_predictions, test_probabilities = evaluate(
        model, test_loader, criterion, device
    )

    report = classification_report(
        test_targets,
        test_predictions,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    test_metrics = calculate_metrics(
        test_targets, test_predictions, test_probabilities, len(labels)
    )
    majority_baseline = float(
        np.bincount(test_targets, minlength=len(labels)).max() / len(test_targets)
    )
    metrics: dict[str, Any] = {
        "device": str(device),
        "classes": labels,
        "samples": {
            "total": int(len(targets)),
            "train": int(len(train_indices)),
            "validation": int(len(validation_indices)),
            "test": int(len(test_indices)),
        },
        "recordings": {
            "total": int(len(np.unique(groups))),
            "train": int(len(np.unique(groups[train_indices]))),
            "validation": int(len(np.unique(groups[validation_indices]))),
            "test": int(len(np.unique(groups[test_indices]))),
        },
        "test_metrics": {
            "cross_entropy_loss": float(test_loss),
            **test_metrics,
            "majority_class_accuracy_baseline": majority_baseline,
            "balanced_accuracy_uniform_guess_baseline": float(1 / len(labels)),
        },
        "per_class_report": report,
        "best_validation_loss": best_validation_loss,
        "epochs_completed": len(history),
        "split_method": "5-fold StratifiedGroupKFold; folds 0/1 held out for test/validation",
        "seed": SEED,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "model_state_dict": best_state,
            "class_names": labels,
            "sample_rate": SAMPLE_RATE,
            "clip_seconds": CLIP_SECONDS,
            "n_mels": N_MELS,
            "architecture": "BirdAudioCNN",
        },
        output_dir / "best_model.pt",
    )

    matrix = confusion_matrix(
        test_targets, test_predictions, labels=list(range(len(labels)))
    )
    normalized_matrix = confusion_matrix(
        test_targets,
        test_predictions,
        labels=list(range(len(labels))),
        normalize="true",
    )
    pd.DataFrame(matrix, index=labels, columns=labels).rename_axis(
        "true_label"
    ).to_csv(output_dir / "test_confusion_matrix_counts.csv")
    pd.DataFrame(normalized_matrix, index=labels, columns=labels).rename_axis(
        "true_label"
    ).to_csv(output_dir / "test_confusion_matrix_normalized.csv")
    class_report = {
        key: value for key, value in report.items() if isinstance(value, dict)
    }
    pd.DataFrame.from_dict(class_report, orient="index").rename_axis("label").to_csv(
        output_dir / "per_class_metrics.csv"
    )
    figure_width = max(10, len(labels) * 0.45)
    for matrix_values, filename, title in (
        (matrix, "test_confusion_matrix.png", "Held-out confusion matrix (counts)"),
        (
            normalized_matrix,
            "test_confusion_matrix_normalized.png",
            "Held-out confusion matrix (row-normalized)",
        ),
    ):
        _, axis = plt.subplots(figsize=(figure_width, figure_width * 0.82))
        image = axis.imshow(matrix_values, cmap="Blues", aspect="auto")
        axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        axis.set_yticks(range(len(labels)), labels, fontsize=7)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_title(title)
        figure = axis.figure
        figure.colorbar(image, ax=axis, shrink=0.75)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=150)
        plt.close(figure)

    history_frame = pd.DataFrame(history)
    _, axes = plt.subplots(2, 2, figsize=(13, 9))
    for metric, title, axis in (
        ("loss", "Cross-entropy loss", axes[0, 0]),
        ("accuracy", "Accuracy", axes[0, 1]),
        ("balanced_accuracy", "Balanced accuracy", axes[1, 0]),
        ("f1_macro", "Macro F1", axes[1, 1]),
    ):
        axis.plot(history_frame["epoch"], history_frame[f"train_{metric}"], label="Train")
        axis.plot(
            history_frame["epoch"],
            history_frame[f"validation_{metric}"],
            label="Validation",
        )
        axis.set_xlabel("Epoch")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_metric_curves.png", dpi=150)
    plt.close()

    print(f"Test metrics: {metrics['test_metrics']}")
    print(f"Model and reports saved to: {output_dir}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--segments-csv", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-segments", type=int, default=30)
    parser.add_argument("--min-recordings", type=int, default=10)
    parser.add_argument("--max-per-recording", type=int, default=3)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_per_recording < 1:
        parser.error("--epochs, --batch-size, and --max-per-recording must be positive.")
    if args.min_segments < 1 or args.min_recordings < SPLITS:
        parser.error(
            f"--min-segments must be positive and --min-recordings must be at least {SPLITS}."
        )

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else data_dir / DEFAULT_OUTPUT_DIRNAME
    )
    segments_path = (
        args.segments_csv.expanduser().resolve()
        if args.segments_csv
        else data_dir / "eda_output" / "bird_segments.csv"
    )
    segments, eligible_labels = select_segments(
        segments_path,
        data_dir,
        args.min_segments,
        args.min_recordings,
        args.max_per_recording,
    )
    print(
        f"Selected {len(segments)} segments across {len(eligible_labels)} labels "
        f"(at least {args.min_segments} segments and {args.min_recordings} recordings per label; "
        f"maximum {args.max_per_recording} segments per label/recording)."
    )
    features, targets, groups, labels, decode_errors = extract_features(
        segments, output_dir
    )

    group_counts: dict[int, set[str]] = defaultdict(set)
    for target, group in zip(targets, groups):
        group_counts[int(target)].add(str(group))
    supported_indices = [
        index for index in range(len(labels)) if len(group_counts[index]) >= SPLITS
    ]
    if len(supported_indices) < 2:
        raise RuntimeError(
            f"Fewer than two classes remain with at least {SPLITS} recordings after decoding."
        )
    removed_labels = [label for index, label in enumerate(labels) if index not in supported_indices]
    if removed_labels:
        print(f"Excluded labels with too few decodable recordings: {removed_labels}")
    keep = np.isin(targets, supported_indices)
    old_to_new = {old: new for new, old in enumerate(supported_indices)}
    features = features[keep]
    groups = groups[keep]
    targets = np.asarray([old_to_new[int(value)] for value in targets[keep]], dtype=np.int64)
    labels = [labels[index] for index in supported_indices]
    print(f"Final training data: {len(targets)} clips, {len(labels)} classes.")
    train(
        features,
        targets,
        groups,
        labels,
        output_dir,
        args.epochs,
        args.batch_size,
    )
    print(f"Skipped recordings due to audio decoding errors: {len(decode_errors)}")


if __name__ == "__main__":
    main()
