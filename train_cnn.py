"""Train and evaluate a grouped-split CNN on the Olly Powell bird dataset."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import kagglehub
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
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
    decode_audio,
    fixed_length_crop,
    log_mel_spectrogram,
    load_species_names,
    source_recording_id,
)

DATASET = "ollypowell/new-zealand-bird-sound"
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INDEX = PROJECT_DIR / "dataset_report" / "audio_index.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "olly_training_outputs"
SEED = 42
SPLITS = 5
NON_BIRD_LABELS = {"weta"}


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


def select_training_clips(
    index_path: Path,
    species_names: dict[str, dict[str, str]],
    min_clips: int,
    min_recordings: int,
    max_clips_per_class: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[str], dict[str, dict[str, str]]]:
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Missing audio index {index_path}. Run EDA.py before training."
        )
    frame = pd.read_csv(index_path)
    required = {"rel", "ok", "label"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{index_path} is missing columns {sorted(missing_columns)}. "
            "Regenerate it by running EDA.py."
        )
    frame = frame[frame["ok"].astype(bool)].copy()
    frame["label"] = frame["label"].astype(str).str.casefold()
    frame["source_id"] = frame["rel"].map(source_recording_id)
    support = frame.groupby("label").agg(
        clips=("label", "size"), recordings=("source_id", "nunique")
    )
    labels = sorted(
        support.index[
            (support["clips"] >= min_clips)
            & (support["recordings"] >= min_recordings)
        ].tolist()
    )
    labels = [label for label in labels if label not in NON_BIRD_LABELS]
    if len(labels) < 2:
        raise ValueError(
            "Fewer than two species meet the training support criteria "
            f"(min_clips={min_clips}, min_recordings={min_recordings})."
        )

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    class_names: dict[str, dict[str, str]] = {}
    eligible = frame[frame["label"].isin(labels)]
    for label in labels:
        class_rows = eligible[eligible["label"] == label]
        class_names[label] = species_names.get(
            label, {"common_name": "", "scientific_name": ""}
        )

        one_per_recording: list[dict[str, Any]] = []
        for _, rows in class_rows.groupby("source_id", sort=True):
            records = rows.to_dict(orient="records")
            one_per_recording.append(rng.choice(records))
        rng.shuffle(one_per_recording)
        selected.extend(one_per_recording[:max_clips_per_class])

    return selected, labels, class_names


def extract_features(
    clips: list[dict[str, Any]],
    labels: list[str],
    dataset_root: Path,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]]]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    features: list[np.ndarray] = []
    targets: list[int] = []
    groups: list[str] = []
    errors: list[dict[str, str]] = []

    for index, row in enumerate(clips, start=1):
        relative = Path(str(row["rel"]))
        audio_path = dataset_root / relative
        try:
            audio = decode_audio(audio_path)
        except (OSError, RuntimeError, ValueError, sf.SoundFileRuntimeError, subprocess.SubprocessError) as exc:
            errors.append({"file": str(relative), "error": str(exc)[:500]})
            continue
        feature = log_mel_spectrogram(fixed_length_crop(audio))
        features.append(feature)
        targets.append(label_to_index[str(row["label"])])
        groups.append(source_recording_id(str(row["rel"])))
        if index % 500 == 0 or index == len(clips):
            print(f"Processed {index:,}/{len(clips):,} selected audio clips")

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(errors, columns=["file", "error"]).to_csv(
        output_dir / "decode_errors.csv", index=False
    )
    if errors:
        print(f"Audio decode errors: {len(errors)} (details in decode_errors.csv)")
    if not features:
        raise RuntimeError("No audio features were extracted.")
    return (
        np.stack(features).astype(np.float32, copy=False),
        np.asarray(targets, dtype=np.int64),
        np.asarray(groups, dtype=str),
        errors,
    )


def make_group_splits(
    targets: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(
        n_splits=SPLITS, shuffle=True, random_state=SEED
    )
    folds = list(splitter.split(np.zeros(len(targets)), targets, groups))
    test_indices = folds[0][1]
    validation_indices = folds[1][1]
    train_indices = np.concatenate([fold[1] for fold in folds[2:]])
    for name, indices in (
        ("train", train_indices),
        ("validation", validation_indices),
        ("test", test_indices),
    ):
        if len(np.unique(targets[indices])) < 2:
            raise ValueError(f"The {name} split contains fewer than two classes.")
    partitions = [set(groups[indices]) for indices in (train_indices, validation_indices, test_indices)]
    if (
        partitions[0] & partitions[1]
        or partitions[0] & partitions[2]
        or partitions[1] & partitions[2]
    ):
        raise RuntimeError("Source recording leakage detected between dataset splits.")
    return train_indices, validation_indices, test_indices


def calculate_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    number_of_classes: int,
) -> dict[str, float]:
    top_k = min(3, number_of_classes)
    top_k_predictions = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
    class_indices = list(range(number_of_classes))
    matthews = float(matthews_corrcoef(targets, predictions))
    kappa = float(cohen_kappa_score(targets, predictions))
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "precision_macro": float(
            precision_score(targets, predictions, average="macro", zero_division=0)
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
        "log_loss": float(log_loss(targets, probabilities, labels=class_indices)),
        "matthews_correlation": matthews if np.isfinite(matthews) else 0.0,
        "cohen_kappa": kappa if np.isfinite(kappa) else 0.0,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    sample_count = 0
    targets_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1)
            total_loss += criterion(logits, targets).item() * targets.size(0)
            sample_count += targets.size(0)
            targets_all.append(targets.cpu().numpy())
            predictions_all.append(logits.argmax(dim=1).cpu().numpy())
            probabilities_all.append(probabilities.cpu().numpy())
    return (
        total_loss / sample_count,
        np.concatenate(targets_all),
        np.concatenate(predictions_all),
        np.concatenate(probabilities_all),
    )


def save_confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    labels: list[str],
    output_dir: Path,
) -> None:
    indices = list(range(len(labels)))
    matrices = (
        (
            confusion_matrix(targets, predictions, labels=indices),
            "test_confusion_matrix.png",
            "test_confusion_matrix_counts.csv",
            "Held-out confusion matrix (counts)",
        ),
        (
            confusion_matrix(
                targets, predictions, labels=indices, normalize="true"
            ),
            "test_confusion_matrix_normalized.png",
            "test_confusion_matrix_normalized.csv",
            "Held-out confusion matrix (row-normalized)",
        ),
    )
    figure_width = max(12, len(labels) * 0.38)
    for values, image_name, csv_name, title in matrices:
        pd.DataFrame(values, index=labels, columns=labels).rename_axis(
            "true_label"
        ).to_csv(output_dir / csv_name)
        _, axis = plt.subplots(figsize=(figure_width, figure_width * 0.78))
        image = axis.imshow(values, cmap="Blues", aspect="auto")
        axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
        axis.set_yticks(range(len(labels)), labels, fontsize=6)
        axis.set_xlabel("Predicted species code")
        axis.set_ylabel("True species code")
        axis.set_title(title)
        axis.figure.colorbar(image, ax=axis, shrink=0.75)
        axis.figure.tight_layout()
        axis.figure.savefig(output_dir / image_name, dpi=150)
        plt.close(axis.figure)


def train_model(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    labels: list[str],
    class_names: dict[str, dict[str, str]],
    output_dir: Path,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    train_indices, validation_indices, test_indices = make_group_splits(targets, groups)
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets))
    train_counts = np.bincount(targets[train_indices], minlength=len(labels))
    sampler = WeightedRandomSampler(
        torch.as_tensor(1.0 / train_counts[targets[train_indices]], dtype=torch.double),
        num_samples=len(train_indices),
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=pin_memory,
    )
    train_metrics_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = BirdAudioCNN(len(labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    best_loss = float("inf")
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

        train_loss, train_y, train_pred, train_prob = evaluate(
            model, train_metrics_loader, criterion, device
        )
        val_loss, val_y, val_pred, val_prob = evaluate(
            model, validation_loader, criterion, device
        )
        scheduler.step(val_loss)
        train_metrics = calculate_metrics(train_y, train_pred, train_prob, len(labels))
        val_metrics = calculate_metrics(val_y, val_pred, val_prob, len(labels))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"validation_{key}": value for key, value in val_metrics.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"Epoch {epoch:02d}/{epochs}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_accuracy={val_metrics['accuracy']:.4f}, "
            f"val_macro_f1={val_metrics['f1_macro']:.4f}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    if best_state is None:
        raise RuntimeError("Training finished without a validation checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    test_loss, test_y, test_pred, test_prob = evaluate(
        model, test_loader, criterion, device
    )
    code_labels = list(range(len(labels)))
    display_labels = [
        f"{class_names[label]['common_name']} ({label})"
        if class_names[label]["common_name"]
        else label
        for label in labels
    ]
    report = classification_report(
        test_y,
        test_pred,
        labels=code_labels,
        target_names=display_labels,
        output_dict=True,
        zero_division=0,
    )
    test_metrics = calculate_metrics(test_y, test_pred, test_prob, len(labels))
    metrics: dict[str, Any] = {
        "dataset": DATASET,
        "excluded_non_bird_labels": sorted(NON_BIRD_LABELS),
        "device": str(device),
        "classes": [
            {"code": label, **class_names[label]} for label in labels
        ],
        "samples": {
            "total": int(len(targets)),
            "train": int(len(train_indices)),
            "validation": int(len(validation_indices)),
            "test": int(len(test_indices)),
        },
        "source_recordings": {
            "total": int(len(np.unique(groups))),
            "train": int(len(np.unique(groups[train_indices]))),
            "validation": int(len(np.unique(groups[validation_indices]))),
            "test": int(len(np.unique(groups[test_indices]))),
        },
        "test_metrics": {
            "cross_entropy_loss": float(test_loss),
            **test_metrics,
            "majority_class_accuracy_baseline": float(
                np.bincount(test_y, minlength=len(labels)).max() / len(test_y)
            ),
            "balanced_accuracy_uniform_guess_baseline": float(1 / len(labels)),
        },
        "per_class_report": report,
        "best_validation_loss": best_loss,
        "epochs_completed": len(history),
        "split_method": (
            "5-fold StratifiedGroupKFold grouped by collection and source filename; "
            "fold 0 test, fold 1 validation, folds 2-4 train"
        ),
        "seed": SEED,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    class_report = {
        key: value for key, value in report.items() if isinstance(value, dict)
    }
    pd.DataFrame.from_dict(class_report, orient="index").rename_axis("species").to_csv(
        output_dir / "per_class_metrics.csv"
    )
    torch.save(
        {
            "model_state_dict": best_state,
            "class_codes": labels,
            "class_names": class_names,
            "sample_rate": SAMPLE_RATE,
            "clip_seconds": CLIP_SECONDS,
            "n_mels": N_MELS,
            "architecture": "BirdAudioCNN",
        },
        output_dir / "best_model.pt",
    )
    save_confusion_matrix(test_y, test_pred, display_labels, output_dir)

    history_frame = pd.DataFrame(history)
    _, axes = plt.subplots(2, 2, figsize=(13, 9))
    for key, title, axis in (
        ("loss", "Cross-entropy loss", axes[0, 0]),
        ("accuracy", "Accuracy", axes[0, 1]),
        ("balanced_accuracy", "Balanced accuracy", axes[1, 0]),
        ("f1_macro", "Macro F1", axes[1, 1]),
    ):
        axis.plot(history_frame["epoch"], history_frame[f"train_{key}"], label="Train")
        axis.plot(
            history_frame["epoch"],
            history_frame[f"validation_{key}"],
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
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--index-csv", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-clips", type=int, default=100)
    parser.add_argument("--min-recordings", type=int, default=50)
    parser.add_argument("--max-clips-per-class", type=int, default=150)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.min_clips, args.max_clips_per_class) < 1:
        parser.error("epochs, batch size, and clip limits must be positive.")
    if args.min_recordings < SPLITS:
        parser.error(f"--min-recordings must be at least {SPLITS}.")
    dataset_root = (
        args.data_dir.expanduser().resolve()
        if args.data_dir
        else Path(kagglehub.dataset_download(DATASET))
    )
    index_path = args.index_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    clips, labels, class_names = select_training_clips(
        index_path,
        load_species_names(dataset_root),
        args.min_clips,
        args.min_recordings,
        args.max_clips_per_class,
        SEED,
    )
    print(
        f"Selected {len(clips):,} clips across {len(labels)} species "
        f"(at least {args.min_clips} clips and {args.min_recordings} source recordings; "
        f"maximum {args.max_clips_per_class} clips per species)."
    )
    features, targets, groups, errors = extract_features(
        clips, labels, dataset_root, output_dir
    )
    supported = sorted(set(targets.tolist()))
    if len(supported) != len(labels):
        labels = [labels[index] for index in supported]
        class_names = {label: class_names[label] for label in labels}
        remap = {old: new for new, old in enumerate(supported)}
        targets = np.asarray([remap[int(value)] for value in targets], dtype=np.int64)
        print(f"Some classes had no decodable files; retained {len(labels)} classes.")
    print(
        f"Final data: {len(targets):,} clips, {len(labels)} classes, "
        f"{len(np.unique(groups)):,} source recordings."
    )
    train_model(
        features,
        targets,
        groups,
        labels,
        class_names,
        output_dir,
        args.epochs,
        args.batch_size,
    )
    print(f"Skipped audio files due to decoding errors: {len(errors)}")


if __name__ == "__main__":
    main()
