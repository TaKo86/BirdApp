"""train a small CNN on the shared eight-species log-Mel cache."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from training_common import CLASSES, SETTINGS, load_dataset, save_scores, start_run, training_parser, write_json


class MelDataset(Dataset):
    """read one spectrogram at a time from the disk-backed cache."""

    def __init__(self, cache_path: Path, rows: pd.DataFrame) -> None:
        self.cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        self.indices = rows.cache_row.to_numpy()
        self.targets = rows.label.map({label: i for i, label in enumerate(CLASSES)}).to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int]:
        mel = np.array(self.cache[self.indices[position]], dtype=np.float32)
        if not np.isfinite(mel).all():
            raise ValueError("Non-finite CNN input in the feature cache.")
        return torch.from_numpy(mel[None]), int(self.targets[position])


class BirdCNN(nn.Module):
    """use Warner's three convolution blocks and a small classification head."""

    def __init__(self) -> None:
        super().__init__()
        blocks = []
        for before, after, dropout in ((1, 32, 0.10), (32, 64, 0.15), (64, 128, 0.0)):
            blocks.extend([nn.Conv2d(before, after, 3, padding=1, bias=False),
                           nn.BatchNorm2d(after), nn.ReLU(inplace=True), nn.MaxPool2d(2)])
            if dropout:
                blocks.append(nn.Dropout2d(dropout))
        # the third block uses adaptive pooling instead of another max pool
        blocks.pop()
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d((4, 8)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.40), nn.Linear(128 * 4 * 8, len(CLASSES)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def evaluate(model: BirdCNN, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray]:
    """keep prediction order identical to the rows used to build the loader."""
    model.eval()
    total_loss, scores = 0.0, []
    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs.to(device))
            total_loss += float(nn.functional.cross_entropy(logits, targets.to(device), reduction="sum").item())
            scores.append(logits.softmax(dim=1).cpu().numpy())
    return total_loss / len(loader.dataset), np.concatenate(scores)


def main() -> None:
    """select the checkpoint on validation loss and optionally score test data."""
    parser = training_parser("cnn")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience, args.threads) < 1 or args.learning_rate <= 0:
        parser.error("Epochs, batch size, patience, threads and learning rate must be positive.")
    random.seed(SETTINGS["seed"])
    np.random.seed(SETTINGS["seed"])
    torch.manual_seed(SETTINGS["seed"])
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SETTINGS["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(selected_device if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch installation. Use --device cpu or a compatible CUDA build.")
    frame, config, _, _ = load_dataset(args.data)
    start_run(args, config, "cnn")
    train_rows, val_rows = frame[frame.split == "train"], frame[frame.split == "val"]
    training = MelDataset(args.data / "mel.npy", train_rows)
    counts = np.bincount(training.targets, minlength=len(CLASSES))
    sampler = WeightedRandomSampler(torch.as_tensor(1.0 / counts[training.targets], dtype=torch.double),
        len(training), replacement=True, generator=torch.Generator().manual_seed(SETTINGS["seed"]))
    # zero loader workers avoids duplicating memmaps when Windows spawns processes
    train_loader = DataLoader(training, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(MelDataset(args.data / "mel.npy", val_rows), batch_size=args.batch_size, shuffle=False)
    model = BirdCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    best_loss, stale, history = float("inf"), 0, []
    started = time.perf_counter()
    print(f"Training CNN on {device}: {len(train_rows):,} training clips, {len(val_rows):,} validation clips.", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for step, (inputs, targets) in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(inputs.to(device)), targets.to(device))
            if not torch.isfinite(loss):
                raise ValueError("CNN loss became non-finite.")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(targets)
            if step % 100 == 0:
                print(f"Epoch {epoch}: batch {step}/{len(train_loader)}", flush=True)
        val_loss, probabilities = evaluate(model, val_loader, device)
        from sklearn.metrics import accuracy_score, f1_score
        predicted = np.asarray(list(CLASSES))[probabilities.argmax(axis=1)]
        history.append({"epoch": epoch, "sampled_train_loss": loss_sum / len(training),
            "val_loss": val_loss, "val_accuracy": float(accuracy_score(val_rows.label, predicted)),
            "val_macro_f1": float(f1_score(val_rows.label, predicted, average="macro", zero_division=0))})
        pd.DataFrame(history).to_csv(args.out / "training_history.csv", index=False)
        print(f"Epoch {epoch}: val loss={val_loss:.4f}, macro F1={history[-1]['val_macro_f1']:.4f}", flush=True)
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            torch.save({"state_dict": model.state_dict(), "classes": list(CLASSES),
                "dataset": config, "best_epoch": epoch}, args.out / "model.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(args.out / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    metrics = {"fit_seconds": time.perf_counter() - started, "device": str(device),
               "best_epoch": checkpoint["best_epoch"], "test_evaluated": args.evaluate_test}
    for split in (["val", "test"] if args.evaluate_test else ["val"]):
        rows = frame[frame.split == split]
        loader = DataLoader(MelDataset(args.data / "mel.npy", rows), batch_size=args.batch_size, shuffle=False)
        loss, probabilities = evaluate(model, loader, device)
        predicted = np.asarray(list(CLASSES))[probabilities.argmax(axis=1)]
        metrics[split] = save_scores(args.out, split, rows, predicted, probabilities,
            train_rows.label.value_counts().idxmax(), "probability")
        metrics[split]["cross_entropy"] = loss
    write_json(args.out / "metrics.json", metrics)
    print(f"Saved CNN model and reports to {args.out}.")


if __name__ == "__main__":
    main()
