"""
Download and examine the Kaggle dataset ollypowell/new-zealand-bird-sound.

Setup:
    pip install kagglehub soundfile librosa pandas matplotlib
    # Kaggle credentials, either:
    #   export KAGGLE_USERNAME=... KAGGLE_KEY=...
    # or put kaggle.json in ~/.kaggle/ (Kaggle > Settings > API > Create New Token)

Run:
    python explore_nz_birds.py
"""
from pathlib import Path

import kagglehub
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf

DATASET = "ollypowell/new-zealand-bird-sound"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
OUT_DIR = Path("dataset_report")
ENDANGERED_LABELS: set[str] = set()


def count_endangered_crops(audio: pd.DataFrame, endangered_labels: set[str]) -> int:
    return int(audio["label"].isin(endangered_labels).sum())


def download() -> Path:
    # kagglehub caches under ~/.cache/kagglehub, so re-runs don't re-download.
    path = Path(kagglehub.dataset_download(DATASET))
    print(f"Dataset at: {path}")
    return path


def list_files(root: Path) -> pd.DataFrame:
    rows = []
    for p in root.rglob("*"):
        if p.is_file():
            rows.append({
                "path": p,
                "rel": p.relative_to(root),
                "ext": p.suffix.lower(),
                "parent": p.parent.name,
                "size_kb": p.stat().st_size / 1024,
            })
    return pd.DataFrame(rows)


def audio_info(path: Path) -> dict:
    try:
        info = sf.info(str(path))  # header only, no decoding: fast
        return {"sr": info.samplerate, "channels": info.channels,
                "duration_s": info.duration, "ok": True}
    except Exception:
        # soundfile can't read mp3/m4a on some builds; fall back to librosa
        try:
            y, sr = librosa.load(str(path), sr=None, mono=False)
            n = y.shape[-1]
            return {"sr": sr, "channels": 1 if y.ndim == 1 else y.shape[0],
                    "duration_s": n / sr, "ok": True}
        except Exception as e:
            return {"sr": None, "channels": None, "duration_s": None,
                    "ok": False, "err": str(e)[:80]}


def main():
    root = download()
    OUT_DIR.mkdir(exist_ok=True)

    files = list_files(root)
    print("\n== File types ==")
    print(files["ext"].value_counts().to_string())

    # Any metadata/label files? Look at them before assuming folder = label.
    tables = files[files["ext"].isin({".csv", ".tsv", ".json", ".txt"})]
    print("\n== Possible metadata files ==")
    print(tables[["rel", "size_kb"]].to_string(index=False) if len(tables) else "none")
    for p in tables[tables["ext"] == ".csv"]["path"].head(3):
        print(f"\n-- head of {p.name} --")
        print(pd.read_csv(p).head())

    audio = files[files["ext"].isin(AUDIO_EXTS)].copy()
    print(f"\nAudio files: {len(audio)}")
    if audio.empty:
        return

    # Probe every file's header (cheap) for sr / channels / duration.
    info = pd.DataFrame([audio_info(p) for p in audio["path"]], index=audio.index)
    audio = audio.join(info)

    # Default label guess: parent folder name. Verify against metadata if it exists.
    audio["label"] = audio["parent"]

    print("\n== Unreadable files ==")
    print(audio[~audio["ok"]][["rel"]].to_string(index=False) or "none")

    good = audio[audio["ok"]]
    print("\n== Sample rates ==")
    print(good["sr"].value_counts().to_string())
    print("\n== Channels ==")
    print(good["channels"].value_counts().to_string())
    print("\n== Duration (s) ==")
    print(good["duration_s"].describe().to_string())

    counts = good["label"].value_counts()
    print(f"\n== Classes: {len(counts)} ==")
    print(counts.to_string())
    print(f"\nImbalance (max/min): {counts.max() / counts.min():.1f}x")

    if ENDANGERED_LABELS:
        endangered_count = count_endangered_crops(good, ENDANGERED_LABELS)
        print(f"\nEndangered sound crops: {endangered_count}")
    else:
        print("\nSet ENDANGERED_LABELS to the endangered species label codes "
              "to count their readable crops.")

    good.drop(columns=["path"]).to_csv(OUT_DIR / "audio_index.csv", index=False)

    # Class balance plot
    ax = counts.sort_values().plot.barh(figsize=(8, max(4, len(counts) * 0.25)))
    ax.set_xlabel("clips")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "class_balance.png", dpi=120)
    plt.close()

    # One mel-spectrogram per class (first 12 classes) for a quick eyeball check
    sample = good.groupby("label").head(1).head(12)
    cols = 3
    rows = -(-len(sample) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), squeeze=False)
    for ax, (_, r) in zip(axes.ravel(), sample.iterrows()):
        y, sr = librosa.load(str(r["path"]), sr=None, mono=True)
        S = librosa.power_to_db(
            librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=1.0
        )
        librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="mel", ax=ax)
        ax.set_title(str(r["label"])[:30], fontsize=9)
    for ax in axes.ravel()[len(sample):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "spectrogram_samples.png", dpi=110)
    plt.close()

    # Near-duplicate / leakage hint: crops from the same source recording
    # usually share a filename stem. Check for that before you split.
    stems = good["rel"].astype(str).str.replace(r"[_-]?\d+$", "", regex=True)
    stems = stems.str.rsplit(".", n=1).str[0]
    print(f"\nUnique filename stems: {stems.nunique()} of {len(good)} clips")
    print("If stems << clips, many crops share a source recording: "
          "split by stem (GroupShuffleSplit), not by clip.")

    print(f"\nReports saved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()