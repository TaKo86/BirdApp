""" convert the clean recordings into model-ready log-mel spectrograms (+ MFCC features)

    python make_spectrograms.py --manifest output/manifest_clean.csv --out output

Writes
    output/specs/<label>/<clip>_c<k>.npy   float16 log-mel, shape (N_MELS, 313), values in [-80, 0] dB
    output/spectrogram_index.csv           one row per spectrogram: path, label, split, group + mfcc_* columns
    output/spectrogram_config.json         every parameter used, so the run is reproducible
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

SR = 32000
WINDOW_S = 5.0  # every spectrogram covers exactly 5 s
MIN_CHUNK_S = 2.0  # a trailing piece shorter than this is dropped (the first piece is always kept, zero-padded)
MAX_CHUNKS = 3  # cap per recording so long xeno-canto files do not dominate
N_FFT, HOP, N_MELS = 2048, 512, 128
FMIN, FMAX = 50, 15000
N_MFCC = 20
SEED = 42


def chunks(y, sr=SR):
    w = int(WINDOW_S * sr)
    out = []
    for start in range(0, max(len(y), 1), w):
        c = y[start : start + w]
        if out and len(c) < MIN_CHUNK_S * sr:
            break
        out.append(np.pad(c, (0, w - len(c))))
        if len(out) == MAX_CHUNKS:
            break
    return out


def log_mel(chunk):
    mel = librosa.feature.melspectrogram(
        y=chunk, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmin=FMIN, fmax=FMAX
    )
    return librosa.power_to_db(mel, ref=np.max)  # loudness-normalised: loudest bin = 0 dB


def process(row, out_dir):
    try:
        y, _ = librosa.load(row["path"], sr=SR, mono=True)  # resamples only if the file is not 32 kHz
    except Exception as e:
        return [], f"{row['path']}: {e}"
    rows = []
    for k, c in enumerate(chunks(y)):
        db = log_mel(c)
        mfcc = librosa.feature.mfcc(S=db, n_mfcc=N_MFCC)
        rel = Path("specs") / row["primary_label"] / f"{row['source'][:7]}_{Path(row['filename']).stem}_c{k}.npy"
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / rel, db.astype(np.float16))
        info = {
            "spec_path": rel.as_posix(),
            "label": row["primary_label"],
            "split": row["split"],
            "group": row["group"],
            "source": row["source"],
            "audio_path": row["path"],
            "chunk": k,
        }
        info.update({f"mfcc_mean_{i}": v for i, v in enumerate(mfcc.mean(axis=1))})
        info.update({f"mfcc_std_{i}": v for i, v in enumerate(mfcc.std(axis=1))})
        rows.append(info)
    return rows, None


def _work(args):
    return process(*args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("output/manifest_clean.csv"))
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--max-per-class", type=int, default=500, help="cap recordings per species (0 = no cap)")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    m = pd.read_csv(args.manifest)
    if args.max_per_class:
        shuffled = m.sample(frac=1, random_state=SEED)
        capped = shuffled.groupby("primary_label").head(args.max_per_class)
        # the cap must not drop a species' only val/test recording, so keep one clip per (species, split) too
        keep = capped.index.union(shuffled.groupby(["primary_label", "split"]).head(1).index)
        m = m.loc[keep]
    print(f"{len(m)} recordings, {m.primary_label.nunique()} species")

    rows, errors = [], []
    jobs = [(r, args.out) for r in m.to_dict("records")]
    with Pool(args.workers) as pool:
        for got, err in tqdm(pool.imap_unordered(_work, jobs, chunksize=16), total=len(jobs), desc="spectrograms"):
            rows.extend(got)
            if err:
                errors.append(err)

    index = pd.DataFrame(rows).sort_values("spec_path")
    index.to_csv(args.out / "spectrogram_index.csv", index=False)
    config = {k: v for k, v in globals().items() if k.isupper()}
    config.update(manifest=str(args.manifest), max_per_class=args.max_per_class, frames=1 + int(WINDOW_S * SR) // HOP)
    (args.out / "spectrogram_config.json").write_text(json.dumps(config, indent=2))
    if errors:
        (args.out / "spectrogram_errors.txt").write_text("\n".join(errors), encoding="utf-8")
    print(f"wrote {len(index)} spectrograms ({len(errors)} files failed)")
    print(index.groupby("split").size().to_string())


if __name__ == "__main__":
    main()
