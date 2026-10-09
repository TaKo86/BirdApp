"""collect, check and filter the raw DOC / xeno-canto audio into one clean manifest.

    python prepare_dataset.py --data-root .. --out output

Writes
    output/manifest_all.csv    every clip + QC metrics + drop_reason ("" = kept)
    output/manifest_clean.csv  clips that passed every check (input for make_spectrograms.py)
    output/qc_report.txt       summary numbers for the report
"""
import argparse
import hashlib
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

DOC_LICENCE = "CC BY 4.0 (DOC, via kaggle.com/datasets/ollypowell/new-zealand-bird-sound)"
SOURCES = {  # dataset folder -> (metadata file, licence / provenance)
    "DOC_001_Tier1": ("001_metadata.csv", DOC_LICENCE),
    "DOC_002_DuncanBayParinga": ("002_metadata.csv", DOC_LICENCE),
    "DOC_003_XenoCanto": ("003_metadata.csv", "xeno-canto, Creative Commons licence per recording"),
}
NON_BIRD = {"badbird", "spybird", "weta"}  # placeholder / noise / insect labels
AMBIGUOUS = {"larus", "parake", "coot1", "oyster1", "x00458"}  # genus-level or hybrid labels
MIN_SECONDS = 1.0
SILENCE_RMS = 1e-4  # about -80 dBFS
CLIPPED_FRACTION = 0.001  # >0.1% of samples at full scale


def load_metadata(data_root):
    frames = []
    for folder, (csv, licence) in SOURCES.items():
        base = data_root / folder / folder
        if not base.exists():
            print("skipping missing dataset", base)
            continue
        m = pd.read_csv(base / csv, usecols=["filename", "primary_label", "secondary_labels"])
        m["source"] = folder
        m["licence"] = licence
        m["path"] = [str(base / "train_audio" / f) for f in m.filename]
        frames.append(m)
    df = pd.concat(frames, ignore_index=True)
    # Clips cut from the same recording end in _000, _001 ... and must never be split across train/test
    df["group"] = df.source + ":" + df.filename.map(lambda f: Path(f).stem.rsplit("_", 1)[0])
    is_xc = df.source == "DOC_003_XenoCanto"
    df["xc_id"] = df.filename.str.extract(r"_(\d+)_\d+\.\w+$")[0].where(is_xc)
    df["source_url"] = ("https://xeno-canto.org/" + df.xc_id).where(is_xc, "")
    return df


def qc(path):
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as e:
        return {"readable": False, "error": str(e)[:200]}
    mono = x.mean(axis=1)
    return {
        "readable": True,
        "sr": sr,
        "channels": x.shape[1],
        "duration": len(x) / sr,
        "rms": float(np.sqrt(np.mean(mono**2))) if len(mono) else 0.0,
        "clipped": float(np.mean(np.abs(x) >= 0.999)) if len(x) else 0.0,
        "audio_md5": hashlib.md5(x.tobytes()).hexdigest(),
    }


def xeno_canto_licences(ids, cache):
    """Read licence + citation from each recording's xeno-canto page (API v3 needs a key, the page does not).
    Results are cached in `cache` so a rerun only fetches new ids."""
    import time

    import requests

    out = pd.read_csv(cache, dtype=str).set_index("xc_id") if cache.exists() else pd.DataFrame(columns=["licence", "citation"])
    for i in tqdm(sorted(set(ids) - set(out.index)), desc="xeno-canto licences"):
        try:
            r = requests.get(f"https://xeno-canto.org/{i}", timeout=30)
            lic = re.search(r"<h2>License</h2>\s*<p><a href='([^']+)'", r.text) if r.ok else None
            cit = re.search(r"<h2>Citation</h2><p>(.*?)</p>", r.text) if r.ok else None
            out.loc[i] = [lic.group(1) if lic else "not found", cit.group(1) if cit else ""]
        except Exception as e:
            out.loc[i] = [f"lookup failed: {e}"[:120], ""]
        time.sleep(0.5)
    out.index.name = "xc_id"
    out.to_csv(cache)
    return out


def drop_reasons(df, min_clips):
    reasons = pd.Series("", index=df.index)

    def flag(mask, why):
        reasons[mask & (reasons == "")] = why

    flag(~df.readable.astype(bool), "unreadable")
    flag(df.primary_label.isin(NON_BIRD), "non_bird_label")
    flag(df.primary_label.isin(AMBIGUOUS), "ambiguous_label")
    flag(df.licence == "not found", "licence_not_found")
    flag(df.duration < MIN_SECONDS, "too_short")
    flag(df.rms < SILENCE_RMS, "silent")
    flag(df.clipped > CLIPPED_FRACTION, "clipped")
    flag(df.audio_md5.notna() & df.audio_md5.duplicated(), "duplicate_audio")
    counts = df[reasons == ""].primary_label.value_counts()
    flag(df.primary_label.map(counts).fillna(0) < min_clips, "too_few_clips")
    return reasons


def split_of(group):
    h = int(hashlib.md5(group.encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"


def assign_splits(clean):
    """Map each recording (group) to train/val/test, about 80/10/10, so that every species with >= 3 recordings
    has at least one recording in each split. A group is never split, so there is no leakage."""
    split = {}
    groups = clean.groupby("primary_label").group.unique()
    for label in groups.index[groups.map(len).argsort()]:  # rarest species first: they have the least room
        mine = sorted(groups[label], key=lambda g: hashlib.md5(g.encode()).hexdigest())
        have = {split[g] for g in mine if g in split}
        free = [g for g in mine if g not in split]
        for s in ("train", "test", "val"):
            if s not in have and free:
                split[free.pop(0)] = s
        for g in free:
            split[g] = split_of(g)
    return split


def report(df, clean, names):
    lines = [
        f"total clips: {len(df)}",
        f"kept clips: {len(clean)} ({len(clean) / len(df):.1%})",
        f"kept species: {clean.primary_label.nunique()}",
        "",
        "dropped by reason:",
        df.drop_reason.replace("", "(kept)").value_counts().to_string(),
        "",
        "kept clips per source:",
        clean.source.value_counts().to_string(),
        "",
        "split (clips):",
        clean.split.value_counts().to_string(),
        "",
        "sample rates seen:",
        df.sr.value_counts().to_string(),
        "",
        "labels missing from bird_naming_map.csv:",
        ", ".join(sorted(set(df.primary_label) - set(names.eBird))),
        "",
        "naming map rows sharing a scientific name (check these labels):",
        names[names.ScientificName.duplicated(keep=False)].sort_values("ScientificName").to_string(index=False),
        "",
        "kept clips per species (smallest first):",
        clean.groupby(["primary_label", "common_name"], dropna=False).size().sort_values().to_string(),
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent, help="folder holding the DOC_* datasets")
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--min-clips", type=int, default=10, help="drop species with fewer clean clips (BirdNET used 10)")
    ap.add_argument("--limit", type=int, help="only check N random clips per source (quick trial run)")
    ap.add_argument("--xc-licences", action="store_true", help="look up the CC licence of each xeno-canto recording")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_metadata(args.data_root)
    if args.limit:
        df = df.groupby("source", group_keys=False).sample(frac=1, random_state=0).groupby("source").head(args.limit)
        df = df.reset_index(drop=True)

    if args.xc_licences:
        lic = xeno_canto_licences(df.xc_id.dropna(), args.out / "xc_licences.csv")
        df.loc[df.xc_id.notna(), "licence"] = df.xc_id.map(lic.licence)
        df["citation"] = df.xc_id.map(lic.citation)

    with Pool(args.workers) as pool:
        metrics = list(tqdm(pool.imap(qc, df.path, chunksize=64), total=len(df), desc="checking audio"))
    df = pd.concat([df, pd.DataFrame(metrics)], axis=1)
    for col in ("sr", "duration", "rms", "clipped", "audio_md5", "error"):
        if col not in df:
            df[col] = np.nan

    names = pd.read_csv(args.data_root / "DOC_001_Tier1" / "DOC_001_Tier1" / "bird_naming_map.csv")
    names = names.apply(lambda s: s.str.strip())
    first = names.drop_duplicates("eBird").set_index("eBird")
    df["common_name"] = df.primary_label.map(first.CommonName)
    df["scientific_name"] = df.primary_label.map(first.ScientificName)

    df["drop_reason"] = drop_reasons(df, args.min_clips)
    df["split"] = df.group.map(assign_splits(df[df.drop_reason == ""])).fillna("")
    clean = df[df.drop_reason == ""]

    df.to_csv(args.out / "manifest_all.csv", index=False)
    clean.to_csv(args.out / "manifest_clean.csv", index=False)
    text = report(df, clean, names)
    (args.out / "qc_report.txt").write_text(text, encoding="utf-8")
    print(text[:1500])
    print(f"\nwrote {args.out / 'manifest_clean.csv'} ({len(clean)} clips)")


if __name__ == "__main__":
    main()
