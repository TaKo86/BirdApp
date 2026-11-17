"""save the proposed bird class list and split counts as a CSV for the group."""

import argparse
from pathlib import Path

import pandas as pd

# proposed eight classes; use --classes to export a different selection
DEFAULT_CLASSES = ["morepo2", "nezbel1", "tui1", "nezfan1", "tomtit1", "gryger1", "kea1", "riflem1"]


def main() -> None:
    """count recordings and windows, then save a table that opens in Excel."""
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--classes", nargs="+", help="specific labels to include")
    scope.add_argument("--all-classes", action="store_true", help="include every class in the feature index")
    parser.add_argument("--out", type=Path, default=Path("output/class_list.csv"))
    args = parser.parse_args()

    index = pd.read_csv("output/spectrogram_index.csv", usecols=["label", "split", "group"])
    names = pd.read_csv("output/manifest_clean.csv", usecols=["primary_label", "common_name"])
    names = names.drop_duplicates("primary_label").set_index("primary_label")["common_name"]
    labels = sorted(index["label"].unique()) if args.all_classes else (args.classes or DEFAULT_CLASSES)
    missing = set(labels) - set(index["label"])
    if missing:
        parser.error(f"Labels not found in the feature index: {', '.join(sorted(missing))}")

    selected = index[index["label"].isin(labels)]
    counts = selected.groupby(["label", "split"]).agg(
        recordings=("group", "nunique"), windows=("group", "size")
    ).unstack("split", fill_value=0)

    table = pd.DataFrame({"Label": list(dict.fromkeys(labels))}).set_index("Label")
    table["Species"] = table.index.map(names)
    table["Species"] = table["Species"].fillna(pd.Series(table.index, index=table.index))
    for measure in ("recordings", "windows"):
        for split, display in (("train", "Training"), ("val", "Validation"), ("test", "Test")):
            values = counts.get((measure, split), pd.Series(dtype="int64"))
            table[f"{display} {measure}"] = values.reindex(table.index, fill_value=0).astype(int)
    table = table.reset_index()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # no extra index column; utf-8-sig helps Excel read accented bird names
    table.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))
    print(f"\nSaved {len(table)} classes to {args.out}")


if __name__ == "__main__":
    main()
