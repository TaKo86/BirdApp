"""list the validation clips the saved model got wrong, with their audio paths."""

import argparse
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    """run validation predictions again and export the clips worth listening to."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("output/rf_baseline"))
    parser.add_argument("--manifest", type=Path, default=Path("output/manifest_clean.csv"))
    args = parser.parse_args()

    bundle = joblib.load(args.run_dir / "model.joblib")
    index = pd.read_csv(args.run_dir / "selected_index.csv")
    # only inspect validation recordings while choosing model settings
    validation = index[index["split"] == "val"].copy()
    if validation.empty:
        parser.error("No validation rows found in the saved index.")
    model = bundle["model"]
    features = validation[bundle["feature_columns"]]
    validation["predicted_label"] = model.predict(features)
    validation["predicted_probability"] = model.predict_proba(features).max(axis=1)
    errors = validation[validation["label"] != validation["predicted_label"]].copy()
    errors = errors.rename(columns={"label": "dataset_label"})

    names = {}
    if args.manifest.is_file():
        manifest = pd.read_csv(args.manifest, usecols=["primary_label", "common_name"])
        names = manifest.dropna(subset=["common_name"]).drop_duplicates("primary_label").set_index(
            "primary_label"
        )["common_name"].to_dict()
    errors["dataset_species"] = errors["dataset_label"].map(lambda label: names.get(label, label))
    errors["predicted_species"] = errors["predicted_label"].map(lambda label: names.get(label, label))
    seconds = float(bundle["preprocessing"]["WINDOW_S"])
    # chunk 0 is the first window, chunk 1 is the next, and so on
    errors["start_seconds"] = errors["chunk"] * seconds
    errors["window_end_seconds"] = errors["start_seconds"] + seconds
    errors["audio_exists"] = errors["audio_path"].map(lambda path: Path(path).is_file())
    errors = errors.sort_values(["predicted_probability", "spec_path"], ascending=[False, True])
    errors.insert(0, "review_id", range(1, len(errors) + 1))
    columns = [
        "review_id", "dataset_species", "predicted_species", "predicted_probability",
        "audio_path", "start_seconds", "window_end_seconds", "chunk", "audio_exists",
        "dataset_label", "predicted_label", "group", "spec_path",
    ]
    output = args.run_dir / "val_misclassified.csv"
    errors[columns].to_csv(output, index=False, encoding="utf-8-sig")
    print(f"{len(errors)} misclassified windows out of {len(validation)} validation windows.")
    print(f"Saved to {output}")
    print("\nFirst 10 errors, starting with the highest predicted probabilities:")
    print(errors[["review_id", "dataset_species", "predicted_species", "predicted_probability"]].head(10).to_string(index=False))
    print("\nUse audio_path to find each recording and start_seconds to locate the window.")
    print("Short windows were padded with silence, so window_end_seconds may exceed the file duration.")
    print("The dataset label is the supplied label, not a guarantee that only that bird is audible.")


if __name__ == "__main__":
    main()
