"""check training and leakage protection with small made-up feature tables."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_random_forest import feature_columns, read_index, select_classes, train


def example_index() -> pd.DataFrame:
    """make two separable classes with different recordings in every split."""
    rows = []
    rng = np.random.default_rng(42)
    for split in ("train", "val", "test"):
        for label, offset in (("bird_a", -5), ("bird_b", 5)):
            for i in range(6):
                row = {"label": label, "split": split, "group": f"{label}_{split}_{i}",
                       "spec_path": f"{label}/{split}_{i}.npy"}
                row.update({f"mfcc_{stat}_{j}": offset + rng.normal(0, 0.1)
                            for stat in ("mean", "std") for j in range(20)})
                rows.append(row)
    return pd.DataFrame(rows)


class RandomForestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.frame = example_index()
        self.index = self.folder / "spectrogram_index.csv"
        self.frame.to_csv(self.index, index=False)
        (self.folder / "spectrogram_config.json").write_text(json.dumps({"N_MFCC": 20}))

    def run_cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(ROOT / "train_random_forest.py"), "--index", str(self.index), *args],
            capture_output=True, text=True, cwd=ROOT,
        )

    def test_recording_overlap_is_rejected(self) -> None:
        self.frame.loc[self.frame["split"] == "val", "group"] = self.frame.loc[0, "group"]
        self.frame.to_csv(self.index, index=False)
        with self.assertRaisesRegex(ValueError, "overlap"):
            read_index(self.index)

    def test_duplicate_windows_are_rejected(self) -> None:
        self.frame.loc[1, "spec_path"] = self.frame.loc[0, "spec_path"]
        self.frame.to_csv(self.index, index=False)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_index(self.index)

    def test_missing_class_in_split_is_rejected(self) -> None:
        subset = self.frame[~((self.frame["label"] == "bird_b") & (self.frame["split"] == "test"))]
        with self.assertRaisesRegex(ValueError, "missing from test"):
            select_classes(subset, None)

    def test_unknown_class_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown classes"):
            select_classes(self.frame, ["not_a_label"])

    def test_feature_order_and_invalid_values(self) -> None:
        columns = feature_columns(self.frame, 20)
        self.assertEqual(columns[9:12], ["mfcc_mean_9", "mfcc_mean_10", "mfcc_mean_11"])
        self.assertEqual(columns[20], "mfcc_std_0")
        self.assertEqual(len(columns), 40)
        self.frame.loc[0, "mfcc_mean_0"] = np.inf
        with self.assertRaisesRegex(ValueError, "infinite"):
            feature_columns(self.frame, 20)

    def test_only_training_rows_affect_model(self) -> None:
        columns = feature_columns(self.frame, 20)
        first, _, _ = train(self.frame, columns, trees=5, jobs=1)
        changed = self.frame.copy()
        changed.loc[changed["split"] != "train", columns] = 999
        second, _, _ = train(changed, columns, trees=5, jobs=1)
        for a, b in zip(first.estimators_, second.estimators_):
            np.testing.assert_array_equal(a.tree_.threshold, b.tree_.threshold)
            np.testing.assert_array_equal(a.tree_.value, b.tree_.value)

    def test_inspect_does_not_train(self) -> None:
        output = self.folder / "inspect"
        result = self.run_cli("--inspect", "--out", str(output))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("recordings", result.stdout)
        self.assertFalse(output.exists())

    def test_scope_must_be_explicit(self) -> None:
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Choose --classes", result.stderr)

    def test_training_artifacts_and_explicit_test_evaluation(self) -> None:
        for evaluate_test in (False, True):
            with self.subTest(evaluate_test=evaluate_test):
                output = self.folder / str(evaluate_test)
                args = ["--classes", "bird_a", "bird_b", "--trees", "5", "--jobs", "1", "--out", str(output)]
                if evaluate_test:
                    args.append("--evaluate-test")
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                scores = json.loads((output / "metrics.json").read_text())
                self.assertEqual("test" in scores, evaluate_test)
                self.assertEqual(scores["val"]["random_forest"]["accuracy"], 1.0)
                self.assertEqual(scores["val"]["majority_baseline"]["accuracy"], 0.5)
                bundle = joblib.load(output / "model.joblib")
                predictions = bundle["model"].predict(self.frame[bundle["feature_columns"]])
                np.testing.assert_array_equal(predictions, self.frame["label"])
                saved = pd.read_csv(output / "selected_index.csv")
                pd.testing.assert_series_equal(saved["split"], self.frame["split"])
                second = self.run_cli(*args)
                self.assertNotEqual(second.returncode, 0)
                self.assertIn("already exists", second.stderr)


if __name__ == "__main__":
    unittest.main()
