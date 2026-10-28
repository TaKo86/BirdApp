# Olly Powell bird CNN: results

This report belongs to Kaggle's
[`ollypowell/new-zealand-bird-sound`](https://www.kaggle.com/datasets/ollypowell/new-zealand-bird-sound),
not the CACOPHONY dataset. The full artifacts are saved under
[`olly_training_outputs/`](./olly_training_outputs/).

## Dataset EDA

The current inventory reports 271,365 readable mono FLAC clips across 133
primary-label codes and 57,874 inferred source recordings. Clip duration is
typically short (median five seconds); source clip names include numbered
segments, so grouped source IDs are used for splitting. The largest class has
50,010 clips, a 50,010-to-1 raw imbalance against the rarest labels.

The dataset includes a `weta` label mapped to Tree Weta, which is an insect.
This class is not included in the bird CNN. The training defaults require at
least 100 clips and 50 inferred source recordings per species, then cap the
sample at 150 clips per species and one clip per species/source recording. That
results in 42 bird classes for this run.

## Held-out evaluation

The test set contains 1,087 clips from 948 inferred source recordings. All
selected clips from a source recording are kept within one split.

| Metric | Held-out result |
|---|---:|
| Accuracy | 19.8% |
| Balanced accuracy | 21.4% |
| Macro precision | 20.4% |
| Macro recall | 21.4% |
| Macro F1 | 18.8% |
| Weighted F1 | 18.4% |
| Top-3 accuracy | 36.3% |
| Cross-entropy loss | 3.300 |
| Log loss | 3.296 |
| Matthews correlation | 0.180 |
| Cohen's kappa | 0.179 |
| Majority-class accuracy baseline | 2.8% |
| Uniform-guess balanced-accuracy reference | 2.4% |

These are baseline results on one grouped split. Accuracy is above the
majority-class reference, but macro F1 remains low; the model still confuses
many species. Per-class scores and support are important context, and these
metrics do not establish generalization to new recordists, locations, or
collection conditions.

## Plots

### Training and validation metrics

![Cross-entropy, accuracy, balanced accuracy, and macro F1 by epoch](./olly_training_outputs/training_metric_curves.png)

### Confusion matrix (raw counts)

![Held-out confusion matrix with raw counts](./olly_training_outputs/test_confusion_matrix.png)

### Confusion matrix (row-normalized)

![Held-out confusion matrix normalized by true species](./olly_training_outputs/test_confusion_matrix_normalized.png)

## Output files

- [`best_model.pt`](./olly_training_outputs/best_model.pt) — CNN checkpoint
  and class-name mapping.
- [`metrics.json`](./olly_training_outputs/metrics.json) — full test metrics,
  split counts, baseline values, and per-class report.
- [`per_class_metrics.csv`](./olly_training_outputs/per_class_metrics.csv) —
  per-species precision, recall, F1, and test support.
- [`training_history.csv`](./olly_training_outputs/training_history.csv) and
  [`training_metric_curves.png`](./olly_training_outputs/training_metric_curves.png)
  — per-epoch train/validation measures and plots.
- [`test_confusion_matrix_counts.csv`](./olly_training_outputs/test_confusion_matrix_counts.csv)
  and
  [`test_confusion_matrix_normalized.csv`](./olly_training_outputs/test_confusion_matrix_normalized.csv)
  — numeric confusion matrix files.
- [`decode_errors.csv`](./olly_training_outputs/decode_errors.csv) — files
  skipped because of decoding errors; none occurred in this run.

EDA tables and charts are in [`dataset_report/`](./dataset_report/). The
training decisions, label filter, and evaluation limitations are described in
[`OLLY_CNN_TRAINING_NOTES.md`](./OLLY_CNN_TRAINING_NOTES.md).
