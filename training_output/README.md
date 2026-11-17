# Model training results

These are report snapshots from the completed eight-species experiments.
The original runs, trained models, per-file predictions and feature caches stay
locally under the ignored `output/` directory. The `run_info.json` files retain
their original run paths for provenance; the reports are copied without edits.

All three models used the same 171,471 audio files and saved split:
102,883 training, 34,294 validation and 34,294 test clips. Source groups inferred
from collection and filename were kept together using five-fold
StratifiedGroupKFold with seed 42 (approximately 60/20/20).

The eight species are Morepork, Bellbird, Tui, Fantail, Tomtit, Grey Warbler,
Kea and Rifleman. Each file supplied one centred four-second segment at
22,050 Hz, padded if short. SVM and RF used 20 MFCC means and 20 standard
deviations; CNN used 128-bin log-Mel spectrograms from the same segments.

## Held-out test results

| Metric | Linear SVM | Random Forest | CNN |
|---|---:|---:|---:|
| Accuracy | 48.45% | 54.75% | 67.85% |
| Balanced accuracy | 37.19% | 37.81% | 65.45% |
| Macro F1 | 33.69% | 39.20% | 60.29% |
| Weighted F1 | 47.18% | 52.45% | 67.99% |
| Top-3 accuracy | 74.61% | 82.45% | 92.14% |

The training-selected majority-class baseline always predicted Morepork and
achieved 29.17% test accuracy. CNN performed best among these three pipelines
on this shared test selection. Perch has not been evaluated in these reports.

## Runs and evidence

- [SVM metrics](svm/metrics.json): standardised MFCCs, linear SVM, C=1,
  balanced class weights. Fitting took approximately 5.5 seconds.
- [RF metrics](rf/metrics.json): 200 trees and balanced class weights.
  Fitting took approximately 35.8 seconds.
- [CNN metrics](cnn/metrics.json): three convolution blocks, weighted training
  sampling and AdamW. Training stopped at epoch 20; the best checkpoint was
  epoch 14, selected using validation loss. Training and validation took
  approximately 52.6 minutes on CUDA. See [training history](cnn/training_history.csv).

Times exclude shared feature preparation. RF/SVM fitting times do not include
evaluation, so they are not directly equivalent to the CNN timing.

Each folder also contains `run_info.json`, validation/test per-class scores,
and count confusion matrices (rows are actual labels, columns are predictions).
The run metadata records hyperparameters, library versions, preprocessing,
class order and source-code hashes.

Shared source manifest SHA-256:

```text
a0713e36299012a54a9fb117c319907a00976ad527f0e3df5dae12e008b44259
```

See the [main README](../README.md#comparing-svm-random-forest-and-cnn-on-all-eight-species)
for preparation, training and test commands. The exact source manifest is stored
locally at `output/eight_species_data/manifest.csv`; it is not included here.
Preserve that file separately when sharing or reproducing these experiments.

The CNN run used `torch 2.15.0+cu132`; the committed dependency lock selects
CPU PyTorch. On the original NVIDIA machine, the CUDA build was installed with:

```cmd
uv pip install --python .venv\Scripts\python.exe "torch==2.15.0" --torch-backend=auto --reinstall-package torch
.\.venv\Scripts\python.exe train_cnn.py --device cuda --evaluate-test --out output/cnn_full_gpu
```

The backend selected by `auto` depends on the machine. Check GPU availability
before running, and use a new output directory for a new experiment. A normal
`uv sync` or `uv run` may restore the CPU build specified by the lockfile.

## Limits of the comparison

Scores are per file, not per independent source, and class sizes are unequal.
Source grouping prevents an inferred recording from crossing splits but does
not prove separation by site, recordist or duplicate audio across collections.
This pipeline includes all readable, non-empty, finite audio for the eight
primary labels; it does not apply the original script's silence, clipping or
duplicate-audio filters. It is a new comparison, not Warner's one-clip-per-source
experiment, so those earlier results should not be treated as a direct ranking.
