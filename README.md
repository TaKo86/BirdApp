# BirdApp 

## 1. Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is installed automatically

```bash
pip install uv
```

```bash
uv sync
```

This creates `.venv` with the exact package versions from `uv.lock`

Add new packages with `uv add <name>` (not `pip install`), so the lock file stays in sync

## 2. Data

The audio is not in the repo. Download it from Kaggle (45 GB, CC BY 4.0, free Kaggle account needed):
https://www.kaggle.com/datasets/ollypowell/new-zealand-bird-sound

Either click Download on the page, or use the Kaggle CLI
(needs an API token in `~/.kaggle/kaggle.json`):

```bash
uv run --with kaggle kaggle datasets download -d ollypowell/new-zealand-bird-sound
```

Unzip `archive.zip` into the repo folder

## 3. Run

Open `bird_pipeline.ipynb`, select the `.venv` interpreter and run all cells

## Licence

- DOC recordings: CC BY 4.0 (Department of Conservation, via Kaggle)
- xeno-canto recordings: Creative Commons licence per recording (see `--xc-licences`)

## Running the demo UI

No dataset, trained model or API server is needed:

```bash
uv sync
uv run streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501).

## Comparing SVM, Random Forest and CNN on all eight species

These scripts use Morepork, Bellbird, Tui, Fantail, Tomtit, Grey Warbler, Kea
and Rifleman. Prepare the data once, then point all three trainers at the same
saved manifest. The notebook and original preprocessing scripts are unchanged.
The new experiment has its own output directory and preprocessing settings.

Install the separate training groups (Python 3.11 is supported):

```cmd
uv sync --group training --group cnn
```

Prepare every available file with one of the eight primary labels. Replace the
path if needed; it must contain the three original nested `DOC_*` collections:

```cmd
uv run --group training --group cnn python prepare_model_data.py --data-root "E:\BIRD CALL CROPS" --workers 2
```

There is no 500-file cap and no one-file-per-source sampling. The current local
metadata contains 171,471 files for these labels, before decode failures. Empty,
unreadable or non-finite audio is logged in `decode_errors.csv` and excluded for
all models. This preparation does not apply the original script's silence,
clipping or duplicate-audio filters. Multiple files from the same inferred
recording are kept, but stay together in one split. The result is a new
comparison, not an exact reproduction of Warner's source-sampled experiment.

Each file supplies one centred four-second segment at 22,050 Hz; shorter files
are centre-padded. This uses the centre-crop approach and basic Mel settings
from Warner's `audio_utils.py`, implemented here with librosa rather than
TorchAudio. Mel settings are 128 bins, FFT 1024, hop 256, HTK scale, no Mel
filter normalization, 50-11,025 Hz, and a -80 dB floor. RF/SVM use 20 MFCC means
followed by 20 standard deviations. CNN uses the log-Mel array, normalized by
its own mean and standard deviation. Exact settings and library versions are
saved in `dataset.json`. Features from the old five-second pipeline are not
used in this experiment.

Preparation writes `output/eight_species_data/manifest.csv` with filenames,
labels, source groups, segment offsets, cache rows and fixed split assignments.
Five-fold StratifiedGroupKFold (seed 42) supplies fold 0 for test, fold 1 for
validation, and folds 2-4 for training: approximately 60/20/20, not necessarily
equal class counts. All eight classes must appear in every split. Group IDs
are inferred from collection and filename; this does not guarantee separation
by site, recordist, equipment, or duplicate recordings across collections.

The spectrogram cache is disk-backed, approximately 14.1 GiB for 171,471 files,
plus metadata, MFCCs and trained models. Preparation may take a long time.
If interrupted, repeat the same command with `--resume`; completed batches are
reused. Keep the same environment when resuming. A completed dataset is reused
directly by the trainers and does not need preparing again.

Run these fixed baseline experiments to get validation and test metrics:

```cmd
uv run --group training --group cnn python train_svm.py --evaluate-test
uv run --group training --group cnn python train_rf.py --evaluate-test
uv run --group training --group cnn python train_cnn.py --evaluate-test
```

If experimenting with hyperparameters, omit `--evaluate-test` and use validation
scores only. Freeze the settings before final test evaluation. Once test results
are viewed, do not use them to tune the models. Use a new `--out` directory for
each experiment; existing model output folders are never overwritten.

The SVM defaults to `StandardScaler` + `LinearSVC` (C=1, balanced class weights).
The scaler learns from training rows only. `--kernel rbf` is available, but a full
RBF SVM can take substantially longer; the linear and RBF versions should be
named separately in the report. SVM top-3 uses decision scores, not calibrated
probabilities. A convergence warning stops the run rather than saving an
unfinished fit. RF defaults to 200 trees, balanced class weights and four jobs.
Both scripts expose basic hyperparameters in `--help`.

CNN uses Warner's three-block architecture with AdamW, balanced training
sampling and validation-loss checkpoint selection. Defaults are 25 epochs,
batch size 32 and early-stopping patience 6. It reads the cache in batches and
uses CUDA when available, otherwise CPU. CPU training on the full dataset can
be slow. `--device cpu`, `--device cuda`, `--batch-size`, `--epochs` and
`--threads` are available. `uv sync` does not guarantee a CUDA-enabled PyTorch
installation; `--device cuda` checks availability before starting. GPU runs can
still differ slightly across hardware/library versions. The best checkpoint,
not necessarily the last epoch, is evaluated.

Results are saved separately in `output/svm_full`, `output/rf_full` and
`output/cnn_full`. Each contains:

- `metrics.json`: accuracy, macro precision/recall/F1, balanced accuracy,
  weighted F1, top-3 accuracy and a majority-class baseline selected from training.
- `val_per_class.csv` and, when requested, `test_per_class.csv`.
- Count confusion matrices and per-file predictions for each evaluated split.
- A model artifact and `run_info.json` containing settings, versions and the
  shared manifest fingerprint. CNN also saves `training_history.csv`.

Compare runs with the same manifest fingerprint. Scores are per audio file,
not per inferred source. Keep the natural class imbalance visible by reporting
macro F1 and balanced accuracy alongside accuracy. To compare with Perch, use
the same test file/segment list, its required input preprocessing, and document
how predictions are mapped or restricted to these eight species. Previous
experiment scores with different examples/splits are not directly comparable.

Synthetic checks (no real bird recordings required):

```cmd
uv run --group training --group cnn python -m unittest discover -s tests -p test_model_training.py -v
```

If `uv` is not on PATH in this checkout, replace it in these commands with
`.\.cache\uv-bin\uv.exe`. That local executable is not part of the repository.
All data, caches and model outputs stay under the already-ignored `output/`.
