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

## Training a Random Forest baseline

Use a feature branch. The existing data preparation and spectrogram scripts are
unchanged. Replace the dataset path below with the folder containing the three
`DOC_*` folders (keep their original nested directory structure).

```powershell
uv sync --group training
uv run python prepare_dataset.py --data-root "E:\BIRD CALL CROPS" --out output --workers 4
uv run python make_spectrograms.py --manifest output/manifest_clean.csv --out output --workers 4
uv run --group training python train_random_forest.py --inspect
```

Preprocessing reads the audio and can take a while. Skip those two preprocessing
commands if you already have matching `spectrogram_index.csv` and
`spectrogram_config.json` files. The Random Forest only needs these two files;
the CNN also needs the matching spectrogram arrays. The default spectrogram run
caps input recordings at roughly 500 per class, while retaining split coverage.

Agree on the same classes and split with your teammates before comparing models.
Use the inspection table to check independent recording counts, not just window
counts. A class with only one validation/test recording gives a weak estimate.
Choose roughly 8-12 adequately represented species if that matches the approved
scope. Replace the example labels below with your agreed list (at least two):

```powershell
uv run --group training python train_random_forest.py --classes LABEL1 LABEL2 --out output/rf_baseline
```

Alternatively, `--all-classes` explicitly uses every class in the index. The
trainer checks group separation and class coverage, fits 200 trees on training
rows, and evaluates validation rows only. Features are the 20 MFCC means followed
by the 20 standard deviations; metadata columns are never used as inputs.

Each output directory contains `model.joblib` (model, class order, feature order
and preprocessing settings), `metrics.json`, `run_info.json`,
`selected_index.csv`, `feature_importances.csv` and a validation confusion matrix
CSV (rows are actual labels, columns are predictions). Metrics are per window,
not per recording. A majority-class baseline is included for comparison. Output
is under the already-ignored `output/` directory; don't commit models or data.
Use a new output directory for each run; existing runs are never overwritten.

Use validation results for model choices. Once those choices are frozen, rerun
with identical settings, a new `--out` directory, and `--evaluate-test` to also
produce final test metrics. All compared models must use the same selected rows
and splits. The Streamlit app still uses mock predictions until integration is
implemented; the trained model does not connect to it automatically.

```powershell
uv run --group training python -m unittest discover -s tests -v
```

If `uv` is not on PATH, use `python -m uv` if installed in that Python environment.
On this development checkout, `.\.cache\uv-bin\uv.exe` is also available; that
local tool is ignored and is not included when cloning the repository.
