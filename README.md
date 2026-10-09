# BirdApp 

## 1. Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is installed automatically

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