# Bird audio classification

The workspace includes an EDA and CNN training workflow for Kaggle dataset
[`ollypowell/new-zealand-bird-sound`](https://www.kaggle.com/datasets/ollypowell/new-zealand-bird-sound).
It uses the collection metadata to map primary species-code labels, then trains
a grouped-split CNN on log-mel spectrograms.

## Run

```powershell
python -m pip install -r requirements.txt
python EDA.py
python train_cnn.py
```

By default KaggleHub downloads or reads the cached dataset. Pass
`--data-dir <path>` to either script to use a local dataset folder. EDA writes
the audio index, class counts, summary, and charts to `dataset_report/`.
Training writes the checkpoint, held-out metrics, training history, per-class
metrics, and confusion matrices to `olly_training_outputs/`.

See [OLLY_TRAINING_RESULTS.md](./OLLY_TRAINING_RESULTS.md) for the visual
results and [OLLY_CNN_TRAINING_NOTES.md](./OLLY_CNN_TRAINING_NOTES.md) for
methodology and caveats.
