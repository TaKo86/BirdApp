# CACOPHONY bird-audio CNN

The scripts use the local dataset at `E:\CACOPHONY` by default. Audio files are
paired with JSON annotations stored in `.txt` sidecars. The numbered
directories are storage shards, not class labels; labels are read from
manually tagged `all.bird.*` tracks.

## Run

Install the requirements, using matching PyTorch and TorchAudio builds for the
machine's operating system and CUDA version:

```powershell
python -m pip install -r requirements.txt
python EDA.py --data-dir E:\CACOPHONY
python train_cnn.py --data-dir E:\CACOPHONY
```

EDA reports and the segment index are written to `E:\CACOPHONY\eda_output`.
Training output, including the checkpoint, held-out metrics, training history,
per-class metrics, training curves, and raw/normalized confusion matrices, is
written to `E:\CACOPHONY\cnn_output`. Training requires the segment index
produced by EDA. See [CNN_TRAINING_NOTES.md](./CNN_TRAINING_NOTES.md) for the
methodology, metric definitions, and limitations. The latest run is copied into
[`training_outputs/`](./training_outputs/), with a visual summary in
[`TRAINING_RESULTS.md`](./TRAINING_RESULTS.md).

The CNN uses 4-second, 22.05 kHz log-mel inputs centered on annotated events.
It retains labels with at least 30 segments from at least 10 recordings, then
limits a label to three segments per recording. Five-fold stratified grouped
splitting keeps each source recording in only one of the train, validation, or
test partitions. These thresholds can be changed with command-line options.

## EDA findings for the current dataset

- 2,077 audio files: 2,045 M4A and 32 MP3.
- 2,067 valid annotation sidecars, eight missing sidecars, and two malformed
  sidecars.
- Valid sidecar durations have a 60-second median.
- 7,595 manually tagged bird segments across 63 labels. The class distribution
  is strongly imbalanced; the most common label has 1,654 segments.
- 23 labels meet the default training support thresholds, yielding 4,283
  candidate segments after limiting repeated recordings.

The training metrics are estimates on held-out recordings; they should not be
interpreted as evidence of generalization to new regions, devices, or species
not represented in this dataset.
