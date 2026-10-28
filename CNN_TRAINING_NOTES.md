# CNN training and evaluation notes

This note explains the data interpretation, modeling decisions, evaluation
metrics, and limitations of the CACOPHONY bird-audio classifier. The scripts
and report outputs are intended to make the experiment reproducible and its
scores interpretable, not to claim that one training run establishes a
production-ready bird detector.

## What the data contains

The dataset is rooted at `E:\CACOPHONY`. Its two-character directories are
storage shards, not species labels. Each recording may have a `.txt` JSON
sidecar containing a `Tracks` array. Labels are taken only from human-created
track tags whose taxonomy path starts with `all.bird.`; automatic predictions,
generic labels (`bird`, `unidentified`, `other`), and non-bird tags are not
used as training targets.

The EDA found 2,077 audio files, 2,067 parseable sidecars, eight missing
sidecars, two malformed sidecars, and 7,595 manually tagged bird segments over
63 labels. Recording durations have a 60-second median. The most common label
has 1,654 segments, versus one segment for several rare labels, so an
unfiltered 63-class model would be dominated by classes that have too little
recording-level evidence to evaluate reliably.

## Training examples and features

- Each manually tagged segment becomes one single-label example. The example
  is a fixed four-second window centered on the annotation interval, padded
  with silence at file boundaries. This retains brief lead-in/decay context
  while giving every CNN input the same shape.
- FFmpeg decodes the M4A and MP3 files to mono 22.05 kHz floating-point audio.
  A 128-bin log-mel spectrogram (1,024-point FFT, 256-sample hop, 50 Hz lower
  edge) is the model input. Mel features compress the frequency axis in a way
  commonly useful for audio classification; per-example standardization
  reduces sensitivity to recording loudness.
- To keep class estimates supported, the default includes labels with at least
  30 annotated segments from at least 10 different recordings. Up to three
  segments per label and recording are sampled, reducing the effect of a
  single prolific recording. This leaves 23 candidate classes and 4,283
  candidate segments in the observed dataset.
- Two files were undecodable in the completed run and are logged in
  `E:\CACOPHONY\cnn_output\decode_errors.csv`. Their examples are skipped with
  an explicit count, rather than silently treated as valid audio.

## Model and optimization

The model is a compact 2D CNN operating on time-frequency images. Three
convolutional blocks (32, 64, then 128 channels) learn local spectral and
temporal patterns; batch normalization and ReLU stabilize/activate the blocks,
max pooling reduces the feature-map size, and an adaptive 4-by-8 pooling layer
preserves coarse time/frequency structure for the class head. Dropout regularizes
the feature extractor and classifier.

Training uses cross-entropy, AdamW (initial learning rate `0.001`, weight decay
`0.0001`), and a `ReduceLROnPlateau` scheduler that halves the learning rate
after validation loss stops improving. A weighted random sampler makes
minority classes more likely to appear in training batches. Training-set
metrics are evaluated on the original, unresampled training examples, so those
curves reflect the observed class distribution rather than the balanced
sampler.

The best checkpoint is selected by validation cross-entropy loss. The test
partition is not used to select epochs or tune hyperparameters. Reproducibility
is aided by fixed Python, NumPy, and PyTorch seeds; GPU kernels may still
introduce small run-to-run variation.

## Split and leakage control

The split is a seeded five-fold `StratifiedGroupKFold`: one fold is held out
for test, a separate fold for validation, and the remaining three for
training. `recording_id` is the group, so all annotated segments from a
recording stay in only one partition. This matters because nearby events from
the same recording share background noise, microphone, and habitat conditions;
a random segment split would make evaluation optimistic.

This grouping does **not** guarantee independence by site, date, recorder, or
geographic region. A stronger claim about generalization would require a
separate location/time/device holdout and ideally external recordings.

## Reported measures

The saved `metrics.json` contains held-out test results and a per-class
classification report. The output folder also contains
`per_class_metrics.csv`, `test_confusion_matrix_counts.csv`, and
`test_confusion_matrix_normalized.csv`.

- **Cross-entropy loss / log loss** assess the quality of predicted
  probabilities; smaller is better. Cross-entropy is also the checkpoint
  selection criterion.
- **Accuracy** is the fraction of correct predictions. It can hide poor
  performance on less frequent classes.
- **Balanced accuracy** averages recall across classes, giving each class equal
  weight.
- **Macro precision, recall, and F1** compute the class metric separately for
  each label, then average the labels equally. Macro F1 is the main summary when
  class balance matters.
- **Weighted F1** averages per-class F1 weighted by test support. It reflects
  the observed frequency mix and should be read alongside macro F1.
- **Top-3 accuracy** counts a prediction as correct when the true class is in
  the three highest-probability classes. It is supplementary and does not
  replace top-1 accuracy.
- **Matthews correlation coefficient** and **Cohen's kappa** summarize
  multiclass agreement while accounting for chance agreement; larger is better.
- **Per-class precision, recall, F1, and support** reveal which labels work
  and which fail. A zero score for a low-support test class is uncertain and
  should not be read as a precise estimate of its true performance.
- The raw confusion matrix shows example counts; its row-normalized version
  shows the fraction of each true class predicted as each class. Diagonal
  values are correct classifications; off-diagonal cells expose confusions.
- The metric curves chart training and validation loss, accuracy, balanced
  accuracy, and macro F1 over epochs. A widening train/validation gap can
  indicate overfitting.
- The majority-class accuracy and uniform-guess balanced-accuracy values are
  reference baselines in `metrics.json`, not additional trained models.

The current grouped test split has 856 examples from 240 recordings. Its
reported accuracy is compared to the majority-class baseline, and its balanced
accuracy is compared to `1 / number_of_classes`. Use the exact values from
`E:\CACOPHONY\cnn_output\metrics.json` for the latest run; do not rely on a
metric copied into this note because training settings or data may change.

## Outputs and reproduction

Run EDA before training so the sidecar-derived segment index is current:

```powershell
python EDA.py --data-dir E:\CACOPHONY
python train_cnn.py --data-dir E:\CACOPHONY
```

`E:\CACOPHONY\cnn_output` contains:

- `best_model.pt` — best validation-loss checkpoint, class names, and feature
  parameters.
- `metrics.json` — test metrics, per-class results, split counts, and baselines.
- `per_class_metrics.csv` — spreadsheet-friendly class report.
- `training_history.csv` and `training_metric_curves.png` — epoch-by-epoch
  train/validation scores and plots.
- `test_confusion_matrix.png` and `test_confusion_matrix_normalized.png` —
  count and row-normalized test matrices.
- `test_confusion_matrix_counts.csv` and
  `test_confusion_matrix_normalized.csv` — numeric matrix values.
- `decode_errors.csv` — source files which failed decoding.

## Interpretation and next steps

A 23-class score is a baseline for this dataset and split, not evidence that
the model generalizes to every bird species, habitat, or recording device.
Class imbalance, annotation noise, rare test classes, and shared locations or
devices can all affect the result. Inspect the normalized confusion matrix and
per-class support before deciding whether a class is useful.

For stronger evaluation, collect more independently recorded examples for
under-supported species, define a site/date/device holdout before tuning, and
compare multiple seeded runs. Hyperparameter or threshold choices should be
made using training/validation data only; preserve a final untouched test set
for the final reported estimate.
