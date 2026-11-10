# Olly Powell bird dataset: CNN training notes

This experiment uses Kaggle dataset
[`ollypowell/new-zealand-bird-sound`](https://www.kaggle.com/datasets/ollypowell/new-zealand-bird-sound),
not the separate CACOPHONY recordings. EDA and model artifacts are written
inside this workspace to `dataset_report/` and `olly_training_outputs/`.

## What is in this dataset

The downloaded dataset contains three collections (`DOC_001_Tier1`,
`DOC_002_DuncanBayParinga`, and `DOC_003_XenoCanto`). Audio is FLAC under
`train_audio/<species_code>/`. Collection metadata CSVs specify a primary
species code and secondary labels; the accompanying bird naming maps provide
common and scientific names. EDA uses the primary label rather than guessing
from an arbitrary directory name.

The inventory contains 271,365 FLAC clips across 133 primary-label codes. Clip
duration is usually short (median five seconds), and class sizes are extremely
uneven. In the previous EDA inventory, the largest class had 50,010 clips.
Most files are source-recording crops: numbered suffixes in the filenames
identify crops/segments from a shared source. A random clip split would let
related audio appear in both train and test data and inflate evaluation.

## Training subset and labels

The default experiment includes only labels with at least 100 indexed clips and
at least 50 inferred source recordings. On the current inventory, 43 labels
meet those criteria, but the naming map identifies `weta` as Tree Weta
(`Hemideina sp.`), which is an insect rather than a bird. That label is
explicitly excluded, leaving 42 bird classes. The model samples at most 150
clips per class and at most one clip from any inferred source recording for
each class. This produces a manageable, approximately class-balanced training
set while retaining diverse recordings. These are configurable using
command-line arguments; excluded rare and non-bird labels are not evaluated by
this model.

Each FLAC is read as mono, resampled to 22.05 kHz, and center-cropped or
zero-padded to four seconds. The input is a 128-bin log-mel spectrogram using a
1,024-point FFT and 256-sample hop. The log-mel representation exposes
time-frequency patterns of bird calls while keeping the CNN input compact.
Per-clip normalization limits the effect of overall recording loudness.

## Model and optimization

The model is a 2D CNN with three convolutional blocks (32, 64, and 128
channels), batch normalization, ReLU activations, max pooling, and dropout. An
adaptive 4-by-8 pooling step preserves coarse time/frequency structure before
the classifier. The model predicts one primary-label class per clip.

Training minimizes cross-entropy using AdamW (learning rate `0.001`, weight
decay `0.0001`). A weighted sampler balances class contributions in training
batches. Epoch metrics are computed on the original, unresampled train set.
`ReduceLROnPlateau` halves the learning rate after validation loss stalls, and
the best checkpoint is chosen by validation cross-entropy with early stopping
after six non-improving epochs. A fixed seed supports repeatability, although
GPU kernels can still introduce small numerical variation.

## Split and leakage prevention

A seeded five-fold `StratifiedGroupKFold` uses the dataset collection plus
filename stem with a trailing crop index removed as the recording group. One
fold is held out for testing, a second for validation, and the remaining three
for training. A runtime assertion verifies that no inferred source group
appears across partitions.

This safeguards against crop-level leakage but does not ensure independence
across collection, location, recordist, date, or equipment. The collections may
also differ in recording conditions. A stronger generalization claim needs an
explicitly held-out collection/site or external recordings.

## Metrics and saved reports

- **Accuracy**: fraction of correct top-1 predictions.
- **Balanced accuracy**: mean per-class recall, useful when classes differ in
  support.
- **Macro precision, recall, and F1**: equal-weight average across classes.
  Macro F1 is the key summary when minority classes matter.
- **Weighted F1**: class F1 averaged according to test support.
- **Top-3 accuracy**: share of clips where the correct label is in the three
  highest-scoring predictions.
- **Cross-entropy / log loss**: quality of predicted probabilities; lower is
  better and it is used for checkpoint selection.
- **Matthews correlation** and **Cohen's kappa**: overall agreement measures
  adjusted for class prevalence or chance agreement.
- **Per-class precision, recall, F1, and support**: expose class-specific
  errors and show how many test examples each estimate is based on.
- **Count confusion matrix**: raw counts of true versus predicted labels.
  **Row-normalized confusion matrix** shows the fraction of each true class
  assigned to each predicted class.
- **Train/validation curves**: cross-entropy, accuracy, balanced accuracy,
  and macro F1 per epoch; a train/validation gap may indicate overfitting.

`metrics.json` includes a majority-class accuracy reference and uniform-guess
balanced-accuracy reference. These are baselines, not other trained models.

## Optional class-list experiment

`train_cnn.py` accepts `--class-list <csv>` to restrict training to labels in a
CSV's `Label` column. For example, `class_list.csv` selects eight species. Its
recording/window count columns are descriptive only: because it contains no
recording IDs, the trainer cannot reproduce those exact partitions. It samples
the requested labels from the indexed dataset and creates fresh source-grouped
train, validation, and test splits.

The eight-species experiment used a cap of 500 recordings per class and
processed 4,000 clips (2,400 train, 800 validation, 800 test) from 3,759
inferred sources. Its test accuracy was 45.4%, macro F1 was 43.8%, and top-3
accuracy was 72.9%, compared with 12.5% for both the test majority-class and
uniform balanced-accuracy references. These scores are not a direct
like-for-like comparison with the 42-class run: the number of candidate
species, selected clips, and resulting split differ. Results are saved in
`class_list_training_outputs/`.

The CNN run does not train on all 271,365 indexed files. The 42-class run
selected at most 150 clips per class (5,438 total); the class-list run selected
at most 500 per class (4,000 total). Audio is decoded and converted to
log-mel features once before training, then the small CNN trains for up to 25
epochs on the precomputed features using CUDA when available. This smaller
sample and GPU training explain why these runs take minutes rather than
processing the entire dataset. Runtime for a separate earlier experiment
cannot be compared without its sample counts, preprocessing, hardware, and
logs.

### All available recordings for the eight listed species

The follow-up run removed the per-class cap for `class_list.csv` by passing
`--max-clips-per-class 0`. It selected 76,946 clips across the eight listed
species, representing 47,520 unique inferred source groups. The grouped split
contained 46,167 training clips, 15,390 validation clips, and 15,389 test
clips. Training used CUDA on an NVIDIA GeForce GTX 1070, stopped after 24
epochs, selected the epoch-18 checkpoint using validation loss, and skipped
no audio files.

| Test metric | Result |
|---|---:|
| Accuracy | 69.3% |
| Balanced accuracy | 59.9% |
| Macro precision | 57.9% |
| Macro recall | 59.9% |
| Macro F1 | 57.8% |
| Weighted F1 | 68.7% |
| Top-3 accuracy | 90.7% |
| Cross-entropy loss | 0.924 |
| Majority-class accuracy reference | 39.6% |
| Uniform-guess balanced-accuracy reference | 12.5% |

These are encouraging results for this eight-species task, not a direct
like-for-like comparison with the 500-per-species experiment: the full-data
test split is much larger and reflects the natural class imbalance, while the
smaller run's test split had equal support per species. Macro F1 and balanced
accuracy are therefore important alongside accuracy. The report, checkpoint,
per-class scores, plots, and confusion matrices are in
`class_list_all_data_outputs/`.

The selected log-mel features were stored in a temporary disk-backed cache
instead of being accumulated in RAM. The cache was removed after evaluation;
only the model and reports remain. To reproduce the all-recordings run:

```powershell
python train_cnn.py --class-list class_list.csv --max-clips-per-class 0 --output-dir class_list_all_data_outputs
```

## Reproduce

Install dependencies. For GPU training, use PyTorch and TorchAudio builds
compatible with the machine's CUDA setup:

```powershell
python -m pip install -r requirements.txt
python EDA.py
python train_cnn.py
```

To train only the classes in a CSV and save separate outputs:

```powershell
python train_cnn.py --class-list class_list.csv --max-clips-per-class 500 --output-dir class_list_training_outputs
```

KaggleHub uses its local cache when available; otherwise it may require
internet access and Kaggle authorization. You can pass `--data-dir` to either
script to use a previously downloaded copy. Run EDA before training so
`dataset_report/audio_index.csv` and naming metadata are current.

## Caveats

The labels are primary labels in the supplied metadata; secondary labels are
not modeled here. The selected subset and single-label objective do not fully
represent recordings with multiple audible species. Rare labels excluded by
the support filter need more independent recordings before a meaningful
class-specific score can be reported. The metrics are baseline estimates for
the grouped split and should not be treated as evidence of performance on
unseen habitats or recording equipment.
