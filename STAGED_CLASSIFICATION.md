# Acoustic-trait staged bird classifier

## Goal

This experiment tests whether routing recordings through acoustic groups
improves species identification over a direct flat classifier. The data source
is Kaggle's
[`ollypowell/new-zealand-bird-sound`](https://www.kaggle.com/datasets/ollypowell/new-zealand-bird-sound),
the same base dataset used by the existing CNN.

The pipeline uses measurable acoustic features to form groups rather than
assigning groups by taxonomy or hand-picked species lists. It is an experiment:
the grouping and routing strategy is selected on validation recordings, and
the result is compared with a flat classifier on the same untouched test
recordings. It does not assume that a hierarchy will improve accuracy.

## Bird selection and sample limits

By default, species must have at least 100 readable clips and 50 inferred
source recordings in the base dataset. Tree Weta (`weta`) is excluded as a
non-bird. Labels also need both common and scientific names in the supplied
naming map so an unidentified label is not silently treated as a bird.

To keep classes balanced and avoid using multiple crops from one source in the
same species, training selects at most one clip per source recording and at
most 150 source recordings per species. These are configurable command-line
limits. The script writes the selected train, validation, and test recording
and clip counts to `staged_model_outputs/eligible_species.csv`.

The support threshold is applied before sampling. Having 50 source recordings
does not guarantee strong estimates for every class; the split counts and
per-species held-out results should be reviewed before using the model in an
application.

## Acoustic traits and grouping

Each four-second mono clip is summarized with:

- median and variability of spectral centroid, a measure related to spectral
  brightness;
- median and variability of spectral bandwidth;
- median 50% and 85% spectral rolloff frequencies;
- spectral flatness and entropy, rough indicators of noise-like versus
  concentrated/tonal energy;
- dominant-frequency median and variability;
- RMS energy, zero-crossing rate, and spectral flux.

These are objective measurements, not guaranteed species-defining traits.
Pitch, tone, and call patterns vary within a species and can change with
background noise, distance, habitat, and recording equipment. The features
summarize the whole clip and do not yet isolate individual calls.

For each candidate group count, the script standardizes feature values and
clusters species-level feature centroids using K-means, fitted on training
recordings only. A random-forest router predicts those acoustic groups; a
separate random-forest expert predicts species within each group. Validation
macro F1 selects the group count and whether to route to the top one or top two
groups. Routing to two groups is a deliberate hedge against a wrong hard
first-stage decision; expert probabilities are combined with router
probabilities.

The script also trains a flat random-forest classifier on the same features.
This tests whether staging helps beyond using the acoustic features alone.
The existing CNN remains a separate reference experiment; it uses
log-mel spectrograms rather than these summary features.

## Split and evaluation

One clip per species/source is sampled before a seeded five-fold
`StratifiedGroupKFold`. Fold 0 is test, fold 1 is validation, and folds 2–4
are training. Source recording IDs include the collection and strip trailing
crop numbers, so crops from a source cannot cross partitions.

Group count and routing width are chosen using validation macro F1. The chosen
pipeline and flat baseline are then refit on train plus validation data and
evaluated once on the held-out test partition. Compare macro F1, balanced
accuracy, per-species recall, and the flat baseline—not accuracy alone.

This split prevents inferred source-recording leakage but does not guarantee
independence across location, recordist, date, equipment, or collection. A
stronger generalization assessment should hold out an entire collection/site
or use external recordings. The single-label objective also does not model
multiple species sounding at once, and this classifier always chooses among
its supported classes unless the application adds a rejection/unknown rule.

## Observed run

The initial run selected 42 named bird species and 5,438 clips. It selected
five acoustic groups and routed to one group based on validation macro F1.
On the held-out 1,087-clip test split, the staged model reached 12.4% accuracy
and 12.2% macro F1. The flat classifier using the same acoustic features
reached 12.0% accuracy and 11.9% macro F1. Thus staging produced a small
improvement over this feature-based baseline, but not a strong classifier.

The existing CNN report records 19.8% accuracy and 18.8% macro F1 on its
grouped test split. This staged feature-based run does not surpass that
reference. These are baseline experiments, not reliable production
performance estimates; do not describe the staged model as accurate on the
strength of this result. The test data should remain held out while changing
features, group counts, routing, or eligibility thresholds.

## Run

After running EDA to create the audio index:

```powershell
python train_staged_classifier.py
```

For an already downloaded dataset:

```powershell
python train_staged_classifier.py --data-dir <dataset-folder>
```

Optional arguments include `--min-clips`, `--min-recordings`,
`--max-recordings-per-class`, and `--group-counts` (comma-separated candidate
group counts; the default candidates are 3, 5, and 7).

Outputs in `staged_model_outputs/` include:

- `staged_bird_model.joblib` — the fitted router and specialist models;
- `metrics.json` — validation selection and test comparison against the flat
  baseline;
- `eligible_species.csv` — exact split support per selected species;
- `acoustic_groups.csv` — species-to-acoustic-group assignments;
- `staged_per_class_metrics.csv` and `test_predictions.csv` — held-out results;
- `decode_errors.csv` — any selected files that could not be decoded.

The saved model can be loaded with `joblib.load`. To score an audio clip,
resample it to 22.05 kHz mono, extract features with
`extract_acoustic_traits`, and pass the resulting feature row to
`predict_staged_proba` with the saved `staged_model` and
`selected_routing_groups` values.
