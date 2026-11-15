# How-to guides

Task-oriented tutorials for typical uses of this repository. Inputs, outputs and options are
documented in [`USER_GUIDE.md`](USER_GUIDE.md). Installation and the commands that regenerate the
paper's figures and tables are in the top-level [`README.md`](../README.md) ("How to Run").

Run everything from the repository root, with the packages in `requirements.txt` installed. The
code imports modules as `src.*`, so use `python -m ...`.

## 1. Get the data and check that it loads

1. Download `dataset_train.h5`, `dataset_val.h5` and `dataset_test.h5` (about 60 MB, CC BY-NC 4.0)
   from the Zenodo record [10.5281/zenodo.24137901](https://doi.org/10.5281/zenodo.24137901).
2. Put them directly in `data/` (`data/dataset_train.h5`, etc.), the path every script reads.
3. Check them:

   ```bash
   python -c "
   import h5py, numpy as np
   for s in ('train', 'val', 'test'):
       with h5py.File(f'data/dataset_{s}.h5', 'r') as f:
           print(s, {k: f[k].shape for k in f}, 'EQ/EX =', np.bincount(f['y'][:]))
   "
   ```

   which should print:

   ```text
   train {'X': (6334, 600, 3), 'event_id': (6334,), 'station': (6334,), 'y': (6334,)} EQ/EX = [2850 3484]
   val {'X': (1444, 600, 3), 'event_id': (1444,), 'station': (1444,), 'y': (1444,)} EQ/EX = [650 794]
   test {'X': (1327, 600, 3), 'event_id': (1327,), 'station': (1327,), 'y': (1327,)} EQ/EX = [565 762]
   ```

## 2. Train and evaluate a single model (quickstart)

`examples/quickstart.py` computes FFT features, fits FastMap (k = 120, Euclidean) and
`SVC(C=1000, gamma=0.01, kernel='rbf', class_weight='balanced')` on the training split, evaluates
on validation and test, and calibrates the threshold on the validation decision values.

```bash
python -m examples.quickstart --data-dir data --out-dir /path/to/new_dir --prior-ex 0.95
```

| Option | Default | Meaning |
|---|---|---|
| `--data-dir` | `data` | Directory holding the three `dataset_*.h5` files |
| `--out-dir` | new directory under the system temp dir | Where all outputs go (created if missing) |
| `--prior-ex` | none | Deployment explosion prior P(EX) in (0, 1); adds the recalibrated threshold to the report |
| `--seed` | 42 | NumPy seed for FastMap pivot selection |
| `--verbose` | off | Show FastMap's per-dimension progress lines |

It writes only into `--out-dir` (Python may also add a git-ignored `examples/__pycache__/`):

| File | Content |
|---|---|
| `model.joblib` | dict with `fastmap`, `scaler`, `svm`, `params`, `val_accuracy`, `test_accuracy`, `calibrator` |
| `decision_values.npz` | `val`, `y_val`, `test`, `y_test`: SVM decision values and labels |
| `test_predictions.csv` | per test window: index, event ID, true label, decision value, predictions at t = 0, at the validation-prior threshold and (with `--prior-ex`) at the deployment threshold |
| `summary.json` | accuracies, thresholds, recalls and timings |

Example output on a laptop CPU (wall-clock 39 s; peak memory about 0.7 GB; the `Loaded ...`
lines and output paths are omitted):

```text
[1] Loading data
    train:  6334 windows  (EQ 2850, EX 3484)
    val  :  1444 windows  (EQ 650, EX 794)
    test :  1327 windows  (EQ 565, EX 762)

[2] Computing FFT features
    feature matrix (train): (6334, 903)

[3] Fitting FastMap (k=120, euclidean) on the training split
    fit_transform(train): 28.4 s; transform(val + test): 7.0 s

[4] Training SVC{'C': 1000, 'gamma': 0.01, 'kernel': 'rbf', 'class_weight': 'balanced'}
    trained in 0.5 s; 2995 support vectors

[5] Default threshold (t = 0)
    validation accuracy: 0.9093
    test accuracy:       0.9209
    test confusion matrix [rows true EQ/EX, cols predicted EQ/EX]: [[521, 44], [61, 701]]

[6] Bayes threshold calibration on validation decision values
    Gaussian fits: EQ N(-1.001, 0.752), EX N(+0.977, 0.669)
    validation priors P(EX) = 0.550 -> threshold -0.0345
    test accuracy at this threshold: 0.9209

    Deployment prior P(EX) = 0.950 -> threshold -0.6993
                             EQ recall  EX recall  prior-weighted acc
      default (t = 0)           0.9221     0.9199              0.9201
      calibrated                0.7009     0.9895              0.9751
      gain: +5.50 percentage points; test windows predicted EX: 745 at t = 0, 923 calibrated

[7] Wrote model.joblib, decision_values.npz, test_predictions.csv, summary.json
Total wall-clock time: 37.5 s
```

The test accuracy (92.09%), thresholds (−0.034; −0.699 at P(EX) = 0.95) and gain (+5.50 pp)
match the README's values from `src/plot_gmm_analysis.py` and `src/plot_threshold_calibration.py`,
which use the same protocol and seed; another `--seed` changes them slightly. The paper's 92.31%
single model is the ablation model, retrained on train + validation (`USER_GUIDE.md`, 6.1).

## 3. Train the 5-voter ensemble

`python -m src.ablation` trains an ensemble for every configuration of the sweep but does not
save the voters. To train and keep the selected ensemble (FFT + Euclidean, k = 120, five voters,
each with its own FastMap fitted on train + validation, majority vote), use:

```python
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.ablation import to_frequency_domain
from src.dataloader import load_data
from src.fastmap import FastMap

(X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_data("data")
F_trval = to_frequency_domain(np.vstack([X_tr, X_va]))
y_trval = np.concatenate([y_tr, y_va])
F_test = to_frequency_domain(X_te)

np.random.seed(42)                      # optional; src/ablation.py does not seed
voters = []
for i in range(5):
    fm = FastMap(n_components=120, dist_func="euclidean")
    E = fm.fit_transform(F_trval)       # prints one line per dimension
    sc = StandardScaler().fit(E)
    svm = SVC(C=1000, gamma=0.01, kernel="rbf", class_weight="balanced").fit(sc.transform(E), y_trval)
    voters.append({"fastmap": fm, "scaler": sc, "svm": svm})

votes = np.array([v["svm"].predict(v["scaler"].transform(v["fastmap"].transform(F_test)))
                  for v in voters])
y_pred = (votes.sum(axis=0) >= 3).astype(int)     # majority of 5 voters
print("ensemble test accuracy:", np.mean(y_pred == y_te))
joblib.dump(voters, "/path/to/ensemble.joblib")
```

Example output on a laptop CPU: each voter trains in about 36 s (3.4 min in total, matching the
paper's "about 3 min"); the five voters score the 1,327 test windows in 19 s (about 14 ms per
window); per-voter test accuracies are 92.2–93.2% and the ensemble reaches **94.12%** (published
run: 93.90%; the difference comes from the random pivots). The saved file is 23 MB.

## 4. Classify new windows with a trained model

Any model file with the keys `fastmap`, `scaler` and `svm` works: `model.joblib` from the
quickstart, or `output/ablation/models/ablation_FFT_euclidean_k120_single.joblib` written by
`python -m src.ablation` (the `output/` directory is not tracked by git, so that file exists only
after you run the ablation).

```python
import joblib
import numpy as np
from src.ablation import to_frequency_domain

model = joblib.load("/path/to/model.joblib")           # run from the repository root
X_new = ...                                            # (N, 600, 3) array, see checklist below
Z = model["scaler"].transform(model["fastmap"].transform(to_frequency_domain(X_new)))
d = model["svm"].decision_function(Z)                  # > 0 leans explosion
labels = (d > 0).astype(int)                           # 0 = earthquake, 1 = explosion
labels_cal = model["calibrator"].predict(d)            # quickstart model only (section 5)
```

On a laptop CPU, the FFT, FastMap projection and SVM take 3.6 s for the 1,327 test windows (about
2.7 ms per window). The ablation model files print `InconsistentVersionWarning` when loaded with a
scikit-learn version other than 1.5.2 (`USER_GUIDE.md`, section 7).

Checklist for `X_new` (the model does no preprocessing and no checks; see `USER_GUIDE.md`, 2.2):

- 600 samples at 200 Hz, starting 0.5 s before the phase pick (P or S) at that station;
- three components in the order Z, N, E, as float arrays of shape `(N, 600, 3)`;
- bandpassed 1–20 Hz, with the filter transient outside the window;
- each channel divided by its peak absolute amplitude.

For an ensemble, compute the votes as in section 3 and take the majority.

## 5. Recalibrate the decision threshold for a known class prior

If the explosion share in deployment is known, `BayesThresholdCalibrator` shifts the SVM
threshold (0) without retraining. Fit it on decision values and labels of a held-out labeled set
(here the validation split), not the training data.

From the command line, pass `--prior-ex` to the quickstart (section 2). In code, reusing the
saved decision values:

```python
import numpy as np
from src.classifier import BayesThresholdCalibrator

dv = np.load("/path/to/new_dir/decision_values.npz")
p_ex = 0.90
cal = BayesThresholdCalibrator().fit(dv["val"], dv["y_val"], priors={0: 1 - p_ex, 1: p_ex})
print(f"threshold = {cal.threshold:+.4f}")              # -0.5232 for the run in section 2
y_pred = cal.predict(dv["test"])                       # or the decision values of new windows
```

Things to know:

- The calibration assumes only the class mix changes; the decision-value distribution of each
  class must stay as on the calibration set.
- A larger P(EX) lowers the threshold. Accuracy weighted by the deployment prior rises, but
  earthquake recall falls (0.92 to 0.70 at P(EX) = 0.95 in section 2). If earthquake recall
  matters more than overall accuracy, set the threshold from a target recall instead, e.g.
  `scipy.stats.norm.ppf(0.95, cal.mu_eq, cal.sigma_eq)` keeps about 95% of earthquakes below it
  if the fitted Gaussian holds.
- The class only searches between the two class means. For very large priors (P(EX) = 0.99 here)
  the threshold stops at the earthquake mean (−1.001) instead of the unrestricted crossing
  (−1.077 in `src/plot_threshold_calibration.py`); see `USER_GUIDE.md`, 4.4.

## 6. Retrain on another network's labeled windows

1. **Cut and preprocess windows** as in `USER_GUIDE.md`, 2.2: 200 Hz, 3.0 s from 0.5 s before
   each P or S pick, 1–20 Hz bandpass, per-channel peak normalization, channels Z, N, E.
2. **Label** each window 0 (earthquake) or 1 (explosion); labels must be exactly these integers.
3. **Split by event** (no event may have windows in more than one split) and **write**
   `dataset_train.h5`, `dataset_val.h5`, `dataset_test.h5` with the keys and dtypes of
   `USER_GUIDE.md`, 2.1:

   ```python
   import h5py, numpy as np
   with h5py.File("my_data/dataset_train.h5", "w") as f:
       f.create_dataset("X", data=X.astype(np.float32))           # (N, 600, 3)
       f.create_dataset("y", data=y.astype(np.int64))             # 0 / 1
       f.create_dataset("event_id", data=event_id.astype(np.int32))
       f.create_dataset("station", data=np.array(stations, dtype="S10"))
   # same for dataset_val.h5 and dataset_test.h5
   ```

   Alternatively, write all windows to one file `seismic_classifier_dataset.h5` (same keys) and
   run `python /path/to/repo/SeismicDataset/split_by_event.py` in its directory: it splits the
   event IDs 70/15/15 at random (seed 42, not stratified by class) and writes the three files there.
4. **Train and evaluate** with the paper's hyper-parameters:

   ```bash
   python -m examples.quickstart --data-dir my_data --out-dir /path/to/my_run
   ```

5. **Re-tune if needed.** The hyper-parameters were selected on this dataset. To repeat the
   search, loop over `ParameterGrid(ABLATION_CONFIG["svm_param_grid"])` (from
   `src.ablation_config`), fit an `SVC` on the scaled training embedding for each combination and
   keep the one with the best validation accuracy, as `run_single_experiment` in `src/ablation.py`
   does. To run the full sweep instead, point `data_dir` in `src/ablation_config.py` at your
   files; note that `src/ablation.py` writes to `output/ablation/` and overwrites earlier results.

Retraining also works for other window lengths `T` (the feature length becomes
`(T // 2 + 1) * 3`) or sampling rates, but models trained on this dataset cannot score such
windows (wrong length) or would score them on shifted frequency bins (other rate), and the
hyper-parameters were not tuned for them. `ncc_distance` assumes 600 x 3 windows and
`BaselineClassifier` three channels.
