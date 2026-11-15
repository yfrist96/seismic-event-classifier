# User Guide

Reference for the inputs, outputs, options and expected behaviour of the code in `src/`.
For step-by-step tasks see [`HOWTO.md`](HOWTO.md); for installation and the commands that
regenerate the paper's results see the top-level [`README.md`](../README.md).

All commands are run from the repository root, as modules (`python -m src.<name>`), because the
code imports `src.*` by absolute name.

## 1. Requirements

| Item | Requirement |
|---|---|
| Python | 3.9+ (tested only on 3.12) |
| Packages (`requirements.txt`) | numpy, h5py, scikit-learn, scipy, matplotlib, joblib, pandas (`pandas` is used only by six scripts in `src/post_review/`: `event_level_predictions`, `export_test_predictions`, `export_advisor_table`, `list_candidate_waveforms`, `p_only_and_distance`, `plot_station_map`) |
| Optional | `cartopy`, only for `plot_station_map` |
| Hardware | CPU only; no GPU. The three splits take about 60 MB on disk; the quickstart peaks at about 0.7 GB of memory. FastMap and the SVM use effectively one core (CPU time equals wall-clock time in the quickstart) |

Why the minimum versions in `requirements.txt` are what they are:

- `scipy.stats.mode(..., keepdims=False)` (used by `src/ablation.py` and three `post_review` scripts)
  needs SciPy 1.9 or later.
- `TSNE(max_iter=...)` in `src/plot_decision_boundary.py` needs scikit-learn 1.5 or later, which
  itself needs Python 3.9 or later.
- `groupby(...).apply(..., include_groups=False)` in `event_level_predictions.py` needs pandas 2.2
  or later.

Tested environment for the timings in this guide and in `HOWTO.md`: Python 3.12.13, NumPy 2.4.3,
SciPy 1.17.1, scikit-learn 1.8.0, h5py 3.16.0, matplotlib 3.10.8, joblib 1.5.3, pandas 3.0.2,
on a laptop CPU.

## 2. Input data

### 2.1 HDF5 files

The pipeline reads three files, `dataset_train.h5`, `dataset_val.h5` and `dataset_test.h5`, from
`data/` (Zenodo, DOI [10.5281/zenodo.24137901](https://doi.org/10.5281/zenodo.24137901)). Each
file holds four aligned datasets whose first dimension is the number of windows `N`:

| Key | Shape | dtype | Content |
|---|---|---|---|
| `X` | `(N, 600, 3)` | float32 | Waveform window: 600 samples x 3 components, channel order Z, N, E |
| `y` | `(N,)` | int64 | Label: **0 = Earthquake, 1 = Explosion** |
| `event_id` | `(N,)` | int32 | Source event ID (shared by all windows of one event) |
| `station` | `(N,)` | bytes (`S10`) | Station code, e.g. `b"GEM"` |

Only `X` and `y` are required by the models; `src/dataloader.py` also reads `event_id` when
present. Within `src/`, `station` is read only by the `post_review` scripts. The files carry no
attributes and no phase field (see `data/README.md` and `data/phase_labels.csv` for P/S labels).

Splits are disjoint at the event level: no event contributes windows to more than one split.
Window counts: train 6,334 (2,850 EQ / 3,484 EX), validation 1,444 (650 / 794), test 1,327
(565 / 762).

### 2.2 Preprocessing assumed upstream

The models do no filtering, resampling or normalization of waveforms; they assume windows prepared
as by `SeismicDataset/400_extract_event_wfs_antelope.py`:

1. Resample each trace to **200 Hz**.
2. Bandpass **1–20 Hz** (ObsPy `trace.filter("bandpass", freqmin=1, freqmax=20)`), with a 5 s
   lead-in that is trimmed off afterwards so filter transients fall outside the window.
3. Trim to start **0.5 s before an analyst phase pick** (P or S) at that station.
4. **Normalize each channel to its peak absolute amplitude** (ObsPy `trace.normalize()`), so values
   lie in [-1, 1].
5. Keep the first 600 samples (3.0 s). A missing component, or a trace shorter than 600 samples,
   is left as zeros.

New data must follow the same recipe to be used with a trained model (see `HOWTO.md`, section 6).

## 3. Features

`src/ablation.py` defines the two feature transforms (identical copies of `to_frequency_domain`
also exist in the plotting and export scripts):

| Function | Input | Output | Steps |
|---|---|---|---|
| `to_frequency_domain(X)` | `(N, T, C)` array | `(N, (T//2+1)*C)` float64 | `np.fft.rfft` along time; magnitude; `log10(. + 1e-6)`; flatten; L2-normalize each row (`sklearn.preprocessing.normalize`) |
| `flatten_time_domain(X)` | `(N, T, C)` | `(N, T*C)` | reshape only |

For 600-sample, 3-channel windows the FFT feature vector has 301 x 3 = **903** entries. The flatten
is row-major, so channels are interleaved: entry `3*j + c` is frequency bin `j` (frequency
`j * 200/600 = j/3` Hz, 0–100 Hz) of channel `c` (0 = Z, 1 = N, 2 = E).

## 4. API reference

### 4.1 `src/dataloader.py`

- `load_split(file_path)` → `(X, y, event_ids)`. Reads `X` and `y` fully into memory; `event_ids`
  is `None` if the file has no `event_id` key. Raises `FileNotFoundError` if the file is missing.
  Prints the loaded shapes.
- `load_data(data_dir="./seismic_datasets")` → `((X_train, y_train), (X_val, y_val), (X_test, y_test))`.
  Loads `dataset_{train,val,test}.h5` from `data_dir`. The default directory is **not** the
  repository's `data/`, so pass `"data"` explicitly. Event IDs are dropped.

### 4.2 `src/distances.py`

Each function takes two arrays and returns a float. All except `ncc_distance` flatten their inputs.

| Function | FastMap name | Definition in the code |
|---|---|---|
| `euclidean_distance` | `'euclidean'` | L2 norm of the difference |
| `cosine_distance` | `'cosine'` | 1 − cosine similarity; returns 1.0 if either vector is all zeros |
| `ncc_distance` | `'ncc'` | 1 − mean over channels of max \|normalized cross-correlation\|; reshapes 1-D input to `(600, 3)`, so it only works on raw 600 x 3 windows or their 1,800-value flattening |
| `wasserstein_fourier_distance` | `'wasserstein_fourier_distance'` | Each vector divided by its sum, then `scipy.stats.wasserstein_distance(u, v)`; the vectors are passed as sample values, not as weights over frequency bins |
| `likelihood_ratio_distance` | `'likelihood_ratio_distance'` | Symmetrized sum of `r − log r − 1` over element ratios; shifts both inputs to be positive if either has negative values |
| `kulczynski_distance` | `'kulczynski'` | 1 − ½(Σmin/Σu + Σmin/Σv); 1.0 if either sum is 0 |
| `soergel_distance` | `'soergel'` | Σ\|u−v\| / Σmax(u, v); 0.0 if the denominator is 0 |
| `lorentzian_distance` | `'lorentzian'` | Σ log(1 + \|u−v\|) |
| `canberra_distance` | `'canberra'` | `scipy.spatial.distance.canberra` |

The paper's ablation uses only Euclidean, Lorentzian and Canberra.

### 4.3 `src/fastmap.py` — `FastMap`

```python
FastMap(n_components=2, dist_func='euclidean')
```

- `dist_func` must be one of the FastMap names in the table above, otherwise `ValueError`.
- `fit_transform(X)` → `(N, n_components)` float64. `X` is a 2-D array (one object per row).
  For each dimension it picks a random object, takes pivot A as the object farthest from it and
  pivot B as the object farthest from A (distances measured in the residual space left by earlier
  dimensions), then projects every object with the cosine law. Prints one progress line per
  dimension. There is no separate `fit()`.
- `transform(X_new)` → `(M, n_components)`. Deterministic: it uses the stored pivots
  (`pivots`), their distances (`pivot_dists`) and their embeddings (`pivot_embeddings`).
- Cost: `fit_transform` makes 3N distance evaluations per dimension plus a pure-Python
  O(N k²) loop; `transform` makes 2 distance evaluations per object per dimension. Measured on
  a laptop CPU for k = 120: 28 s to fit 6,334 FFT vectors, about 2.5 ms per window to transform.
- The pivots are stored as views of the rows of `X`; do not modify the fitted array in place.

### 4.4 `src/classifier.py`

**`BayesThresholdCalibrator()`** — Bayes-optimal decision threshold on SVM decision values.

- `fit(decision_vals, y, priors=None)` → `self`. Fits a normal distribution to each class's values
  with `scipy.stats.norm.fit` (maximum likelihood) and finds the threshold `x` where
  `priors[0]·N(x; mu_eq, sigma_eq) = priors[1]·N(x; mu_ex, sigma_ex)`. `y` must use 0/1 labels.
  `priors` is a dict `{0: P(EQ), 1: P(EX)}`; the default is the class frequencies in `y`. The
  priors are not checked (only their ratio affects the threshold).
- Root search: `brentq` between the two class means. With extreme priors the crossing lies
  beyond a class mean, so the bracket is then widened outward one standard deviation at a time
  (up to 20) until it contains a sign change; the weighted densities cross at most twice, and this
  finds the crossing nearest the bulk of the data. If no crossing exists (one weighted density
  dominates everywhere), it falls back to the point between the means where the two weighted
  densities are closest. Example (validation fits of the quickstart model): P(EX) = 0.99 gives
  −1.077, the same as `src/plot_threshold_calibration.py`.
- `predict(decision_vals)` → `(decision_vals > threshold).astype(int)`.
- Attributes after `fit`: `threshold`, `mu_eq`, `sigma_eq`, `mu_ex`, `sigma_ex`, `priors`,
  `is_calibrated`.

**`FastMapSVMClassifier(k=2, dist_func='euclidean', C=1.0, kernel='rbf')`** — convenience wrapper
holding `fastmap` (a `FastMap`), `scaler` (`StandardScaler`), `svm`
(`SVC(C=C, kernel=kernel, probability=True)`) and `calibrator`. Methods: `fit(X, y)`,
`calibrate(X_cal, y_cal, priors=None)`, `predict(X)` (uses the calibrated threshold once
`calibrate` has been called). It cannot set `gamma` or `class_weight`, so it does **not** reproduce
the selected model; the scripts use only its `fastmap` attribute and build the SVM themselves.

**`BaselineClassifier()`** — Random Forest (`n_estimators=100, random_state=42`) on 12 features
(mean, std, max, min of each channel of the raw window). Methods `fit(X, y)` and `predict(X)` take
raw `(N, 600, 3)` windows.

### 4.5 `src/ablation.py` (functions)

- `run_single_experiment(X_train, y_train, X_val, y_val, X_test, y_test, dist, k, exp_name, exp_idx)`
  → `(report, best_params, best_val_accuracy)`. Fits FastMap on the train features, selects SVM
  hyper-parameters on validation accuracy, retrains the SVM on train + validation embeddings and
  evaluates on test. It **writes** an embedding pair and a model file under
  `ABLATION_CONFIG["output_dir"]` (whose `embeddings/` and `models/` folders must already exist;
  `run_ablation` creates them).
- `run_ensemble(X_train, y_train, X_val, y_val, X_test, y_test, dist, k, best_params, n_models, ...)`
  trains `n_models` voters, each with its own FastMap fitted on train + validation, and returns a
  `classification_report` dict for the majority vote. It writes nothing and keeps no models.
- `run_ablation()` runs the whole sweep (section 6).

## 5. Configuration: `src/ablation_config.py`

`ABLATION_CONFIG` is a plain dict read by `src/ablation.py` only. Edit it to change the sweep.

| Key | Default | Effect |
|---|---|---|
| `distances` | `["euclidean", "lorentzian", "canberra"]` | FastMap distance names to sweep |
| `dimensions` | `[2, 10, 30, 60, 80, 120]` | FastMap `k` values |
| `use_fft` | `[True, False]` | Feature domains: FFT features and/or flattened raw waveform |
| `use_ensemble` | `[True, False]` | **Not read by any code**: every configuration runs both single and ensemble |
| `ensemble_n_models` | `5` | Voters per ensemble |
| `seed` | `42` | NumPy seed set once at the start of the sweep, so reruns are repeatable; `None` for unseeded. The published results came from an unseeded run |
| `svm_param_grid` | 12 combinations | RBF: `C` ∈ {1000, 5000, 10000, 20000}, `gamma=0.01`, `class_weight` ∈ {`'balanced'`, `None`}; polynomial: `C=1000`, `gamma=0.01`, `degree` ∈ {2, 3}, `coef0` ∈ {0, 1}, `class_weight='balanced'` |
| `output_dir` | `<repo>/output/ablation` | Where results are written |
| `data_dir` | `<repo>/data` | Where the HDF5 splits are read |

Hyper-parameters are chosen by validation accuracy; the first combination in `ParameterGrid` order
wins ties. For the selected FFT + Euclidean k = 120 configuration the chosen values are
`C=1000, gamma=0.01, kernel='rbf', class_weight='balanced'` (stored under `params` in the saved
model), the same values hard-coded in the analysis scripts.

## 6. Scripts

`output/` and `data/` are not tracked by git; scripts create the `output/` sub-directories they
need. Every script writes to fixed paths and overwrites existing files without asking. The
scripts below were not timed for this guide: run times are estimates from the cost of the same
steps (FastMap fit, transform, SVM) measured with the quickstart and ensemble in `HOWTO.md`.

### 6.1 `src/`

| Command | Reads | Writes | Run time |
|---|---|---|---|
| `python -m src.ablation` | `data/*.h5` | `output/ablation/models/ablation_{FFT,time}_{dist}_k{k}_single.joblib` (36 single models; ensembles are not saved); `embeddings/{exp}_train.npy`, `{exp}_test.npy`; `results/{exp}_report.json` (72), `results/baseline_report.json`, `results/ablation_summary_partial.json` (after every configuration), `results/ablation_summary.json` | Not timed; estimated one to a few hours. Prints progress and an ETA |
| `python -m src.plot_ablation` | `output/ablation/results/ablation_summary.json` | 13 figures in `output/ablation/plots/` (`.png` 300 dpi + `.pdf`; list in README) | Seconds (estimate) |
| `python -m src.plot_decision_boundary` | `data/*.h5` | `output/decision_boundary_plots/`: `tsne_clusters`, `decision_histogram`, `decision_boundary_best2dims`, `misclassified_tsne`, `average_spectra`, `spectral_difference`, `example_waveforms`, `example_spectra` (`.png` + `.pdf`). **Overwrites the paper's Fig. 2 files** `example_waveforms`/`example_spectra` in that directory | About 1–2 min (estimate) |
| `python -m src.plot_gmm_analysis` | `data/*.h5` | `output/decision_boundary_plots/decision_function_analysis.{png,pdf}`, `decision_gaussian_histogram.{png,pdf}`; statistics printed | About 1 min (estimate; same model as the quickstart) |
| `python -m src.plot_threshold_calibration` | `data/*.h5` | `output/decision_boundary_plots/threshold_calibration.{png,pdf}`, `threshold_calibration_results.json`; table printed | About 1 min (estimate; same model as the quickstart) |

Model protocols differ between scripts, which is why they report different single-model test
accuracies:

| Where | FastMap fitted on | SVM trained on | Seed | Test accuracy |
|---|---|---|---|---|
| `ablation.py`, single | train | train + val (after tuning on val) | 42 (published run: none) | 92.31% (published run) |
| `ablation.py`, ensemble voters | train + val | train + val | 42 (published run: none) | 93.90% (published run) |
| `plot_decision_boundary.py` | train + val | train + val | 42 | printed by the script |
| `plot_gmm_analysis.py`, `plot_threshold_calibration.py`, `examples/quickstart.py` | train | train | 42 | 92.09% |

### 6.2 `src/post_review/`

| Command | Reads | Writes | Run time |
|---|---|---|---|
| `python -m src.post_review.export_test_predictions` | `data/*.h5`, `output/ablation/models/ablation_FFT_euclidean_k120_single.joblib` | `output/test_predictions.csv`: one row per test window with `window_index, event_id, station, true_label, true_class, single_pred_label/_class, single_decision_score, calibrated_pred_label/_class, ensemble_pred_label/_class, ensemble_vote_confidence, ensemble_mean_decision_score`. Trains a fresh 5-voter ensemble (NumPy seed 42; the published run was unseeded) | About 4 min (estimate from the measured ensemble timing in `HOWTO.md`, section 3) |
| `python -m src.post_review.event_level_predictions` | `output/test_predictions.csv` | `output/event_level_predictions.csv` (one row per event; majority vote for single, calibrated and ensemble predictions); summary printed | Seconds |
| `python -m src.post_review.export_advisor_table` | saved single model, `data/dataset_val.h5`, `data/dataset_test.h5`, `output/test_predictions.csv` | `output/advisor_table.csv` (validation rows: labels only; test rows: ensemble mean decision score thresholded with a calibrator fitted on the single model's validation scores) | Seconds |
| `python -m src.post_review.p_only_and_distance [--draws 200] [--seed 0]` | `data/*.h5`, `output/test_predictions.csv`, `SeismicDataset/gsi_event_match.csv`, `SeismicDataset/station_metadata.csv` | `data/phase_labels.csv` (**writes into `data/`**), `output/p_only_distance_results.json` | Seconds |
| `python -m src.post_review.time_distances [--pairs 300] [--repeats 200] [--fit-n 1000] [--fit-k 10]` | `data/dataset_test.h5` | `output/timing/distance_timing.json` (per-call times of `euclidean_distance` on log-spectra and `ncc_distance` on waveforms, FastMap fit and transform times with each, and the software versions) | About 1 min with the defaults; several minutes with `--fit-k 120` |
| `python -m src.post_review.list_candidate_waveforms` | `data/*.h5` | `window_inventory.csv`, `event_inventory.csv`, `top_candidates.csv` in the examples folder under `output/` (the script's `OUTPUT_DIR`) | Seconds |
| `python -m src.post_review.plot_advisor_examples` | `data/dataset_train.h5` | Fig. 2: `example_waveforms.{pdf,png}`, `example_spectra.{pdf,png}` in the examples folder under `output/` (the default `output_dir` of `plot_example_pair`), **not** in `output/decision_boundary_plots/`. Fig. 6 comes from calling `plot_spectra_and_ratio_pair(...)` (see README), which writes `output/decision_boundary_plots/spectra_and_ratio_by_channel.{pdf,png}` | Seconds |
| `python -m src.post_review.extract_example_waveforms` | `data/*.h5` | Example waveform and spectrum figures in the examples folder under `output/` (see `src/post_review/README.md`); it can overwrite files of the same name there | Seconds |
| `python -m src.post_review.plot_figure5_updated` | `output/ablation/results/*_report.json` | `output/ablation/plots/overall_ranking.{png,pdf}`: the same files as `plot_ablation` (Supplementary Fig. S2), which it overwrites | Seconds |
| `python -m src.post_review.plot_station_map` | `output/station_list.csv`, `data/mainfaultsisrael/Main_Faults.shp` | `output/station_map.{png,pdf}`. Needs `cartopy`; neither input is produced by code in this repository | Seconds |

`event_level_predictions`, `export_advisor_table` and `p_only_and_distance` need
`output/test_predictions.csv`, which needs the saved model from `src.ablation`.

### 6.3 `SeismicDataset/`

Scripts that built the dataset from the GSI waveform archive (extraction, station selection,
event-level split); they are not needed to use the released HDF5 files. The extraction script
needs ObsPy, pandas, tqdm and access to the archive. `build_training_dataset.py` and
`split_by_event.py` need only h5py, NumPy and scikit-learn and read and write fixed file names in
the current directory.

## 7. Expected behaviour

**Randomness.** FastMap draws one random starting object per dimension from NumPy's global RNG
(`np.random.randint` in `FastMap.fit_transform`). `src/ablation.py` and the ensemble in
`export_test_predictions.py` seed it once (42) before training, so their reruns are repeatable;
the published results came from unseeded runs, so a rerun differs slightly from them (two
independent fits of the best ensemble gave 93.90% and 93.52%). `src/plot_gmm_analysis.py`,
`src/plot_threshold_calibration.py` and `src/plot_decision_boundary.py` call `np.random.seed(42)`
before fitting, and `examples/quickstart.py` does the same (option `--seed`). With the released
data, the quickstart run documented in `HOWTO.md` reproduces the 92.09% test accuracy and the
thresholds that the README reports from the two analysis scripts. `FastMap.transform`, the
`StandardScaler` and `SVC` (without `probability=True`) are deterministic.

**Default decision rule.** `SVC.predict` returns Explosion (1) when `decision_function > 0`.
`BayesThresholdCalibrator.predict` returns 1 when the value exceeds the calibrated threshold.

**Ensemble.** Five voters vote by majority (`scipy.stats.mode`), so there are no ties. In
`ablation_summary.json` an ensemble row's `val_accuracy` is that of the single model whose
hyper-parameters it reuses; the ensemble itself is not evaluated on validation.

**Event-level votes.** `event_level_predictions.py` and `p_only_and_distance.py` take the mode of
the window labels; on a tie `scipy.stats.mode` returns the smaller label, i.e. Earthquake.

**Saved models.** A model file is a joblib-pickled dict with keys `fastmap`, `scaler`, `svm`,
`params`, `val_accuracy`, `test_accuracy`. Unpickling needs `src.fastmap` importable (run from the
repository root). The authors' ablation model files (not distributed with the repository) were pickled
with scikit-learn 1.5.2; loading such a file with another version emits
`InconsistentVersionWarning`. With scikit-learn 1.8.0 the FFT + Euclidean k = 120 model still
reproduces its stored test accuracy (0.9231). Load only pickle
files you trust.

**No input validation.** The code does not check window length, sampling rate, channel order,
normalization or label values. Labels must be the integers 0 and 1: the calibrator selects classes
with `y == 0` and `y == 1`, and `src/ablation.py` reads the report keys `'0'` and `'1'`.

**Console output.** `load_split` prints each file's shapes, and `FastMap.fit_transform` prints one
line per dimension (120 lines for k = 120).
