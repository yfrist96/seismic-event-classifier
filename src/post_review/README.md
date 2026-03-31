# Post-Review Scripts

Scripts created after the advisor review meeting to prepare deliverables and update paper figures. All scripts should be run from the project root directory.

## Scripts

### extract_example_waveforms.py

Extracts 3 example waveforms (2 earthquakes, 1 explosion) with time-domain and FFT spectra plotted side by side. Also exports raw data as CSV for re-plotting.

```bash
python -m src.post_review.extract_example_waveforms
```

**Output:** `output/example_waveforms_for_ittai/`
- `example_{1,2,3}_{earthquake,explosion}.png` and `.pdf` — side-by-side waveform + spectrum plots
- `waveform_{1,2,3}_{earthquake,explosion}.csv` — raw time-domain data (time, Z, N, E)
- `spectrum_{1,2,3}_{earthquake,explosion}.csv` — FFT data (freq, Z/N/E log-spectral amplitude)

### export_test_predictions.py

Runs inference on the full test set using both the saved single model and a freshly trained 5-voter ensemble. Exports predictions with metadata and confidence scores.

```bash
python -m src.post_review.export_test_predictions
```

**Output:** `output/test_predictions.csv`

Columns: `window_index`, `event_id`, `station`, `true_label`, `true_class`, `single_pred_label`, `single_pred_class`, `single_decision_score`, `ensemble_pred_label`, `ensemble_pred_class`, `ensemble_vote_confidence`, `ensemble_mean_decision_score`

### plot_figure5_updated.py

Generates the updated Figure 5 (overall ranking) showing top 5 FFT models, the best time-series model, and the Random Forest baseline.

```bash
python -m src.post_review.plot_figure5_updated
```

**Output:** `output/ablation/plots/overall_ranking.png` and `.pdf`
