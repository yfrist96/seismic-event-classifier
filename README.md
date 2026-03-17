# Seismic Event Classifier

A machine learning system for classifying seismic events as **Earthquakes** or **Explosions** using FastMap dimensionality reduction, FFT-based feature engineering, and SVM classification with ensemble voting.

## Project Goal

- Classify seismic waveforms as earthquakes or explosions
- Evaluate the impact of frequency-domain features (FFT) vs raw time-series
- Conduct a systematic ablation study across distance metrics, embedding dimensions, and ensemble methods

## Project Structure

```text
seismic-event-classifier/
├── data/
│   ├── dataset_train.h5                  # Training split (6,334 samples)
│   ├── dataset_val.h5                    # Validation split (1,444 samples)
│   └── dataset_test.h5                   # Test split (1,327 samples)
├── output/
│   └── ablation/
│       ├── models/                       # Saved models (.joblib)
│       ├── embeddings/                   # FastMap embeddings (.npy)
│       ├── results/                      # Per-experiment JSON + ablation_summary.json
│       ├── plots/                        # All comparison plots
│       └── tb_logs/                      # TensorBoard logs
├── src/
│   ├── config.py                         # Main experiment configuration
│   ├── ablation_config.py                # Ablation study configuration
│   ├── ablation.py                       # Ablation study runner
│   ├── plot_ablation.py                  # Plot generation from ablation results
│   ├── main.py                           # Single-config experiment runner
│   ├── dataloader.py                     # HDF5 data loading
│   ├── distances.py                      # 9 distance metric implementations
│   ├── fastmap.py                        # FastMap algorithm
│   ├── classifier.py                     # FastMapSVM + Baseline classifiers
│   └── visualize.py                      # Embedding visualization (2D, t-SNE)
├── Colab_Results.ipynb                   # Early Colab experiments
├── requirements.txt
└── .gitignore
```

## Pipeline

```
Raw Seismic Data (N, 600, 3)     600 timesteps x 3 channels (Z, N, E)
        |
   [FFT branch]                  [Time branch]
        |                              |
  rfft -> |magnitude|            flatten (N, 1800)
  -> log10 -> L2 norm
  -> flatten (N, 903)
        |                              |
        +--------- FastMap (N, k) -----+
                       |
                 StandardScaler
                       |
              SVM (tuned on val set)
                       |
             Retrain on train + val
                       |
            [Single] or [Ensemble x5]
                       |
              Final eval on test set
```

### Data Split Protocol

| Split | Samples | Role |
|-------|---------|------|
| **Train** | 6,334 | FastMap fitting + SVM training |
| **Validation** | 1,444 | Hyperparameter selection (no cross-validation) |
| **Test** | 1,327 | Final evaluation only |

After selecting the best hyperparameters on the validation set, the model is retrained on **train + val combined** before evaluating on test. The test set is never seen during tuning.

## Ablation Study

We swept across 4 axes to isolate the contribution of each design choice:

| Axis | Values |
|------|--------|
| Feature domain | FFT (log-magnitude spectrum) vs raw time-series |
| Distance metric | Euclidean, Lorentzian, Canberra |
| FastMap dimensions (k) | 2, 10, 30, 60, 80, 120 |
| Model type | Single SVM vs Ensemble (5 voters, majority vote) |

**Total: 72 experiments + 1 baseline** (36 configs x single/ensemble + Random Forest baseline)

### Results: Top Configurations

| Rank | Config | Test Acc | Val Acc |
|------|--------|----------|---------|
| 1 | FFT + Euclidean, k=120, ensemble | **93.75%** | 91.62% |
| 2 | FFT + Euclidean, k=80, ensemble | 92.61% | 90.86% |
| 3 | FFT + Euclidean, k=60, ensemble | 92.54% | 88.37% |
| 4 | FFT + Euclidean, k=120, single | 91.64% | 91.62% |
| 5 | FFT + Euclidean, k=80, single | 91.03% | 90.86% |
| - | Baseline (Random Forest) | 62.62% | 64.54% |

### Key Findings

**1. FFT is the single most important factor**

FFT features outperform time-domain features by 10-15 percentage points across all distance metrics and k values. The best time-domain result (Euclidean, k=120, ensemble: 80.18%) is still far below the worst competitive FFT result.

This works because FFT makes the representation **shift-invariant** -- seismic events have variable arrival times, so time-domain distances are dominated by alignment noise. FFT captures *what* energy is present, not *when* it arrives.

**2. Euclidean distance dominates on FFT features**

| Distance | Best Test Acc (ensemble) |
|----------|------------------------|
| Euclidean | 93.75% |
| Canberra | 89.30% |
| Lorentzian | 87.04% |

Euclidean outperforms alternatives by 4-7% in the FFT domain. Lorentzian and Canberra, which were designed for robustness to outliers and relative differences, don't help when the features are already well-normalized by the FFT+L2 pipeline.

**3. More dimensions help, with diminishing returns**

| k | Euclidean FFT (ensemble) |
|---|--------------------------|
| 2 | 60.14% |
| 10 | 86.36% |
| 30 | 90.81% |
| 60 | 92.54% |
| 80 | 92.61% |
| 120 | 93.75% |

k=2 is essentially useless. The jump from k=10 to k=30 is large (+4.5%), then gains taper off. k=80-120 is the sweet spot.

Notably, Lorentzian *degrades* at higher k (peaks at k=30, drops at k=60+), suggesting it doesn't scale well with many dimensions.

**4. Ensemble consistently adds 2-3%**

Ensemble voting (5 independent FastMap projections with different random pivots) improves accuracy on every single configuration. The boost ranges from 1-8%, with the largest gains on weaker base models.

**5. Validation accuracy tracks test accuracy well**

The val-vs-test scatter plot shows strong correlation, confirming the validation set is a reliable proxy for generalization and the model is not overfitting to the validation split.

### Per-Class Performance

Best model (FFT + Euclidean, k=120, ensemble):
- Earthquake F1: 0.928
- Explosion F1: 0.945

The model is slightly better at detecting explosions, likely because their spectral signatures (sharp high-frequency peaks) are more distinctive than the diffuse energy patterns of earthquakes.

## Generated Plots

All plots are saved to `output/ablation/plots/` by running `python -m src.plot_ablation`:

| Plot | Description |
|------|-------------|
| `fft_vs_time.png` | FFT vs time-domain accuracy by distance metric |
| `accuracy_vs_k_fft.png` | Accuracy scaling with k for each distance (FFT) |
| `accuracy_vs_k_time.png` | Same for time domain |
| `single_vs_ensemble.png` | Scatter: single vs ensemble accuracy for all configs |
| `ensemble_gain.png` | Percentage gain from ensemble by distance and k |
| `f1_per_class.png` | Per-class F1 scores for best configs |
| `heatmap_fft_single.png` | Accuracy heatmap: distance x k (FFT, single) |
| `heatmap_fft_ensemble.png` | Same for ensemble |
| `heatmap_time_single.png` | Accuracy heatmap (time domain, single) |
| `heatmap_time_ensemble.png` | Same for ensemble |
| `val_vs_test.png` | Validation vs test accuracy (overfitting check) |
| `overall_ranking.png` | Top 15 experiments ranked |
| `domain_gap.png` | FFT advantage in percentage points at each k |

## How to Run

### Setup

```bash
conda create -n seismic python=3.12 -y
conda activate seismic
pip install -r requirements.txt
```

Place the HDF5 data files in `data/`.

### Run Ablation Study

```bash
python -m src.ablation
```

Progress is printed with ETA after each experiment. Partial results are saved incrementally.

### Generate Plots

```bash
python -m src.plot_ablation
```

### Run Single Experiment

Edit `src/config.py` to set your desired configuration, then:

```bash
python -m src.main
```

### TensorBoard

```bash
tensorboard --logdir output/ablation/tb_logs
```

### Model Loading

```python
import joblib
model = joblib.load("output/ablation/models/ablation_FFT_euclidean_k120_single.joblib")
svm = model['svm']
scaler = model['scaler']
fastmap = model['fastmap']
```

## Distance Metrics

| Metric | Description |
|--------|-------------|
| Euclidean | Standard L2 norm |
| Lorentzian | Log-based, robust to outliers: sum(log(1 + \|u-v\|)) |
| Canberra | Relative differences: sum(\|u-v\| / (\|u\|+\|v\|)) |
| Cosine | 1 - cosine similarity |
| NCC | Normalized cross-correlation (waveform alignment) |
| Wasserstein | Earth mover's distance on normalized spectra |
| Kulczynski | Intersection-based similarity |
| Soergel | Jaccard-like: sum(\|u-v\|) / sum(max(u,v)) |
| Likelihood Ratio | KL-divergence based |

## Requirements

- Python 3.12+
- numpy, h5py, scikit-learn, scipy, matplotlib, tensorboard, tensorboardX, joblib
