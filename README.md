# Seismic Event Classifier

A machine learning classifier using **FastMap embeddings + FFT + SVM ensembles** to distinguish between natural earthquakes and man-made explosions based on seismic signal characteristics.

## Project Goal

- Classify seismic events as **Earthquakes** or **Explosions** using the FastMapSVM framework
- Compare FastMapSVM against a baseline model to evaluate performance
- Analyze the impact of different distance metrics and embedding dimensions

## Project Structure
```text
seismic-event-classifier/
├── data/
│   ├── dataset_train.h5              # Training split (HDF5)
│   ├── dataset_val.h5                # Validation split (HDF5)
│   └── dataset_test.h5               # Test split (HDF5)
├── output/
│   ├── models/                       # Saved models (.joblib)
│   ├── embeddings/                   # FastMap embeddings (.npy)
│   ├── results/                      # Classification reports (.json)
│   ├── plots/                        # Visualization plots
│   └── tb_logs/                      # TensorBoard logs
├── src/
│   ├── __init__.py
│   ├── config.py                     # Centralized configuration
│   ├── dataloader.py                 # HDF5 data loading pipeline
│   ├── distances.py                  # Distance metric implementations
│   ├── fastmap.py                    # FastMap algorithm
│   ├── classifier.py                 # SVM wrapper & Baseline classifier
│   ├── main.py                       # Main experiment runner
│   └── visualize.py                  # Embedding visualization (2D, t-SNE)
├── Colab_Results.ipynb               # Google Colab experiment notebook
├── requirements.txt
├── .gitignore
└── README.md
```

## Pipeline

```
Raw HDF5 (N, 600, 3)
  → FFT → Log-magnitude → L2 normalization
    → FastMap embedding (N, k)
      → StandardScaler
        → SVM (tuned on validation set)
          → Retrain on train+val with best params
            → [Single Model] or [Ensemble: 5 voters, majority vote]
              → Final evaluation on test set
```

### Data Split Usage

| Split | Purpose |
|-------|---------|
| **Train** | Train FastMap embeddings and SVM models |
| **Validation** | Hyperparameter tuning (replaces cross-validation on train) |
| **Test** | Final evaluation only — never seen during tuning |

After the best hyperparameters are selected using the validation set, the model is **retrained on train+val combined** before final test evaluation. This maximizes training data while keeping the test set truly unseen.

## Implemented Models

### FastMapSVM (Core Approach)
- Projects FFT-transformed waveform data into a low-dimensional Euclidean space (k dimensions) using FastMap
- Preserves domain-specific distances during dimensionality reduction
- **Distance Metrics:** Euclidean, Lorentzian, Canberra, Cosine, NCC, Wasserstein, Kulczynski, Soergel, Likelihood Ratio
- **Ensemble:** 5 independent FastMap+SVM voters with majority vote for improved stability

### Baseline Model (Comparison)
- Random Forest on statistical features (Mean, Std, Max, Min per channel)
- Provides a performance benchmark to validate FastMap's value

## Results

| Model | Accuracy |
|-------|----------|
| Baseline (Random Forest) | ~62.8% |
| FastMap + Time Correlation | ~66.7% |
| FastMap + FFT + SVM (single, k=60) | ~91.0% |
| FastMap + FFT + SVM (ensemble, k=80) | ~92.8% |

## Model Saving

Trained models are saved as `.joblib` files in `output/models/`, containing:
- FastMap object (with learned pivots)
- StandardScaler
- Trained SVM
- Best hyperparameters
- Validation and test accuracy

Load a saved model:
```python
import joblib
model = joblib.load("output/models/fastmap_FFT_euclidean_k80_tuned.joblib")
svm = model['svm']
scaler = model['scaler']
fastmap = model['fastmap']
```

## TensorBoard

Training metrics are logged to `output/tb_logs/`. To view:
```bash
tensorboard --logdir output/tb_logs
```

Logged metrics:
- Validation accuracy per grid search combination
- Test accuracy and per-class F1 scores (Earthquake/Explosion)
- Hyperparameter configurations via the HParams dashboard

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Place Data
Put the HDF5 files in the `data/` directory:
- `dataset_train.h5`
- `dataset_val.h5`
- `dataset_test.h5`

### 3. Configure
Edit `src/config.py` to set distance metrics, dimensions, and SVM grid search parameters.

### 4. Run Experiments
```bash
python -m src.main
```

### 5. Visualize Embeddings
```bash
python -m src.visualize
```

## Key Insight: The Frequency Domain Breakthrough

Initially, the model relied on raw time-series waveforms. Seismic events have variable start times, so standard distance metrics on raw data failed to account for time shifts — resulting in ~62% accuracy.

By converting signals to **FFT log-magnitude spectrograms**, we made the model **shift-invariant**: it focuses on spectral energy content rather than when the event started.

- **Explosions:** Higher frequency energy, sharp spectral peaks
- **Earthquakes:** Lower frequency, distributed energy

This single change jumped accuracy from ~63% to 91%+.

### What Worked
- **FFT conversion** — eliminated time-shift variability
- **Higher dimensions (k=80–120)** — captured fine spectral details
- **High SVM C values (1000–20000)** — tight decision boundaries for complex frequency clusters
- **Ensemble voting** — improved stability and accuracy

### What Didn't Work
- **Frequency filtering (high-pass/band-pass)** — earthquakes and explosions share many frequencies; hard filters cut useful information
- **Raw time-series data** — stuck at ~60% due to time-shift problem
- **Euclidean distance on time data** — point-to-point comparison fails when peaks don't align
