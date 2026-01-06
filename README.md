# seismic-event-classifier
A machine learning classifier using Support Vector Machines (SVM) to distinguish between natural earthquakes and man-made explosions based on seismic signal characteristics.
Project Goal

    Classify seismic events as Earthquakes or Explosions using the FastMapSVM framework.

Compare FastMapSVM against a standard baseline model to evaluate performance.

Analyze the impact of different distance metrics (Euclidean vs. Correlation) and embedding dimensions (k)

## Project Structure
```text
seismic_project/
├── data/
│   └── seismic_classifier_dataset.h5   # The processed HDF5 dataset
├── output/
│   ├── models/                         # Trained .pkl models
│   ├── embeddings/                     # FastMap embeddings (.npy)
│   ├── configs/                        # Experiment configurations (.json)
│   └── plots/                          # Generated visualization plots
├── src/
│   ├── __init__.py
│   ├── dataloader.py                   # Data loading & Event-ID splitting
│   ├── distances.py                    # Euclidean & Normalized Cross-Correlation metrics
│   ├── fastmap.py                      # FastMap algorithm implementation
│   ├── classifier.py                   # SVM wrapper & Baseline classifier
│   └── main.py                         # Main experiment execution script
├── requirements.txt                    # Project dependencies
└── README.md       
```

### Project Overview
This project discriminates between **Earthquakes** and **Explosions** using seismic waveform data. We implemented the **FastMapSVM** framework—a method that combines the FastMap dimensionality reduction algorithm with Support Vector Machines (SVM)—and compared it against a standard baseline model.

### Implemented Models
1.  **FastMapSVM (The Core Approach):**
    * **Algorithm:** Projects complex waveform data into a low-dimensional Euclidean space ($k$ dimensions) while preserving domain-specific distances.
    * **Distance Metrics:** Implemented two custom metrics:
        * **Euclidean Distance:** Standard geometric distance.
        * **Normalized Cross-Correlation (NCC):** A domain-specific metric that measures waveform similarity regardless of amplitude scaling.
    * **Dimensions ($k$):** Evaluated across $k \in \{2, 5, 10, 20\}$.

2.  **Baseline Model (The Comparison):**
    * **Algorithm:** Random Forest Classifier.
    * **Features:** Extracts statistical features (Mean, Std, Max, Min) from raw waveforms instead of using embeddings.
    * **Goal:** Provides a performance benchmark to validate if FastMap adds value.

### Preliminary Results (Debug Run)
We performed a "sanity check" run on a small subset of the data (**100 samples**) to verify the pipeline. Even with this tiny dataset, FastMap showed promising results compared to the baseline.

* **Baseline (Random Forest):** 60% Accuracy
* **FastMapSVM (Euclidean, k=10):** **62% Accuracy**

*Note: The debug run produced some "UndefinedMetricWarning" logs because 100 samples are insufficient for the SVM to fully learn class boundaries. These warnings will resolve during the full run.*

### How to Run the Full Experiment
The code is currently set to "Debug Mode" to ensure it runs quickly on CPUs. To run the full experiment on the complete dataset (6,000+ events):

1.  Open `src/main.py`.
2.  **Delete or Comment Out** the Debug Block (Lines ~35–46).
3.  Run the script: `python -m src.main`