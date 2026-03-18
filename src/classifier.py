from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from src.fastmap import FastMap
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


class BayesThresholdCalibrator:
    """Calibrates an SVM decision boundary using Bayes-optimal threshold.

    Fits a Gaussian to each class's decision function values and finds the
    threshold where the weighted posteriors cross. This accounts for class
    prior imbalance — the SVM boundary at 0 assumes equal priors, while
    this threshold shifts toward the minority class.

    Usage:
        calibrator = BayesThresholdCalibrator()
        calibrator.fit(decision_vals, y_true)
        y_pred = calibrator.predict(decision_vals_test)

    The priors default to the empirical class frequencies. To simulate a
    different deployment distribution (e.g. 95% explosions), pass
    custom priors: calibrator.fit(vals, y, priors={0: 0.05, 1: 0.95})
    """

    def __init__(self):
        self.threshold = 0.0
        self.mu_eq = None
        self.sigma_eq = None
        self.mu_ex = None
        self.sigma_ex = None
        self.priors = None
        self.is_calibrated = False

    def fit(self, decision_vals, y, priors=None):
        """Fit Gaussians per class and compute the Bayes-optimal threshold.

        Args:
            decision_vals: SVM decision function output, shape (n_samples,).
            y: true labels (0=Earthquake, 1=Explosion).
            priors: optional dict {0: float, 1: float} for custom class priors.
                    Defaults to empirical frequencies from y.
        """
        eq_vals = decision_vals[y == 0]
        ex_vals = decision_vals[y == 1]

        self.mu_eq, self.sigma_eq = norm.fit(eq_vals)
        self.mu_ex, self.sigma_ex = norm.fit(ex_vals)

        if priors is not None:
            self.priors = priors
        else:
            n_total = len(y)
            self.priors = {0: len(eq_vals) / n_total, 1: len(ex_vals) / n_total}

        def posterior_diff(x):
            return (self.priors[0] * norm.pdf(x, self.mu_eq, self.sigma_eq) -
                    self.priors[1] * norm.pdf(x, self.mu_ex, self.sigma_ex))

        try:
            self.threshold = brentq(posterior_diff, self.mu_eq, self.mu_ex)
        except ValueError:
            # Fallback: grid search between the means
            x_grid = np.linspace(self.mu_eq, self.mu_ex, 10000)
            diffs = np.abs([posterior_diff(x) for x in x_grid])
            self.threshold = x_grid[np.argmin(diffs)]

        self.is_calibrated = True
        return self

    def predict(self, decision_vals):
        """Predict class labels using the calibrated threshold."""
        return (decision_vals > self.threshold).astype(int)


class FastMapSVMClassifier:
    def __init__(self, k=2, dist_func='euclidean', C=1.0, kernel='rbf'):
        self.k = k
        self.fastmap = FastMap(n_components=k, dist_func=dist_func)
        self.svm = SVC(C=C, kernel=kernel, probability=True)
        self.scaler = StandardScaler()
        self.calibrator = BayesThresholdCalibrator()

    def fit(self, X_train, y_train):
        print(f"  [FastMap] Embedding train set into {self.k}d...")
        X_emb = self.fastmap.fit_transform(X_train)

        print(f"  [SVM] Training classifier...")
        X_scaled = self.scaler.fit_transform(X_emb)
        self.svm.fit(X_scaled, y_train)
        return self

    def calibrate(self, X_cal, y_cal, priors=None):
        """Calibrate the decision threshold using a calibration set.

        Args:
            X_cal: calibration features (same domain as training data).
            y_cal: calibration labels.
            priors: optional custom class priors for deployment.
        """
        X_emb = self.fastmap.transform(X_cal)
        X_scaled = self.scaler.transform(X_emb)
        decision_vals = self.svm.decision_function(X_scaled)
        self.calibrator.fit(decision_vals, y_cal, priors=priors)
        print(f"  [Calibration] Bayes-optimal threshold: {self.calibrator.threshold:.4f}")
        return self

    def predict(self, X_test):
        X_emb = self.fastmap.transform(X_test)
        X_scaled = self.scaler.transform(X_emb)
        if self.calibrator.is_calibrated:
            decision_vals = self.svm.decision_function(X_scaled)
            return self.calibrator.predict(decision_vals)
        return self.svm.predict(X_scaled)

""" this is the basleine classifer to show we have better results using the fastmap + svm approach """
class BaselineClassifier:
    """
    Random Forest on statistical features (Mean, Std, Max, Min).
    No FastMap involved.
    """

    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)

    def _extract_features(self, X):
        # Flattening (N, 600, 3) -> (N, 1800) is one way,
        # but statistical aggregation is more robust for RF.
        features = []
        for i in range(len(X)):
            sample = X[i]  # (600, 3)
            feats = []
            # Calculate stats for each channel (Z, N, E)
            for col in range(3):
                channel_data = sample[:, col]
                feats.append(np.mean(channel_data))
                feats.append(np.std(channel_data))
                feats.append(np.max(channel_data))
                feats.append(np.min(channel_data))
            features.append(feats)
        return np.array(features)

    def fit(self, X, y):
        print("  [Baseline] Extracting features and training RF...")
        X_feat = self._extract_features(X)
        self.clf.fit(X_feat, y)
        return self

    def predict(self, X):
        X_feat = self._extract_features(X)
        return self.clf.predict(X_feat)