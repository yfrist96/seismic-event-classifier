from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from src.fastmap import FastMap
import numpy as np


class FastMapSVMClassifier:
    def __init__(self, k=2, dist_func='euclidean', C=1.0, kernel='rbf'):
        self.k = k
        self.fastmap = FastMap(n_components=k, dist_func=dist_func)
        self.svm = SVC(C=C, kernel=kernel, probability=True)
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train):
        print(f"  [FastMap] Embedding train set into {self.k}d...")
        X_emb = self.fastmap.fit_transform(X_train)

        print(f"  [SVM] Training classifier...")
        X_scaled = self.scaler.fit_transform(X_emb)
        self.svm.fit(X_scaled, y_train)
        return self

    def predict(self, X_test):
        X_emb = self.fastmap.transform(X_test)
        X_scaled = self.scaler.transform(X_emb)
        return self.svm.predict(X_scaled)

""" this is the basleine classifer to show we ahve better results using the fastmap + svm approach """
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