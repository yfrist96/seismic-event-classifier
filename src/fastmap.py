import numpy as np


class FastMap:
    def __init__(self, n_components, dist_func, verbose=True):
        self.n_components = n_components
        self.verbose = verbose

        # Handle string names for distance functions
        if isinstance(dist_func, str):
            if dist_func == 'euclidean':
                self.dist_func = self._euclidean_dist
            elif dist_func == 'correlation':
                self.dist_func = self._correlation_dist
            else:
                raise ValueError(f"Unknown metric: {dist_func}")
        else:
            self.dist_func = dist_func

        self.pivots_ = []
        self.X_train_ = None

    def _euclidean_dist(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def _correlation_dist(self, x1, x2):
        """
        1 - Pearson Correlation.
        Flattening (600,3) -> (1800,) captures global shape similarity.
        """
        x1_flat = x1.flatten()
        x2_flat = x2.flatten()

        # Centering
        x1_c = x1_flat - np.mean(x1_flat)
        x2_c = x2_flat - np.mean(x2_flat)

        norm1 = np.linalg.norm(x1_c)
        norm2 = np.linalg.norm(x2_c)

        if norm1 == 0 or norm2 == 0:
            return 1.0  # Max distance if signal is flat/silent

        num = np.dot(x1_c, x2_c)
        # Clip to [-1, 1] to avoid numerical errors going outside range
        corr = np.clip(num / (norm1 * norm2), -1.0, 1.0)
        return 1.0 - corr

    def _get_dist(self, x1, x2):
        return self.dist_func(x1, x2)

    def fit_transform(self, X):
        n_samples = X.shape[0]
        self.X_train_ = X
        embeddings = np.zeros((n_samples, self.n_components))
        self.pivots_ = []

        for dim in range(self.n_components):
            if self.verbose:
                print(f"  [FastMap] Processing Dimension {dim + 1}/{self.n_components}...")

            # 1. Pivot Selection
            idx_a = np.random.randint(0, n_samples)
            idx_b = self._furthest_point(X, embeddings, dim, idx_a)
            idx_a = self._furthest_point(X, embeddings, dim, idx_b)  # Refine A

            # 2. Compute Projected Distance (guard against 0)
            dist_ab = self._projected_dist(X[idx_a], X[idx_b], embeddings[idx_a], embeddings[idx_b], dim)
            if dist_ab < 1e-9:
                dist_ab = 1.0  # Prevent division by zero

            # 3. Project all points
            col_vals = []
            for i in range(n_samples):
                d_ai = self._projected_dist(X[idx_a], X[i], embeddings[idx_a], embeddings[i], dim)
                d_bi = self._projected_dist(X[idx_b], X[i], embeddings[idx_b], embeddings[i], dim)

                # Cosine Law
                val = (d_ai ** 2 + dist_ab ** 2 - d_bi ** 2) / (2 * dist_ab)
                col_vals.append(val)

            embeddings[:, dim] = col_vals

            # 4. Store Pivot Info (Cleaned up)
            self.pivots_.append({
                'idx_a': idx_a,
                'idx_b': idx_b,
                'dist_ab': dist_ab,
                # Store strictly previous coords required for next steps
                'coords_a': embeddings[idx_a, :dim].copy(),
                'coords_b': embeddings[idx_b, :dim].copy()
            })

        return embeddings

    def transform(self, X_test):
        n_test = X_test.shape[0]
        embeddings = np.zeros((n_test, self.n_components))

        for dim in range(self.n_components):
            pivot_info = self.pivots_[dim]
            idx_a = pivot_info['idx_a']
            idx_b = pivot_info['idx_b']
            dist_ab = pivot_info['dist_ab']

            # Raw train samples
            pivot_a_raw = self.X_train_[idx_a]
            pivot_b_raw = self.X_train_[idx_b]

            # Previous embedding coordinates (Projected space)
            prev_coords_a = pivot_info['coords_a']  # Now correctly sized (:dim)
            prev_coords_b = pivot_info['coords_b']

            for i in range(n_test):
                # 1. Raw Distances
                d_ai_raw = self._get_dist(pivot_a_raw, X_test[i])
                d_bi_raw = self._get_dist(pivot_b_raw, X_test[i])

                # 2. Projection Correction (Subtract previous dimensions)
                correction_ai = np.sum((prev_coords_a - embeddings[i, :dim]) ** 2)
                correction_bi = np.sum((prev_coords_b - embeddings[i, :dim]) ** 2)

                # 3. Apply Correction & CLAMP to 0 (Fixes numerical instability)
                d_ai_sq = max(0.0, d_ai_raw ** 2 - correction_ai)
                d_bi_sq = max(0.0, d_bi_raw ** 2 - correction_bi)

                # 4. Cosine Law
                val = (d_ai_sq + dist_ab ** 2 - d_bi_sq) / (2 * dist_ab)
                embeddings[i, dim] = val

        return embeddings

    def _projected_dist(self, obj_a, obj_b, coord_a, coord_b, current_dim):
        raw_dist = self._get_dist(obj_a, obj_b)
        sum_sq_diff = 0.0
        if current_dim > 0:
            diffs = coord_a[:current_dim] - coord_b[:current_dim]
            sum_sq_diff = np.sum(diffs ** 2)

        projected_sq = raw_dist ** 2 - sum_sq_diff
        return 0.0 if projected_sq < 0 else np.sqrt(projected_sq)

    def _furthest_point(self, X, embeddings, dim, idx_ref):
        max_dist = -1.0
        best_idx = -1
        # Random sample of 100 points for speed, or check all for accuracy
        # Checking all is safer for <10k samples
        for i in range(X.shape[0]):
            d = self._projected_dist(X[idx_ref], X[i], embeddings[idx_ref], embeddings[i], dim)
            if d > max_dist:
                max_dist = d
                best_idx = i
        return best_idx