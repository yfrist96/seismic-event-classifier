import numpy as np
from src.distances import euclidean_distance, ncc_distance


class FastMap:
    def __init__(self, n_components=2, dist_func='euclidean'):
        self.k = n_components
        self.pivots = []  # Stores (X[idx_a], X[idx_b])
        self.pivot_dists = []  # Stores dist_ab
        # NEW: Store embeddings of the pivots to project test data correctly
        self.pivot_embeddings = []

        if dist_func == 'euclidean':
            self.dist_fn = euclidean_distance
        elif dist_func == 'correlation':
            self.dist_fn = ncc_distance
        else:
            raise ValueError(f"Unknown distance function: {dist_func}")

    def _get_dist(self, obj_a, obj_b):
        return self.dist_fn(obj_a, obj_b)

    def fit_transform(self, X):
        N = len(X)
        embeddings = np.zeros((N, self.k))
        self.pivots = []
        self.pivot_dists = []
        self.pivot_embeddings = []  # Reset

        for dim in range(self.k):
            print(f"  [FastMap] Processing Dimension {dim + 1}/{self.k}...")

            # 1. Heuristic Pivot Selection
            idx_rand = np.random.randint(0, N)

            # Furthest from random
            dists_from_rand = np.array([self._projected_dist(X, idx_rand, i, embeddings, dim) for i in range(N)])
            idx_a = np.argmax(dists_from_rand)

            # Furthest from Pivot A
            dists_from_a = np.array([self._projected_dist(X, idx_a, i, embeddings, dim) for i in range(N)])
            idx_b = np.argmax(dists_from_a)

            dist_ab = dists_from_a[idx_b]
            if dist_ab < 1e-9: dist_ab = 1e-9

            # Store pivots and their distance
            self.pivots.append((X[idx_a], X[idx_b]))
            self.pivot_dists.append(dist_ab)

            # 2. Projection
            dists_from_b = np.array([self._projected_dist(X, idx_b, i, embeddings, dim) for i in range(N)])
            col_vals = (dists_from_a ** 2 + dist_ab ** 2 - dists_from_b ** 2) / (2 * dist_ab)
            embeddings[:, dim] = col_vals

            # NEW: Save the embeddings of the chosen pivots for this dimension
            # We need the full row of embeddings for A and B up to this point
            self.pivot_embeddings.append((embeddings[idx_a].copy(), embeddings[idx_b].copy()))

        return embeddings

    def _projected_dist(self, X, i, j, embeddings, current_dim):
        d_orig = self._get_dist(X[i], X[j])
        if current_dim == 0:
            return d_orig

        dist_sq_projected = 0
        for d in range(current_dim):
            dist_sq_projected += (embeddings[i, d] - embeddings[j, d]) ** 2

        val = d_orig ** 2 - dist_sq_projected
        return np.sqrt(max(0, val))

    def transform(self, X_test):
        N = len(X_test)
        embeddings = np.zeros((N, self.k))

        for dim in range(self.k):
            pivot_a, pivot_b = self.pivots[dim]
            pivot_a_emb, pivot_b_emb = self.pivot_embeddings[dim]  # Retrieve pivot embeddings
            dist_ab = self.pivot_dists[dim]

            for i in range(N):
                # 1. Measure original distance
                d_ai_orig = self._get_dist(pivot_a, X_test[i])
                d_bi_orig = self._get_dist(pivot_b, X_test[i])

                # 2. Adjust distance based on previous dimensions (Projected Distance)
                # D_new^2 = D_old^2 - sum((x_pivot - x_test)^2)

                sub_a = 0
                sub_b = 0
                for d in range(dim):
                    sub_a += (embeddings[i, d] - pivot_a_emb[d]) ** 2
                    sub_b += (embeddings[i, d] - pivot_b_emb[d]) ** 2

                d_ai = np.sqrt(max(0, d_ai_orig ** 2 - sub_a))
                d_bi = np.sqrt(max(0, d_bi_orig ** 2 - sub_b))

                # 3. Cosine Law
                x_val = (d_ai ** 2 + dist_ab ** 2 - d_bi ** 2) / (2 * dist_ab)
                embeddings[i, dim] = x_val

        return embeddings