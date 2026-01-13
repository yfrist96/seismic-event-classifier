import numpy as np
from scipy.signal import correlate


def euclidean_distance(x1, x2):
    """
    Computes Euclidean distance between two flattened waveform arrays.
    """
    return np.linalg.norm(x1.flatten() - x2.flatten())

def cosine_distance(x1, x2):
    """
    Computes Cosine Distance (1 - Cosine Similarity).
    Formula: 1 - (u . v) / (|u| * |v|)
    """
    # Flatten inputs (essential for FFT features which might come in as 2D)
    u = x1.flatten()
    v = x2.flatten()
    
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    
    # Safety check: if one vector is all zeros (empty signal), distance is max (1.0)
    if norm_u == 0 or norm_v == 0:
        return 1.0
        
    similarity = dot_product / (norm_u * norm_v)
    
    # Clip to [-1, 1] to handle tiny floating point errors
    similarity = np.clip(similarity, -1.0, 1.0)
    
    return 1.0 - similarity


def ncc_distance(x1, x2):
    """
    Normalized Cross-Correlation Distance (Eq. 8 in the Article).
    D(x1, x2) = 1 - max_correlation
    """
    # Ensure inputs are shaped correctly (600, 3)
    if x1.ndim == 1:
        x1 = x1.reshape(600, 3)
    if x2.ndim == 1:
        x2 = x2.reshape(600, 3)

    # 1. Normalize signals (Zero mean, Unit variance) to enable comparison
    # Add epsilon to prevent division by zero
    x1_norm = (x1 - np.mean(x1, axis=0)) / (np.std(x1, axis=0) + 1e-10)
    x2_norm = (x2 - np.mean(x2, axis=0)) / (np.std(x2, axis=0) + 1e-10)

    max_corrs = []

    # 2. Compute Correlation per component (Z, N, E)
    for ch in range(3):
        # 'valid' mode is faster but 'full' is safer for max alignment
        corr = correlate(x1_norm[:, ch], x2_norm[:, ch], mode='full')

        # Normalize by signal length to keep value between -1 and 1
        corr /= x1.shape[0]

        # We take the absolute max correlation (similarity)
        max_corrs.append(np.max(np.abs(corr)))

    # 3. Average similarity across 3 channels
    avg_sim = np.mean(max_corrs)

    # 4. Convert to Distance (Dissimilarity)
    return 1 - avg_sim