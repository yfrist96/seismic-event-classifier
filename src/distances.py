import numpy as np
from scipy.signal import correlate
from scipy.stats import wasserstein_distance as scipy_wasserstein
from scipy.spatial import distance as scipy_dist

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

def wasserstein_fourier_distance(u, v):
    u = u.flatten()
    v = v.flatten()

    u_norm = u / (np.sum(u) + 1e-9)
    v_norm = v / (np.sum(v) + 1e-9)

    return scipy_wasserstein(u_norm, v_norm)

def likelihood_ratio_distance(u, v):
    u = u.flatten()
    v = v.flatten()

    # Shift to ensure positivity if input is log-scaled with negatives
    # This may not be needed but just in case
    if np.min(u) < 0 or np.min(v) < 0:
        offset = min(np.min(u), np.min(v))
        u = u - offset + 1e-6
        v = v - offset + 1e-6
    
    ratio_uv = (u + 1e-9) / (v + 1e-9)
    ratio_vu = (v + 1e-9) / (u + 1e-9)

    d_uv = np.sum(ratio_uv - np.log(ratio_uv)-1)
    d_vu = np.sum(ratio_vu - np.log(ratio_vu)-1)

    return 0.5*(d_uv + d_vu)

def kulczynski_distance(u, v):
    """
    Similar to Soergel but averages the relative differences.
    """
    u = u.flatten()
    v = v.flatten()
    intersection = np.sum(np.minimum(u, v))
    sum_u = np.sum(u)
    sum_v = np.sum(v)
    
    if sum_u == 0 or sum_v == 0:
        return 1.0
        
    # Kulczynski Similarity is 0.5 * (Int/SumU + Int/SumV)
    similarity = 0.5 * ((intersection / sum_u) + (intersection / sum_v))
    return 1.0 - similarity

def soergel_distance(u, v):
    """
    Bounded [0,1]. Similar to Jaccard.
    Sum(|u - v|) / Sum(max(u, v))
    """
    u = u.flatten()
    v = v.flatten()
    numerator = np.sum(np.abs(u - v))
    denominator = np.sum(np.maximum(u, v))
    
    if denominator == 0:
        return 0.0
    return numerator / denominator

def lorentzian_distance(u, v):
    """
    Robust to outliers (glitches).
    Sum( log(1 + |u_i - v_i|) )
    """
    diff = np.abs(u.flatten() - v.flatten())
    return np.sum(np.log(1 + diff))

def canberra_distance(u, v):
    """
    Sensitive to data near zero. Good for relative changes.
    Sum( |u_i - v_i| / (|u_i| + |v_i|) )
    """
    # flatten and ensure no division by zero
    u = u.flatten()
    v = v.flatten()
    return scipy_dist.canberra(u, v)