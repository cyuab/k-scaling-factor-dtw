import numpy as np
from numba import njit

@njit
def nearest_neighbor_interpolation(ts, L):
    """
    Parameters
    ----------
    ts : time series
    L : desired length of output time series
    """
    k = len(ts)
    result = np.empty(L, dtype=ts.dtype)
    for j in range(L):
        idx = int(np.ceil((j + 1) * k / L)) - 1  # 1-based to 0-based
        result[j] = ts[idx]
    return result

def precision_at_k(distances, true_index, k):
    # Get the indices of the top-k smallest distances
    top_k_indices = sorted(range(len(distances)), key=lambda x: distances[x])[:k]

    # Check if the true match is among them
    return 1 if true_index in top_k_indices else 0