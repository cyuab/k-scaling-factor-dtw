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
        idx = int(np.ceil((j + 1) * k / L)) - 1  # 0-based to 1-based to 0-based
        result[j] = ts[idx]
    return result

def precision_at_k(distances, true_index, k):
    # Get the indices of the top-k smallest distances
    top_k_indices = sorted(range(len(distances)), key=lambda x: distances[x])[:k]

    # Check if the true match is among them
    return 1 if true_index in top_k_indices else 0

@njit
def nearest_neighbor_search(query, dataset, r, l, P, dist_method, dist_func):
    """
    query: shape (m,)
    dataset: shape (N, n)  (assuming equal length for simplicity, or list of arrays)
    """
    bsf = np.inf
    best_idx = -1
    total_dist_calls = 0
    
    for k in range(len(dataset)):
        candidate = dataset[k]
        
        # Pass the current BSF into the distance function
        dist, count_dist_calls, cuts = dist_func(query, candidate, r, l, P, dist_method, bsf)
        
        total_dist_calls += count_dist_calls
        
        # If we found a closer match, update BSF
        if dist < bsf:
            bsf = dist
            best_idx = k
            # print(f"New best found at index {k}: {bsf}")
            
    return best_idx, bsf, total_dist_calls