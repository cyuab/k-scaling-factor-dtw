def precision_at_k(distances, true_index, k):
    # Get the indices of the top-k smallest distances
    top_k_indices = sorted(range(len(distances)), key=lambda x: distances[x])[:k]

    # Check if the true match is among them
    return 1 if true_index in top_k_indices else 0


###
###
###

import numpy as np
from pyts.metrics import dtw as _pyts_dtw_p
from dtaidistance import ed as dtaidistance_ed
from dtaidistance import dtw, dtw_visualisation
from sklearn.preprocessing import StandardScaler
import math
import time


def normalize(ts):
    mean = np.mean(ts)
    std = np.std(ts)
    return (ts - mean) / std


def normalize_legacy_2(series):
    mean = series.mean()  # Calculate the mean
    std = series.std(
        ddof=0
    )  # Calculate the population standard deviation instead of the default sample standard deviation

    # Apply z-normalization formula: (x - mean) / std
    normalized_series = (series - mean) / std

    return normalized_series


def normalize_legacy_3(series):
    # Reshape the series to 2D (required by StandardScaler)
    reshaped = series.values.reshape(-1, 1)

    # Apply z-normalization
    scaler = StandardScaler()
    normalized = scaler.fit_transform(reshaped)

    # Convert back to pandas Series
    return pd.Series(normalized.flatten(), index=series.index)


def pyts_dtw(ts1, ts2, r=0.1):
    return _pyts_dtw_p(ts1, ts2, method="fast", options={"radius": r})
    # return _pyts_dtw_p(ts1, ts2, method='sakoechiba', options={'window_size': r})


def dtai_ed(a, b, l=1, r=0.1):
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    # https://dtaidistance.readthedocs.io/en/latest/usage/ed.html
    return dtaidistance_ed.distance(a, b)


def dtai_dtw(a, b, l=1, r=0.1):
    if isinstance(r, float):
        # print("r is a float.")
        minlen = min(len(a), len(b))
        window = int(minlen * r)
    elif isinstance(r, int):
        # Do something when r is an int
        # print("r is an integer.")
        window = r
    else:
        raise ValueError("r must be either an integer or a float.")
    return dtw.distance(a, b, window=window)


def nearest_neighbor_interpolation_legacy(ts, new_len):
    ts = np.asarray(ts)
    k = len(ts)
    indices = [
        int(np.ceil(j * k / new_len)) - 1 for j in range(1, new_len + 1)
    ]  # Why -1? 1-based (in the paper) to 0-based (default in Python)
    return ts[indices]


def us_usdtw_p(Q, C, l, r, L, distance_method="ed"):
    # Scaling both time series
    # m = len(Q)
    # n = len(C)
    # L = min(np.ceil(l * m), n)

    Q_scaled = nearest_neighbor_interpolation(Q, L)
    C_scaled = nearest_neighbor_interpolation(C, L)

    # Compute distance based on the chosen method
    if distance_method == "dtw":
        dist = dtai_dtw(Q_scaled, C_scaled, l, r)
    elif distance_method == "ed":
        dist = dtai_ed(Q_scaled, C_scaled)
    else:
        raise ValueError(f"Unsupported distance method: {distance_method}")

    return dist


def us_usdtw(Q, C, l, r, L, distance_method="ed"):
    m = len(Q)
    n = len(C)
    best_so_far = np.inf
    for k in range(math.ceil(m / l), min(math.ceil(l * m), n) + 1):
        C_prefix = C[:k]
        dist = us_usdtw_p(Q, C_prefix, l, r, L, distance_method)
        if dist < best_so_far:
            best_so_far = dist
            best_k = k
    return best_so_far, best_k


###
###
###

# Vectorized version with NumPy
import numpy as np


def nearest_neighbor_interpolation(ts, new_len):
    ts = np.asarray(ts)
    k = len(ts)
    indices = np.rint(np.linspace(0, k - 1, new_len)).astype(int)
    return ts[indices]


def nearest_neighbor_interpolation_legacy_2(ts, new_len):
    ts = np.asarray(ts)
    k = len(ts)
    # Compute indices symmetrically
    indices = [int(round(j * (k - 1) / (new_len - 1))) for j in range(new_len)]
    return ts[indices]


# https://stackoverflow.com/questions/66934748/how-to-stretch-an-array-to-a-new-length-while-keeping-same-value-distribution
def linear_interpolation(array: np.ndarray, new_len: int) -> np.ndarray:
    la = len(array)
    return np.interp(np.linspace(0, la - 1, num=new_len), np.arange(la), array)
