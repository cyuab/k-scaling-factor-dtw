import numpy as np
from pyts.metrics import dtw as _pyts_dtw_p 
from dtaidistance import ed as dtaidistance_ed
from dtaidistance import dtw, dtw_visualisation

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
        window=r
    else:
        raise ValueError("r must be either an integer or a float.")
    return dtw.distance(a, b, window=window)

# https://stackoverflow.com/questions/66934748/how-to-stretch-an-array-to-a-new-length-while-keeping-same-value-distribution
def linear_interpolation(array: np.ndarray, new_len: int) -> np.ndarray:
    la = len(array)
    return np.interp(np.linspace(0, la - 1, num=new_len), np.arange(la), array)

def nearest_neighbor_interpolation(ts, new_len):
    ts = np.asarray(ts)
    k = len(ts)
    indices = [int(np.ceil(j * k / new_len)) - 1 for j in range(1, new_len + 1)]  # Why -1? 1-based (in the paper) to 0-based (default in Python)
    return ts[indices]

def us_usdtw_p(Q, C, l, r, distance_method="ed"):
    # Scaling both time series
    m = len(Q)
    n = len(C)
    L = min(np.ceil(l * m), n)

    Q_scaled = nearest_neighbor_interpolation(Q, L)
    C_scaled = nearest_neighbor_interpolation(C, L)

    # Compute distance based on the chosen method
    if distance_method == 'dtw':
        dist = dtai_dtw(Q_scaled, C_scaled, l, r)
    elif distance_method == 'ed':
        dist = dtai_ed(Q_scaled, C_scaled)
    else:
        raise ValueError(f"Unsupported distance method: {distance_method}")

    return dist

def us_usdtw(Q, C, l, r, distance_method='ed'):
    m = len(Q)
    n = len(C)
    best_so_far = np.inf
    for k in range(math.ceil(m/l), min(math.ceil(l*m), n)+1):
        C_prefix = C[:k]
        dist = us_usdtw_p(Q, C_prefix, l, r, distance_method)
        if dist < best_so_far:
            best_so_far = dist
            best_k = k
    return best_so_far, best_k

def lb_keogh(ts_query, ts_candidate, radius):
    lower, upper = lb_keogh_envelope(ts_query, radius)
    lb_sum = 0
    for i in range(len(ts_candidate)):
        if ts_candidate[i] > upper[i]:
            lb_sum += (ts_candidate[i] - upper[i])**2
        elif ts_candidate[i] < lower[i]:
            lb_sum += (ts_candidate[i] - lower[i])**2
    return np.sqrt(lb_sum)

def lb_keogh_envelope(ts, radius):
    n = len(ts)
    upper = np.zeros(n)
    lower = np.zeros(n)
    for i in range(n):
        start = max(0, i - radius)
        end = min(n, i + radius + 1)
        lower[i] = np.min(ts[start:end])
        upper[i] = np.max(ts[start:end])
    return lower, upper

###

import math



def normalize(ts):
    mean = np.mean(ts)
    std = np.std(ts)
    return (ts - mean) / std

def pyts_dtw(ts1, ts2, r=10):
    return _pyts_dtw_p(ts1, ts2, method='fast', options={'radius': r})
    # return _pyts_dtw_p(ts1, ts2, method='sakoechiba', options={'window_size': r})













