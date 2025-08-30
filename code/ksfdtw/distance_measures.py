import numpy as np
from numba import njit
import math

from aeon.distances import (
    euclidean_distance as aeon_euclidean_distance,
    dtw_distance as aeon_dtw_distance,
)

from ksfdtw.utils import nearest_neighbor_interpolation


def hello_world():
    print("Hello, world!")


@njit
def ed(a, b):
    assert len(a) == len(
        b
    ), "Euclidean distance (ED) requires time series of equal length!"

    dist = 0.0
    for i in range(len(a)):
        diff = a[i] - b[i]
        dist += diff * diff
    return dist
    # return math.sqrt(dist)


# Compute Euclidean distances to all training samples
def ed_legacy(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def usdtw__prime(Q, C, L, r, distance_method="dtw"):
    C_scaled = nearest_neighbor_interpolation(C, L)
    Q_scaled = nearest_neighbor_interpolation(Q, L)
    if distance_method == "ed":
        return aeon_euclidean_distance(Q_scaled, C_scaled)
    elif distance_method == "dtw":
        return aeon_dtw_distance(Q_scaled, C_scaled, window=r)
    else:
        raise ValueError("Unknown distance method: {}".format(distance_method))


def usdtw(Q, C, l, L, r, distance_method="dtw"):
    m = len(Q)
    n = len(C)
    best_so_far = np.inf
    for k in range(math.ceil(m / l), min(math.ceil(l * m), n) + 1):
        C_prefix = C[:k]
        dist = usdtw__prime(Q, C_prefix, L, r, distance_method)
        if dist < best_so_far:
            best_so_far = dist
            best_k = k
    return best_so_far, best_k
