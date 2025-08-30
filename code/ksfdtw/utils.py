# Vectorized version with NumPy
import numpy as np


def hello_world():
    print("Hello, world! 2")


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


def precision_at_k(distances, true_index, k):
    # Get the indices of the top-k smallest distances
    top_k_indices = sorted(range(len(distances)), key=lambda x: distances[x])[:k]

    # Check if the true match is among them
    return 1 if true_index in top_k_indices else 0
