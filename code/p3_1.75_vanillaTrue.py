# %%
import numpy as np
import pandas as pd
from numba import njit
import math
import matplotlib.pyplot as plt
import os
from aeon.utils.numba.general import z_normalise_series_2d
from aeon.distances import get_distance_function
import time

# %%
# %load_ext autoreload
# %autoreload 2
from ksfdtw.distance_measures import (
    psdtw_prime_vanilla as psdtw_prime_vanilla,
    psdtw_prime_lb_shen as psdtw_prime_lb_shen,
    cut_based_distance as cut_based_distance,
)
from ksfdtw.utils import precision_at_k

# %% [markdown]
# # Import Dataset

# %%
dataset_name = "GunPoint"
P = 3
l = 1.75
data = np.load(
    f"../data_processed/{dataset_name}_P{P}_uniform.npz",
    allow_pickle=True,
)
X_train_trans_uniform_concatenated = data["X_train_trans_uniform_concatenated"]

data = np.load(
    f"../data_processed/{dataset_name}_P{P}_l{l}_random.npz",
    allow_pickle=True,
)
X_train_trans_random_concatenated = data["X_train_trans_random_concatenated"]

# %% [markdown]
# ## Z-normalise the transformed series

# %%
X_train_trans_uniform_concatenated = z_normalise_series_2d(
    X_train_trans_uniform_concatenated
)
X_train_trans_random_concatenated = z_normalise_series_2d(
    X_train_trans_random_concatenated
)
instance_idx = 0

# %% [markdown]
# ## Assign query and target sets

# %%
# Query set
query_set = X_train_trans_random_concatenated

# Target set
target_set = X_train_trans_uniform_concatenated
if len(query_set) != len(target_set):
    raise ValueError("query_set and target_set have different sizes!")

# %% [markdown]
# # Searching with PSED, PSDTW

# %%
dist_method = 1  # 0 for PSED, 1 for PSDTW
vanilla = True  # True for vanilla, False for lb_shen
if vanilla:
    dist_func_p = lambda Q, C: psdtw_prime_vanilla(
        Q, C, l=l, P=P, r=0.1, dist_method=dist_method
    )
else:
    dist_func_p = lambda Q, C: psdtw_prime_lb_shen(
        Q, C, l=l, P=P, r=0.1, dist_method=dist_method
    )

# %% [markdown]
# ## Warmup for numba

# %%
def dist_func(Q, C):
    dist, _, _ = dist_func_p(Q, C)
    return dist

# %%
dist_func(
    X_train_trans_uniform_concatenated[instance_idx],
    X_train_trans_random_concatenated[instance_idx],
)
start = time.time()
dist_func(
    X_train_trans_uniform_concatenated[instance_idx],
    X_train_trans_random_concatenated[instance_idx],
)
end = time.time()
elapsed_time = end - start
print("Elapsed time for a single distance computation: " + str(elapsed_time))

# %% [markdown]
# ## Precision@k

# %%
all_distances = []
all_count_dist_calls = []
all_cuts = []
start = time.time()
precision_at_1, precision_at_3, precision_at_5, precision_at_7 = 0, 0, 0, 0
for i in range(0, len(query_set)):
    results = [dist_func_p(query_set[i], x) for x in target_set]
    dist_arr, count_dist_calls_arr, cuts_arr = zip(*results)
    distances = np.array(dist_arr)

    # store per-iteration results
    all_distances.append(distances)
    all_count_dist_calls.append(count_dist_calls_arr)
    all_cuts.append(cuts_arr)

    precision_at_1 += precision_at_k(distances, i, 1)
    precision_at_3 += precision_at_k(distances, i, 3)
    precision_at_5 += precision_at_k(distances, i, 5)
    precision_at_7 += precision_at_k(distances, i, 7)
print(
    f"{precision_at_1 / len(query_set):.2f},",
    f"{precision_at_3 / len(query_set):.2f},",
    f"{precision_at_5 / len(query_set):.2f},",
    f"{precision_at_7 / len(query_set):.2f}",
)
end = time.time()
elapsed_time = end - start
print("Elapsed time: " + str(elapsed_time))

# %%
os.makedirs("../outputs", exist_ok=True)
np.savez(
    f"../outputs/{dataset_name}_P{P}_l{l}_dist_method{dist_method}_vanilla{vanilla}.npz",
    all_distances=np.array(all_distances, dtype=object),
    all_count_dist_calls=np.array(all_count_dist_calls, dtype=object),
    all_cuts=np.array(all_cuts, dtype=object),
    precision_at_1=precision_at_1 / len(query_set),
    precision_at_3=precision_at_3 / len(query_set),
    precision_at_5=precision_at_5 / len(query_set),
    precision_at_7=precision_at_7 / len(query_set),
    elapsed_time=elapsed_time,
)




