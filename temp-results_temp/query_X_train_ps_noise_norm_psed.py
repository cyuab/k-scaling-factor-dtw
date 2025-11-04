# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from numba import njit
import os

from aeon.utils.numba.general import z_normalise_series_2d

# from aeon.distances import euclidean_distance
# from aeon.distances import dtw_distance
from aeon.distances import get_distance_function

# %%

from ksfdtw.distance_measures import (
    usdtw_prime as ksfdtw_usdtw_prime,
    psdtw_prime_vanilla as psdtw_prime_vanilla,
    psedd_prime as psedd_prime,
    psedd_prime_test as psedd_prime_test,
    cut_based_distance as cut_based_distance,
)
from ksfdtw.utils import precision_at_k

# %% [markdown]
# # Import Dataset

# %%
# A neat way to load the dataset, but more complicated to use
# data = np.load("../data_intermediate/GunPoint_preprocessed_P_3_l_2.0_len_150.npz")
# data_dict = {key: data[key] for key in data.files}

# A old way to load the dataset
data = np.load(
    "../data_intermediate/GunPoint_ps_P_3_l_2.0_len_150.npz",
    allow_pickle=True,
)
X_train_scaled = data["X_train_scaled"]
X_train_ps = data["X_train_ps"]
X_train_ps_noise = data["X_train_ps_noise"]
y_train = data["y_train"]
X_test_scaled = data["X_test_scaled"]
X_test_ps = data["X_test_ps"]
X_test_ps_noise = data["X_test_ps_noise"]
y_test = data["y_test"]
X_train_cuts = data["X_train_cuts"].tolist()
X_train_ps_cuts = data["X_train_ps_cuts"].tolist()
X_test_cuts = data["X_test_cuts"].tolist()
X_test_ps_cuts = data["X_test_ps_cuts"].tolist()

# %%
# X_train_scaled.shape, X_train_ps.shape, X_train_ps_noise.shape

# %%
X_train_scaled_norm = z_normalise_series_2d(X_train_scaled)
X_train_ps_norm = z_normalise_series_2d(X_train_ps)
X_train_ps_noise_norm = z_normalise_series_2d(X_train_ps_noise)

# %%
instance_idx = 0
# plt.plot(X_train_scaled[instance_idx])
# plt.plot(X_train_ps[instance_idx])
# plt.plot(X_train_ps_noise[instance_idx])
# plt.show()
# plt.plot(X_train_scaled_norm[instance_idx])
# plt.plot(X_train_ps_norm[instance_idx])
# plt.plot(X_train_ps_noise_norm[instance_idx])
# plt.show()

# %% [markdown]
# # Querying

# %%
# *** Change here 1 ***
# Query set
# query_set = X_train_ps
# query_set = X_train_ps_norm
# query_set = X_train_ps_noise
query_set = X_train_ps_noise_norm


# Target set
# target_set = X_train_scaled
target_set = X_train_scaled_norm
if len(query_set) != len(target_set):
    raise ValueError("query_set and target_set have different sizes!")

# %% [markdown]
# ## Precision@k

# %% [markdown]
# Compute $P@k$ for querying $Q \in$ `query_set` using `method_name` on `target_set`

# %%
# --------------------- Step 1 ---------------------
# *psed* dist_method=0
# *psdtw* dist_method=1

dist_func_p = lambda Q, C: psdtw_prime_vanilla(Q, C, l=2, P=3, r=0.1, dist_method=0)

# *psdtw*
# method_name = "psdtw"
# dist_func_p = lambda Q, C: psdtw_prime_vanilla(Q, C, l=2, P=3, r=0.1, dist_method=1)


# --------------------- Step 2 ---------------------
# def dist_func(Q, C):
#     dist, _, _ = dist_func_p(Q, C)
#     return dist

# *psedd*
# method_name = "psedd"
# dist_func = lambda Q, C: psedd_prime(Q, C, l=2, P=3, r=0.1)

# %%
all_distances = []
all_count_dist_calls = []
all_cuts = []

# %%


import time

print("Start")
start = time.time()
precision_at_1, precision_at_3, precision_at_5, precision_at_7 = 0, 0, 0, 0
for i in range(0, len(query_set)):
    # *** Change here 3 ***
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
print(elapsed_time)

# %%
# *** Change here 4 ***

# Ensure the folder exists
os.makedirs("../results_temp", exist_ok=True)

np.savez(
    "../results_temp/X_train_ps_noise_norm_psed.npz",
    all_distances=np.array(all_distances, dtype=object),
    all_count_dist_calls=np.array(all_count_dist_calls, dtype=object),
    all_cuts=np.array(all_cuts, dtype=object),
    precision_at_1=precision_at_1 / len(query_set),
    precision_at_3=precision_at_3 / len(query_set),
    precision_at_5=precision_at_5 / len(query_set),
    precision_at_7=precision_at_7 / len(query_set),
    elapsed_time=elapsed_time,
)

# %%
import datetime

print(f"This notebook was last run end-to-end on: {datetime.datetime.now()}\n")
###
###
###
