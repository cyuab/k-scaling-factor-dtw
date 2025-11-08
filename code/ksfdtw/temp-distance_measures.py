import numpy as np
import math
from numba import njit

from aeon.distances import (
    euclidean_distance as aeon_euclidean_distance,
    dtw_distance as aeon_dtw_distance,
    adtw_distance as aeon_adtw_distance,
    ddtw_distance as aeon_ddtw_distance,
    erp_distance as aeon_erp_distance,
    edr_distance as aeon_edr_distance,
    lcss_distance as aeon_lcss_distance,
    manhattan_distance as aeon_manhattan_distance,
    minkowski_distance as aeon_minkowski_distance,
    msm_distance as aeon_msm_distance,
    sbd_distance as aeon_sbd_distance,
    shape_dtw_distance as aeon_shape_dtw_distance,
    squared_distance as aeon_squared_distance,
    twe_distance as aeon_twe_distance,
    wddtw_distance as aeon_wddtw_distance,
    wdtw_distance as aeon_wdtw_distance,
)

from .lower_bounds import lb_shen, lb_shen_without_last



@njit
def psdtw_prime_vanilla_lb_cache(Q, C, l, P, r, dist_method):
    print("psdtw_prime_vanilla_lb_cache")
    count_dist_calls = 0
    m = len(Q)
    n = len(C)
    assert m == n, "m should be equal to n"
    l_root = math.sqrt(l)
    L_avg = m / P
    L_min = max(1, int(math.ceil(L_avg / l_root)))
    L_max = min(int(math.floor(L_avg * l_root)), m)
    # print(L_max, L_min)

    # Minimum cost to align the **first i** elements of Q (i.e., Q[:i]) and the **first j** elements of C (i.e., C[:j]) using **P** exactly segments
    # Each segment satisfies the length constraint
    D = np.full((m + 1, n + 1, P + 1), np.inf)
    D[0, 0, 0] = 0.0
    D_cut = np.full((m + 1, n + 1, P + 1, 2), -1, dtype=np.int64)

    # Flattened cache
    cache_size = (m + 1) * (m + 1) * (n + 1) * (n + 1)
    dist_cache = -np.ones(cache_size, dtype=np.float64)  # -1 means "not computed"

    lb = 0

    for p in range(1, P + 1):
        L_acc_min = L_min * p  # p segments take at least "L_min" points
        L_acc_max = L_max * p

        for i in range(L_acc_min, min(L_acc_max, m) + 1):
            for L_Q in range(L_min, L_max + 1):
                i_prime = i - L_Q
                L_C_min = max(L_min, int(math.ceil(L_Q / l)))
                L_C_max = min(int(math.floor(L_Q * l)), L_max)
                for j in range(L_acc_min, min(L_acc_max, n) + 1):
                    lb = lb_shen_prefix(
                        Q[i_prime:i][::-1], C[j - L_C_min : j][::-1], l=l, r=r
                    )
                    if lb > D[i][j][p]:  # best_so_far
                        continue
                    for L_C in range(L_C_min, L_C_max + 1):
                        j_prime = j - L_C
                        D_cost = D[i_prime, j_prime, p - 1]
                        # Lower bounds
                        if np.isinf(D_cost):
                            # print("Skipping due to D_cost = inf!")
                            continue
                        elif D_cost + lb > D[i][j][p]:  # best_so_far
                            # print("Skipping due to D_cost > best_so_far!")
                            continue
                        if L_C > L_C_min:
                            lb += lb_shen_incremental(
                                Q[i_prime:i][::-1], C[j_prime:j], l, r
                            )
                            if D_cost + lb > D[i][j][p]:
                                continue
                        # print(
                        #     f"Computing usdtw_prime for Q[{i_prime}:{i}] and C[{j_prime}:{j}]"
                        # )
                        # print(f"L_Q = {L_Q}, L_C = {L_C}")
                        # print(f"L_C_min = {L_C_min}, L_C_max = {L_C_max}")

                        # idx = cache_index(i_prime, i, j_prime, j)
                        idx = (
                            ((i_prime * (m + 1) + i) * (n + 1) + j_prime) * (n + 1)
                        ) + j

                        if dist_cache[idx] < 0:  # not computed yet
                            dist_cache[idx] = usdtw_prime(
                                Q[i_prime:i],
                                C[j_prime:j],
                                L=L_max,
                                r=r,
                                dist_method=dist_method,
                            )
                            count_dist_calls += 1
                        else:
                            # Have computed before
                            pass
                        dist_cost = dist_cache[idx]
                        # D[i, j, p] = min(D[i, j, p], D_cost + dist_cost)
                        cur_cost = D_cost + dist_cost
                        if cur_cost < D[i, j, p]:
                            D[i, j, p] = cur_cost
                            D_cut[i, j, p, 0] = i_prime
                            D_cut[i, j, p, 1] = j_prime
    cuts = np.zeros((P, 4), dtype=np.int64)
    i, j, p = m, n, P
    while p > 0:
        i_prime = D_cut[i, j, p, 0]
        j_prime = D_cut[i, j, p, 1]
        cuts[p - 1, 0] = i_prime
        cuts[p - 1, 1] = i
        cuts[p - 1, 2] = j_prime
        cuts[p - 1, 3] = j
        i, j, p = i_prime, j_prime, p - 1
    return D[m, n, P], count_dist_calls, cuts


@njit
def psdtw_prime_vanilla_lb_testing2(Q, C, l, P, r, dist_method):
    print("psdtw_prime_vanilla_lb_testing")
    count_dist_calls = 0
    m = len(Q)
    n = len(C)
    assert m == n, "m should be equal to n"
    l_root = math.sqrt(l)
    L_avg = m / P
    L_min = max(1, int(math.ceil(L_avg / l_root)))
    L_max = min(int(math.floor(L_avg * l_root)), m)

    # Minimum cost to align the **first i** elements of Q (i.e., Q[:i]) and the **first j** elements of C (i.e., C[:j]) using **P** exactly segments
    # Each segment satisfies the length constraint
    D = np.full((m + 1, n + 1, P + 1), np.inf)
    D[0, 0, 0] = 0.0
    D_cut = np.full((m + 1, n + 1, P + 1, 2), -1, dtype=np.int64)

    for p in range(1, P + 1):
        L_acc_min = L_min * p  # p segments take at least "L_min" points
        L_acc_max = L_max * p

        for i in range(L_acc_min, min(L_acc_max, m) + 1):
            for L_Q in range(L_min, L_max + 1):
                i_prime = i - L_Q
                L_C_min = max(L_min, int(math.ceil(L_Q / l)))
                L_C_max = min(int(math.floor(L_Q * l)), L_max)

                for j in range(L_acc_min, min(L_acc_max, n) + 1):
                    for L_C in range(L_C_min, L_C_max + 1):
                        j_prime = j - L_C
                        D_cost = D[i_prime, j_prime, p - 1]

                        # Skip if previous cost is infinite
                        if np.isinf(D_cost):
                            continue

                        # Skip if previous cost already exceeds current best
                        if D_cost > D[i][j][p]:
                            continue

                        # Compute actual distance (same as vanilla version)
                        dist_cost = usdtw_prime(
                            Q[i_prime:i],
                            C[j_prime:j],
                            L=L_max,
                            r=r,
                            dist_method=dist_method,
                        )
                        count_dist_calls += 1

                        cur_cost = D_cost + dist_cost
                        if cur_cost < D[i, j, p]:
                            D[i, j, p] = cur_cost
                            D_cut[i, j, p, 0] = i_prime
                            D_cut[i, j, p, 1] = j_prime

    cuts = np.zeros((P, 4), dtype=np.int64)
    i, j, p = m, n, P
    while p > 0:
        i_prime = D_cut[i, j, p, 0]
        j_prime = D_cut[i, j, p, 1]
        cuts[p - 1, 0] = i_prime
        cuts[p - 1, 1] = i
        cuts[p - 1, 2] = j_prime
        cuts[p - 1, 3] = j
        i, j, p = i_prime, j_prime, p - 1
    return D[m, n, P], count_dist_calls, cuts


@njit
def psedd_prime(Q, C, l, P, r):
    _, _, cuts = psdtw_prime_vanilla(Q, C, l, P, r, dist_method=0)
    m = len(Q)
    l_root = math.sqrt(l)
    L_avg = m / P
    L_max = min(int(math.floor(L_avg * l_root)), m)
    dist = 0.0
    for cut in cuts:
        # print(cut[0], cut[1], cut[2], cut[3])
        dist_cost = usdtw_prime(
            Q[cut[0] : cut[1]],
            C[cut[2] : cut[3]],
            L=L_max,
            r=r,
            dist_method=1,
        )
        dist += dist_cost
    return dist


@njit
def psedd_prime_test(Q, C, l, P, r):
    _, _, cuts = psdtw_prime_vanilla(Q, C, l, P, r, dist_method=0)
    print(cuts)
    m = len(Q)
    l_root = math.sqrt(l)
    L_avg = m / P
    L_max = min(int(math.floor(L_avg * l_root)), m)
    dist = 0.0
    for cut in cuts:
        # print(cut[0], cut[1], cut[2], cut[3])
        dist_cost = usdtw_prime(
            Q[cut[0] : cut[1]],
            C[cut[2] : cut[3]],
            L=L_max,
            r=r,
            dist_method=1,
        )
        dist += dist_cost
    return dist


###
###
###


# @njit
# def psdtw_prime_cache_flattened_array(Q, C, l, P, r, dist_method=0):
#     m, n = len(Q), len(C)
#     l_sqrt = math.sqrt(l)
#     L_Q_avg = m / P
#     L_Q_gmin = int(math.floor(L_Q_avg / l_sqrt))
#     L_Q_gmax = int(math.ceil(L_Q_avg * l_sqrt))
#     L_C_avg = n / P
#     L_C_gmin = int(math.floor(L_C_avg / l_sqrt))
#     L_C_gmax = int(math.ceil(L_C_avg * l_sqrt))

#     # Minimum cost to align the **first i** elements of Q and the **first j** elements of C using **P** exactly segments
#     D = np.full((m + 1, n + 1, P + 1), np.inf)
#     D[0, 0, 0] = 0.0

#     # Flattened cache
#     cache_size = (m + 1) * (m + 1) * (n + 1) * (n + 1)
#     dist_cache = -np.ones(cache_size, dtype=np.float64)  # -1 means "not computed"

#     def cache_index(i_prime, i, j_prime, j):
#         return (((i_prime * (m + 1) + i) * (n + 1) + j_prime) * (n + 1)) + j

#     for p in range(1, P + 1):
#         L_Q_prevs_min = (
#             p - 1
#         ) * L_Q_gmin  # (p-1) segments take at least "(p - 1) * L_Q_gmin" points
#         L_Q_prevs_w_cur_min = (
#             L_Q_prevs_min + L_Q_gmin
#         )  # p segments take at least "L_Q_gmin" more points
#         L_Q_prevs_max = (p - 1) * L_Q_gmax
#         L_Q_prevs_w_cur_max = L_Q_prevs_max + L_Q_gmax

#         for i in range(L_Q_prevs_w_cur_min, min(L_Q_prevs_w_cur_max, m) + 1):
#             for L_Q in range(L_Q_gmin, L_Q_gmax + 1):
#                 if i - (L_Q + L_Q_prevs_min) < 0:
#                     continue
#                 i_prime = i - L_Q
#                 L_C_min = int(math.floor(L_Q / l))
#                 L_C_max = int(math.ceil(L_Q * l))
#                 L_C_prevs_min = (p - 1) * L_C_gmin
#                 L_C_prevs_w_cur_min = L_C_prevs_min + L_C_gmin
#                 L_C_prevs_max = (p - 1) * L_C_gmax
#                 L_C_prevs_w_cur_max = L_C_prevs_max + L_C_gmax
#                 for j in range(L_C_prevs_w_cur_min, min(L_C_prevs_w_cur_max, n) + 1):
#                     for L_C in range(L_C_min, L_C_max + 1):
#                         if j - (L_C + L_C_prevs_min) < 0:
#                             continue
#                         j_prime = j - L_C
#                         D_cost = D[i_prime, j_prime, p - 1]

#                         idx = cache_index(i_prime, i, j_prime, j)
#                         if dist_cache[idx] < 0:  # not computed yet
#                             dist_cache[idx] = usdtw_prime(
#                                 Q[i_prime:i],
#                                 C[j_prime:j],
#                                 L=max(L_Q_gmax, L_C_gmax),
#                                 r=r,
#                                 dist_method=dist_method,
#                             )
#                         else:
#                             # print("Using cached value")
#                             pass
#                         dist_cost = dist_cache[idx]
#                         # D[i, j, p] = min(D[i, j, p], D_cost + dist_cost)
#                         new_cost = D_cost + dist_cost
#                         if new_cost < D[i, j, p]:
#                             D[i, j, p] = new_cost
#     return D[m, n, P]


# @njit
# def row_min(D, qi_st, p):
#     n = D.shape[1]  # length along j
#     min_val = np.inf
#     for j in range(n):
#         val = D[qi_st, j, p - 1]
#         if val < min_val:
#             min_val = val
#     return min_val


# @njit
# def psdtw_prime_cache_dict(Q, C, l, P, r, dist_method=0):
#     print("Using psdtw_prime_cache_dict 2")
#     count_dist_calls = 0
#     m, n = len(Q), len(C)
#     l_sqrt = math.sqrt(l)
#     L_Q_avg = m / P
#     L_Q_gmin = int(math.floor(L_Q_avg / l_sqrt))
#     L_Q_gmax = int(math.ceil(L_Q_avg * l_sqrt))
#     L_C_avg = n / P
#     L_C_gmin = int(math.floor(L_C_avg / l_sqrt))
#     L_C_gmax = int(math.ceil(L_C_avg * l_sqrt))

#     # DP table: min cost aligning first i of Q with first j of C using p segments
#     D = np.full((m + 1, n + 1, P + 1), np.inf)
#     D[0, 0, 0] = 0.0

#     # Dictionary cache (keyed by indices)
#     dist_cache = {}

#     def get_dist(i_prime, i, j_prime, j):
#         key = (i_prime, i, j_prime, j)
#         if key not in dist_cache:
#             dist_cache[key] = usdtw_prime(
#                 Q[i_prime:i],
#                 C[j_prime:j],
#                 L=max(L_Q_gmax, L_C_gmax),
#                 r=r,
#                 dist_method=dist_method,
#             )
#         return dist_cache[key]

#     for p in range(1, P + 1):
#         L_Q_prevs_min = (p - 1) * L_Q_gmin
#         L_Q_prevs_w_cur_min = L_Q_prevs_min + L_Q_gmin
#         L_Q_prevs_max = (p - 1) * L_Q_gmax
#         L_Q_prevs_w_cur_max = L_Q_prevs_max + L_Q_gmax

#         for i in range(L_Q_prevs_w_cur_min, min(L_Q_prevs_w_cur_max, m) + 1):
#             for L_Q in range(L_Q_gmin, L_Q_gmax + 1):
#                 if i - (L_Q + L_Q_prevs_min) < 0:
#                     continue
#                 i_prime = i - L_Q

#                 L_C_min = int(math.floor(L_Q / l))
#                 L_C_max = int(math.ceil(L_Q * l))
#                 L_C_prevs_min = (p - 1) * L_C_gmin
#                 L_C_prevs_w_cur_min = L_C_prevs_min + L_C_gmin
#                 L_C_prevs_max = (p - 1) * L_C_gmax
#                 L_C_prevs_w_cur_max = L_C_prevs_max + L_C_gmax
#                 for j in range(L_C_prevs_w_cur_min, min(L_C_prevs_w_cur_max, n) + 1):

#                     # lower_bound = np.min(D[i_prime, :, p - 1])
#                     # print(f"LB for D[{i},{j},{p}] is {lower_bound}")
#                     # print(f"Current best D[{i},{j},{p}] is {D[i][j][p]}")
#                     # if lower_bound > D[i][j][p]:  # best_so_far
#                     #     print("Skipping due to LB")
#                     #     continue

#                     for L_C in range(L_C_min, L_C_max + 1):
#                         if j - (L_C + L_C_prevs_min) < 0:
#                             continue
#                         j_prime = j - L_C
#                         D_cost = D[i_prime, j_prime, p - 1]
#                         # Start of Lower bounds
#                         if np.isinf(D_cost):
#                             # print("Skipping due to LB 1")
#                             continue
#                         if D_cost > D[i][j][p]:  # best_so_far
#                             # print("Skipping due to LB 2")
#                             continue
#                         lower_bound = lb_shen(Q[i_prime:i], C[j_prime:j], l=2.0, r=0.1)
#                         if D_cost + lower_bound > D[i][j][p]:
#                             # print("Skipping due to LB 3")
#                             continue
#                         # End of Lower bounds
#                         # Expensive call
#                         dist_cost = get_dist(i_prime, i, j_prime, j)
#                         count_dist_calls += 1
#                         new_cost = D_cost + dist_cost
#                         if new_cost < D[i, j, p]:
#                             D[i, j, p] = new_cost

#     return D[m, n, P], count_dist_calls


# # @njit
# def psdtw_prime_lb_w_counting(Q, C, l, P, r, dist_method=0):
#     count_dist_calls = 0
#     m, n = len(Q), len(C)
#     l_sqrt = math.sqrt(l)
#     L_Q_avg = m / P
#     L_Q_gmin = int(math.floor(L_Q_avg / l_sqrt))
#     L_Q_gmax = int(math.ceil(L_Q_avg * l_sqrt))
#     L_C_avg = n / P
#     L_C_gmin = int(math.floor(L_C_avg / l_sqrt))
#     L_C_gmax = int(math.ceil(L_C_avg * l_sqrt))

#     # Minimum cost to align the **first i** elements of Q and the **first j** elements of C using **P** exactly segments
#     D = np.full((m + 1, n + 1, P + 1), np.inf)
#     D[0, 0, 0] = 0.0
#     for p in range(1, P + 1):
#         L_Q_prevs_min = (
#             p - 1
#         ) * L_Q_gmin  # (p-1) segments take at least "(p - 1) * L_Q_gmin" points
#         L_Q_prevs_w_cur_min = (
#             L_Q_prevs_min + L_Q_gmin
#         )  # p segments take at least "L_Q_gmin" more points
#         L_Q_prevs_max = (p - 1) * L_Q_gmax
#         L_Q_prevs_w_cur_max = L_Q_prevs_max + L_Q_gmax

#         for i in range(L_Q_prevs_w_cur_min, min(L_Q_prevs_w_cur_max, m) + 1):
#             for L_Q in range(L_Q_gmin, L_Q_gmax + 1):
#                 if i - (L_Q + L_Q_prevs_min) < 0:
#                     continue
#                 i_prime = i - L_Q

#                 L_C_min = int(math.floor(L_Q / l))
#                 L_C_max = int(math.ceil(L_Q * l))
#                 L_C_prevs_min = (p - 1) * L_C_gmin
#                 L_C_prevs_w_cur_min = L_C_prevs_min + L_C_gmin
#                 L_C_prevs_max = (p - 1) * L_C_gmax
#                 L_C_prevs_w_cur_max = L_C_prevs_max + L_C_gmax

#                 lower_bound = np.min(D[i_prime, :, p - 1])
#                 for j in range(L_C_prevs_w_cur_min, min(L_C_prevs_w_cur_max, n) + 1):

#                     # print(f"LB for D[{i},{j},{p}] is {lower_bound}")
#                     # lower_bound = row_min(D, i_prime, p)
#                     if lower_bound > D[i][j][p]:  # best_so_far
#                         # print("Skipping due to LB")
#                         continue

#                     for L_C in range(L_C_min, L_C_max + 1):
#                         if j - (L_C + L_C_prevs_min) < 0:
#                             continue
#                         j_prime = j - L_C
#                         D_cost = D[i_prime, j_prime, p - 1]
#                         lower_bound = lb_shen(Q[i_prime:i], C[j_prime:j], l=2.0, r=0.1)
#                         if D_cost + lower_bound > D[i][j][p]:
#                             # print("Skipping due to LB 2")
#                             continue
#                         dist_cost = usdtw_prime(
#                             Q[i_prime:i],
#                             C[j_prime:j],
#                             L=max(L_Q_gmax, L_C_gmax),
#                             r=r,
#                             dist_method=dist_method,
#                         )
#                         count_dist_calls += 1
#                         # D[i, j, p] = min(D[i, j, p], D_cost + dist_cost)
#                         new_cost = D_cost + dist_cost
#                         if new_cost < D[i, j, p]:
#                             D[i, j, p] = new_cost
#     # return D[m, n, P]
#     return D[m, n, P], count_dist_calls

# @njit
# def psdtw_prime_cache_dict(Q, C, l, P, r, dist_method=0):
#     m, n = len(Q), len(C)
#     l_sqrt = math.sqrt(l)
#     L_Q_avg = m / P
#     L_Q_gmin = int(math.floor(L_Q_avg / l_sqrt))
#     L_Q_gmax = int(math.ceil(L_Q_avg * l_sqrt))
#     L_C_avg = n / P
#     L_C_gmin = int(math.floor(L_C_avg / l_sqrt))
#     L_C_gmax = int(math.ceil(L_C_avg * l_sqrt))

#     # DP table: min cost aligning first i of Q with first j of C using p segments
#     D = np.full((m + 1, n + 1, P + 1), np.inf)
#     D[0, 0, 0] = 0.0

#     # Dictionary cache (keyed by indices)
#     dist_cache = {}

#     def get_dist(i_prime, i, j_prime, j):
#         key = (i_prime, i, j_prime, j)
#         if key not in dist_cache:
#             dist_cache[key] = usdtw_prime(
#                 Q[i_prime:i],
#                 C[j_prime:j],
#                 L=max(L_Q_gmax, L_C_gmax),
#                 r=r,
#                 dist_method=dist_method,
#             )
#         return dist_cache[key]

#     for p in range(1, P + 1):
#         L_Q_prevs_min = (p - 1) * L_Q_gmin
#         L_Q_prevs_w_cur_min = L_Q_prevs_min + L_Q_gmin
#         L_Q_prevs_max = (p - 1) * L_Q_gmax
#         L_Q_prevs_w_cur_max = L_Q_prevs_max + L_Q_gmax

#         for i in range(L_Q_prevs_w_cur_min, min(L_Q_prevs_w_cur_max, m) + 1):
#             for L_Q in range(L_Q_gmin, L_Q_gmax + 1):
#                 if i - (L_Q + L_Q_prevs_min) < 0:
#                     continue
#                 i_prime = i - L_Q
#                 L_C_min = int(math.floor(L_Q / l))
#                 L_C_max = int(math.ceil(L_Q * l))
#                 L_C_prevs_min = (p - 1) * L_C_gmin
#                 L_C_prevs_w_cur_min = L_C_prevs_min + L_C_gmin
#                 L_C_prevs_max = (p - 1) * L_C_gmax
#                 L_C_prevs_w_cur_max = L_C_prevs_max + L_C_gmax
#                 for j in range(L_C_prevs_w_cur_min, min(L_C_prevs_w_cur_max, n) + 1):
#                     for L_C in range(L_C_min, L_C_max + 1):
#                         if j - (L_C + L_C_prevs_min) < 0:
#                             continue
#                         j_prime = j - L_C
#                         D_cost = D[i_prime, j_prime, p - 1]
#                         if np.isinf(D_cost):
#                             continue
#                         dist_cost = get_dist(i_prime, i, j_prime, j)
#                         new_cost = D_cost + dist_cost
#                         if new_cost < D[i, j, p]:
#                             D[i, j, p] = new_cost

#     return D[m, n, P]
