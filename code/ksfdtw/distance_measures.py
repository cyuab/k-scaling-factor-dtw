import numpy as np
import math
from numba import njit

from aeon.distances import (
    dtw_distance as aeon_dtw_distance,
    shape_dtw_distance as aeon_shape_dtw_distance,
    ddtw_distance as aeon_ddtw_distance,
    wdtw_distance as aeon_wdtw_distance,
    wddtw_distance as aeon_wddtw_distance,
    adtw_distance as aeon_adtw_distance,
    erp_distance as aeon_erp_distance,
    edr_distance as aeon_edr_distance,
    msm_distance as aeon_msm_distance,
    twe_distance as aeon_twe_distance,
    lcss_distance as aeon_lcss_distance,
    euclidean_distance as aeon_euclidean_distance,
    manhattan_distance as aeon_manhattan_distance,
    minkowski_distance as aeon_minkowski_distance,
    sbd_distance as aeon_sbd_distance,
)

from .lower_bounds import lb_shen


@njit
def ed(Q, C):
    if len(Q) != len(C):
        raise ValueError("Not equal length!")
    dist = 0.0
    for i in range(len(Q)):
        diff = Q[i] - C[i]
        dist += diff * diff
    return dist


@njit
def dtw(Q, C, r):
    m, n = len(Q), len(C)
    if abs(n - m) > r:
        raise ValueError("abs(n-m) > r!")
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, m + 1):
        for j in range(max(1, i - r), min(n, i + r) + 1):
            cost = (Q[i - 1] - C[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[m, n]


# https://stackoverflow.com/questions/66934748/how-to-stretch-an-array-to-a-new-length-while-keeping-same-value-distribution
def linear_interpolation(array: np.ndarray, new_len: int) -> np.ndarray:
    la = len(array)
    return np.interp(np.linspace(0, la - 1, num=new_len), np.arange(la), array)


@njit
def nearest_neighbor_interpolation(T, new_len):
    k = len(T)
    indices = np.rint(np.linspace(0, k - 1, new_len)).astype(np.int64)
    out = np.empty(new_len, dtype=T.dtype)
    for i in range(new_len):
        out[i] = T[indices[i]]
    return out


def nearest_neighbor_interpolation_2d(ts, new_len):
    ts = np.asarray(ts)
    k = len(ts)
    # Compute indices symmetrically
    indices = [int(round(j * (k - 1) / (new_len - 1))) for j in range(new_len)]
    return ts[indices]


def nearest_neighbor_interpolation_legacy_3(ts, new_len):
    ts = np.asarray(ts)
    k = len(ts)
    indices = np.rint(np.linspace(0, k - 1, new_len)).astype(int)
    return ts[indices]


def nearest_neighbor_interpolation_legacy_4(ts, new_len):
    ts = np.asarray(ts)
    k = len(ts)
    indices = [
        int(np.ceil(j * k / new_len)) - 1 for j in range(1, new_len + 1)
    ]  # Why -1? 1-based (in the paper) to 0-based (default in Python)
    return ts[indices]


@njit
def usdtw_prime(Q, C, L, r, dist_method):
    Q_scaled = nearest_neighbor_interpolation(Q, L)
    C_scaled = nearest_neighbor_interpolation(C, L)
    if dist_method == 0:
        # return ed(Q_scaled, C_scaled)
        return aeon_euclidean_distance(Q_scaled, C_scaled) ** 2
    elif dist_method == 1:
        # return dtw(Q_scaled, C_scaled, r)
        return aeon_dtw_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 2:
        return aeon_shape_dtw_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 3:
        return aeon_wdtw_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 4:
        return aeon_wddtw_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 5:
        return aeon_adtw_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 6:
        return aeon_erp_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 7:
        return aeon_edr_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 8:
        return aeon_msm_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 9:
        return aeon_twe_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 10:
        return aeon_lcss_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 11:
        return aeon_manhattan_distance(Q_scaled, C_scaled)
    elif dist_method == 12:
        return aeon_minkowski_distance(Q_scaled, C_scaled)
    elif dist_method == 13:
        return aeon_sbd_distance(Q_scaled, C_scaled)
    else:
        raise ValueError("Invalid distance method!")


@njit
def usdtw(Q, C, l, L, r, dist_method=0):
    m, n = len(Q), len(C)
    best_so_far = np.inf
    best_k = -1
    for k in range(math.ceil(m / l), min(math.floor(l * m), n) + 1):
        C_prefix = C[:k]
        dist = usdtw_prime(Q, C_prefix, L, r, dist_method)
        if dist < best_so_far:
            best_so_far = dist
            best_k = k
    return best_so_far, best_k


@njit
def psdtw_prime_vanilla(Q, C, l, P, r, dist_method):
    # print("Using psdtw_prime_test")
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
                        # Lower bounds
                        if np.isinf(D_cost):
                            # print("Skipping due to D_cost = inf!")
                            continue
                        if D_cost > D[i][j][p]:  # best_so_far
                            # print("Skipping due to D_cost > best_so_far!")
                            continue
                        # print(
                        #     f"Computing usdtw_prime for Q[{i_prime}:{i}] and C[{j_prime}:{j}]"
                        # )
                        # print(f"L_Q = {L_Q}, L_C = {L_C}")
                        # print(f"L_C_min = {L_C_min}, L_C_max = {L_C_max}")
                        dist_cost = usdtw_prime(
                            Q[i_prime:i],
                            C[j_prime:j],
                            L=L_max,
                            r=r,
                            dist_method=dist_method,
                        )
                        count_dist_calls += 1
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
            dist_method=0,
        )
        dist += dist_cost
    return dist


@njit
def cut_based_distance(Q, C, l, P, r, dist_method, cuts):
    m = len(Q)
    L_avg = m / P
    l_root = math.sqrt(l)
    L_max = min(int(math.floor(L_avg * l_root)), m)
    # print(cuts.shape)
    dist = 0.0
    for cut in cuts:
        print(cut[0], cut[1], cut[2], cut[3])
        dist_cost = usdtw_prime(
            Q[cut[0] : cut[1]],
            C[cut[2] : cut[3]],
            L=L_max,
            r=r,
            dist_method=dist_method,
        )
        dist += dist_cost
    return dist


###
###
###


@njit
def psdtw_prime_cache_flattened_array(Q, C, l, P, r, dist_method=0):
    m, n = len(Q), len(C)
    l_sqrt = math.sqrt(l)
    L_Q_avg = m / P
    L_Q_gmin = int(math.floor(L_Q_avg / l_sqrt))
    L_Q_gmax = int(math.ceil(L_Q_avg * l_sqrt))
    L_C_avg = n / P
    L_C_gmin = int(math.floor(L_C_avg / l_sqrt))
    L_C_gmax = int(math.ceil(L_C_avg * l_sqrt))

    # Minimum cost to align the **first i** elements of Q and the **first j** elements of C using **P** exactly segments
    D = np.full((m + 1, n + 1, P + 1), np.inf)
    D[0, 0, 0] = 0.0

    # Flattened cache
    cache_size = (m + 1) * (m + 1) * (n + 1) * (n + 1)
    dist_cache = -np.ones(cache_size, dtype=np.float64)  # -1 means "not computed"

    def cache_index(i_prime, i, j_prime, j):
        return (((i_prime * (m + 1) + i) * (n + 1) + j_prime) * (n + 1)) + j

    for p in range(1, P + 1):
        L_Q_prevs_min = (
            p - 1
        ) * L_Q_gmin  # (p-1) segments take at least "(p - 1) * L_Q_gmin" points
        L_Q_prevs_w_cur_min = (
            L_Q_prevs_min + L_Q_gmin
        )  # p segments take at least "L_Q_gmin" more points
        L_Q_prevs_max = (p - 1) * L_Q_gmax
        L_Q_prevs_w_cur_max = L_Q_prevs_max + L_Q_gmax

        for i in range(L_Q_prevs_w_cur_min, min(L_Q_prevs_w_cur_max, m) + 1):
            for L_Q in range(L_Q_gmin, L_Q_gmax + 1):
                if i - (L_Q + L_Q_prevs_min) < 0:
                    continue
                i_prime = i - L_Q
                L_C_min = int(math.floor(L_Q / l))
                L_C_max = int(math.ceil(L_Q * l))
                L_C_prevs_min = (p - 1) * L_C_gmin
                L_C_prevs_w_cur_min = L_C_prevs_min + L_C_gmin
                L_C_prevs_max = (p - 1) * L_C_gmax
                L_C_prevs_w_cur_max = L_C_prevs_max + L_C_gmax
                for j in range(L_C_prevs_w_cur_min, min(L_C_prevs_w_cur_max, n) + 1):
                    for L_C in range(L_C_min, L_C_max + 1):
                        if j - (L_C + L_C_prevs_min) < 0:
                            continue
                        j_prime = j - L_C
                        D_cost = D[i_prime, j_prime, p - 1]

                        idx = cache_index(i_prime, i, j_prime, j)
                        if dist_cache[idx] < 0:  # not computed yet
                            dist_cache[idx] = usdtw_prime(
                                Q[i_prime:i],
                                C[j_prime:j],
                                L=max(L_Q_gmax, L_C_gmax),
                                r=r,
                                dist_method=dist_method,
                            )
                        else:
                            # print("Using cached value")
                            pass
                        dist_cost = dist_cache[idx]
                        # D[i, j, p] = min(D[i, j, p], D_cost + dist_cost)
                        new_cost = D_cost + dist_cost
                        if new_cost < D[i, j, p]:
                            D[i, j, p] = new_cost
    return D[m, n, P]


@njit
def row_min(D, qi_st, p):
    n = D.shape[1]  # length along j
    min_val = np.inf
    for j in range(n):
        val = D[qi_st, j, p - 1]
        if val < min_val:
            min_val = val
    return min_val


@njit
def psdtw_prime_cache_dict(Q, C, l, P, r, dist_method=0):
    print("Using psdtw_prime_cache_dict 2")
    count_dist_calls = 0
    m, n = len(Q), len(C)
    l_sqrt = math.sqrt(l)
    L_Q_avg = m / P
    L_Q_gmin = int(math.floor(L_Q_avg / l_sqrt))
    L_Q_gmax = int(math.ceil(L_Q_avg * l_sqrt))
    L_C_avg = n / P
    L_C_gmin = int(math.floor(L_C_avg / l_sqrt))
    L_C_gmax = int(math.ceil(L_C_avg * l_sqrt))

    # DP table: min cost aligning first i of Q with first j of C using p segments
    D = np.full((m + 1, n + 1, P + 1), np.inf)
    D[0, 0, 0] = 0.0

    # Dictionary cache (keyed by indices)
    dist_cache = {}

    def get_dist(i_prime, i, j_prime, j):
        key = (i_prime, i, j_prime, j)
        if key not in dist_cache:
            dist_cache[key] = usdtw_prime(
                Q[i_prime:i],
                C[j_prime:j],
                L=max(L_Q_gmax, L_C_gmax),
                r=r,
                dist_method=dist_method,
            )
        return dist_cache[key]

    for p in range(1, P + 1):
        L_Q_prevs_min = (p - 1) * L_Q_gmin
        L_Q_prevs_w_cur_min = L_Q_prevs_min + L_Q_gmin
        L_Q_prevs_max = (p - 1) * L_Q_gmax
        L_Q_prevs_w_cur_max = L_Q_prevs_max + L_Q_gmax

        for i in range(L_Q_prevs_w_cur_min, min(L_Q_prevs_w_cur_max, m) + 1):
            for L_Q in range(L_Q_gmin, L_Q_gmax + 1):
                if i - (L_Q + L_Q_prevs_min) < 0:
                    continue
                i_prime = i - L_Q

                L_C_min = int(math.floor(L_Q / l))
                L_C_max = int(math.ceil(L_Q * l))
                L_C_prevs_min = (p - 1) * L_C_gmin
                L_C_prevs_w_cur_min = L_C_prevs_min + L_C_gmin
                L_C_prevs_max = (p - 1) * L_C_gmax
                L_C_prevs_w_cur_max = L_C_prevs_max + L_C_gmax
                for j in range(L_C_prevs_w_cur_min, min(L_C_prevs_w_cur_max, n) + 1):

                    # lower_bound = np.min(D[i_prime, :, p - 1])
                    # print(f"LB for D[{i},{j},{p}] is {lower_bound}")
                    # print(f"Current best D[{i},{j},{p}] is {D[i][j][p]}")
                    # if lower_bound > D[i][j][p]:  # best_so_far
                    #     print("Skipping due to LB")
                    #     continue

                    for L_C in range(L_C_min, L_C_max + 1):
                        if j - (L_C + L_C_prevs_min) < 0:
                            continue
                        j_prime = j - L_C
                        D_cost = D[i_prime, j_prime, p - 1]
                        # Start of Lower bounds
                        if np.isinf(D_cost):
                            # print("Skipping due to LB 1")
                            continue
                        if D_cost > D[i][j][p]:  # best_so_far
                            # print("Skipping due to LB 2")
                            continue
                        lower_bound = lb_shen(Q[i_prime:i], C[j_prime:j], l=2.0, r=0.1)
                        if D_cost + lower_bound > D[i][j][p]:
                            # print("Skipping due to LB 3")
                            continue
                        # End of Lower bounds
                        # Expensive call
                        dist_cost = get_dist(i_prime, i, j_prime, j)
                        count_dist_calls += 1
                        new_cost = D_cost + dist_cost
                        if new_cost < D[i, j, p]:
                            D[i, j, p] = new_cost

    return D[m, n, P], count_dist_calls


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
