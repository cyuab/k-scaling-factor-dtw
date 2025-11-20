import numpy as np
from numba import njit
import math
from .utils import nearest_neighbor_interpolation

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

@njit
def euclidean_distance(x, y):
    """
    Calculate Euclidean distance between two sequences using numba.

    Parameters:
    x, y: array-like sequences of the same length

    Returns:
    float: Euclidean distance
    """
    if len(x) != len(y):
        raise ValueError("Sequences must have the same length")

    distance = 0.0
    for i in range(len(x)):
        distance += (x[i] - y[i]) ** 2

    return distance
@njit
def dtw(Q, C, window=None):
    m, n = len(Q), len(C)

    # Convert fractional window to absolute value
    if window is None:
        r = max(m, n)  # No constraint if window is None
    else:
        r = int(window * max(m, n))  # Convert fraction to absolute value

    if abs(n - m) > r:
        raise ValueError("abs(n-m) > r!")

    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, m + 1):
        for j in range(max(1, i - r), min(n, i + r) + 1):
            cost = (Q[i - 1] - C[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[m, n]


@njit
def usdtw_prime(Q, C, L, r, dist_method):
    Q_scaled = nearest_neighbor_interpolation(Q, L)
    C_scaled = nearest_neighbor_interpolation(C, L)
    if dist_method == 0:
        return aeon_squared_distance(Q_scaled, C_scaled)
    elif dist_method == 1:
        return aeon_dtw_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 2:
        return aeon_adtw_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 3:
        return aeon_ddtw_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 4:
        return aeon_erp_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 5:
        return aeon_edr_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 6:
        return aeon_lcss_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 7:
        return aeon_manhattan_distance(Q_scaled, C_scaled)
    elif dist_method == 8:
        return aeon_minkowski_distance(Q_scaled, C_scaled)
    elif dist_method == 9:
        return aeon_msm_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 10:
        return aeon_sbd_distance(Q_scaled, C_scaled)
    elif dist_method == 11:
        return aeon_shape_dtw_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 12:
        return aeon_twe_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 13:
        return aeon_wddtw_distance(Q_scaled, C_scaled, window=r)
    elif dist_method == 14:
        return aeon_wdtw_distance(Q_scaled, C_scaled, window=r)
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
                            continue
                        if D_cost > D[i][j][p]:  # best_so_far
                            continue
                        dist_cost = usdtw_prime(
                            Q[i_prime:i][::-1],
                            C[j_prime:j][::-1],
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


@njit(inline="always")
def lb_shen_prefix(Q, C, l, r):
    m = len(Q)
    r_int = int(r * max(len(Q), len(C)))  # Convert fraction to integer value
    dist_total = (Q[0] - C[0]) ** 2
    for j in range(1, m):
        start = int(max(0, np.ceil(j / l) - r_int))
        end = int(min(np.ceil(j * l) + r_int, m - 1))
        min_dist = (C[j] - Q[start]) ** 2
        for k in range(start + 1, end + 1):
            d = (C[j] - Q[k]) ** 2
            if d < min_dist:
                min_dist = d
        dist_total += min_dist
    return dist_total


@njit(inline="always")
def lb_shen_incremental(Q, C, l, r):
    m = len(Q)
    j_last = len(C) - 1
    start = int(max(0, np.ceil(j_last / l) - r))
    end = int(min(np.ceil(j_last * l) + r, m - 1))
    min_dist = (C[j_last] - Q[start]) ** 2
    for k in range(start + 1, end + 1):
        d = (C[j_last] - Q[k]) ** 2
        if d < min_dist:
            min_dist = d
    return min_dist


@njit
def psdtw_prime_lb_shen(Q, C, l, r, P, dist_method):
    print("Using psdtw_prime_test3")
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
                    lb = 0.0
                    for L_C in range(L_C_min, L_C_max + 1):
                        j_prime = j - L_C
                        D_cost = D[i_prime, j_prime, p - 1]
                        # Lower bounds
                        if np.isinf(D_cost):
                            continue
                        if D_cost > D[i][j][p]:  # best_so_far
                            continue
                        if L_C == L_C_min:
                            lb = lb_shen_prefix(
                                Q[i_prime:i][::-1], C[j_prime:j][::-1], l=l, r=r
                            )
                        else:
                            # lb += lb_shen_incremental(
                            #     Q[i_prime:i][::-1], C[j_prime:j][::-1], l=l, r=r
                             lb = lb_shen_prefix(
                                Q[i_prime:i][::-1], C[j_prime:j][::-1], l=l, r=r
                            )
                            # lb = lb_shen_prefix(Q[i_prime:i][::-1], C[j_prime:j][::-1], l=l, r=r)
                        if D_cost + lb > D[i][j][p]:  # best_so_far
                            continue
                        dist_cost = usdtw_prime(
                            Q[i_prime:i][::-1],
                            C[j_prime:j][::-1],
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
def psdtw_prime_lb_shen_test_temp(Q, C, l, r, P, dist_method):
    # print("psdtw_prime_lb_shen_test")
    count_dist_calls = 0
    m = len(Q)
    n = len(C)
    l_root = math.sqrt(l)
    L_Q_gavg = m / P
    L_Q_gmin = max(1, int(math.ceil(L_Q_gavg / l_root)))
    L_Q_gmax = min(int(math.floor(L_Q_gavg * l_root)), m)
    L_C_gavg = n / P
    L_C_gmin = max(1, int(math.ceil(L_C_gavg / l_root)))
    L_C_gmax = min(int(math.floor(L_C_gavg * l_root)), n)
    L_gmax = max(L_Q_gmax, L_C_gmax)

    # Minimum cost to align the **first i** elements of Q (i.e., Q[:i]) and the **first j** elements of C (i.e., C[:j]) using **P** exactly segments
    # Each segment satisfies the length constraint
    D = np.full((m + 1, n + 1, P + 1), np.inf)
    D[0, 0, 0] = 0.0
    D_cut = np.full((m + 1, n + 1, P + 1, 2), -1, dtype=np.int64)

    for p in range(1, P + 1):
        # p segments in Q take at least (L_Q_gmin * p) points
        # L_acc_max = L_max * p

        for i in range(L_Q_gmin * p, min(L_Q_gmax * p, m) + 1):
            for L_Q in range(L_Q_gmin, L_Q_gmax + 1):
                i_prime = i - L_Q
                if i_prime < 0:
                    continue
                Q_segment = Q[i_prime:i][::-1]
        

                L_C_min = max(L_C_gmin, int(math.ceil(L_Q / l)))
                L_C_max = min(int(math.floor(L_Q * l)), L_C_gmax)

                r_int = int(
                    r * max(len(Q_segment), L_C_max)
                )  # Convert fraction to integer value
                # print("r_int:", r_int)
                windows_sorted = []
                for k in range(0, L_C_max):
                    # print("k:", k)
                    start = int(max(0, np.ceil(k / l) - r_int))
                    end = int(min(np.ceil(k * l) + r_int + 1, len(Q_segment)))
                    window = Q_segment[start:end]
                    windows_sorted.append(np.sort(window))
                # print("windows_sorted:", windows_sorted)

                for j in range(L_C_gmin * p, min(L_C_gmax * p, n) + 1):
                    for L_C in range(L_C_min, L_C_max + 1):
                        j_prime = j - L_C
                        if j_prime < 0:
                            continue
                        D_cost = D[i_prime, j_prime, p - 1]
                        # Lower bounds
                        if np.isinf(D_cost):
                            continue
                        if D_cost > D[i][j][p]:  # D[i][j][p] stores the best_so_far
                            continue
                        C_segment = C[j_prime:j][::-1]
                        LB = 0.0
                        if len(C_segment) == L_C_min:
                            for k in range(0, L_C_min):
                                delta_val = delta(C_segment[k], windows_sorted[k])
                                LB += delta_val
                        else:
                            delta_val = delta(
                                C_segment[L_C - 1], windows_sorted[L_C - 1]
                            )
                            # print(delta_val)
                            LB += delta_val
                        if D_cost + LB >= D[i][j][p]:
                            continue
                        # dist_cost = usdtw_prime(
                        #     Q[i_prime:i][::-1],
                        #     C[j_prime:j][::-1],
                        #     L=L_gmax,
                        #     r=r,
                        #     dist_method=dist_method,
                        # )
                        dist_cost = usdtw_prime(
                            Q_segment,
                            C_segment,
                            L=L_gmax,
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
def cut_based_distance(Q, C, l, r, P, dist_method, cuts):
    m = len(Q)
    L_avg = m / P
    l_root = math.sqrt(l)
    L_max = min(int(math.floor(L_avg * l_root)), m)
    # print(cuts.shape)
    dist = 0.0
    for cut in cuts:
        # print(cut[0], cut[1], cut[2], cut[3])
        dist_cost = usdtw_prime(
            Q[cut[0] : cut[1]][::-1],
            C[cut[2] : cut[3]][::-1],
            L=L_max,
            r=r,
            dist_method=dist_method,
        )
        # dist_cost = distance_measure_prime(
        #     Q[cut[0] : cut[1]],
        #     C[cut[2] : cut[3]],
        #     r=r,
        #     dist_method=dist_method,
        # )
        dist += dist_cost
    return dist