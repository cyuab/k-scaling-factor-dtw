import numpy as np
from numba import njit


@njit
def lb_dummy(Q, C, l, r):
    return 0.0


@njit
def lb_kim(Q, C):
    return max(
        (Q[0] - C[0]) ** 2,
        (Q[-1] - C[-1]) ** 2,
        (np.max(Q) - np.max(C)) ** 2,
        (np.min(Q) - np.min(C)) ** 2,
    )


@njit
def lb_kim_first_last(Q, C):
    return (Q[0] - C[0]) ** 2 + (Q[-1] - C[-1]) ** 2


@njit
def lb_keogh(Q, C, r):
    n = len(Q)
    lb_sum = 0.0
    for i in range(len(C)):
        # compute envelope on-the-fly
        start = 0 if i - r < 0 else i - r  # start = max(0, i - radius)
        end = n if i + r + 1 > n else i + r + 1  # end = min(n, i + radius + 1)

        lower_entry = Q[start]
        upper_entry = Q[start]
        for k in range(start + 1, end):
            val = Q[k]
            if val < lower_entry:
                lower_entry = val
            elif val > upper_entry:
                upper_entry = val

        # accumulate deviation
        cur_val = C[i]
        if cur_val > upper_entry:
            diff = cur_val - upper_entry
            lb_sum += diff * diff
        elif cur_val < lower_entry:
            diff = cur_val - lower_entry
            lb_sum += diff * diff

    return lb_sum


def lb_keogh_legacy(ts_query, ts_candidate, radius):
    lower, upper = lb_keogh_envelope_legacy(ts_query, radius)
    lb_sum = 0
    for i in range(len(ts_candidate)):
        if ts_candidate[i] > upper[i]:
            lb_sum += (ts_candidate[i] - upper[i]) ** 2
        elif ts_candidate[i] < lower[i]:
            lb_sum += (ts_candidate[i] - lower[i]) ** 2
    return np.sqrt(lb_sum)


@njit
def lb_keogh_envelope(T, r):
    n = len(T)
    upper = np.empty(n, dtype=np.float64)
    lower = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = 0 if i - r < 0 else i - r  # start = max(0, i - radius)
        end = n if i + r + 1 > n else i + r + 1  # end = min(n, i + radius + 1)
        lower_entry = T[start]
        upper_entry = T[start]
        for k in range(start + 1, end):
            val = T[k]
            if val < lower_entry:
                lower_entry = val
            elif val > upper_entry:
                upper_entry = val
        lower[i] = lower_entry
        upper[i] = upper_entry
    return lower, upper


def lb_keogh_envelope_legacy(ts, radius):
    n = len(ts)
    upper = np.zeros(n)
    lower = np.zeros(n)
    for i in range(n):
        start = max(0, i - radius)
        end = min(n, i + radius + 1)
        lower[i] = np.min(ts[start:end])
        upper[i] = np.max(ts[start:end])
    return lower, upper


@njit
def lb_shen(q, c, l=2.0, r=0.1):
    # convert float radius to int
    r = int(min(len(q), len(c)) * r) if r < 1 else int(r)
    m = len(q)
    n = len(c)

    dist_total = (q[0] - c[0]) ** 2

    max_j = min(int(np.ceil(l * m)), n - 1)
    for j in range(1, max_j):
        start = int(max(0, np.ceil(j / l) - r))
        end = int(min(np.ceil(j * l) + r, m - 1))

        min_dist = (c[j] - q[start]) ** 2
        for k in range(start + 1, end + 1):
            d = (c[j] - q[k]) ** 2
            if d < min_dist:
                min_dist = d
        dist_total += min_dist

    dist_total += (q[-1] - c[-1]) ** 2
    return dist_total


@njit
def lb_shen_without_last(q, c, l=2.0, r=0.1):
    r = int(min(len(q), len(c)) * r) if r < 1 else int(r)
    m = len(q)
    n = len(c)

    dist_total = (q[0] - c[0]) ** 2

    max_j = min(int(np.ceil(l * m)), n - 2)
    for j in range(1, max_j):
        start = int(max(0, np.ceil(j / l) - r))
        end = int(min(np.ceil(j * l) + r, m - 1))

        min_dist = (c[j] - q[start]) ** 2
        for k in range(start + 1, end + 1):
            d = (c[j] - q[k]) ** 2
            if d < min_dist:
                min_dist = d
        dist_total += min_dist

    return dist_total


###
###
###


def lb_shen_legacy(q, c, l=1, r=0.1):
    if isinstance(r, float):
        # print("r is a float.")
        minlen = min(len(q), len(c))
        r = int(minlen * r)
    elif isinstance(r, int):
        # Do something when r is an int
        # print("r is an integer.")
        # window=r
        pass
    else:
        raise ValueError("r must be either an integer or a float.")
    m = len(q)
    n = len(c)
    dist = lambda a, b: (a - b) ** 2

    dist_total = 0
    # window_contrib.append((0, 0))
    # print(dist(q[0], c[0]))
    dist_total += dist(q[0], c[0])

    for j in range(1, min(np.ceil(l * m), n - 1)):
        start = int(max(0, np.ceil(j / l) - r))
        end = int(min(np.ceil(j * l) + r, m - 1))

        # print(j, start, end)
        q_window = q[start : end + 1]
        # print("q_window ", q_window)
        min_dist = np.min([dist(c[j], q_k) for q_k in q_window])
        # print(min_dist)
        # print(min_dist)
        # min_dist = np.min([dist(c[j], q_k) for q_k in q_window])
        # window_contrib.append((j, q_window[argmin]))
        dist_total += min_dist
    # print(dist(q[-1], c[-1]))
    dist_total += dist(q[-1], c[-1])
    return dist_total


def lb_shen_without_last_legacy(q, c, l=1, r=0.1):
    if isinstance(r, float):
        # print("r is a float.")
        minlen = min(len(q), len(c))
        r = int(minlen * r)
    elif isinstance(r, int):
        # Do something when r is an int
        # print("r is an integer.")
        # window=r
        pass
    else:
        raise ValueError("r must be either an integer or a float.")
    m = len(q)
    n = len(c)
    dist = lambda a, b: (a - b) ** 2

    dist_total = 0
    # window_contrib.append((0, 0))
    # print(dist(q[0], c[0]))
    dist_total += dist(q[0], c[0])

    for j in range(1, min(np.ceil(l * m), n - 2)):
        start = int(max(0, np.ceil(j / l) - r))
        end = int(min(np.ceil(j * l) + r, m - 1))

        # print(j, start, end)
        q_window = q[start : end + 1]
        # print("q_window ", q_window)
        min_dist = np.min([dist(c[j], q_k) for q_k in q_window])
        # print(min_dist)
        # print(min_dist)
        # min_dist = np.min([dist(c[j], q_k) for q_k in q_window])
        # window_contrib.append((j, q_window[argmin]))
        dist_total += min_dist
    # print(dist(q[-1], c[-1]))
    # dist_total += dist(q[-1], c[-1])
    return np.sqrt(dist_total)
