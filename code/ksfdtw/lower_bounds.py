###
###
###
import numpy as np


def lb_keogh(ts_query, ts_candidate, radius):
    lower, upper = lb_keogh_envelope(ts_query, radius)
    lb_sum = 0
    for i in range(len(ts_candidate)):
        if ts_candidate[i] > upper[i]:
            lb_sum += (ts_candidate[i] - upper[i]) ** 2
        elif ts_candidate[i] < lower[i]:
            lb_sum += (ts_candidate[i] - lower[i]) ** 2
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


def ensure_non_zero_len(a):
    if len(a) == 0:
        raise ValueError("Error: Zero length time series.")


def lb_dummy(q, c, l=1, r=0.1):
    ensure_non_zero_len(q)
    ensure_non_zero_len(c)
    return 0


def lb_kim(q, c, l=1, r=0.1):
    ensure_non_zero_len(q)
    ensure_non_zero_len(c)
    return np.sqrt(
        max(
            (q[0] - c[0]) ** 2,
            (q[-1] - c[-1]) ** 2,
            (max(q) - max(c)) ** 2,
            (min(q) - min(c)) ** 2,
        )
    )


def lb_kim_fl(q, c, l=1, r=0.1):
    ensure_non_zero_len(q)
    ensure_non_zero_len(c)
    return np.sqrt((q[0] - c[0]) ** 2 + (q[-1] - c[-1]) ** 2)


def lb_shen(q, c, l=1, r=0.1):
    ensure_non_zero_len(q)
    ensure_non_zero_len(c)
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
    return np.sqrt(dist_total)


def lb_shen_without_last(q, c, l=1, r=0.1):
    ensure_non_zero_len(q)
    ensure_non_zero_len(c)
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
