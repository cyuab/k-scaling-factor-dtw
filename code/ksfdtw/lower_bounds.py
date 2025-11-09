import numpy as np
from numba import njit

# |Q| = m, |C| = n
# q_i, c_j
# l: scaling factor of US
# r: radius for the band of DTW

@njit
def lb_kim(Q, C, r=1, l=1):  # l, r are unused.
    return max(
        (Q[0] - C[0]) ** 2,
        (Q[-1] - C[-1]) ** 2,
        (np.max(Q) - np.max(C)) ** 2,
        (np.min(Q) - np.min(C)) ** 2,
    )


@njit
def lb_kim_fl(Q, C, r=1, l=1):  # l, r are unused.
    return (Q[0] - C[0]) ** 2 + (Q[-1] - C[-1]) ** 2


def lb_keogh_envelope(T, r):
    n = len(T)
    upper = np.zeros(n)
    lower = np.zeros(n)
    for i in range(n):
        start = max(0, i - r)
        end = min(n, i + r + 1)
        lower[i] = np.min(T[start:end])
        upper[i] = np.max(T[start:end])
    return lower, upper

def lb_keogh(Q, C, r, l=1):  # l is unused.
    L_Q, U_Q = lb_keogh_envelope(Q, r)
    lb_sum = 0
    for i in range(len(C)):
        if C[i] > U_Q[i]:
            lb_sum += (C[i] - U_Q[i]) ** 2
        elif C[i] < L_Q[i]:
            lb_sum += (C[i] - L_Q[i]) ** 2
    return lb_sum


@njit
def lb_keogh_envelope_njit(T, r):
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


@njit
def lb_keogh_njit(Q, C, r, l=1):  # l is unused.
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


def lb_shen(Q, C, r, l):  # l is unused.
    m = len(Q)
    n = len(C)
    dist = lambda a, b: (a - b) ** 2

    lb_sum = 0
    lb_sum += dist(Q[0], C[0])

    for j in range(1, min(np.floor(l * m), n - 1)):
        start = int(max(0, np.ceil(j / l) - r))
        end = int(min(np.ceil(j * l) + r, m - 1))
        q_window = Q[start : end + 1]
        min_dist = np.min([dist(C[j], q_k) for q_k in q_window])
        lb_sum += min_dist

    lb_sum += dist(Q[-1], C[-1])

    return lb_sum


@njit
def lb_shen_njit(Q, C, r, l):  # l is unused.
    m = len(Q)
    n = len(C)
    lb_sum = (Q[0] - C[0]) ** 2

    max_j = min(int(np.ceil(l * m)), n - 1)
    
    for j in range(1, max_j):
        start = int(max(0, np.ceil(j / l) - r))
        end = int(min(np.ceil(j * l) + r, m - 1))

        min_dist = (C[j] - Q[start]) ** 2  # Start with the first element in the window
        for k in range(start + 1, end + 1):
            dist = (C[j] - Q[k]) ** 2
            if dist < min_dist:
                min_dist = dist
        lb_sum += min_dist

    lb_sum += (Q[-1] - C[-1]) ** 2
    return lb_sum