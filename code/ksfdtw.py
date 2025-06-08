import numpy as np
from pyts.metrics import dtw as _pyts_dtw_p 
from dtaidistance import ed as dtaidistance_ed
from dtaidistance import dtw, dtw_visualisation
import math
import time


def normalize(ts):
    mean = np.mean(ts)
    std = np.std(ts)
    return (ts - mean) / std

def pyts_dtw(ts1, ts2, r=0.1):
    return _pyts_dtw_p(ts1, ts2, method='fast', options={'radius': r})
    # return _pyts_dtw_p(ts1, ts2, method='sakoechiba', options={'window_size': r})

def dtai_ed(a, b, l=1, r=0.1):
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    # https://dtaidistance.readthedocs.io/en/latest/usage/ed.html
    return dtaidistance_ed.distance(a, b)

def dtai_dtw(a, b, l=1, r=0.1):
    if isinstance(r, float):
        # print("r is a float.")
        minlen = min(len(a), len(b))
        window = int(minlen * r)
    elif isinstance(r, int):
        # Do something when r is an int
        # print("r is an integer.")
        window=r
    else:
        raise ValueError("r must be either an integer or a float.")
    return dtw.distance(a, b, window=window)

# https://stackoverflow.com/questions/66934748/how-to-stretch-an-array-to-a-new-length-while-keeping-same-value-distribution
def linear_interpolation(array: np.ndarray, new_len: int) -> np.ndarray:
    la = len(array)
    return np.interp(np.linspace(0, la - 1, num=new_len), np.arange(la), array)

def nearest_neighbor_interpolation_legacy(ts, new_len):
    ts = np.asarray(ts)
    k = len(ts)
    indices = [int(np.ceil(j * k / new_len)) - 1 for j in range(1, new_len + 1)]  # Why -1? 1-based (in the paper) to 0-based (default in Python)
    return ts[indices]
    
def nearest_neighbor_interpolation(ts, new_len):
    ts = np.asarray(ts)
    k = len(ts)
    # Compute indices symmetrically
    indices = [int(round(j * (k - 1) / (new_len - 1))) for j in range(new_len)]
    return ts[indices]

def us_usdtw_p(Q, C, l, r, L, distance_method="ed"):
    # Scaling both time series
    # m = len(Q)
    # n = len(C)
    # L = min(np.ceil(l * m), n)

    Q_scaled = nearest_neighbor_interpolation(Q, L)
    C_scaled = nearest_neighbor_interpolation(C, L)

    # Compute distance based on the chosen method
    if distance_method == 'dtw':
        dist = dtai_dtw(Q_scaled, C_scaled, l, r)
    elif distance_method == 'ed':
        dist = dtai_ed(Q_scaled, C_scaled)
    else:
        raise ValueError(f"Unsupported distance method: {distance_method}")

    return dist

def us_usdtw(Q, C, l, r, L, distance_method='ed'):
    m = len(Q)
    n = len(C)
    best_so_far = np.inf
    for k in range(math.ceil(m/l), min(math.ceil(l*m), n)+1):
        C_prefix = C[:k]
        dist = us_usdtw_p(Q, C_prefix, l, r, L, distance_method)
        if dist < best_so_far:
            best_so_far = dist
            best_k = k
    return best_so_far, best_k

def lb_keogh(ts_query, ts_candidate, radius):
    lower, upper = lb_keogh_envelope(ts_query, radius)
    lb_sum = 0
    for i in range(len(ts_candidate)):
        if ts_candidate[i] > upper[i]:
            lb_sum += (ts_candidate[i] - upper[i])**2
        elif ts_candidate[i] < lower[i]:
            lb_sum += (ts_candidate[i] - lower[i])**2
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
    return np.sqrt(max((q[0]-c[0])**2,(q[-1]-c[-1])**2,(max(q)-max(c))**2,(min(q)-min(c))**2))
    
def lb_kim_fl(q, c, l=1, r=0.1):
    ensure_non_zero_len(q)
    ensure_non_zero_len(c)
    return np.sqrt((q[0]-c[0])**2+(q[-1]-c[-1])**2)

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
    dist = lambda a, b: (a - b)**2

    dist_total = 0
    # window_contrib.append((0, 0))
    # print(dist(q[0], c[0]))
    dist_total += dist(q[0], c[0]) 
    
    for j in range(1, min(np.ceil(l*m), n - 1)):
        start = int(max(0, np.ceil(j/l) - r))
        end = int(min(np.ceil(j*l) + r, m-1))
        
        # print(j, start, end)
        q_window = q[start:end+1]
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
    dist = lambda a, b: (a - b)**2

    dist_total = 0
    # window_contrib.append((0, 0))
    # print(dist(q[0], c[0]))
    dist_total += dist(q[0], c[0]) 
    
    for j in range(1, min(np.ceil(l*m), n-2)):
        start = int(max(0, np.ceil(j/l) - r))
        end = int(min(np.ceil(j*l) + r, m-1))
        
        # print(j, start, end)
        q_window = q[start:end+1]
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



def ps_distance_p(Q, C, l, r, L, P, distance_method='ed', lower_bound_method=lb_kim_fl):

    no_of_iteration = 0
    m, n = len(Q), len(C)
    LQ_avg, LC_avg = np.floor(m/P), np.floor(n/P) # Expected length
    # Minimum cost to align the **first i** elements of Q and the **first j** elements of C using P exactly segments
    D = np.full((m+1, n+1, P+1), np.inf) 

    # backtrack = np.full((m+1, n+1, P+1, 2), -1, dtype=int)  # The last dimension is used to store a pair of cutting points
     # Parent table to trace back: stores (prev_i, prev_j)
    parent = [[[(None, None) for _ in range(P + 1)] for _ in range(n + 1)] for _ in range(m + 1)] #backtracking

    # Distance cache: (qi_st, i, cj_st, j) → distance
    lb_cache = {}
    dist_cache = {}

    D[0][0][0] = 0 # Base case

    # total_iters = P * m * n
    # progress = tqdm(total=total_iters, desc="DP Progress") # For progress tracking
    # start_time = time.time() # Timing

    for p in range(1, P+1):
        for i in range(1, m+1):
            for j in range(1, n+1):
                
                # progress.update(1)
                # print("p i j", p, i, j)
                best_so_far = np.inf
                for lc in range(int(np.ceil(LQ_avg/l)), min(int(np.ceil(LQ_avg*l)), i)+1):
                    # j is not long enough for the segment in C to exist for this length =lc segment in Q 
                    if j < int(np.ceil(lc/l)):
                        continue
                    qi_st = i-lc

                    # Cascading
                    lower_bound = np.min(D[qi_st, :, p-1]) 
                    if lower_bound > D[i][j][p]: # best_so_far
                        continue
                    lower_bound = lower_bound + lb_shen_without_last(Q[qi_st:i][::-1], C[j-int(np.ceil(lc/l)):j][::-1], l, r)
                    if lower_bound > D[i][j][p]: # best_so_far
                        continue

                    for lq in range(int(np.ceil(lc/l)), min(int(np.ceil(lc*l)), j)+1):
                    # for lq in range(max(1, int(np.ceil(LC_avg/l))), min(int(np.ceil(LC_avg*l)), j)+1):
                        # if (lc/lq >l) or (lq/lc >l): # scaling constraint
                        #     continue
                        # print("Q_p C_p Q_p/C_p", Q_p, C_p, Q_p/C_p)
                        cj_st = j-lq
                        # Q[cj_st:i]
                        prev_cost = D[qi_st][cj_st][p-1]
                        # Cascading
                        if prev_cost == np.inf: # Stop expensive computation
                            continue
                        if prev_cost > D[i][j][p]: # what we currently have is the upper bound of the real value
                            continue
                        key = (qi_st, i, cj_st, j)
                        # lower bound pruning
                        if key not in lb_cache:
                            lb_cache[key] = lower_bound_method(Q[qi_st:i], C[cj_st:j], l, r)
                        if lb_cache[key] + prev_cost >= D[i][j][p]:
                            continue  # Skip: can't improve
                        if key not in dist_cache:
                            dist_cache[key] = us_usdtw_p(Q[qi_st:i], C[cj_st:j], l, r, L, distance_method)
                            no_of_iteration = no_of_iteration + 1
                            # dist_cache[key] = dtaidistance_ed.distance(Q[qi_st:i], C[cj_st:j])
                        cost = dist_cache[key]
                        # test
                        # temp1=lb_shen(Q[qi_st:i], C[j-int(np.ceil(lc/l)):j], l, r)
                        # temp2=us_usdtw_p(Q[qi_st:i], C[cj_st:j], l, r, distance_method)
                        # if temp1 > temp2:
                        #     print("i, j qi_st cj_st lb, usdtw",i, j, qi_st, cj_st, temp1, temp2)
                        #     print("Fuck. Here you are.")
                        
                        new_cost = prev_cost + cost
                        if new_cost < D[i][j][p]:
                            D[i][j][p] = new_cost
                            parent[i][j][p] = (qi_st, cj_st)
                        # D[i][j][p] = min(D[i][j][p], D[qi_st][cj_st][p-1]+cost)
    
    # progress.close()
    # elapsed = time.time() - start_time
    # print(f"Total iterations: {no_of_iteration}")
    # print(f"\nTotal time: {elapsed:.2f} seconds")
    # Backtracking to recover the cutting points
    cuts = []
    i, j, p = m, n, P
    while p > 0:
        qi_st, cj_st = parent[i][j][p]
        cuts.append(((qi_st, i), (cj_st, j)))  # Q segment, C segment
        i, j, p = qi_st, cj_st, p - 1
    cuts.reverse()
    return D[m][n][P], cuts, no_of_iteration
    # return D[m][n][P]

    # # Base case
    # p = 1 
    # for i_e in range(2, m+1): # e: ending, we need (m+1) to enumerate until m (inclusive)
    #     for j_e in range(2, n+1): # same as above
    #         m_p, n_p = i_e-0, j_e-0 # lengths of the two subsequence under comparison
    #         if (m_p > l*(n_p)) or (n_p > l*(m_p)): # scaling constraint
    #             continue
    #         if m_p < np.floor(s_Q/l) or n_p < np.floor(s_C/l): # min length of each subsequence constraint
    #             continue
    #         if m_p > np.floor(s_Q*l) or n_p > np.floor(s_C*l): # max length of each subsequence constraint
    #             continue
    #         dist = us_usdtw_p(Q[0:i_e], C[0:j_e], l, "dtw") # Sequences from 0 to i_e (j_e) (exclusive)
    #         #
    #         # dist, _ = lb_shen_argmin_q(Q[0:i_e], C[0:j_e], l)


    #         D[i_e-1, j_e-1, p-1] = dist # i_e-1 and j_e-1 as the indics for ending of Q, C in D are inclusive. 
    #         # p -1 as it counts from 0.
    #         backtrack[i_e-1, j_e-1, p-1] = [0, 0]
    #         no_of_iteration = no_of_iteration + 1
    # # Recursive case
    # for p in range(2, P+1):
    #     # Assigned length of previous subsequence constraint
    #     Q_occupied = 0
    #     C_occupied = 0
    #     Q_occupied = int(np.floor((s_Q/l)*(p-1)))
    #     C_occupied = int(np.floor((s_C/l)*(p-1)))
    #     for i_e in range(Q_occupied+2, m+1): # (Q_occupied+1) is the first available time stamps
    #         for j_e in range(C_occupied+2, n+1): # same as above
    #             best_so_far = np.inf
    #             best_indices = (-1, -1)
    #             # Iterate over all previous indices
    #             for i_s in range(Q_occupied, i_e+1):
    #                 for j_s in range(C_occupied, j_e+1):
    #                     m_p, n_p = i_e-i_s, j_e-j_s
    #                     if m_p < 2 or n_p < 2: # Their lengths should be at least 2. This constraint is also included in subsequence constraint
    #                         continue
    #                     if (m_p > l*(n_p)) or (n_p > l*(m_p)): # scaling constraint
    #                         continue    
    #                     if m_p < np.floor(s_Q/l) or n_p < np.floor(s_C/l): # min length of each subsequence constraint
    #                         continue
    #                     if m_p > np.floor(s_Q*l) or n_p > np.floor(s_C*l): # max length of each subsequence constraint
    #                         continue
    #                     dist = D[i_s-1, j_s-1, (p-1)-1] + us_usdtw_p(Q[i_s:i_e], C[j_s:j_e], l, "dtw")
    #                     # dist_cur, _ = lb_shen_argmin_q(Q[i_s:i_e], C[j_s:j_e], l)
    #                     # dist = D[i_s-1, j_s-1, (p-1)-1] + dist_cur

    #                     if dist < best_so_far:
    #                         best_so_far = dist
    #                         best_indices = (i_s, j_s)
    #                     no_of_iteration = no_of_iteration + 1
    #             D[i_e-1, j_e-1, p-1] = best_so_far
    #             backtrack[i_e-1, j_e-1, p-1] = best_indices
    # return D, backtrack


def ps_distance_p_without_prune(Q, C, l, r, L, P, distance_method='ed', lower_bound_method=lb_kim_fl):

    no_of_iteration = 0
    m, n = len(Q), len(C)
    LQ_avg, LC_avg = np.floor(m/P), np.floor(n/P) # Expected length
    # Minimum cost to align the **first i** elements of Q and the **first j** elements of C using P exactly segments
    D = np.full((m+1, n+1, P+1), np.inf) 

    # backtrack = np.full((m+1, n+1, P+1, 2), -1, dtype=int)  # The last dimension is used to store a pair of cutting points
     # Parent table to trace back: stores (prev_i, prev_j)
    parent = [[[(None, None) for _ in range(P + 1)] for _ in range(n + 1)] for _ in range(m + 1)] #backtracking

    # Distance cache: (qi_st, i, cj_st, j) → distance
    lb_cache = {}
    dist_cache = {}

    D[0][0][0] = 0 # Base case

    # total_iters = P * m * n
    # progress = tqdm(total=total_iters, desc="DP Progress") # For progress tracking
    # start_time = time.time() # Timing

    for p in range(1, P+1):
        for i in range(1, m+1):
            for j in range(1, n+1):
                
                # progress.update(1)
                # print("p i j", p, i, j)
                best_so_far = np.inf
                for lc in range(int(np.ceil(LQ_avg/l)), min(int(np.ceil(LQ_avg*l)), i)+1):
                    # j is not long enough for the segment in C to exist for this length =lc segment in Q 
                    if j < int(np.ceil(lc/l)):
                        continue
                    qi_st = i-lc

                    # # Cascading
                    # lower_bound = np.min(D[qi_st, :, p-1]) 
                    # if lower_bound > D[i][j][p]: # best_so_far
                    #     continue
                    # lower_bound = lower_bound + lb_shen_without_last(Q[qi_st:i][::-1], C[j-int(np.ceil(lc/l)):j][::-1], l, r)
                    # if lower_bound > D[i][j][p]: # best_so_far
                    #     continue

                    for lq in range(int(np.ceil(lc/l)), min(int(np.ceil(lc*l)), j)+1):
                    # for lq in range(max(1, int(np.ceil(LC_avg/l))), min(int(np.ceil(LC_avg*l)), j)+1):
                        # if (lc/lq >l) or (lq/lc >l): # scaling constraint
                        #     continue
                        # print("Q_p C_p Q_p/C_p", Q_p, C_p, Q_p/C_p)
                        cj_st = j-lq
                        # Q[cj_st:i]
                        prev_cost = D[qi_st][cj_st][p-1]
                        # Cascading
                        if prev_cost == np.inf: # Stop expensive computation
                            continue
                        if prev_cost > D[i][j][p]: # what we currently have is the upper bound of the real value
                            continue
                        key = (qi_st, i, cj_st, j)
                        # lower bound pruning
                        # if key not in lb_cache:
                        #     lb_cache[key] = lower_bound_method(Q[qi_st:i], C[cj_st:j], l, r)
                        # if lb_cache[key] + prev_cost >= D[i][j][p]:
                        #     continue  # Skip: can't improve
                        #if key not in dist_cache:
                        dist_cache[key] = us_usdtw_p(Q[qi_st:i], C[cj_st:j], l, r, L, distance_method)
                        no_of_iteration = no_of_iteration + 1
                            # dist_cache[key] = dtaidistance_ed.distance(Q[qi_st:i], C[cj_st:j])
                        cost = dist_cache[key]
                        # test
                        # temp1=lb_shen(Q[qi_st:i], C[j-int(np.ceil(lc/l)):j], l, r)
                        # temp2=us_usdtw_p(Q[qi_st:i], C[cj_st:j], l, r, distance_method)
                        # if temp1 > temp2:
                        #     print("i, j qi_st cj_st lb, usdtw",i, j, qi_st, cj_st, temp1, temp2)
                        #     print("Fuck. Here you are.")
                        
                        new_cost = prev_cost + cost
                        if new_cost < D[i][j][p]:
                            D[i][j][p] = new_cost
                            parent[i][j][p] = (qi_st, cj_st)
                        # D[i][j][p] = min(D[i][j][p], D[qi_st][cj_st][p-1]+cost)
    
    # progress.close()
    # elapsed = time.time() - start_time
    # print(f"Total iterations: {no_of_iteration}")
    # print(f"\nTotal time: {elapsed:.2f} seconds")
    # Backtracking to recover the cutting points
    cuts = []
    i, j, p = m, n, P
    while p > 0:
        qi_st, cj_st = parent[i][j][p]
        cuts.append(((qi_st, i), (cj_st, j)))  # Q segment, C segment
        i, j, p = qi_st, cj_st, p - 1
    cuts.reverse()
    return D[m][n][P], cuts, no_of_iteration