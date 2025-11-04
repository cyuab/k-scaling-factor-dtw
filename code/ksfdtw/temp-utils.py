import numpy as np
from sklearn.preprocessing import StandardScaler


def normalize(ts):
    mean = np.mean(ts)
    std = np.std(ts)
    return (ts - mean) / std


def normalize_legacy_2(series):
    mean = series.mean()  # Calculate the mean
    std = series.std(
        ddof=0
    )  # Calculate the population standard deviation instead of the default sample standard deviation

    # Apply z-normalization formula: (x - mean) / std
    normalized_series = (series - mean) / std

    return normalized_series


def normalize_legacy_3(series):
    # Reshape the series to 2D (required by StandardScaler)
    reshaped = series.values.reshape(-1, 1)

    # Apply z-normalization
    scaler = StandardScaler()
    normalized = scaler.fit_transform(reshaped)

    # Convert back to pandas Series
    return pd.Series(normalized.flatten(), index=series.index)


def precision_at_k(distances, true_index, k):
    # Get the indices of the top-k smallest distances
    top_k_indices = sorted(range(len(distances)), key=lambda x: distances[x])[:k]

    # Check if the true match is among them
    return 1 if true_index in top_k_indices else 0


def piecewise_time_warp(
    ts,
    n_segments=4,
    stretch_range=(0.5, 2.0),
    random_state=None,
    return_metadata=False,
):
    """
    Introduce a piecewise time-warping distortion while preserving the overall shape.

    The generated sequence misaligns salient events so that Euclidean distance grows,
    whereas DTW can realign the warped sections and recover a low cost.

    Parameters
    ----------
    ts : array-like
        1D time series to distort.
    n_segments : int, default=4
        Number of contiguous segments that will each be stretched or compressed.
    stretch_range : tuple(float, float), default=(0.5, 2.0)
        Min and max multiplicative stretch to sample for each segment.
        Values < 1 compress, values > 1 stretch the segment.
    random_state : int or numpy.random.Generator, optional
        Seed or RNG for reproducibility.
    return_metadata : bool, default=False
        When True, also return the sampled breakpoints and stretch factors.

    Returns
    -------
    warped : np.ndarray
        Distorted time series with the same length as `ts`.
    metadata : dict
        Only returned when `return_metadata` is True. Contains
        `breakpoints` (segment boundaries) and `stretch_factors`.

    Notes
    -----
    The algorithm samples `n_segments-1` interior breakpoints, applies an
    independent stretch factor to each segment, concatenates the warped blocks,
    and finally interpolates back to the original length. This keeps the output
    on the same grid while creating local misalignments that DTW can resolve.
    """
    ts = np.asarray(ts, dtype=float)
    if ts.ndim != 1:
        raise ValueError("`ts` must be a 1D array-like object.")
    length = ts.size
    if length < 2:
        raise ValueError("`ts` must contain at least two samples.")

    if n_segments < 1:
        raise ValueError("`n_segments` must be a positive integer.")
    if n_segments == 1:
        if return_metadata:
            return ts.copy(), {"breakpoints": [0, length], "stretch_factors": [1.0]}
        return ts.copy()

    low, high = stretch_range
    if low <= 0 or high <= 0:
        raise ValueError("`stretch_range` values must be positive.")
    if low > high:
        raise ValueError("`stretch_range[0]` cannot exceed `stretch_range[1]`.")

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    # Sample unique breakpoints and ensure they are sorted with boundaries added.
    interior_points = np.arange(1, length)
    if n_segments - 1 >= interior_points.size:
        # Degenerates to single segment if series is too short.
        breakpoints = [0, length]
        n_segments = 1
    else:
        selected = rng.choice(
            interior_points, size=n_segments - 1, replace=False
        )
        breakpoints = [0] + sorted(int(x) for x in selected) + [length]

    warped_segments = []
    stretch_factors = []
    for start, end in zip(breakpoints[:-1], breakpoints[1:]):
        segment = ts[start:end]
        seg_len = segment.size
        if seg_len == 0:
            continue
        factor = float(rng.uniform(low, high))
        stretch_factors.append(factor)
        new_len = max(1, int(round(seg_len * factor)))
        if new_len == seg_len:
            warped_segments.append(segment.copy())
            continue
        warped_segment = np.interp(
            np.linspace(0, seg_len - 1, num=new_len),
            np.arange(seg_len),
            segment,
        )
        warped_segments.append(warped_segment)

    if not warped_segments:
        warped = ts.copy()
    else:
        concatenated = np.concatenate(warped_segments)
        warped = np.interp(
            np.linspace(0, concatenated.size - 1, num=length),
            np.arange(concatenated.size),
            concatenated,
        )

    if return_metadata:
        meta = {"breakpoints": breakpoints, "stretch_factors": stretch_factors}
        return warped, meta
    return warped


###
###
###
