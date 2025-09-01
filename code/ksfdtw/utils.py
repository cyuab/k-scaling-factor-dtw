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


###
###
###
