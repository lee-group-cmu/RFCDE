import numpy as np


def weighted_quantile(x, weights, quantile):
    """Calculates the weighted quantile

    Arguments
    ---------
    x : numpy array
       An array of values.
    weights : numpy array
       The weights associated with each value.
    quantile : float
       The quantile to be calculated.

    Returns
    -------
    float
       The weighted quantile
    """
    perm = np.argsort(x)
    sorted_weights = weights[perm]
    ecdf = np.cumsum(sorted_weights) / sum(weights)
    return np.interp(quantile, ecdf, x[perm])
