"""Functions for weighted kernel density estimation."""

import numpy as np

def _gauss_kernel(vecs):
    """Evaluates Gaussian kernel.

    Arguments
    ---------
    vecs : numpy matrix
       A matrix of differences from the target point.

    Returns
    -------
    float
        The gaussian kernel evaluated from each row in `vecs`.

    """
    return np.exp(-(vecs ** 2).sum(1) / 2.0) / np.sqrt(2.0 * np.pi)

def kde(responses, grid, weights, bandwidth):
    """Calculates the weighted kernel density estimate.

    Arguments
    ---------
    responses : numpy matrix
       The training responses; each row corresponds to an observation,
       each column corresponds to a variable.
    grid : numpy matrix
        The grid points at which the KDE is evaluated.
    weights : numpy array
        A vector of weights used in the kernel density estimate. Has
        the same length as the number of rows in `responses`.
    bandwidth : float
        The bandwidth for the kernel density estimate.

    Returns
    -------
    numpy array
       The density evaluated at the grid points.

    """
    n_grid = grid.shape[0]

    tot_weight = sum(weights)

    density = np.zeros(n_grid)
    for igrid in range(n_grid):
        vecs = (grid[igrid, :] - responses) / bandwidth
        density[igrid] = np.sum(weights * _gauss_kernel(vecs))
    return density / (bandwidth * tot_weight)
