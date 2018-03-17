"""Functions for weighted kernel density estimation."""

import numpy as np
import scipy.stats

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
    n_grid, n_dim = grid.shape
    density = np.zeros(n_grid)
    if isinstance(bandwidth, (float, int)):
        bandwidth = bandwidth ** 2 * np.eye(n_dim)
    for igrid in range(n_grid):
        dist = scipy.stats.multivariate_normal(mean = grid[igrid, :],
                                               cov = bandwidth)
        density[igrid] = np.sum(weights * dist.pdf(responses))
    return density / np.sum(weights)
