"""Functions for weighted kernel density estimation."""

# Copyright Taylor Pospisil 2018.
# Distributed under MIT License (http://opensource.org/licenses/MIT)

import numpy as np
import statsmodels.api as sm

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
    bandwidth : numpy array or string
        The bandwidth for the kernel density estimate; array specifies
        the diagonal of the bandwidth matrix. Strings include
        "scott", "silverman", and "normal_reference" for univariate densities and
        "normal_reference", "cv_ml", and "cv_ls" for multivariate densities.

    Returns
    -------
    numpy array
       The density evaluated at the grid points.

    """
    n_grid, n_dim = grid.shape
    n_obs, _ = responses.shape
    density = np.zeros(n_grid)

    responses = responses[weights > 0, :]
    weights = weights[weights > 0]

    if n_dim == 1:
        kde = sm.nonparametric.KDEUnivariate(responses[:, 0])
        kde.fit(bw = bandwidth, weights = weights.astype(float), fft = False)
        return kde.evaluate(grid[:, 0])
    else:
        if isinstance(bandwidth, (float, int)):
            bandwidth = [bandwidth] * n_dim
        ## Doesn't take weights so just "resample"
        weights = weights * n_obs // sum(weights)
        ids = np.repeat(range(len(weights)), weights.astype(int))
        kde = sm.nonparametric.KDEMultivariate(responses[ids, :],
                                               var_type = "c" * n_dim,
                                               bw = bandwidth)
        return kde.pdf(grid)
