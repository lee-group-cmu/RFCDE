import numpy as np
import rfcde
import pytest


def cde_loss(cdes, z_grid, true_z):
    """Calculates conditional density estimation loss on holdout data

    Arguments
    ---------
    cdes : numpy matrix
        A matrix of conditional density estimates. Each column
        corresponds to a grid point, each row corresponds to an
        observation.
    z_grid : numpy array/matrix
        The grid points at which `cdes` is evaluated.
    true_z : numpy array/matrix
        The true z values corresponding to the rows of cdes.

    Returns
    -------
    float
        The CDE loss (up to a constant) for the CDE estimator on the
        holdout data.

    """
    n_obs, _ = cdes.shape

    term1 = np.mean(np.trapz(cdes**2, z_grid))

    nns = [np.argmin(np.abs(z_grid - true_z[ii])) for ii in range(n_obs)]
    term2 = np.mean(cdes[range(n_obs), nns])
    return term1 - 2 * term2


def test_beta_example_performance():
    def generate_data(n):
        x = 5.0 * np.random.random((n, 2))
        z = np.random.beta(x[:, 0] + 5, x[:, 1] + 5, n)
        return x, z

    x_train, z_train = generate_data(1000)
    x_test, z_test = generate_data(1000)

    n_trees = 100
    mtry = 2
    min_size = 20
    n_basis = 15
    bandwidth = 0.1

    forest = rfcde.RFCDE(n_trees=n_trees,
                         mtry=mtry,
                         node_size=min_size,
                         n_basis=n_basis)
    forest.train(x_train, z_train)

    n_grid = 1000
    z_grid = np.linspace(0, 1, n_grid)
    density = forest.predict(x_test, z_grid, bandwidth)
    assert cde_loss(density, z_grid, z_test) < -1.8
