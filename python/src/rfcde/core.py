"""Provides the RFCDE class for fitting RFCDE models."""

from warnings import warn

import numpy as np

from .basis_functions import evaluate_basis
from .kde import kde
from .ForestWrapper import ForestWrapper

# Helper function
def _box(responses, box_min, box_max):
    """Projects responses from box [box_min, box_max] to [0, 1].

    Arguments
    ---------
    responses : numpy array
       An array of values in [box_min, box_max] to be projected.
    box_min : float
       The minimum value of a box containing responses.
    box_max : float
       The maximum value of a box containing responses.

    Returns
    -------
    numpy array
       responses projected onto [0, 1].
    """
    return (responses - box_min) / (box_max - box_min)

class RFCDE(object):
    """Object for RFCDE.

    Arguments
    ---------
    n_trees : integer
        The number of trees to train.
    mtry : integer
        The number of variables to evaluate at each split.
    node_size : integer
       The minimum number of observations in each leaf node.
    n_basis : integer
       The number of basis functions used for split density estimates.
    basis_system : {'cosine'}
       The basis system for split density estimates.

    Attributes
    ----------
    n_trees : integer
        The number of trees to train.
    mtry : integer
        The number of variables to evaluate at each split.
    node_size : integer
       The minimum number of observations in each leaf node.
    n_basis : integer
       The number of basis functions used for split density estimates.
    basis_system : {'cosine'}
       The basis system for split density estimates.
    z_train : numpy array/matrix
       The training responses. Each value/row corresponds to an observation.
    forest : ForestWrapper
       Wrapped C++ forest

    """
    def __init__(self, n_trees, mtry, node_size, n_basis=15,
                 basis_system='cosine'):
        self.n_trees = n_trees
        self.mtry = mtry
        self.node_size = node_size
        self.n_basis = n_basis
        self.z_train = None
        self.basis_system = basis_system
        self.forest = ForestWrapper()

    def train(self, x_train, z_train):
        """Train RFCDE object on training data.

        Arguments
        ---------
        x_train : numpy array/matrix
           The training covariates. Each value/row corresponds to an
           observation.
        z_train : numpy array/matrix
           The training responses. Each value/row corresponds to an
           observation.

        """
        # Coerce to matrices
        if len(x_train.shape) == 1:
            x_train = x_train.reshape((len(x_train), 1))
        if len(z_train.shape) == 1:
            z_train = z_train.reshape((len(z_train), 1))

        self.z_train = z_train

        z_min = z_train.min(0)
        z_max = z_train.max(0)
        if self.mtry > x_train.shape[1]:
            warn("mtry larger than number of covariates; \
            setting mtry to number of covariates", RuntimeWarning)
            self.mtry = x_train.shape[1]

        z_basis = evaluate_basis(_box(z_train, z_min, z_max), self.n_basis,
                                 self.basis_system)

        self.forest.train(np.asfortranarray(x_train), np.asfortranarray(z_basis),
                          self.n_trees,
                          self.mtry, self.node_size)

    def weights(self, x_new):
        """Calculate weights from forest tree structure.

        Arguments
        ---------
        x_test : numpy array
            A new observation.

        Returns
        -------
        numpy array
            The weights of each training point for the new observation.
        """
        return self.forest.weights(x_new)

    def oob_weights(self):
        """Calculates out-of-bag weights from forest tree structure.

        Returns
        -------
        numpy matrix
            A matrix with element [ii, jj] being the out-of-bag weight
            for training point jj when predicting for training point
            ii.

        Raises
        ------
        ValueError
            If the forest was not fit with out-of-bag samples.

        """
        if not self.fit_oob:
            raise ValueError("Forest was not fit with out-of-bag samples")
        return self.forest.oob_weights()

    def predict(self, x_new, z_grid, bandwidth):
        """Calculate KDE conditional density estimate for new observations.

        Arguments
        ---------
        x_new : numpy array/matrix
           The covariates for the new observations. Each row/value
           corresponds to an observation. Must have the same
           dimensionality as the training covariates.
        z_grid : numpy array/matrix
           The grid points at which to estimate the conditional
           densities.
        bandwidth : float
           The bandwidth for the kernel density estimates.

        Returns
        -------
        numpy matrix
           A matrix of conditional density estimates; each column
          corresponds to a grid point, each row corresponds to an
          observation.

        """
        # Coerce to matrices
        if len(z_grid.shape) == 1:
            z_grid = z_grid.reshape((len(z_grid), 1))
        if len(x_new.shape) == 1:
            x_new = x_new.reshape((1, len(x_new)))

        n_test = x_new.shape[0]
        n_grid = z_grid.shape[0]
        cde = np.zeros((n_test, n_grid))
        for idx in range(n_test):
            weights = self.weights(x_new[idx, :])
            cde[idx, :] = kde(self.z_train, z_grid, weights, bandwidth)
        return cde
