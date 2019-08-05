"""Provides the RFCDE class for fitting RFCDE models."""

# Copyright Taylor Pospisil 2018.
# Distributed under MIT License (http://opensource.org/licenses/MIT)

from warnings import warn

import numpy as np

from .basis_functions import evaluate_basis
from .kde import kde
from .weighted_quantile import weighted_quantile
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
    min_loss_delta : float
       The minimum change in loss for a split.
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
    min_loss_delta : float
       The minimum change in loss for a split.
    n_basis : integer
       The number of basis functions used for split density estimates.
    basis_system : {'cosine'}
       The basis system for split density estimates.
    z_train : numpy array/matrix
       The training responses. Each value/row corresponds to an observation.
    fit_oob: boolean
       Whether the forest has fit out-of-bag samples.
    n_var : integer
       Number of training covariates
    lens : numpy array
       The lengths of functional variables; scalar variables will have a length of 1.
    forest : ForestWrapper
       Wrapped C++ forest

    """

    def __init__(self,
                 n_trees,
                 mtry,
                 node_size,
                 min_loss_delta=0.0,
                 n_basis=15,
                 basis_system='cosine'):
        self.n_trees = n_trees
        self.mtry = mtry
        self.node_size = node_size
        self.min_loss_delta = min_loss_delta
        self.n_basis = n_basis
        self.z_train = None
        self.basis_system = basis_system
        self.lens = None
        self.forest = ForestWrapper()

    def train(self, x_train, z_train, lens=None, flambda=1.0, fit_oob=False):
        """Train RFCDE object on training data.

        Arguments
        ---------
        x_train : numpy array/matrix
           The training covariates. Each value/row corresponds to an
           observation.
        z_train : numpy array/matrix
           The training responses. Each value/row corresponds to an
           observation.
        lens : numpy array
           The lengths of functional variables. Defaults to treating
           each variable as a scalar.
        flambda : float
           The functional splitting parameter
        fit_oob : boolean
           Whether to fit out-of-bag observations.

        """
        # Coerce to matrices
        if len(x_train.shape) == 1:
            x_train = x_train.reshape((len(x_train), 1))
        if len(z_train.shape) == 1:
            z_train = z_train.reshape((len(z_train), 1))

        self.n_var = x_train.shape[1]
        self.z_train = z_train

        if not lens:
            lens = np.array([1] * self.n_var, dtype=np.intc)
        if lens.dtype != np.intc:
            lens = lens.astype(np.cint)

        z_min = z_train.min(0)
        z_max = z_train.max(0)
        if self.mtry > x_train.shape[1]:
            warn(
                "mtry larger than number of covariates; \
            setting mtry to number of covariates", RuntimeWarning)
            self.mtry = x_train.shape[1]

        z_basis = evaluate_basis(_box(z_train, z_min, z_max), self.n_basis,
                                 self.basis_system)

        self.forest.train(np.asfortranarray(x_train),
                          np.asfortranarray(z_basis), np.asfortranarray(lens),
                          self.n_trees, self.mtry, self.node_size,
                          self.min_loss_delta, flambda, fit_oob)
        self.fit_oob = fit_oob

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
        if len(x_new.shape) != 1 or len(x_new) != self.n_var:
            raise ValueError("x_new must have same dimensions as x_train")
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
        bandwidth : float or string
           The bandwidth for the kernel density estimates. For
           automatic bandwidth selection use "normal_reference",
           "cv_ml", and "cv_ls" for reference, maximum likelihood
           cross validation, and least-squares cross validation
           respectively.

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
            weights = self.weights(np.ascontiguousarray(x_new[idx, :]))
            cde[idx, :] = kde(self.z_train, z_grid, weights, bandwidth)
        return cde

    def predict_mean(self, x_new):
        """Calculate conditional mean estimate for new observations.

        Arguments
        ---------
        x_new : numpy array/matrix
           The covariates for the new observations. Each row/value
           corresponds to an observation. Must have the same
           dimensionality as the training covariates.

        Returns
        -------
        numpy array
           An array of conditional mean estimates.
        """
        # Coerce to matrix
        if len(x_new.shape) == 1:
            x_new = x_new.reshape((1, len(x_new)))

        n_test = x_new.shape[0]
        means = np.zeros(n_test)
        for idx in range(n_test):
            weights = self.weights(np.ascontiguousarray(x_new[idx, :]))
            means[idx] = np.average(self.z_train.reshape(-1, ),
                                    weights=weights)
        return means

    def predict_quantile(self, x_new, quantile):
        """Calculate conditional quantile estimate for new observations.

        Arguments
        ---------
        x_new : numpy array/matrix
           The covariates for the new observations. Each row/value
           corresponds to an observation. Must have the same
           dimensionality as the training covariates.
        quantile : float
           The quantile to estimate (between 0 and 1).

        Returns
        -------
        numpy array
           An array of conditional quantile estimates.
        """
        # Coerce to matrix
        if len(x_new.shape) == 1:
            x_new = x_new.reshape((1, len(x_new)))

        n_test = x_new.shape[0]
        quantiles = np.zeros(n_test)
        for idx in range(n_test):
            weights = self.weights(np.ascontiguousarray(x_new[idx, :]))
            quantiles[idx] = weighted_quantile(self.z_train.reshape(-1, ),
                                               weights, quantile)
        return quantiles
