"""Functions for evaluation of orthogonal basis functions."""

import numpy as np

def evaluate_basis(responses, n_basis, basis_system):
    """Evaluates basis functions for given values.

    Arguments
    ---------
    responses : numpy array
      An array of responses; must be in [0, 1].
    n_basis : integer
       The number of basis functions to calculate.
    basis_system : {'cosine'}
        String denoting the system of orthogonal basis functions.

    Returns
    -------
    numpy matrix
       The basis functions evaluated at `responses`. Each column
       corresponds to a basis function, each row corresponds to a
       value of `responses`.

    Raises
    ------
    ValueError
        If the basis system isn't recognized.

    """
    systems = {'cosine':cosine_basis}
    try:
        basis_fn = systems[basis_system]
    except KeyError:
        raise ValueError("Basis system {} not recognized".format(basis_system))

    return basis_fn(responses, n_basis)

def cosine_basis(responses, n_basis):
    """Evaluates cosine basis.

    Arguments
    ----------
    responses : array
        An array of responses in [0, 1].
    n_basis : integer
        The number of basis functions to evaluate.

    Returns
    -------
    numpy matrix
        A matrix of cosine basis functions evaluated at `responses`. Each row
        corresponds to a value of `responses`, each column corresponds to a
        basis function.

    """
    n_obs = responses.shape[0]
    basis = np.empty((n_obs, n_basis))

    basis[:, 0] = 1.0
    for col in range(1, n_basis):
        basis[:, col] = np.sqrt(2) * np.cos(np.pi * col * responses).flatten()
    return basis
