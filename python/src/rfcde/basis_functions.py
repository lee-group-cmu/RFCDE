"""Functions for evaluation of orthogonal basis functions."""

import numpy as np

def evaluate_basis(responses, n_basis, basis_system):
    """Evaluates a system of basis functions.

    Arguments
    ----------
    responses : array
        An array of responses in [0, 1].
    n_basis : integer
        The number of basis functions to calculate.
    basis_system : {'cosine', 'Fourier', 'db4'}
        String denoting the system of orthogonal basis functions.

    Returns
    -------
    numpy matrix
       A matrix of basis functions evaluations. Each row corresponds
       to a value of `responses`, each column corresponds to a basis function.

    Raises
    ------
    ValueError
        If the basis system isn't recognized.

    """
    systems = {'cosine' : cosine_basis}
    try:
        basis_fn = systems[basis_system]
    except KeyError:
        raise ValueError("Basis system {} not recognized".format(basis_system))

    n_dim = responses.shape[1]
    if n_dim  == 1:
        return basis_fn(responses, n_basis)
    else:
        if isinstance(n_basis, int):
            n_basis = [n_basis] * n_dim
        return tensor_basis(responses, n_basis, basis_fn)

def tensor_basis(responses, n_basis, basis_fn):
    """Evaluates tensor basis.

    Combines single-dimensional basis functions \phi_{d}(z) to form
    orthogonal tensor basis $\phi(z_{1}, \dots, z_{D}) = \prod_{d}
    \phi_{d}(z_{d})$.

    Arguments
    ---------
    responses : numpy matrix
        A matrix of responses in [0, 1]^(n_dim). Each column
        corresponds to a variable, each row corresponds to an
        observation.
    n_basis : list of integers
        The number of basis function for each dimension. Should have
        the same length as the number of columns of `responses`.
    basis_fn : function
        The function which evaluates the one-dimensional basis
        functions.

    Returns
    -------
    numpy matrix
        Returns a matrix where each column is a basis function and
        each row is an observation.

    """
    n_obs, n_dims = responses.shape

    basis = np.ones((n_obs, np.prod(n_basis)))
    period = 1
    for dim in range(n_dims):
        sub_basis = basis_fn(responses[:, dim], n_basis[dim])
        col = 0
        for _ in range(np.prod(n_basis) // (n_basis[dim] * period)):
            for sub_col in range(n_basis[dim]):
                for _ in range(period):
                    basis[:, col] *= sub_basis[:, sub_col]
                    col += 1
        period *= n_basis[dim]
    return basis


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

    responses = responses.flatten()

    basis[:, 0] = 1.0
    for col in range(1, n_basis):
        basis[:, col] = np.sqrt(2) * np.cos(np.pi * col * responses)
    return basis
