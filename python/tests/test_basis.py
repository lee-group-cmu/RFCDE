import numpy as np
import rfcde
import pytest

def test_cosine_basis_is_orthonormal():
    n_grid  = 10000
    n_basis = 31

    grid = np.linspace(0, 1, n_grid)
    basis = rfcde.basis_functions.cosine_basis(grid, n_basis)

    norms = np.matmul(basis.T, basis) / n_grid

    assert np.all(norms.diagonal() == pytest.approx(1, abs=1e-3))
    np.fill_diagonal(norms, 0.0)
    assert np.abs(norms).max() == pytest.approx(0.0, abs=1e-3)
