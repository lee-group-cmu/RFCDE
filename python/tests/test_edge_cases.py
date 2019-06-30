import numpy as np
import rfcde
import pytest


def test_node_size_is_respected():
    n = 1000
    x = np.random.random((n, 1))
    z = np.random.random(n)

    n_trees = 1
    mtry = 1
    n_basis = 15
    for min_size in [1, 2, 3, 10, 100]:
        forest = rfcde.RFCDE(n_trees=n_trees,
                             mtry=mtry,
                             node_size=min_size,
                             n_basis=n_basis)
        forest.train(x, z)
        assert sum(forest.weights(x[0, :])) >= min_size
