import numpy as np
import rfcde
import pytest


def test_joint_densities_run():
    n_grid = 10000
    n_basis = 31

    def generate_data(n):
        x = np.random.random((n, 2))
        z = np.random.random((n, 2))
        return x, z

    x_train, z_train = generate_data(1000)
    x_test, z_test = generate_data(10)

    n_trees = 100
    mtry = 2
    min_size = 20
    n_basis = 15
    bandwidth = [0.1, 0.1]

    forest = rfcde.RFCDE(n_trees=n_trees,
                         mtry=mtry,
                         node_size=min_size,
                         n_basis=n_basis)
    forest.train(x_train, z_train)

    n_grid = 30
    z1, z2 = np.meshgrid(np.linspace(0, 1, n_grid), np.linspace(0, 1, n_grid))
    z_grid = np.array([z1.flatten(), z2.flatten()]).T
    density = forest.predict(x_test, z_grid, bandwidth)
