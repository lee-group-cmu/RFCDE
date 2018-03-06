context("Test basis functions")

test_that("Cosine basis is orthonormal", {
  n_grid <- 10000
  n_basis <- 31

  z_grid <- seq(0, 1, length.out = n_grid)
  z_basis <- cosine_basis(z_grid, n_basis)

  norms <- crossprod(z_basis) / n_grid
  expect_equal(diag(norms), rep(1, n_basis), tol = 1e-3)
  diag(norms) <- 0
  expect_lt(max(abs(norms)), 1e-2)
})

test_that("Haar basis is orthonormal", {
  n_grid <- 10000
  n_basis <- 31

  z_grid <- seq(0, 1, length.out = n_grid)
  z_basis <- haar_basis(z_grid, n_basis)

  norms <- crossprod(z_basis) / n_grid
  expect_equal(diag(norms), rep(1, n_basis), tol = 1e-3)
  diag(norms) <- 0
  expect_lt(max(abs(norms)), 1e-2)
})
