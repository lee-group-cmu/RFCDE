context("Test type conversions")

test_that("Training works for vectors", {
  set.seed(42)

  n <- 100
  x <- matrix(runif(n), n, 1)
  z <- matrix(runif(n), n, 1)

  n_trees <- 100
  mtry <- 2
  min_size <- 20
  n_basis <- 15

  expect_silent(RFCDE(x, z[, 1], n_trees, mtry, min_size, n_basis))
  expect_silent(RFCDE(x[, 1], z, n_trees, mtry, min_size, n_basis))
  expect_silent(RFCDE(x[, 1], z[, 1], n_trees, mtry, min_size, n_basis))
})

test_that("Prediction works for vectors", {
  set.seed(42)

  n <- 100
  x <- matrix(runif(n), n, 1)
  z <- matrix(runif(n), n, 1)

  z_grid <- seq(0, 1, length.out = 100)

  n_trees <- 100
  mtry <- 2
  min_size <- 20
  n_basis <- 15

  forest <- RFCDE(x, z, n_trees, mtry, min_size, n_basis)

  expect_silent(predict(forest, x, "CDE", z_grid))
  expect_silent(predict(forest, x[, 1], "CDE", z))
})
