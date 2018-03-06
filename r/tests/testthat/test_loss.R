context("Loss functions")

test_that("Loss estimates match integrals", {
  set.seed(42)

  n_train <- 1000
  n_test <- 10
  n_grid <- 1000

  x_train <- matrix(runif(n_train))
  z_train <- matrix(rnorm(n_train, 0, x_train))

  x_test <- matrix(runif(n_test))
  z_test <- matrix(rnorm(n_test, 0, x_test))

  n_trees <- 100
  mtry <- 2
  min_size <- 20
  n_basis <- 15
  bandwidth <- 0.1

  forest <- RFCDE(x_train, z_train, n_trees, mtry, min_size, n_basis)

  z_grid <- seq(-4, 4, length.out = n_grid)
  cdes <- predict(forest, x_test, z_grid, bandwidth = bandwidth)

  expected <- cdetools::cde_loss(cdes, z_grid, z_test)$loss

  actual <- estimate_loss(forest, bandwidth = bandwidth, method = "test",
                          x_test = x_test, z_test = z_test)

  expect_equal(actual, expected, tol = 1e-2)
})
