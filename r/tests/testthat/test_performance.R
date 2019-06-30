context("Test prediction performance")

test_that("Beta example performance", {
  set.seed(42)

  gen_data <- function(n) {
    x <- matrix(runif(n * 2, 0.0, 1.0), n, 2)
    z <- matrix(rbeta(n, x[, 1] + 5, x[, 2] + 5), n, 1)
    return(list(x = x, z = z))
  }

  train_data <- gen_data(1000)
  x_train <- train_data$x
  z_train <- train_data$z

  test_data <- gen_data(1000)
  x_test <- test_data$x
  z_test <- test_data$z

  n_trees <- 100
  mtry <- 2
  min_size <- 20
  n_basis <- 15
  bandwidth <- 0.1

  forest <- RFCDE(x_train, z_train, n_trees = n_trees, mtry = mtry,
                  node_size = min_size, n_basis = n_basis)

  n_grid <- 1000
  z_grid <- seq(0, 5.0, length.out = n_grid)
  density <- predict(forest, x_test, "CDE", z_grid, bandwidth = bandwidth)
  loss <- cdetools::cde_loss(density, z_grid, z_test)$loss

  expect_lt(loss, -1.8)
})
