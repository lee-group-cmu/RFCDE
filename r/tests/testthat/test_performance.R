context("Test prediction performance")

cde_loss <- function(cdes, z_grid, z_test) {
  if (!is.matrix(z_grid)) {
    z_grid <- as.matrix(z_grid)
  }

  if (!is.matrix(z_test)) {
    z_test <- as.matrix(z_test)
  }

  stopifnot(nrow(cdes) == nrow(z_test))
  stopifnot(ncol(cdes) == nrow(z_grid))
  stopifnot(ncol(z_grid) == ncol(z_test))

  z_min <- apply(z_grid, 2, min)
  z_max <- apply(z_grid, 2, max)
  z_delta <- prod(z_max - z_min) / nrow(z_grid)

  integrals <- z_delta * rowSums(cdes ^ 2)

  nn_ids <- vapply(z_test, function(xx) which.min(abs(z_grid - xx)), 1L)
  likeli <- cdes[cbind(seq_len(nrow(z_test)), nn_ids)]

  losses <- integrals - 2 * likeli

  return(mean(losses))
}

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

  forest <- RFCDE(x_train, z_train, n_trees, mtry, min_size, n_basis)

  n_grid <- 1000
  z_grid <- seq(0, 5.0, length.out = n_grid)
  density <- predict(forest, x_test, z_grid, bandwidth = bandwidth)
  loss <- cde_loss(density, z_grid, z_test)

  expect_lt(loss, -1.8)
})
