context("Test edge cases")

test_that("node_size is respected", {
  n <- 1000
  x <- matrix(runif(n))
  z <- matrix(runif(n))

  n_trees <- 1
  mtry <- 1
  n_basis <- 15
  for(min_size in c(1, 2, 3, 10, 100)) {
    forest <- RFCDE(x, z, n_trees, mtry, min_size, n_basis)
    wts <- rep(0L, n)
    expect_gte(sum(weights(forest, x[1, ])), min_size)
  }
})
