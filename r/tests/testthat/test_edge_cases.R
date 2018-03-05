context("Test edge cases")

test_that("Node size is respected", {
  n <- 1000
  x <- matrix(runif(n))
  z <- matrix(runif(n))

  n_trees <- 1
  mtry <- 1
  n_basis <- 15
  for(min_size in c(1, 2, 3, 10, 100)) {
    forest <- RFCDE(x, z, n_trees, mtry, min_size, n_basis)
    expect_gte(sum(weights(forest, x[1, ])), min_size)
  }
})

test_that("Binary splits are respected", {
  n <- 1000
  x <- matrix(sample(1:2, n, replace = TRUE))
  z <- matrix(rnorm(n))

  n_trees <- 1
  mtry <- 1
  min_size <- 20
  n_basis <- 15
  forest <- RFCDE(x, z, n_trees, mtry, min_size, n_basis)
  wts1 <- weights(forest, matrix(1))
  wts2 <- weights(forest, matrix(2))

  expect_true(all(wts1[x != 1] == 0))
  expect_true(all(wts2[x != 2] == 0))
  expect_true(all((wts1 == 0) | (wts2 == 0)))
})
