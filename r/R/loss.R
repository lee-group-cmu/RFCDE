# Copyright Taylor Pospisil 2018.
# Distributed under MIT License (http://opensource.org/licenses/MIT)

#' Calculate loss for a single kernel density estimate
#'
#' @param weights A vector of weights for the training points.
#' @param z_train A matrix of training responses
#' @param z_test A test response
#' @param h A bandwidth
#' @return The CDE loss for the kernel density estimate
single_kde_loss <- function(weights, z_train, z_test, h) {
  n_train <- nrow(z_train)
  n_dim <- ncol(z_train)

  if (n_dim == 1) {
    det <- h
    invh <- 1 / h ^ 2
  } else {
    det <- det(h)
    invh <- solve(h)
  }

  d_train <- matrix(NA, n_train, n_train)
  for (ii in seq_len(n_train)) {
    for (jj in seq_len(n_train)) {
      delta <- z_train[jj, ] - z_train[ii, ]
      d_train[ii, jj] <- t(delta) %*% invh %*% delta
    }
  }

  d_test <- rep(NA, n_train)
  for (jj in seq_len(n_train)) {
    delta <- z_test - z_train[jj, ]
    d_test[jj] <- t(delta) %*% invh %*% delta
  }

  term1 <- sum(tcrossprod(weights) * exp(-d_train / 4))
  term2 <- sum(weights * stats::dnorm(sqrt(d_test), 0, 1) / det)

  term1 <- term1 * (2 * pi) ^ (-n_dim / 2) / (det * sqrt(2))

  return(term1 - 2 * term2)
}

#' Calculate loss for KDE estimates
#'
#' @param weights A matrix of weights for the training points.
#' @param z_train A matrix of training responses
#' @param z_test A matrix of test responses
#' @param bandwidth (optional) A bandwidth or "auto" for automatic
#'   bandwidth selection.
#' @return The CDE loss for the kernel density estimate
kde_loss <- function(weights, z_train, z_test, bandwidth = "auto") {
  losses <- sapply(seq_len(nrow(z_test)), function(ii) {
    bandwidth <- select_bandwidth(z_train, weights[ii, ], bandwidth)
    return(single_kde_loss(weights[ii, ], z_train, z_test[ii, ], bandwidth))
  })
  return(mean(losses))
}

#' Estimate loss for RFCDE
#'
#' @param forest An RFCDE object
#' @param bandwidth (optional) A bandwidth or "auto" for automatic
#'   bandwidth selection.
#' @param method (optional) A string: either "oob" for out-of-bag
#'   weights or "validation" for a validation data set
#' @param x_test (optional) The test covariates if using
#'   `method="test"`.
#' @param z_test (optional) The test responses if using
#'   `method="test"`.

#' @return The CDE loss for the kernel density estimate
estimate_loss <- function(forest, bandwidth = "auto", method = "oob",
                          x_test = NULL, z_test = NULL) {
  if (method == "oob") {
    weights <- oob_weights(forest)
    weights <- weights / rowSums(weights)
    weights[is.nan(weights)] <- 0.0
    return(kde_loss(weights, forest$z_train, forest$z_train, bandwidth))
  } else if (method == "test") {
    weights <- weights(forest, x_test)
    weights <- weights / rowSums(weights)
    weights[is.nan(weights)] <- 0.0
    return(kde_loss(weights, forest$z_train, z_test, bandwidth))
  } else {
    stop("Loss estimation method not recognized")
  }
}
