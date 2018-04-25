# Copyright Taylor Pospisil 2018.
# Distributed under MIT License (http://opensource.org/licenses/MIT)

#' Helper function for kernel density estimation.
#'
#' @param z_train matrix of training responses.
#' @param z_grid matrix of grid points to evaluate densities.
#' @param weights vector of weights.
#' @param bandwidth (optional) Either "auto" for automatic bandwidth
#'   selection or a fixed bandwidth value or matrix. Defaults to "auto".
#' @return A vector of the density estimated at z_grid
kde_estimate <- function(z_train, z_grid, weights, bandwidth = "auto") {
  bandwidth <- select_bandwidth(z_train, weights, bandwidth)

  if (ncol(z_train) == 1) {
    return(ks::kde(z_train, h = bandwidth,
                   eval.points = z_grid, w = weights)$estimate)
  } else {
    return(ks::kde(z_train, H = bandwidth,
                   eval.points = z_grid, w = weights)$estimate)
  }
}

#' Helper function for selecting bandwidth for KDE
#'
#' @param z A matrix of training responses.
#' @param weights A vector of weights for training points.
#' @param method (optional) Either "auto" for automatic bandwidth
#'   selection or a fixed bandwidth value or matrix. Defaults to "auto".
#' @return A bandwidth for KDE
select_bandwidth <- function(z, weights, method = "auto") {
  if (!is.character(method)) {
    return(method)
  }
  w <- weights * sum(weights != 0) / sum(weights)
  if (ncol(z) == 1) {
    return(ks::hpi(z[rep(seq_along(weights), w), ]))
  } else {
    return(ks::Hpi(z[rep(seq_along(weights), w), ]))
  }
}

#' Select a constant bandwidth to minimize CDE loss
#'
#' @param forest A RFCDE object
#' @param bandwidths A list of bandwidths
#' @return The loss associated with each choice of bandwidth
tune_constant_bandwidth <- function(forest, bandwidths) {
  weights <- oob_weights(forest)
  loss <- rep(NA, length(bandwidths))
  for (ii in seq_along(bandwidths)) {
    bandwidth <- bandwidths[[ii]]
    loss[ii] <- kde_loss(weights, forest$z_train, forest$z_train, bandwidth)
  }
  return(loss)
}
