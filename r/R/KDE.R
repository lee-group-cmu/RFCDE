# Copyright Taylor Pospisil 2018.
# Distributed under MIT License (http://opensource.org/licenses/MIT)

#' Helper function for kernel density estimation.
#'
#' @param z_train matrix of training responses.
#' @param z_grid matrix of grid points to evaluate densities.
#' @param weights vector of weights.
#' @param bandwidth (optional) Either "plugin" for bandwidth selection by
#'     plug-in rule, "cv" for cross-validation, or a fixed bandwidth
#'     value or matrix. Defaults to "plugin".
#' @return A vector of the density estimated at z_grid
kde_estimate <- function(z_train, z_grid, weights, bandwidth = "plugin") {
  bandwidth <- select_bandwidth(z_train, weights, bandwidth)

  if (ncol(z_train) == 1) {
    estimates <- ks::kde(z_train, h = bandwidth,
                         eval.points = z_grid, w = weights)$estimate
  } else {
    estimates <- ks::kde(z_train, H = bandwidth,
                         eval.points = z_grid, w = weights)$estimate
  }

  return(pmax(0.0, estimates)) # sometimes density is numerically less
                               # than zero
}

#' Helper function for selecting bandwidth for KDE
#'
#' @param z A matrix of training responses.
#' @param weights A vector of weights for training points.
#' @param method (optional) Either "plugin" for bandwidth selection by
#'     plug-in rule, "cv" for cross-validation, or a fixed bandwidth
#'     value or matrix. Defaults to "plugin".
#' @return A bandwidth for KDE
select_bandwidth <- function(z, weights, method = "plugin") {
  if (!is.character(method)) {
    return(method)
  }
  w <- weights * sum(weights != 0) / sum(weights)
  rep_indices <- rep(seq_along(weights), w)
  if (method == "auto") {
      method <- "plugin"
      warning("method=\"auto\" is deprecated; please use method=\"plugin\"")
  }
  if (method == "plugin") {
      if (ncol(z) == 1) {
          return(ks::hpi(z[rep_indices, ]))
      } else {
          return(ks::Hpi(z[rep_indices, ]))
      }
  } else if (method == "cv") {
      if (ncol(z) == 1) {
          return(ks::hscv(z[rep_indices, ]))
      } else {
          return(ks::Hscv(z[rep_indices, ]))
      }
  }
  stop(paste("method =", method, "not recognized"))
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
