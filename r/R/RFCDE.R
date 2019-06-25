# Copyright Taylor Pospisil 2018.
# Distributed under MIT License (http://opensource.org/licenses/MIT)

#' Fits a conditional density estimate random forest to training data.
#'
#' @param x_train a matrix of training covariates.
#' @param z_train a matrix of training responses.
#' @param lens length of functional data
#' @param n_trees the number of trees in the forest. Defaults to 1000.
#' @param mtry the number of candidate variables to try for each
#'   split. Defaults to the square root of the number of covariates.
#' @param node_size the minimum number of observations in a leaf node.
#'   Defaults to 5.
#' @param n_basis the number of basis functions used in split density
#'   estimates. Defaults to 31.
#' @param basis_system the system of basis functions to use; currently
#'   "cosine" and "Haar" are supported. Defaults to "cosine"
#' @param min_loss_delta the minimum loss for a split. Defaults to 0.0.
#' @param flambda the functional splitting parameter.
#' @param fit_oob whether to fit out-of-bag samples or not. Out-of-bag
#'   samples increase the computation time but allows for estimation
#'   of the prediction loss. Defaults to FALSE.
#' @export
RFCDE <- function(x_train, z_train, lens = rep(1L, ncol(x_train)), #nolint
                  n_trees = 1000, mtry = sqrt(ncol(x_train)),
                  node_size = 5, n_basis = 31, basis_system = "cosine",
                  min_loss_delta = 0.0, flambda = 1.0, fit_oob = FALSE) {
  x_train <- as.matrix(x_train)
  z_train <- as.matrix(z_train)

  mtry <- min(mtry, ncol(x_train))

  stopifnot(sum(lens) == ncol(x_train))

  z_min <- apply(z_train, 2, min)
  z_max <- apply(z_train, 2, max)
  z_basis <- evaluate_basis(box(z_train, z_min, z_max), n_basis, basis_system)

  forest <- methods::new(ForestRcpp)
  forest$train(x_train, z_basis, lens, n_trees, mtry, node_size,
               min_loss_delta, flambda, fit_oob)

  x_names <- colnames(x_train)
  if (is.null(x_names)) {
    x_names <- 1:ncol(x_train)
  }

  return(structure(list(z_train = z_train,
                        x_names = x_names,
                        fit_oob = fit_oob,
                        n_x = ncol(x_train),
                        rcpp = forest), class = "RFCDE"))
}

#' Print method for RFCDE objects
#'
#' @param x A RFCDE object.
#' @param ... Other arguments to print
print.RFCDE <- function(x, ...) {
  cat(format(x, ...), sep = "\n")
}

#' Summary method for RFCDE objects
#'
#' @param x A RFCDE object.
#' @param ... Other arguments to summary
summary.RFCDE <- function(x, ...) {
  print(x)
}

#' Format method for RFCDE objects
#'
#' @param x A RFCDE object.
#' @param ... Other arguments to format
format.RFCDE <- function(x, ...) { #nolint
  return(c("RFCDE object:",
           paste("n_train =", nrow(x$z_train)),
           paste("OOB =", x$fit_oob)))
}

#' Obtains weights from RFCDE object.
#'
#' Provides weights for the training data reflecting co-occurance in
#' leaf nodes of the forest.
#'
#' @usage \method{weights}{RFCDE}(object, newdata, ...)
#'
#' @param object A RFCDE object.
#' @param newdata A vector of test covariates.
#' @param \dots Other arguments
#' @return A vector of weights counting the number of co-occurances in
#'   leaf nodes of the forest.
#' @importFrom stats weights
#' @export
weights.RFCDE <- function(object, newdata, ...) { #nolint
  if (is.vector(newdata)) {
    if (length(newdata) == object$n_x) {
      newdata <- matrix(newdata, nrow = 1)
    } else if (object$n_x == 1) {
      newdata <- matrix(newdata, ncol = 1)
    } else {
      stop("Prediction must have same number of covariates as training.")
    }
  }

  stopifnot(is.matrix(newdata))
  stopifnot(ncol(newdata) == object$n_x)

  wts <- matrix(NA, nrow(newdata), nrow(object$z_train))
  for (ii in seq_len(nrow(newdata))) {
    tmp <- rep(0L, nrow(object$z_train))
    object$rcpp$fill_weights(newdata[ii, ], tmp)
    wts[ii, ] <- tmp
  }
  return(wts)
}

#' Calculate out-of-bag weights.
#'
#' @param forest A RFCDE object.
#' @return A matrix of out-of-bag weights for the training data.
oob_weights <- function(forest) {
  stopifnot(forest$fit_oob)
  n_train <- nrow(forest$z_train)
  weights <- matrix(0L, n_train, n_train)
  forest$rcpp$fill_oob_weights(weights)
  return(weights)
}

#' Predict conditional density estimates for RFCDE objects.
#'
#' @usage \method{predict}{RFCDE}(object, newdata, response, ...)
#'
#' @param object a RFCDE object.
#' @param newdata matrix of test covariates.
#' @param response the type of response to predict; "CDE" for full
#' conditional densities, "mean" for conditional means, "quantile"
#' for conditional quantiles.
#' @param z_grid (optional) grid points at which to evaluate the kernel density.
#' @param bandwidth (optional) bandwidth for kernel density estimates.
#'   Defaults to "auto" for automatic bandwidth selection.
#' @param quantile (optional) quantile to estimate
#' @param \dots additional arguments
#' @importFrom stats predict
#' @export
predict.RFCDE <- function(object, newdata,
                          response = c("CDE", "mean", "quantile"),
                          z_grid = NULL, bandwidth = "auto", quantile = NULL,
                          ...) {
  if (is.vector(newdata)) {
    if (length(newdata) == object$n_x) {
      newdata <- matrix(newdata, nrow = 1)
    } else if (object$n_x == 1) {
      newdata <- matrix(newdata, ncol = 1)
    } else {
      stop("Prediction must have same number of covariates as training.")
    }
  }

  stopifnot(is.matrix(newdata))
  stopifnot(ncol(newdata) == object$n_x)

  n_test <- nrow(newdata)
  n_train <- nrow(object$z_train)
  n_dim <- ncol(object$z_train)

  if (response == "CDE") {
    if (!is.matrix(z_grid)) {
      z_grid <- as.matrix(z_grid)
    }
    stopifnot(ncol(z_grid) == n_dim)

    cde <- matrix(NA, n_test, nrow(z_grid))
    wts <- rep(0L, n_train)
    for (ii in seq_len(n_test)) {
      wts <- weights(object, newdata[ii, , drop = FALSE]) #nolint
      wts <- wts * n_train / sum(wts)
      cde[ii, ] <- kde_estimate(object$z_train, z_grid, wts, bandwidth)
    }
    return(cde)
  } else if (response == "mean") {
    means <- rep(NA, n_test)
    wts <- rep(0L, n_train)
    for (ii in seq_len(n_test)) {
      wts <- weights(object, newdata[ii, , drop = FALSE]) #nolint
      wts <- wts * n_train / sum(wts)
      means[ii] <- weighted.mean(object$z_train, wts)
    }
    return(means)
  } else if (response == "quantile") {
    quantiles <- rep(NA, n_test)
    wts <- rep(0L, n_train)
    for (ii in seq_len(n_test)) {
      wts <- weights(object, newdata[ii, , drop = FALSE]) #nolint
      wts <- wts * n_train / sum(wts)
      quantiles[ii] <- Hmisc::wtd.quantile(object$z_train, weights = wts,
                                           probs = quantile)
    }
    return(quantiles)
  } else {
      stop("Response type not recognized")
  }
}

#' Calculate variable importance measures for RFCDE.
#'
#' @param forest a RFCDE object
#' @param type the type of importance measure; options are "count" and
#'   "loss".
#' @export
variable_importance <- function(forest, type = c("count", "loss")) {
  n_x <- length(forest$x_names)
  imp <- rep(0.0, n_x)
  type <- match.arg(type)
  if (type == "count") {
    forest$rcpp$fill_count_importance(imp)
  } else if (type == "loss") {
    forest$rcpp$fill_loss_importance(imp)
    counts <- rep(0.0, n_x)
    forest$rcpp$fill_count_importance(counts)
    imp <- imp / counts
  }
  names(imp) <- forest$x_names
  return(imp)
}
