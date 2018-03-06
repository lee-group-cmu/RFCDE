#' Fits a conditional density estimate random forest to training data.
#'
#' @param x_train a matrix of training covariates.
#' @param z_train a matrix of training responses.
#' @param n_trees the number of trees in the forest. Defaults to 1000.
#' @param mtry the number of candidate variables to try for each
#'   split. Defaults to the square root of the number of covariates.
#' @param node_size the minimum number of observations in a leaf node.
#'   Defaults to 5.
#' @param n_basis the number of basis functions used in split density
#'   estimates. Defaults to 31.
#' @param basis_system the system of basis functions to use; currently
#'   "cosine" and "Haar" are supported. Defaults to "cosine"
#' @param fit_oob whether to fit out-of-bag samples or not. Out-of-bag
#'   samples increase the computation time but allows for estimation
#'   of the prediction loss. Defaults to FALSE.
#' @export
RFCDE <- function(x_train, z_train, n_trees = 1000, mtry = sqrt(ncol(x_train)),
                  node_size = 5, n_basis = 31, basis_system = "cosine",
                  fit_oob = FALSE) {
  x_train <- as.matrix(x_train)
  z_train <- as.matrix(z_train)

  mtry <- min(mtry, ncol(x_train))

  z_min <- apply(z_train, 2, min)
  z_max <- apply(z_train, 2, max)
  z_basis <- evaluate_basis(box(z_train, z_min, z_max), n_basis, basis_system)

  forest <- methods::new(ForestRcpp)
  forest$train(x_train, z_basis, n_trees, mtry, node_size, fit_oob)

  return(structure(list(z_train = z_train,
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
format.RFCDE <- function(x, ...) {
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
weights.RFCDE <- function(object, newdata, ...) {
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
#' @usage \method{predict}{RFCDE}(object, newdata, z_grid, bandwidth, ...)
#'
#' @param object a RFCDE object.
#' @param newdata matrix of test covariates.
#' @param z_grid grid points at which to evaluate the kernel density.
#' @param bandwidth (optional) bandwidth for kernel density estimates.
#'   Defaults to "auto" for automatic bandwidth selection.
#' @param \dots additional arguments
#' @importFrom stats predict
#' @export
predict.RFCDE <- function(object, newdata, z_grid, bandwidth = "auto", ...) {
  n_train <- nrow(object$z_train)
  n_dim <- ncol(object$z_train)
  stopifnot(ncol(z_grid) == n_dim)

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
  if (!is.matrix(z_grid)) {
    z_grid <- as.matrix(z_grid)
  }

  n_test <- nrow(newdata)

  cde <- matrix(NA, n_test, nrow(z_grid))
  wts <- rep(0L, n_train)
  for (ii in seq_len(n_test)) {
    wts <- weights(object, newdata[ii, , drop = FALSE]) #nolint
    wts <- wts * n_train / sum(wts)
    cde[ii, ] <- kde_estimate(object$z_train, z_grid, wts, bandwidth)
  }

  return(cde)
}
