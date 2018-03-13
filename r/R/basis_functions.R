#' Calculate basis functions for new observations
#'
#' @param z matrix whose row correspond to observations and whose
#'   columns correspond to dimensions. Assumes each column has been
#'   scaled to [0, 1].
#' @param n_basis integer number of how many basis functions should be
#'   calculated for each dimension. If not same length as the number
#'   of columns of `z` will recycle.
#' @param system Basis system to be used. If not same length as the
#'   number of columns of `z` will recycle.
#'
#' @return A matrix of dimension nrow(z) by prod(n_basis) with entries
#'   consisting of the first n_basis functions in the (tensor) basis
#'   evaluated at the points z
evaluate_basis <- function(z, n_basis, system = "cosine") {
  if (!is.matrix(z)) {
    z <- as.matrix(z)
  }

  n_dim <- ncol(z)
  if (n_dim == 1) {
    system <- match.arg(system, c("cosine"))
    basis_function <- switch(system,
                             cosine = cosine_basis,
                             Haar = haar_basis)
    return(basis_function(z, n_basis))
  } else {
    if (length(n_basis) == 1) {
      n_basis <- rep(n_basis, n_dim)
    }
    if (length(system) == 1) {
      system <- rep(system, n_dim)
    }
    return(tensor_basis(z, n_basis, system))
  }
}

#' Calculate tensor basis functions for new observations
#'
#' @param z matrix whose row correspond to observations and whose
#'   columns correspond to dimensions. Assumes each column has been
#'   scaled to [0, 1].
#' @param n_basis integer vector of how many basis functions should be
#'   calculated for each dimension. If not same length as the number
#'   of columns of `z` will recycle.
#' @param system vector of strings indicating the basis system to be
#'   used. Options are "cosine", "Fourier", "Haar", and "Daubechies".
#'   If not same length as the number of columns of `z` will recycle.
#'
#' @return A matrix of dimension nrow(z) by prod(n_basis) with entries
#'   consisting of the first n_basis functions in the tensor basis
#'   evaluated at the points z
tensor_basis <- function(z, n_basis, systems = "Fourier") {
  n_var <- ncol(z)
  stopifnot(length(n_basis) == n_var)
  stopifnot(length(systems) == n_var)

  tensor_basis <- matrix(NA, nrow(z), prod(n_basis))

  bases <- lapply(1:n_var, function(ii) {
    return(evaluate_basis(z[, ii], n_basis[ii], systems[ii]))
  })

  ids <- expand.grid(lapply(1:n_var, function(ii) {
    return(seq_len(n_basis[ii]))
  }))

  for (ii in 1:prod(n_basis)) {
    id <- ids[ii, ]
    tensor_basis[, ii] <- do.call("*", lapply(1:n_var, function(nn) {
      return(bases[[nn]][, id[[nn]]])
    }))
  }

  return(tensor_basis)
}

#' Evaluates cosine basis for new observations
#' @inheritParams evaluate_basis
#' @return A matrix of dimension length(z) by n_basis with entries
#'   consisting of the first n_basis cosine basis functions evaluated
#'   at the points z
cosine_basis <- function(z, n_basis) {
  basis <- matrix(NA, length(z), n_basis)
  basis[, 1] <- 1.0
  for (ii in 1:(n_basis - 1)) {
    basis[, ii + 1] <- sqrt(2) * cospi(ii * z)
  }
  return(basis)
}

#' Evaluates Haar mother wavlet
#' @param x: float, value at which to evaluate the wavlet
#' @return float; the Haar mother wavlet evaluated at x
.haar_phi <- function(x) {
  if (0 <= x && x < 0.5) {
    return(1)
  } else if (0.5 <= x && x < 1) {
    return(-1)
  } else {
    return(0)
  }
}

#' Evaluates Haar basis for new observations
#' @inheritParams calculateBasis
#' @return A matrix of dimension length(z) by n_basis with entries
#'   consisting of the first n_basis Haar basis functions evaluated
#'   at the points z
haar_basis <- function(z, n_basis) {
  basis <- matrix(NA, length(z), n_basis)
  basis[, 1] <- 1.0

  kk <- 0
  jj <- 0

  for (ii in 2:n_basis) {
    if (jj == 2 ^ kk - 1) {
      kk <- kk + 1
      jj <- 0
    } else {
      jj <- jj + 1
    }
    basis[, ii] <- 2 ^ (kk / 2) * sapply(2 ^ kk * z - jj, .haar_phi)
  }

  return(basis)
}
