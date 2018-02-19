#' Evaluate cosine basis
#'
#' @param z points at which to evaluate the basis
#' @param n_basis number of basis functions
#' @return matrix of basis evaluations, each column corresponds to a
#'   basis function, each row corresponds to a z point
cosine_basis <- function(z, n_basis) {
  basis <- matrix(NA, length(z), n_basis)
  basis[, 1] <- 1.0
  for (ii in 1:(n_basis - 1)) {
    basis[, ii + 1] <- sqrt(2) * cospi(ii * z)
  }
  return(basis)
}

.haar_phi <- function(x) {
  if (0 <= x && x < 0.5) {
    return(1)
  } else if (0.5 <= x && x < 1) {
    return(-1)
  } else {
    return(0)
  }
}

#' Evaluate haar basis
#'
#' @param z points at which to evaluate the basis
#' @param n_basis number of basis functions
#' @return matrix of basis evaluations, each column corresponds to a
#'   basis function, each row corresponds to a z point
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
    basis[, ii] <- 2 ^ (kk / 2) * vapply(2 ^ kk * z - jj, .haar_phi, 0.0)
  }
  return(basis)
}
