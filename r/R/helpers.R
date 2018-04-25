# Copyright Taylor Pospisil 2018.
# Distributed under MIT License (http://opensource.org/licenses/MIT)

#' Project from [z_min, z_max] to [0, 1].
#'
#' @param z A vector of grid points to project
#' @param z_min A vector of minimum values for each dimension.
#' @param z_max A vector of maximum values for each dimension.
#' @return A matrix of z projected onto the unit cube.
box <- function(z, z_min, z_max) {
  return(scale(z, z_min, z_max - z_min))
}
