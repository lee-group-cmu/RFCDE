#' RFCDE: Random Forests for Conditional Density Estimation
#'
#' Fits random forest conditional density estimates using conditional
#' density loss for splits.
#'
#' @name RFCDE-package
#' @author Taylor Pospisil
#' @docType package
#' @useDynLib RFCDE
#' @importFrom Rcpp evalCpp
#' @importFrom Rcpp cpp_object_initializer
#' @importClassesFrom Rcpp "C++Object"
#' @importFrom methods new
NULL

.onUnload <- function (libpath) {
  library.dynam.unload("RFCDE", libpath)
}

Rcpp::loadModule("RFCDEModule", TRUE)
