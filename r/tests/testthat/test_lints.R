if (requireNamespace("lintr", quietly = TRUE)) {
  context("Lints")
  test_that("Package Style", {
    lintr::expect_lint_free()
  })
}
