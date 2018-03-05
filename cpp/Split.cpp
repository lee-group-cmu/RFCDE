#include <vector>
#include <random>
#include "Split.h"
#include "helpers.h"

typedef std::vector<int>::iterator ivecit;

Split find_best_split(double* x_train, double* z_basis,
                      const std::vector<int>& weights,
                      ivecit idx_begin, ivecit idx_end,
                      int n_train, int n_basis, int n_var, int mtry, int node_size, int& last_var) {

  // Initialize total_sum and total_weight
  int total_weight = 0;
  std::vector<double> total_sum(n_basis, 0.0);
  for (auto it = idx_begin; it != idx_end; ++it) {
    total_weight += weights[*it];
    for (int bb = 0; bb < n_basis; bb++) {
      total_sum[bb] += z_basis[bb * n_train + *it] * weights[*it];
    }
  }

  Split best_split;

  // Can quit early if not enough weight for split
  if (total_weight < 2 * node_size) {
    return best_split;
  }

  int seed = 43;
  static std::mt19937 rng(seed);
  std::uniform_int_distribution<int> unif(0, n_var - 1);

  // Find best recursive split

  for (int ii = 0; ii < mtry; ii++) {
    int var = unif(rng);
    if (var != last_var) {
      sortby(idx_begin, idx_end, &x_train[var * n_train]);
      last_var = var;
    }

    Split split = evaluate_split(x_train, z_basis, weights, idx_begin, idx_end,
                                 n_train, n_basis, node_size,
                                 total_weight, total_sum);

    if (split.loss < best_split.loss) {
      best_split = split;
      best_split.var = var;
    }
  }

  return best_split;
}


Split evaluate_split(const double* x_train, const double* z_basis,
                     const std::vector<int>& weights,
                     const ivecit idx_begin, const ivecit idx_end,
                     int n_train, int n_basis, int node_size,
                     int total_weight, const std::vector<double>& total_sum) {
  // Finds the best split given an ordering of observations.
  //
  // Find best_split by incrementing through each observation
  // maintaining a running sum of the basis function values. Then
  // evaluating loss as -\sum_{j} \beta_{j}^{2} where
  // \beta_{j} = \sum_{i} w_{i}\beta_{j}(x_{i}) / \sum_{i} w_{i}
  // for each side of the split.
  //
  // Arguments:
  //   x_train: pointer to covariate vector.
  //   z_basis: pointer to basis function evaluations.
  //   weights: vector of bootstrap weights.
  //   idx: pointer to array of valid indices.
  //   n_train: number of observations, length of weights.
  //   n_basis: number of basis functions.
  //   n_idx: number of valid indices.
  //   node_size: minimum weight for a leaf node.
  //   total_weight: sum of weights for valid indices. Computed
  //     outside of this function as it can be reused for other
  //     splits.
  //   total_sum: vector of weighted sums of basis functions. Computed
  //     outside of this function as it can be reused for other
  //     splits.
  //
  // Returns: a Split object containing the best split loss and offset.
  int le_weight = 0;
  std::vector<double> le_sum(n_basis, 0.0);

  Split best_split;
  for (auto it = idx_begin; it != idx_end; ++it) {
    le_weight += weights[*it];
    for (int bb = 0; bb < n_basis; bb++) {
      le_sum[bb] += z_basis[bb * n_train + *it] * weights[*it];
    }

    // Enforce node_size constraint on minimum weight in a leaf node.
    if (le_weight <= node_size || total_weight - le_weight <= node_size) {
      continue;
    }

    double loss = 0.0;
    for (int bb = 0; bb < n_basis; bb++) {
      loss -= le_sum[bb] * le_sum[bb] / le_weight;
      loss -= (total_sum[bb] - le_sum[bb]) * (total_sum[bb] - le_sum[bb]) /
        (total_weight - le_weight);
    }

    if (loss < best_split.loss && x_train[*it] != x_train[*(it + 1)]) {
      best_split.loss = loss;
      best_split.offset = it - idx_begin;
    }
  }
  return best_split;
}
