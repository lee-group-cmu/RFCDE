// Copyright Taylor Pospisil 2018.
// Distributed under MIT License (http://opensource.org/licenses/MIT)

#include <vector>
#include <random>
#include <stdlib.h>
#include "Tree.h"
#include "Node.h"
#include "helpers.h"

void Tree::train(double* x_train, double* z_basis, int* lens,
                 const std::vector<int>& weights,
                 int n_train, int n_var, int n_basis, int mtry, int node_size,
                 double min_loss_delta, double flambda, bool fit_oob) {
  // Train Tree object on training covariates and responses.
  //
  // Arguments:
  //   x_train: pointer to training covariates
  //   z_basis: pointer to basis function evaluatations of training responses.
  //   lens: length of functional variables
  //   weights: vector of bootstrapped weights.
  //   n_train: number of training observations.
  //   n_var: number of training covariates.
  //   n_basis: number of basis functions.
  //   mtry: number of variables to evaluate for each split.
  //   node_size: minimum weight in a leaf node.
  //   min_loss_delta: the minimum change in loss for a split.
  //   flambda: lambda used for grouping functional data
  //   fit_oob: boolean whether to fit out-of-bag samples. Allows
  //     estimation of out-of-bag loss at the cost of increased
  //     computational effort.
  //
  // Side-Effects:
  //   Builds a tree for prediction in root.
  this -> n_train = n_train;

  std::vector<int> valid_idx(n_train);
  for (int ii = 0; ii < n_train; ii++) { valid_idx[ii] = ii; }


  this -> valid_idx = valid_idx;
  this -> wts = weights;

  // Select features
  std::vector<int> starts;
  std::vector<int> ends;
  int idx = 0;
  int lens_id = 0;
  int cur_len = lens[lens_id];

  static auto rng = std::default_random_engine {};
  std::poisson_distribution<int> rpois(flambda);

  while(idx < n_var) {
    int jump = rpois(rng);
    if (jump > cur_len) {
      jump = cur_len;
      // todo: fix this?
    }
    if (jump == 0) { continue; }
    cur_len -= jump;
    starts.push_back(idx);
    idx += jump;
    ends.push_back(idx);
    if (cur_len == 0 && idx != n_var) {
      lens_id += 1;
      cur_len = lens[lens_id];
    }
  }

  n_var = ends.size();
  double* xs_train = (double*) malloc(sizeof(double) * n_var * n_train);
  for (int ii = 0; ii < n_var; ii++) {
    for (int jj = 0; jj < n_train; jj++) {
      double val = 0.0;
      for (int idx = starts[ii]; idx < ends[ii]; ++idx) {
        val += x_train[idx * n_train + jj];
      }
      xs_train[ii * n_train + jj] = val;
    }
  }

  // Determine starting index
  auto start_it = this -> valid_idx.begin();
  if (!fit_oob) {
    // Remove observations with zero weight to avoid performance hit of
    // sorting/summing over them.
    sort_next(this -> valid_idx.begin(), this -> valid_idx.end(), weights.data());
    for (; start_it != this -> valid_idx.end(); ++start_it) { if(weights[*start_it] > 0) { break; } }

    if (start_it == this -> valid_idx.end()) {
      start_it = this -> valid_idx.begin();
    }
  }

  this -> starts = starts;
  this -> ends = ends;

  mtry = std::min(n_var, mtry);
  this -> root.train(xs_train, z_basis, weights, start_it,
                     this -> valid_idx.end(),
                     n_train, n_var, n_basis, mtry, node_size, min_loss_delta);

  free(xs_train);

}

Node Tree::traverse(double* x_test) {
  // Traverses tree to determine id for leaf node.
  //
  // Arguments:
  //   x_test: pointer to a new observation
  //
  // Returns: the leaf node in which x_test ends up.
  Node* cur = &(this -> root);
  while (cur -> split_var != -1) {
    if (calculate_feature(x_test, cur -> split_var) <= cur -> split_value) {
      cur = cur -> le_child;
    } else {
      cur = cur -> gt_child;
    }
  }
  return *cur;
}
