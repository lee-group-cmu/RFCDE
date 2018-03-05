#include <vector>
#include <stdlib.h>
#include "Tree.h"
#include "Node.h"
#include "helpers.h"

void Tree::train(double* x_train, double* z_basis,
                 const std::vector<int>& weights,
                 int n_train, int n_var, int n_basis, int mtry, int node_size,
                 bool fit_oob) {
  // Train Tree object on training covariates and responses.
  //
  // Arguments:
  //   x_train: pointer to training covariates
  //   z_basis: pointer to basis function evaluatations of training responses.
  //   weights: vector of bootstrapped weights.
  //   n_train: number of training observations.
  //   n_var: number of training covariates.
  //   n_basis: number of basis functions.
  //   mtry: number of variables to evaluate for each split.
  //   node_size: minimum weight in a leaf node.
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

  this -> root.train(x_train, z_basis, weights, start_it,
                     this -> valid_idx.end(),
                     n_train, n_var, n_basis, mtry, node_size);
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
    if (x_test[cur -> split_var] <= cur -> split_value) {
      cur = cur -> le_child;
    } else {
      cur = cur -> gt_child;
    }
  }
  return *cur;
}
