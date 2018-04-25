// Copyright Taylor Pospisil 2018.
// Distributed under MIT License (http://opensource.org/licenses/MIT)

#include <vector>
#include <random>
#include "Node.h"
#include "helpers.h"

typedef std::vector<int>::iterator ivecit;

Node::Node() {
  split_value = 0.0;
  loss_delta = 0.0;
  split_var = -1;
  le_child = NULL;
  gt_child = NULL;
}

Node::~Node() {
  if (le_child) delete le_child;
  if (gt_child) delete gt_child;
}

void Node::train(double* x_train, double* z_basis,
                 const std::vector<int>& weights,
                 ivecit valid_idx_begin, ivecit valid_idx_end,
                 int n_train, int n_var, int n_basis, int mtry,
                 int node_size, double min_loss_delta, int last_var) {
  // Trains a node; selects split and recursively trains children.
  //
  // Arguments:
  //   x_train: pointer to training covariates.
  //   z_basis: pointer to training basis evaluations.
  //   weights: vector of bootstrap weights.
  //   valid_idx: pointer to array of valid indices.
  //   n_train: number of observations; length of weights.
  //   n_var: number of variables.
  //   n_basis: number of basis functions.
  //   n_idx: number of valid indices; length of valid_idx.
  //   mtry: number of variables to evaluate for each split.
  //   node_size: minimum weight in each split.
  //   min_loss_delta: the minimum change in loss for a split.
  //   last_var: the variable sorted; reduces redundant sorts.
  //
  // Side-Effects:
  //   Sets the split values and children nodes for the Node. If it's a
  //   leaf it updates groups.

  this -> valid_idx_begin = valid_idx_begin;
  this -> valid_idx_end = valid_idx_end;

  Split best_split = find_best_split(x_train, z_basis, weights, valid_idx_begin,
                                     valid_idx_end, n_train, n_basis, n_var, mtry,
                                     node_size, last_var);

  if (best_split.var == -1) {
    // Couldn't find a split
    return;
  }

  if (best_split.loss_delta < min_loss_delta) {
    // Couldn't find a split that achieves the minimum decrease in loss
    return;
  }
  loss_delta = best_split.loss_delta;

  this -> split_var = best_split.var;
  if (split_var != last_var) {
    sortby(valid_idx_begin, valid_idx_end, &x_train[split_var * n_train]);
    last_var = split_var;
  }

  this -> split_value = x_train[split_var * n_train + *(valid_idx_begin + best_split.offset)];

  // Recursively train children nodes; because splits never reoccur we
  // can send each its respective part of valid_idx and recurse
  // without affecting the other side.
  le_child = new Node;
  le_child -> train(x_train, z_basis, weights,
                    valid_idx_begin, valid_idx_begin + best_split.offset + 1,
                    n_train, n_var, n_basis, mtry, node_size, last_var);

  gt_child = new Node;
  gt_child -> train(x_train, z_basis, weights,
                    valid_idx_begin + best_split.offset + 1, valid_idx_end,
                    n_train, n_var, n_basis, mtry, node_size, last_var);
}

double full_loss(double* x_train, double* z_basis,
                 const std::vector<int>& weights,
                 ivecit idx_begin, ivecit idx_end,
                 int n_train, int n_basis) {

  // Initialize total_sum and total_weight
  int total_weight = 0;
  std::vector<double> total_sum(n_basis, 0.0);
  for (auto it = idx_begin; it != idx_end; ++it) {
    total_weight += weights[*it];
    for (int bb = 0; bb < n_basis; bb++) {
      total_sum[bb] += z_basis[bb * n_train + *it] * weights[*it];
    }
  }

  double loss = 0.0;
  for (int bb = 0; bb < n_basis; bb++) {
    loss -= (total_sum[bb] * total_sum[bb]) / total_weight;
  }
  return loss;
}
