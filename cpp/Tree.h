// Copyright Taylor Pospisil 2018.
// Distributed under MIT License (http://opensource.org/licenses/MIT)

#ifndef TREE_GUARD
#define TREE_GUARD
#include <vector>
#include "Node.h"

class Tree {
 public:
  Node root;
  int n_train;
  std::vector<int> valid_idx;
  std::vector<int> wts;
  std::vector<int> starts;
  std::vector<int> ends;

  void train(double* x_train, double* z_basis, int* lens, const std::vector<int>& weights,
             int n_train, int n_var, int n_basis, int mtry, int node_size,
             double min_loss_delta, double flambda, bool fit_oob);
  Node traverse(double* x_test);

  double calculate_feature(double* x_test, int idx) {
    double val = 0.0;
    for (int ii = this -> starts[idx]; ii < this -> ends[idx]; ++ii) {
      val += x_test[ii];
    }
    return val;
  }

  // Use template since Python uses longs and R uses ints for their
  // integer types.
  template<class INTEGER>
  void update_weights(double* x_test, INTEGER* wt_buf) {
    // Update weights for prediction on new variable.
    //
    // Arguments:
    //   x_test: pointer to test data.
    //   wt_buf: pointer to weights array.
    //
    // Side-Effects: increments the values wt_buf by the prediction
    //   weight derived from this tree.
    Node id = traverse(x_test);
    for(auto it = id.valid_idx_begin; it != id.valid_idx_end; ++it) {
      wt_buf[*it] += wts[*it];
    }
  };

  template<class INTEGER>
  void update_oob_weights_helper(INTEGER* wt_mat, Node* node) {
    if (node -> is_leaf()) {
      int n_train = wts.size();
      for(auto lt = node -> valid_idx_begin; lt != node -> valid_idx_end; ++lt) {
        for(auto rt = node -> valid_idx_begin; rt != lt; ++rt) {
          if (wts[*rt] == 0) { wt_mat[*lt * n_train + *rt] += wts[*lt]; }
          if (wts[*lt] == 0) { wt_mat[*rt * n_train + *lt] += wts[*rt]; }
        }
      }
    } else {
      update_oob_weights_helper(wt_mat, node -> le_child);
      update_oob_weights_helper(wt_mat, node -> gt_child);
    }
  };

  // Use template since Python uses longs and R uses ints for their
  // integer types.
  template<class INTEGER>
  void update_oob_weights(INTEGER* wt_mat) {
    // Traverse the tree filling in pairwise weights for each leaf
    // node.
    update_oob_weights_helper(wt_mat, &(this -> root));
  };


  void update_loss_importance(double* scores, Node* current = NULL) {
    // Update variable importance counts
    //
    // Traverse down the tree and decreased loss for the selected
    // variable.
    //
    // Arguments:
    //   scores: a pointer to a scores vector; should have length equal
    //     to the number of covariates.
    //   current: current node being processed; default is NULL for the
    //     root node.
    //   parent_loss: the loss of the parent node
    // Side-Effects: increases the score of the covariate used to
    //   split the current node; default is overwritten for root node.
    if (current == NULL) {
      current = &(this -> root);
    }

    if (!current -> is_leaf()) {
      int start = this -> starts[current -> split_var];
      int end = this -> ends[current -> split_var];
      for (int idx = start; idx < end; idx++) {
        scores[idx] += current -> loss_delta / (end - start);
      }

      update_loss_importance(scores, current -> le_child);
      update_loss_importance(scores, current -> gt_child);
    }
  };

  void update_count_importance(double* scores, Node* current = NULL) {
    // Update variable importance counts
    //
    // Traverse down the tree and increment for each time a variable
    // is selected.
    //
    // Arguments:
    //   scores: a pointer to a scores vector; should have length equal
    //     to the number of covariates.
    //   current: current node being processed; default is NULL for the
    //     root node.
    // Side-Effects: increments score of the covariate used to split
    //   the current node
    if (current == NULL) { current = &(this -> root); }

    if (!current -> is_leaf()) {
      int start = this -> starts[current -> split_var];
      int end = this -> ends[current -> split_var];
      for (int idx = start; idx < end; idx++) {
        scores[idx] += 1.0 / (end - start);
      }
      update_count_importance(scores, current -> le_child);
      update_count_importance(scores, current -> gt_child);
    }
  };
};

#endif
