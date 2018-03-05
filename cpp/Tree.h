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

  void train(double* x_train, double* z_basis, const std::vector<int>& weights,
             int n_train, int n_var, int n_basis, int mtry, int node_size,
             bool fit_oob);
  Node traverse(double* x_test);

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

};

#endif
