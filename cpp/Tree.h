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
             int n_train, int n_var, int n_basis, int mtry, int node_size);
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
};

#endif
