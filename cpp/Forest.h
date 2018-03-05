#ifndef FOREST_GUARD
#define FOREST_GUARD
#include "Tree.h"

class Forest {
 public:
  std::vector<Tree> trees; // vector of trees in the forest
  bool fit_oob;

  void train(double* x_train, double* z_basis,
             int n_train, int n_var, int n_basis, int n_trees, int mtry,
             int node_size, bool fit_oob);

  // Python uses longs for their integers; use template for easy
  // wrapping.
  template<class INTEGER>
  void fill_weights(double* x_test, INTEGER* wt_buf) {
    for (auto &tree : trees) {
      tree.update_weights(x_test, wt_buf);
    }
  };

  template<class INTEGER>
  void fill_oob_weights(INTEGER* wt_mat) {
    for (auto &tree : trees) {
      tree.update_oob_weights(wt_mat);
    }
  };

};

void draw_weights(std::vector<int>& weights);

#endif
