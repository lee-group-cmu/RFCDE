// Copyright Taylor Pospisil 2018.
// Distributed under MIT License (http://opensource.org/licenses/MIT)

#ifndef NODE_GUARD
#define NODE_GUARD
#include <algorithm>
#include <vector>
#include "Split.h"

typedef std::vector<int>::iterator ivecit;

class Node {
public:
  double split_value; // value used to perform split; 0.0 if leaf node.
  int split_var; // variable used to perform split; -1 if leaf node.
  double loss_delta; // difference in density loss for this split.
  Node* le_child; // pointer to <= child. NULL if no children.
  Node* gt_child; // pointer to > child. NULL if no children.
  ivecit valid_idx_begin;
  ivecit valid_idx_end;

  Node();
  ~Node();

  bool is_leaf() {
    return(this -> split_var == -1);
  }

  void train(double* x_train, double* z_basis,
             const std::vector<int>& weights,
             ivecit valid_idx_begin, ivecit valid_idx_end,
             int n_train, int n_var, int n_basis, int mtry,
             int node_size, double min_loss_delta, int last_var=-1);
};

double full_loss(double* x_train, double* z_basis,
                 const std::vector<int>& weights,
                 ivecit idx_begin, ivecit idx_end,
                 int n_train, int n_basis);

#endif
