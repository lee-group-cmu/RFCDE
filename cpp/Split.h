// Copyright Taylor Pospisil 2018.
// Distributed under MIT License (http://opensource.org/licenses/MIT)

#ifndef SPLIT_GUARD
#define SPLIT_GUARD
#include <vector>

typedef std::vector<int>::iterator ivecit;

class Split {
 public:
  int var;
  int offset;
  double loss_delta;

 Split() : var(-1), offset(-1), loss_delta(0.0) {}
};

Split find_best_split(double* x_train, double* z_basis,
                      const std::vector<int>& weights,
                      ivecit idx_begin, ivecit idx_end,
                      int n_train, int n_basis, int n_var, int mtry, int node_size,
                      int& last_var);

Split evaluate_split(const double* x_train, const double* z_basis,
                     const std::vector<int>& weights,
                     const ivecit idx_begin, const ivecit idx_end,
                     int n_train, int n_basis, int node_size,
                     int total_weight, const std::vector<double>& total_sum);

#endif
