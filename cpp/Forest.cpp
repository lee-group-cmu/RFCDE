#include <random>
#include "Forest.h"
#include "Tree.h"

void Forest::train(double* x_train, double* z_basis, int n_train, int n_var,
                   int n_basis, int n_trees, int mtry, int node_size,
                   bool fit_oob) {
  // Trains a Forest object on training covariates and responses.
  //
  // Arguments:
  //   x_train: pointer to training covariates.
  //   z_basis: pointer to evaluations of basis functions on training
  //     covariates.
  //   n_train: number of training observations.
  //   n_var: number of training covariates.
  //   n_basis: number of training basis functions.
  //   n_trees: number of trees to train.
  //   mtry: number of variables to evaluate for each split.
  //   node_size: minimum weight for a leaf node.
  //   fit_oob: boolean whether to fit out-of-bag samples. Allows
  //     estimation of out-of-bag loss at the cost of increased
  //     computational effort.
  //
  // Side-Effects: populates trees with fitted trees.
  trees.resize(n_trees);
  this -> fit_oob = fit_oob;

  std::vector<int> weights(n_train, 0);

  for (int ii = 0; ii < n_trees; ii++) {
    draw_weights(weights);
    trees[ii].train(x_train, z_basis, weights, n_train, n_var, n_basis, mtry,
                    node_size, fit_oob);
  }
}

void draw_weights(std::vector<int>& weights) {
  // Draw bootstrap weights using Pois(1) random variables.
  //
  // This is an approximation to the Multinomial bootstrap weights.
  //
  // Arguments:
  //   weights: a vector of weights to fill
  //
  // Side-Effects: fills weights with draws from a Pois(1) RV.

  static std::default_random_engine generator;
  static std::poisson_distribution<int> distribution(1.0);

  for (int ii = 0; ii < weights.size(); ii++) {
    weights[ii] = distribution(generator);
  }
}
