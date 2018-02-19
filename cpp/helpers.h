#ifndef HELPER_GUARD
#define HELPER_GUARD

#include <vector>
#include <algorithm>

typedef std::vector<int>::iterator ivecit;

struct SortComparator {
  const double* x;
  SortComparator(const double* x) { this -> x = x; }
  bool operator()(int li, int ri) { return x[li] < x[ri]; }
};

void sortby(ivecit begin, ivecit end, const double* x);


struct IntComparator {
  const int* w;
  IntComparator(const int* w) { this -> w = w; }
  bool operator()(int li, int ri) { return w[li] < w[ri]; }
};

void sort_next(ivecit begin, ivecit end, const int* w);

#endif
