#include "helpers.h"
#include <vector>
#include <algorithm>

void sortby(ivecit begin, ivecit end, const double* x) {
  std::sort(begin, end, SortComparator(x));
}

void sort_next(ivecit begin, ivecit end, const int* w) {
  std::sort(begin, end, IntComparator(w));
}
