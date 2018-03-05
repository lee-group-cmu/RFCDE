cimport cython
from libcpp cimport bool

import numpy as np
cimport numpy as np

cdef extern from "Forest.h":
    cdef cppclass Forest:
        Forest() except +

        # Methods
        void train(double* x_train, double* z_basis,
                   int n_train, int n_var, int n_basis, int n_trees, int mtry,
                   int node_size, bool fit_oob)
        void fill_weights(double* x_test, long* wt_buf);
        void fill_oob_weights(long* wt_mat);

cdef class ForestWrapper:
    """Wrapper for C++ implementation of RFCDE forests.

    Attributes
    ----------
    Cpp_Call : Forest
        The wrapped C++ object
    n_train : integer
        The number of training points.
    """

    cdef Forest* Cpp_Class
    cdef int n_train

    # Boilerplate
    def __init__(self):
        self.n_train = -1
    def __cinit__(self):
        self.Cpp_Class = new Forest()
    def __dealloc__(self):
        del self.Cpp_Class

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def train(self, np.ndarray[double, ndim=2, mode="fortran"] x_train,
              np.ndarray[double, ndim=2, mode="fortran"] z_basis, long n_trees,
              long mtry, long node_size, bool fit_oob=False):
        """Trains RFCDE on training data.

        Arguments
        ---------
        x_train : numpy matrix
            The training covariates. Must be stored in "fortran" mode.
        z_basis : numpy matrix
            The training responses evaluated at basis functions; each
            column corresponds to a basis function, each row
            corresponds to an observation. Must be stored in "fortran"
            mode.
        n_trees : integer
            The number of trees to train.
        mtry : integer
            The number of variables to evaluate at each split.
        node_size : integer
            The minimum number of observations in each leaf node.
        fit_oob : boolean
            Whether to fit out-of-bag samples. Defaults to False.
        """
        self.n_train = x_train.shape[0]

        cdef int n_train = x_train.shape[0]
        cdef int n_var = x_train.shape[1]
        cdef int n_basis = z_basis.shape[1]
        cdef int n_trees_i = n_trees;
        cdef int mtry_i = mtry;
        cdef int node_size_i = node_size;

        # Pass in pointers of numpy matrices/arrays
        self.Cpp_Class.train(&x_train[0,0], &z_basis[0,0], n_train,
                             n_var, n_basis, n_trees_i, mtry_i,
                             node_size_i, fit_oob)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fill_weights(self, np.ndarray[double, ndim=1, mode="c"] x_test,
                     np.ndarray[long, ndim=1, mode="c"] wt_buf):
        """Calculate weights from forest tree structure.

        Arguments
        ---------
        x_test : numpy array
            A new observation.

        wt_buf : numpy array
            An empty buffer to fill with weights. Must have length
            equal to the number of training points.

        Returns
        -------
        numpy array
            The weights of each training point for the new observation.

        """
        self.Cpp_Class.fill_weights(&x_test[0], &wt_buf[0])

    def weights(self, np.ndarray[double, ndim=1, mode="c"] x_test):
        wt_buf = np.zeros(self.n_train, dtype=int)
        self.fill_weights(x_test, wt_buf)
        return wt_buf


    def fill_oob_weights(self, np.ndarray[long, ndim=2, mode="fortran"] wt_mat):
        self.Cpp_Class.fill_oob_weights(&wt_mat[0,0])

    def oob_weights(self):
        wt_mat = np.zeros(self.n_train, self.n_train, dtype=int, order="F")
        self.fill_oob_weights(wt_mat)
        return wt_mat
