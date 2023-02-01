import torch
import numpy as np
from tqdm import tqdm
import scipy

from .kernels import rbf


class KarhunenLoeveExpansion(object):
    """Class for Karhunen-Loeve expansion.

    Implements the Karhunen-Loeve expansion based on appendix B.3 of
    https://arxiv.org/pdf/2209.14125.pdf

    Given a dataset of functions evaluated at a set of points, this class
    computes the Karhunen-Loeve expansion.
    
    Attributes:
        data: A list of data points created by toy_dataset.
        M: Number of eigenvalues to compute.
        kernel: A kernel function that takes two vectors and returns a scalar.
    """
    def __init__(self, data, kernel=rbf):
        self.data = data
        self.x = self.query_points(data)
        self.S = len(self.x)
        self.kernel = kernel
        self.kernel_matrix = self.get_kernel_matrix(self.x)
        self.eigen_val, self.eigen_vec = scipy.linalg.eigh(self.kernel_matrix)
        self.eigen_fn = self.get_eigen_functions()

    def query_points(self, data):
        """Take data pairs as input and return the union of coordinates.

        Args:
            data: A list of data points created toy_dataset.

        Returns:
            A numpy array of shape (s, d) containing the union of coordinates.
        """
        # Initialize coordinates as a set to ensure uniqueness.
        x = set()
        for pair in data:
            if len(pair[0][0].shape) == 0:
                x.update(pair[0])
            else:
                for coord in pair[0]:
                    x.add(tuple(coord))
        return np.array(list(x))

    def get_kernel_matrix(self, x):
        """Compute the covariance matrix using a given kernel function.

        Args:
            kernel: A kernel function that takes two vectors and returns a
            scalar.
            x: A numpy array of shape (s, d) containing n d-dimensional
            vectors.

        Returns:
            A numpy array of shape (s, s) containing the covariance matrix.
        """
        s = x.shape[0]
        cov = np.zeros([s, s], dtype=np.float32)
        for j in range(s):
            cov[:, j] = self.kernel(x, x[j, :])
        return cov / s

    def get_eigen_functions(self):
        """Compute the eigenfunctions of the covariance matrix.
        """
        eigen_fns = []

        for m in range(self.S):
            # Define a lambda function to compute the eigenfunction.
            def fn(x, m=m):
                val = 0.0
                for s in range(self.S):
                    val += (self.eigen_vec[s, m] *
                            self.kernel(x, self.x[s, :]) /
                            (np.sqrt(self.S) * self.eigen_val[m]))
                return val

            eigen_fns.append(fn)

        def eigen_fn(x):
            return np.array([fn(x) for fn in eigen_fns])

        return eigen_fn

    def get_spectral_component(self, Y_x):
        """Compute the spectral component of the KL expansion.
        """
        spectral_comps = []

        Z = np.zeros([self.S, self.data[0][1].shape[1]], dtype=np.float32)
        for m in range(self.S):
            Z[m, :] = 1 / len(Y_x[0]) * np.sum(
                Y_x[1] * self.eigen_val[m]**(-0.5) *
                self.eigen_fn(Y_x[0])[m].reshape(-1, 1),
                axis=0)
        return Z

    def get_spectral_dataset(self):
        Z = []
        for i in range(len(self.data)):
            Z.append(self.get_spectral_component(self.data[i][1]))
        return Z

    def fn_approx(self, Y_x):
        Z = self.get_spectral_component(Y_x)
        return self.eigen_fn(Y_x[0]).T @ np.reshape(self.eigen_val**0.5, (-1, 1)) * Z
