import numpy as np
import scipy
from mpire import WorkerPool

from .kernels import rbf


class KarhunenLoeveExpansion(object):
    """Class for Karhunen-Loeve expansion.

    Implements the Karhunen-Loeve expansion based on appendix B.3 of
    https://arxiv.org/pdf/2209.14125.pdf

    Given a dataset of functions evaluated at a set of points, this class
    computes the Karhunen-Loeve expansion.
    
    Attributes:
        data: A list of data points created by toy_dataset.
        M: Number of eigenvalues to use.
        kernel: A kernel function that takes two vectors and returns a scalar.
    """
    def __init__(self, data, kernel=rbf, M=10, num_workers=8):
        self.data = data
        self.x = self.query_points(data)
        self.S = len(self.x)
        self.d = self.data[0][1].shape[1]
        self.M = M
        self.kernel = kernel
        self.num_workers = num_workers

        if self.S < self.M:
            raise ValueError('Number of points must be larger than number of '
                             'eigenvalues.')

        self.kernel_matrix = self.get_kernel_matrix(self.x)
        self.eigen_val, self.eigen_vec = scipy.linalg.eigh(self.kernel_matrix)
        self.eigen_val = self.eigen_val[::-1]
        self.eigen_vec = self.eigen_vec[:, ::-1]
        if self.eigen_val[self.M] < 1e-7:
            raise ValueError(
                'Use smaller M. Eigenvalue number {} is equal to {}, '
                'which may lead to large numerical errors.'.format(
                    self.M, self.eigen_val[self.M]))
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
        def serial_job(x, idxs):
            cov = np.zeros([len(idxs), self.S], dtype=np.float32)
            for i, idx in enumerate(idxs):
                cov[i, :] = self.kernel(x, x[idx, :])
            return idxs, cov

        split_idxs = np.array_split(np.arange(self.S),
                                    self.num_workers,
                                    axis=0)
        with WorkerPool(n_jobs=self.num_workers,
                        shared_objects=x,
                        start_method='fork') as pool:
            outputs = pool.map(serial_job, split_idxs, progress_bar=False)
        # Unpack the outputs.
        (idxs, covs) = zip(*outputs)
        covs = np.concatenate(covs, axis=0)[np.concatenate(idxs, axis=0), :]
        return covs / self.S

    def get_eigen_functions(self):
        """Compute the eigenfunctions of the covariance matrix.
        """
        eigen_fns = []
        for m in range(self.M):

            def fn(x, m=m):
                val = 0.0
                for s in range(self.S):
                    val += (self.eigen_vec[s, m] *
                            self.kernel(x, self.x[s, :]) /
                            (np.sqrt(self.S) * self.eigen_val[m]))
                return val

            eigen_fns.append(fn)

        def eigen_fn(x, eigen_fns=eigen_fns):
            def serial_job(in_shared, idxs):
                x, eigen_fns = in_shared
                return idxs, [fn(x) for fn in eigen_fns[idxs]]

            split_idxs = np.array_split(np.arange(self.M),
                                        self.num_workers,
                                        axis=0)
            with WorkerPool(n_jobs=self.num_workers,
                            shared_objects=(x, np.array(eigen_fns)),
                            start_method='fork') as pool:
                outputs = pool.map(serial_job, split_idxs, progress_bar=False)

            # Unpack the outputs.
            (idxs, fn_evals) = zip(*outputs)
            fn_evals = [fn_eval for fn_eval in fn_evals if fn_eval]
            fn_evals = np.concatenate(fn_evals,
                                      axis=0)[np.concatenate(idxs, axis=0), :]
            return fn_evals

        return eigen_fn

    def get_spectral_component(self, Y_x):
        """Compute the spectral component of the KL expansion.
        """
        spectral_comps = []

        Z = np.zeros([self.M, self.d], dtype=np.float32)
        eigen_fn_val = self.eigen_fn(Y_x[0])
        for m in range(self.M):
            Z[m, :] = 1 / len(
                Y_x[0]) * np.sum(Y_x[1] * self.eigen_val[m]**(-0.5) *
                                 eigen_fn_val[m].reshape(-1, 1),
                                 axis=0)
        return Z, eigen_fn_val

    def get_spectral_dataset(self):
        Z = []
        for i in range(len(self.data)):
            Z.append(self.get_spectral_component(self.data[i])[0])
        return Z

    def fn_approx(self, Y_x):
        Z, eigen_fn_val = self.get_spectral_component(Y_x)
        fn_val = np.zeros([len(Y_x[0]), self.d], dtype=np.float32)
        for m in range(self.M):
            fn_val += eigen_fn_val[m, :].reshape(
                -1, 1) * Z[m, :] * self.eigen_val[m]**0.5
        return fn_val
