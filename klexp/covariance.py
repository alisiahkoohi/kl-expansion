import torch
import numpy as np

from .kernels import rbf


def compute_covariance_matrix(x, kernel=rbf):
    """Compute the covariance matrix using a given kernel function.

    Args:
        kernel: A kernel function that takes two vectors and returns a scalar.
        This must be implemented as a torch function.

        x: A tensor of shape (s, d) containing n d-dimensional vectors. This
        must be implemented as a torch tensor.

    Returns:
        A tensor of shape (s, s) containing the covariance matrix.
    """
    s = x.shape[0]
    cov = torch.zeros(s, s)
    for i in range(s):
        for j in range(i, s):
            cov[i, j] = kernel(x[i], x[j])
            cov[j, i] = cov[i, j]
    return cov


def query_points(data):
    """Take data pairs as input and return the union of coordinates.

    Args:
        data: A list of data points created toy_dataset.

    Returns:
        A torch tensor of shape (s, d) containing the union of coordinates.
    """
    # Initialize coordinates as a set to ensure uniqueness.
    x = set()
    for pair in data:
        if len(pair[0][0].shape) == 0:
            x.update(pair[0])
        else:
            for coord in pair[0]:
                x.add(tuple(coord))
    return torch.tensor(list(x))
