import torch

# Compute the covariance matrix using a given kernel function.
def compute_covariance_matrix(kernel, x):
    """Compute the covariance matrix using a given kernel function.

    Args:
        kernel: A kernel function that takes two vectors and returns a scalar.
        x: A tensor of shape (s, d) containing n d-dimensional vectors.
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
    """Take data pairs as input and return union of coordinates.

    Args:
        data: A list of data points created toy_dataset.

    Returns:
        An array of shape (s, d) containing the union of coordinates.
    """
    x = []
    for pair in data:
        x.append(np.array(pair[0]))

    # Remove duplicates.
    # TODO: This is a hack. Find a better way to do this.
    L = {array.tobytes(): array for array in x}
    
    return np.array(list(L.values()))


