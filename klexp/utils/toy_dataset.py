import numpy as np

# Creat toy dataset of pairs of coordinates and function values
def toy_dataset(n=200, s=100, d=1, x_range=(-10, 10), same_s=True):
    """Creat quadratic toy dataset of pairs of coordinates and function values.

    This toy dataset is obtained from: https://arxiv.org/pdf/2209.14125.pdf.

    Args:
        n: Number of data points.
        s: Maximum number of points at which functions is evaluated.
        d: Dimension of the input space.
        x_range: Range of the input space.
        same_s: If True, all functions are evaluated at the same s points.

    Returns:
        A list of n data points. Each data point is a tuple of two arrays with
        the first array being the coordinates and the second array being the
        function values.
    """
    data = []
    for i in range(n):
        a = np.random.uniform(-1, 1, size=(d,))
        eps = np.random.randn()
        if d == 1:
            x = np.sort(np.random.uniform(*x_range, size=(s,)))
        else:
            x = np.sort(np.random.uniform(*x_range, size=(s, d)))
        y = a * x **2 + eps
        data.append((x, y))
    return data


