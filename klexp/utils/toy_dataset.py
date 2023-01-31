import numpy as np


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
    x = np.sort(np.random.uniform(*x_range, size=(s, ))).astype(np.float32)
    for i in range(n):
        a = np.random.uniform(-1, 1, size=(d, )).astype(np.float32)
        eps = np.random.randn()
        if not same_s:
            if d == 1:
                x = np.sort(np.random.uniform(*x_range, size=(s, ))).reshape(
                    s, 1).astype(np.float32)
            else:
                x = np.sort(np.random.uniform(*x_range,
                                              size=(s, d))).astype(np.float32)
        y = a * x**2 + eps
        data.append((x, y))
    return data
