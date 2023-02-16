import numpy as np


def toy_dataset(n=200, s=100, d=1, x_range=(-10, 10), eval_pattern='same'):
    """Creat quadratic toy dataset of pairs of coordinates and function values.

    This toy dataset is obtained from: https://arxiv.org/pdf/2209.14125.pdf.

    Args:
        n: Number of data points.
        s: Maximum number of points at which functions is evaluated.
        d: Dimension of the input space.
        x_range: Range of the input space.
        eval_pattern: Whether to evaluate the function on the the same
        coordinates ('same'), random coordinates ('random'), or random
        coordinates with the same size ('same_size'). Default is 'same'.

    Returns:
        A list of n data points. Each data point is a tuple of two arrays with
        the first array being the coordinates and the second array being the
        function values.
    """
    data = []

    if eval_pattern == 'same':
        # x = np.random.uniform(*x_range, size=(s, d)).astype(np.float32)
        x = np.linspace(*x_range, s).repeat(d).reshape(s, d).astype(np.float32)
        for i in range(n):
            # a = np.random.uniform(-1, 1, size=(d, )).astype(np.float32)
            a = np.random.choice([-1.0, 1.0]).astype(np.float32)
            eps = np.random.randn()
            y = a * x**2 + eps
            data.append((x, y))

    if eval_pattern == 'same_size':
        for i in range(n):
            a = np.random.choice([-1.0, 1.0]).astype(np.float32)
            eps = np.random.randn()
            x = np.random.uniform(*x_range, size=(s, d)).astype(np.float32)
            y = a * x**2 + eps
            data.append((x, y))

    if eval_pattern == 'random':
        for i in range(n):
            a = np.random.choice([-1.0, 1.0]).astype(np.float32)
            eps = np.random.randn()
            x = np.random.uniform(*x_range, size=(np.random.randint(s),
                                                  d)).astype(np.float32)
            y = a * x**2 + eps
            data.append((x, y))

    return data


def sort_coordinates(data):
    """Sort coordinates in a toy dataset.

    Args:
        data: A list of data points created by toy_dataset.

    Returns:
        A list of data points with sorted coordinates.
    """
    if data[0][0].shape[1] != 1:
        raise ValueError(
            'Coordinates are multi-dimensional. Sorting not possible.')
    sorted_data = []
    for pair in data:
        x, y = pair
        str_idx = np.argsort(x[: ,0])
        x = x[str_idx, :]
        y = y[str_idx, :]
        sorted_data.append((x, y))
    return sorted_data