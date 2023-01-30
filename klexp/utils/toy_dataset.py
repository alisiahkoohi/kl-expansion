import numpy as np

# Creat toy dataset of pairs of coordinates and function values
def toy_dataset(n=200, s=100, d=1, x_range=(-10, 10)):
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


