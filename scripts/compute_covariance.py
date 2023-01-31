import numpy as np

from klexp.utils import toy_dataset
from klexp import query_points, compute_covariance_matrix

# Set global parameters.
N = 20
S = 10
D = 2
X_RANGE = (-10, 10)
SAME_S = False

if __name__ == "__main__":
    data = toy_dataset(n=N, s=S, d=D, x_range=X_RANGE, same_s=SAME_S)
    x = query_points(data)
    cov = compute_covariance_matrix(x)
    print(cov)
    print(cov.shape)
