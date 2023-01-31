import numpy as np

from klexp.utils import toy_dataset
from klexp import KarhunenLoeveExpansion, rbf

# Set global parameters.
N = 20
S = 10
D = 2
X_RANGE = (-10, 10)
EVAL_PATTERN = 'same_size'

if __name__ == "__main__":
    data = toy_dataset(n=N,
                       s=S,
                       d=D,
                       x_range=X_RANGE,
                       eval_pattern=EVAL_PATTERN)
    kl_exp = KarhunenLoeveExpansion(data, kernel=rbf)
    fns = kl_exp.get_eigen_functions()
    print(fns[0](data[0][0][0]))
