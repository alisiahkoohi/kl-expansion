import numpy as np

from klexp.utils import toy_dataset
from klexp import KarhunenLoeveExpansion, rbf

# Set global parameters.
N = 20
S = 10
D = 1
X_RANGE = (-10, 10)
EVAL_PATTERN = 'same'

if __name__ == "__main__":
    data = toy_dataset(n=N,
                       s=S,
                       d=D,
                       x_range=X_RANGE,
                       eval_pattern=EVAL_PATTERN)
    kl_exp = KarhunenLoeveExpansion(data, kernel=rbf)
    
    from IPython import embed; embed()
    fhat = kl_exp.fn_approx(data[0])
