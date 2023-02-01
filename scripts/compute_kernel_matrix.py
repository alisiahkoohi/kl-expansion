import numpy as np
import matplotlib.pyplot as plt

from klexp.utils import toy_dataset
from klexp import KarhunenLoeveExpansion, rbf

# Random seed.
SEED = 12
np.random.seed(SEED)

# Set global parameters.
N = 20
S = 100
D = 1
M = 20
X_RANGE = (-10, 10)
EVAL_PATTERN = 'same'

if __name__ == "__main__":
    data = toy_dataset(n=N,
                       s=S,
                       d=D,
                       x_range=X_RANGE,
                       eval_pattern=EVAL_PATTERN)
    kl_exp = KarhunenLoeveExpansion(data, kernel=rbf, M=M)

    test_data = toy_dataset(n=1,
                            s=S,
                            d=D,
                            x_range=X_RANGE,
                            eval_pattern=EVAL_PATTERN)
    fhat = kl_exp.fn_approx(test_data[0])

    plt.figure(figsize=(6, 4), dpi=200)
    plt.plot(test_data[0][0],
             test_data[0][1],
             label="True function",
             linewidth=0.8,
             color="black",
             alpha=0.8)
    plt.plot(test_data[0][0],
             fhat,
             label="KL approximation",
             linewidth=0.8,
             color="orange",
             alpha=0.8)
    plt.legend()
    plt.grid()
    plt.show()
