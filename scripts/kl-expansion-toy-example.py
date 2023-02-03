import numpy as np
import matplotlib.pyplot as plt
import os

from klexp.utils import toy_dataset, configsdir, read_config, parse_input_args
from klexp import KarhunenLoeveExpansion, rbf

CONFIG_FILE = 'toy_example.json'

if __name__ == "__main__":
    # Command line arguments.
    args = read_config(os.path.join(configsdir(), CONFIG_FILE))
    args = parse_input_args(args)

    # Random seed.
    np.random.seed(args.seed)

    data = toy_dataset(n=args.n,
                       s=args.s,
                       d=args.d,
                       x_range=args.x_range,
                       eval_pattern=args.eval_pattern)
    kl_exp = KarhunenLoeveExpansion(data, kernel=rbf, M=args.M)

    test_data = toy_dataset(n=5,
                            s=args.s,
                            d=args.d,
                            x_range=args.x_range,
                            eval_pattern=args.eval_pattern)

    plt.figure(figsize=(6, 4), dpi=200)
    for i in range(5):
        fhat = kl_exp.fn_approx(test_data[i])
        plt.plot(test_data[i][0],
                 test_data[i][1],
                 linewidth=0.9,
                 color="black",
                 label='_nolegend_' if i > 0 else 'True function',
                 alpha=0.8)
        plt.plot(test_data[i][0],
                 fhat,
                 linewidth=0.9,
                 label='_nolegend_' if i > 0 else 'KL approximation',
                 color="red",
                 alpha=0.8)
    plt.legend()
    plt.grid()
    plt.show()
