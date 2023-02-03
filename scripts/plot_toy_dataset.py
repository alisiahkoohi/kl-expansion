import matplotlib.pyplot as plt
import os

from klexp.utils import (toy_dataset, sort_coordinates, configsdir,
                         read_config, parse_input_args)

CONFIG_FILE = 'toy_example.json'

if __name__ == "__main__":
    # Command line arguments.
    args = read_config(os.path.join(configsdir(), CONFIG_FILE))
    args = parse_input_args(args)

    data = toy_dataset(n=args.n,
                       s=args.s,
                       d=args.d,
                       x_range=args.x_range,
                       eval_pattern=args.eval_pattern)
    data = sort_coordinates(data)
    plt.figure(figsize=(6, 4), dpi=200)
    for i in range(args.n):
        plt.plot(data[i][0],
                 data[i][1],
                 '.-',
                 linewidth=0.8,
                 color="black",
                 alpha=0.3)
    plt.grid()
    plt.show()
