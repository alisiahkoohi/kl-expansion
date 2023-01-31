import matplotlib.pyplot as plt

from klexp.utils import toy_dataset, sort_coordinates

# Set global parameters.
N = 20
S = 50
D = 1
X_RANGE = (-10, 10)
EVAL_PATTERN = 'same_size'

if __name__ == "__main__":
    data = toy_dataset(n=N, s=S, d=D, x_range=X_RANGE, eval_pattern=EVAL_PATTERN)
    data = sort_coordinates(data)
    plt.figure(figsize=(6, 4), dpi=200)
    for i in range(N):
        plt.plot(data[i][0],
                 data[i][1],
                 '.-',
                 linewidth=0.8,
                 color="black",
                 alpha=0.3)
    plt.grid()
    plt.show()
