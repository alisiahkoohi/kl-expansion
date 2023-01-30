import matplotlib.pyplot as plt

from klexp.utils import toy_dataset

N = 200
S = 100
D = 1
X_RANGE = (-10, 10)
SAME_S = True

if __name__ == "__main__":
    data = toy_dataset(n=N, s=S, d=D, x_range=X_RANGE, same_s=SAME_S)
    for i in range(N):
        plt.plot(data[i][0],
                 data[i][1],
                 linewidth=0.5,
                 color="black",
                 alpha=0.3)
    plt.grid()
    plt.show()
