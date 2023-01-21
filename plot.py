import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def sim_r_squared_k(n, k, reps):
    x = np.ones((n, k + 1))
    y = np.zeros(n)
    result = []
    for _ in range(reps):
        np.random.randn(x[:, :k])
        y += np.sum(x, axis=1)
        for i in range(n):
            y[i] += np.random.randn()
        model = stats.linregress(x, y)
        result.append(model.rvalue ** 2)
    return result


def run(k, reps=10000, maxn=200):
    sizes = range(10, maxn + 1, 10)
    r_squared_q95, r_squared_q5, r_squared_mean = [], [], []
    for s in sizes:
        print(s)
        result = sim_r_squared_k(s, k, reps)
        r_squared_mean.append(np.mean(result))
        r_squared_q5.append(np.quantile(result, 0.05))
        r_squared_q95.append(np.quantile(result, 0.95))

    plt.scatter(sizes, r_squared_mean)
    plt.xlabel("sample size")
    plt.ylabel("R^2")
    plt.plot(sizes, r_squared_q5, color="black")
    plt.plot(sizes, r_squared_q95, color="black")
    plt.show()
