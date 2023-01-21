import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

sizes = np.arange(10, 210, 10)
reps = 10000


def sim_r_squared(n):
    x = np.random.normal(size=n)
    y = 1 + x + np.random.normal(size=n)
    model = stats.linregress(x, y)
    return model.rvalue ** 2


r_squared_q95 = np.empty(len(sizes))
r_squared_q5 = np.empty(len(sizes))
r_squared_mean = np.empty(len(sizes))

for i, size in enumerate(sizes):
    print(size)
    result = np.array([sim_r_squared(size) for _ in range(reps)])
    r_squared_mean[i] = result.mean()
    r_squared_q5[i] = np.quantile(result, 0.05)
    r_squared_q95[i] = np.quantile(result, 0.95)

plt.plot(sizes, r_squared_mean)
plt.ylim(np.min(r_squared_q5), np.max(r_squared_q95))
plt.xlabel("sample size")
plt.ylabel("R^2")
plt.plot(sizes, r_squared_q5)
plt.plot(sizes, r_squared_q95)
plt.show()
