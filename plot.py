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

# kod w języku Julia
# using GLM, Plots, DataFrames, Statistics, Random
# function sim_r²_k(n, k, reps)
# x, y, result = ones(n, k+1), zeros(n), Float64[]
# for _ in 1:reps
# randn!(view(x, :, 1:k)); sum!(y, x)
# for i in 1:n
# y[i] += randn()
# end
# model = lm(x, y)
# push!(result, r²(model))
# end
# return result
# end
# function run(k, reps=10000, maxn=200)
# sizes = 10:10:maxn
# r²_q95, r²_q5, r²_mean = Float64[], Float64[], Float64[]
# @time for s in sizes
# @show s
# result = sim_r²_k(s, k, reps)
# push!(r²_mean, mean(result))
# push!(r²_q5, quantile(result, 0.05))
# push!(r²_q95, quantile(result, 0.95))
# end
# scatter(sizes, r²_mean, xlabel="sample size", ylabel="R²", legend=false)
# plot!(sizes, r²_q5, color="black")
# plot!(sizes, r²_q95, color="black")
# endsing GLM, Plots, DataFrames, Statistics, Random
# function sim_r²_k(n, k, reps)
# x, y, result = ones(n, k+1), zeros(n), Float64[]
# for _ in 1:reps
# randn!(view(x, :, 1:k)); sum!(y, x)
# for i in 1:n
# y[i] += randn()
# end
# model = lm(x, y)
# push!(result, r²(model))
# end
# return result
# end
# function run(k, reps=10000, maxn=200)
# sizes = 10:10:maxn
# r²_q95, r²_q5, r²_mean = Float64[], Float64[], Float64[]
# @time for s in sizes
# @show s
# result = sim_r²_k(s, k, reps)
# push!(r²_mean, mean(result))
# push!(r²_q5, quantile(result, 0.05))
# push!(r²_q95, quantile(result, 0.95))
# end
# scatter(sizes, r²_mean, xlabel="sample size", ylabel="R²", legend=false)
# plot!(sizes, r²_q5, color="black")
# plot!(sizes, r²_q95, color="black")
# end