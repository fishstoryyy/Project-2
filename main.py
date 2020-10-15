# %%
import numpy as np
import pandas as pd
import optmodels
from scipy.stats import random_correlation
import matplotlib.pyplot as plt


# %%
# specify eigenvalues

num_fake_stocks = 10
# eig_v = np.random.rand(num_fake_stocks)
eig_v = np.random.uniform(0.5, 1, num_fake_stocks)
eig_v[-1] = eig_v.shape[0] - np.sum(eig_v[:-1])

np.random.seed(666)
mean_true = np.random.uniform(0, 0.15, num_fake_stocks)
std_true = np.random.uniform(0.15, 0.5, num_fake_stocks).reshape(-1, 1)
corr_true = random_correlation.rvs(eig_v)
cov_true = np.outer(std_true, std_true) * corr_true

benchmark_weight_fake = np.ones(num_fake_stocks) / num_fake_stocks
maximum_deviation_fake = 1


# %%
path = "./output/data.xlsx"
price_table = pd.read_excel(path, sheet_name="stock_close", index_col="Date")
ret_table = pd.read_excel(path, sheet_name="stock_ret", index_col="Date")
cov_table = pd.read_excel(path, sheet_name="stock_cov", index_col=0)
corr_table = pd.read_excel(path, sheet_name="stock_corr", index_col=0)
sigma = np.sqrt(np.diag(cov_table))
num_stocks = price_table.shape[1]
benchmark_weight = np.ones(num_stocks) / num_stocks
maximum_deviation = 1


# %%
# fake data

# MiniVar
MiniVar_weight_fake = optmodels.MiniVar(
    cov_true, num_fake_stocks, benchmark_weight_fake, maximum_deviation_fake
)

# RiskParity
RiskParity_weight_fake = optmodels.RiskParity(
    cov_true, num_fake_stocks, benchmark_weight_fake, maximum_deviation_fake
)

# MaxDiverse
MaxDiverse_weight_fake = optmodels.MaxDiverse(
    cov_true, num_fake_stocks, benchmark_weight_fake, maximum_deviation_fake
)

print("fake data results:")
print("MiniVar")
print(MiniVar_weight_fake)
print()
print("RiskParity")
print(RiskParity_weight_fake)
print()
print("MaxDiverse")
print(MaxDiverse_weight_fake)

# %%
# Performance analysis
Benchmark_returns_fake = np.sum(benchmark_weight_fake*mean_true)
MiniVar_returns_fake = np.sum(MiniVar_weight_fake*mean_true)
RiskParity_returns_fake = np.sum(RiskParity_weight_fake*mean_true)
MaxDiverse_returns_fake = np.sum(MaxDiverse_weight_fake*mean_true)
print()
print("Expected return for artificial data:")
print()
print("Benchmark")
print(Benchmark_returns_fake)
print()
print("MiniVar")
print(MiniVar_returns_fake)
print()
print("RiskParity")
print(RiskParity_returns_fake)
print()
print("MaxDiverse") 
print(MaxDiverse_returns_fake)

ind = np.arange(4)
plt.bar(ind, [Benchmark_returns_fake, RiskParity_returns_fake, MiniVar_returns_fake, MaxDiverse_returns_fake])
plt.ylabel('Expected Returns')
plt.xticks(ind, ('Benchmark', 'MiniVar', 'RiskParity', 'MaxDiverse'))
plt.show()

# %%
# actual data

# MiniVar
MiniVar_weight = optmodels.MiniVar(
    cov_table, num_stocks, benchmark_weight, maximum_deviation
)

# RiskParity
RiskParity_weight = optmodels.RiskParity(
    cov_table, num_stocks, benchmark_weight, maximum_deviation
)

# MaxDiverse
MaxDiverse_weight = optmodels.MaxDiverse(
    cov_table, num_stocks, benchmark_weight, maximum_deviation
)

# HRP
HRP_weight = optmodels.HRP(price_table)
# %%
print()
print("actual data results:")
print("MiniVar")
print(MiniVar_weight)
print()
print("RiskParity")
print(RiskParity_weight)
print()
print("MaxDiverse")
print(MaxDiverse_weight)
print()
print("HRP")
print(HRP_weight)

# %%
# Performance analysis
mean_actual = np.mean(ret_table, axis=0)
Benchmark_returns = np.sum(benchmark_weight*mean_actual)
MiniVar_returns = np.sum(MiniVar_weight*mean_actual)
RiskParity_returns = np.sum(RiskParity_weight*mean_actual)
MaxDiverse_returns = np.sum(MaxDiverse_weight*mean_actual)
HRP_returns = np.sum(HRP_weight*mean_actual)

print()
print("Expected return for actual data:")
print()
print("Benchmark")
print(Benchmark_returns)
print()
print("MiniVar")
print(MiniVar_returns)
print()
print("RiskParity")
print(RiskParity_returns)
print()
print("MaxDiverse")
print(MaxDiverse_returns)
print()
print("HRP")
print(HRP_returns)

ind = np.arange(5)
plt.bar(ind, [Benchmark_returns, RiskParity_returns, MiniVar_returns, MaxDiverse_returns, HRP_returns])
plt.ylabel('Expected Returns')
plt.xticks(ind, ('Benchmark', 'MiniVar', 'RiskParity', 'MaxDiverse', 'HRP'))
plt.show()

# %%
result = pd.DataFrame(
    [MiniVar_weight, RiskParity_weight, MaxDiverse_weight],
    columns=price_table.columns,
    index=["MiniVar", "RiskParity", "MaxDiverse"],
).T

result["HRP"] = HRP_weight
result.plot.barh(stacked=True, figsize=(10, 10))
plt.show()

result.plot.pie(subplots=True, figsize=(30, 5), legend=False)
plt.show()
