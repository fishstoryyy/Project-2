# %%
import numpy as np
import pandas as pd
import optmodels
from scipy.stats import random_correlation
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# %%
# get fake data

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
# Performance analysis - sharpe ratio - rf rate is assumed to be zero
Benchmark_returns_fake = np.sum(benchmark_weight_fake * mean_true)
Benchmark_var_fake = benchmark_weight_fake @ cov_true @ benchmark_weight_fake
Benchmark_sharpe_fake = Benchmark_returns_fake / np.sqrt(Benchmark_var_fake)

MiniVar_returns_fake = np.sum(MiniVar_weight_fake * mean_true)
MiniVar_var_fake = MiniVar_weight_fake @ cov_true @ MiniVar_weight_fake
MiniVar_sharpe_fake = MiniVar_returns_fake / np.sqrt(MiniVar_var_fake)

RiskParity_returns_fake = np.sum(RiskParity_weight_fake * mean_true)
RiskParity_var_fake = RiskParity_weight_fake @ cov_true @ RiskParity_weight_fake
RiskParity_sharpe_fake = RiskParity_returns_fake / np.sqrt(RiskParity_var_fake)

MaxDiverse_returns_fake = np.sum(MaxDiverse_weight_fake * mean_true)
MaxDiverse_var_fake = MaxDiverse_weight_fake @ cov_true @ MaxDiverse_weight_fake
MaxDiverse_sharpe_fake = MaxDiverse_returns_fake / np.sqrt(MaxDiverse_var_fake)

print()
print("sharpe ratio for artificial data:")
print()
print("Benchmark")
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
plt.bar(
    ind,
    [
        Benchmark_sharpe_fake,
        MiniVar_sharpe_fake,
        RiskParity_sharpe_fake,
        MaxDiverse_sharpe_fake,
    ],
)
plt.title("Performance analysis - artificial data")
plt.ylabel("sharpe ratio - annual")
plt.xticks(ind, ("Benchmark", "MiniVar", "RiskParity", "MaxDiverse"))
plt.show()


# %%
# independent fake data, no correlation

num_fake_stocks = 10
# eig_v = np.random.rand(num_fake_stocks)
eig_v = np.random.uniform(0.5, 1, num_fake_stocks)
eig_v[-1] = eig_v.shape[0] - np.sum(eig_v[:-1])

np.random.seed(666)
mean_true = np.random.uniform(0, 0.15, num_fake_stocks)
std_true = np.random.uniform(0.15, 0.5, num_fake_stocks).reshape(-1, 1)
corr_true = random_correlation.rvs(eig_v)
corr_true = np.diag(np.diag(corr_true))
cov_true = np.outer(std_true, std_true) * corr_true
cov_true = np.diag(np.diag(cov_true))  # only keep diagonal entries

benchmark_weight_fake = np.ones(num_fake_stocks) / num_fake_stocks
maximum_deviation_fake = 1

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
# Performance analysis - sharpe ratio - rf rate is assumed to be zero - independent fake data, no correlation
Benchmark_returns_fake = np.sum(benchmark_weight_fake * mean_true)
Benchmark_var_fake = benchmark_weight_fake @ cov_true @ benchmark_weight_fake
Benchmark_sharpe_fake = Benchmark_returns_fake / np.sqrt(Benchmark_var_fake)

MiniVar_returns_fake = np.sum(MiniVar_weight_fake * mean_true)
MiniVar_var_fake = MiniVar_weight_fake @ cov_true @ MiniVar_weight_fake
MiniVar_sharpe_fake = MiniVar_returns_fake / np.sqrt(MiniVar_var_fake)

RiskParity_returns_fake = np.sum(RiskParity_weight_fake * mean_true)
RiskParity_var_fake = RiskParity_weight_fake @ cov_true @ RiskParity_weight_fake
RiskParity_sharpe_fake = RiskParity_returns_fake / np.sqrt(RiskParity_var_fake)

MaxDiverse_returns_fake = np.sum(MaxDiverse_weight_fake * mean_true)
MaxDiverse_var_fake = MaxDiverse_weight_fake @ cov_true @ MaxDiverse_weight_fake
MaxDiverse_sharpe_fake = MaxDiverse_returns_fake / np.sqrt(MaxDiverse_var_fake)

print()
print("sharpe ratio for artificial data:")
print()
print("Benchmark")
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
plt.bar(
    ind,
    [
        Benchmark_sharpe_fake,
        MiniVar_sharpe_fake,
        RiskParity_sharpe_fake,
        MaxDiverse_sharpe_fake,
    ],
)
plt.title("Performance analysis - artificial non-correlated data")
plt.ylabel("sharpe ratio - annual")
plt.xticks(ind, ("Benchmark", "MiniVar", "RiskParity", "MaxDiverse"))
plt.show()
plt.show()


# %%
# get actual data

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
# Performance analysis - actual data - expected return
mean_actual = np.mean(ret_table, axis=0)
Benchmark_returns = np.sum(benchmark_weight * mean_actual)
MiniVar_returns = np.sum(MiniVar_weight * mean_actual)
RiskParity_returns = np.sum(RiskParity_weight * mean_actual)
MaxDiverse_returns = np.sum(MaxDiverse_weight * mean_actual)
HRP_returns = np.sum(HRP_weight * mean_actual)

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
plt.bar(
    ind,
    [
        Benchmark_returns,
        RiskParity_returns,
        MiniVar_returns,
        MaxDiverse_returns,
        HRP_returns,
    ],
)
plt.ylabel("Expected Returns - daily")
plt.xticks(
    ind, ("Benchmark", "MiniVar", "RiskParity", "MaxDiverse", "HRP"), rotation=45
)
plt.show()


# %%
# Performance analysis - sharpe ratio - rf rate is assumed to be zero - actual data
Benchmark_returns = np.sum(benchmark_weight * mean_actual)
Benchmark_var = benchmark_weight @ cov_table @ benchmark_weight
Benchmark_sharpe = Benchmark_returns / np.sqrt(Benchmark_var)

MiniVar_returns = np.sum(MiniVar_weight * mean_actual)
MiniVar_var = MiniVar_weight @ cov_table @ MiniVar_weight
MiniVar_sharpe = MiniVar_returns / np.sqrt(MiniVar_var)

RiskParity_returns = np.sum(RiskParity_weight * mean_actual)
RiskParity_var = RiskParity_weight @ cov_table @ RiskParity_weight
RiskParity_sharpe = RiskParity_returns / np.sqrt(RiskParity_var)

MaxDiverse_returns = np.sum(MaxDiverse_weight * mean_actual)
MaxDiverse_var = MaxDiverse_weight @ cov_table @ MaxDiverse_weight
MaxDiverse_sharpe = MaxDiverse_returns / np.sqrt(MaxDiverse_var)

HRP_returns = np.sum(HRP_weight * mean_actual)
HRP_var = HRP_weight @ cov_table @ HRP_weight
HRP_sharpe = HRP_returns / np.sqrt(HRP_var)


ind = np.arange(5)
plt.bar(
    ind,
    [
        Benchmark_sharpe,
        MiniVar_sharpe,
        RiskParity_sharpe,
        MaxDiverse_sharpe,
        HRP_sharpe,
    ],
)
plt.title("Performance analysis - actual data")
plt.ylabel("sharpe ratio - daily")
plt.xticks(
    ind, ("Benchmark", "MiniVar", "RiskParity", "MaxDiverse", "HRP"), rotation=45
)
plt.show()
plt.show()


# %%
result = pd.DataFrame(
    [MiniVar_weight, RiskParity_weight, MaxDiverse_weight],
    columns=price_table.columns,
    index=["MiniVar", "RiskParity", "MaxDiverse"],
).T


# %%
result["HRP"] = HRP_weight
result.plot.barh(stacked=True, figsize=(10, 10))
plt.show()


# %%
result.plot.pie(subplots=True, figsize=(15, 15), legend=False, layout=(2, 2))
plt.show()


# %%
result.index


# %%

sec_map = {}
ticker_belongs = {ind: yf.Ticker(ind).info["sector"] for ind in result.index}
for ind in result.index:
    tick = yf.Ticker(ind)
    sec = tick.info["sector"]
    if sec in sec_map:
        sec_map[sec].append(ind)
    else:
        sec_map[sec] = [ind]


# %%
sec_map


# %%
ticker_belongs


# %%
result["sector"] = pd.Series(ticker_belongs)


# %%
result.groupby("sector").sum()


# %%

fig, ax = plt.subplots(figsize=(8, 8))
sns.set(font_scale=1.5)
sns.heatmap(result.groupby("sector").sum(), ax=ax)
plt.show()


# %%
