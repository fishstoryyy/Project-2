# %%
import numpy as np
import pandas as pd
import optmodels
from scipy.stats import random_correlation


# %%
# specify eigenvalues

num_fake_stocks = 10
eig_v = np.random.rand(num_fake_stocks)
eig_v[-1] = eig_v.shape[0] - np.sum(eig_v[:-1])

np.random.seed(0)
mean_true = np.random.uniform(0, 0.15, num_fake_stocks)
std_true = np.random.uniform(0.15, 0.5, num_fake_stocks).reshape(-1, 1)
corr_true = random_correlation.rvs(eig_v)
cov_true = std_true @ std_true.T @ corr_true

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