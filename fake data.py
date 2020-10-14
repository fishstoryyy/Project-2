# %%
import numpy as np
from scipy.stats import random_correlation

# %%
# specify eigenvalues
eig_v = np.random.rand(10)
eig_v[-1] = eig_v.shape[0] - np.sum(eig_v[:-1])

np.random.seed(0)
mean_true = np.random.uniform(0, 0.15, 10)
std_true = np.random.uniform(0.15, 0.5, 10).reshape(-1, 1)
corr_true = random_correlation.rvs(eig_v)
cov_true = std_true @ std_true.T @ corr_true
# %%
