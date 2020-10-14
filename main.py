# %%
import numpy as np
import cvxpy as cp
import pandas as pd
from scipy.optimize import minimize

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
mu = pd.read_excel(
    "\output\data.xlsx",
)
pd.r
n = len(mu)
benchmark_weight = np.ones(n) / n
maximum_deviation = 1

# %%
# Minimum-variance portfolio
x = cp.Variable(n)
prob = cp.Problem(
    cp.Minimize(cp.quad_form(x, V)),
    [
        x - benchmark_weight <= maximum_deviation,
        benchmark_weight - x <= maximum_deviation,
        np.ones(n) @ x == 1,
        x >= 0,
    ],
)
prob.solve(solver=cp.GUROBI)
print(x.value)
# %%
# Risk-parity portfolio
def objfun(x):
    tmp = (V * np.matrix(x).T).A1
    risk = x * tmp
    var = sum(risk)
    delta_risk = np.sum((risk - var / n) ** 2)
    return delta_risk


x0 = benchmark_weight
bnds = tuple((0, None) for x in x0)
cons = (
    {"type": "eq", "fun": lambda x: sum(x) - 1},
    {"type": "ineq", "fun": lambda x: x - benchmark_weight + maximum_deviation},
    {"type": "ineq", "fun": lambda x: benchmark_weight - x + maximum_deviation},
)
options = {"disp": False, "maxiter": 1000, "ftol": 1e-20}

# Optimization
res = minimize(
    objfun, x0, bounds=bnds, constraints=cons, method="SLSQP", options=options
)
print(res)

# %%
# Maximum-diversification portfolio
def objfun(x):
    return -sigma.T.dot(x) / np.sqrt(x.T.dot(V).dot(x))


x0 = benchmark_weight
bnds = tuple((0, None) for x in x0)
cons = (
    {"type": "eq", "fun": lambda x: sum(x) - 1},
    {"type": "ineq", "fun": lambda x: x - benchmark_weight + maximum_deviation},
    {"type": "ineq", "fun": lambda x: benchmark_weight - x + maximum_deviation},
)
options = {"disp": False, "maxiter": 1000, "ftol": 1e-20}

# Optimization
res = minimize(
    objfun, x0, bounds=bnds, constraints=cons, method="SLSQP", options=options
)
print(res)


# %%
# Hierarchical risk-parity
