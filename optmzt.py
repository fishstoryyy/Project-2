# %%
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize
# %%

mu = 0.01*np.array([0.27, 0.25, 0.39, 0.88, 0.53, 0.88, 0.79, 0.71]) 
sigma = 0.01*np.array([1.56, 2.01, 5.50, 7.03, 6.22, 7.04, 6.01, 4.30])
correl = np.array([[1.00,	0.92,0.33,0.26,0.28,	0.16,0.29,0.42],
                   [0.92,	1.00,0.26,0.22,0.27,	0.14,0.25,0.36],
                   [0.33,0.26,1.00,0.41,0.30,	0.25,0.58,0.71],
                   [0.26,	0.22,0.41,1.00,0.62,0.42,0.54,0.44],
                   [0.28,	0.27,0.30,0.62,1.00,0.35,0.48,0.34],
                   [0.16,	0.14,0.25,0.42,0.35,	1.00,0.40,0.22],
                   [0.29,	0.25,0.58,0.54,0.48,0.40,1.00,0.56],
                   [0.42,	0.36,0.71,0.44,0.34,	0.22,0.56,1.00]])
V = np.outer(sigma,sigma)*correl
n = len(mu)
benchmark_weight = np.ones(n)/n
maximum_deviation = 1

# %%
# Minimum-variance portfolio
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(cp.quad_form(x,V)),
                 [x - benchmark_weight <= maximum_deviation,
                  benchmark_weight - x <= maximum_deviation,
                  np.ones(n)@x == 1,
                  x>=0])
prob.solve(solver=cp.GUROBI)
print(x.value)
# %%
# Risk-parity portfolio
def objfun(x):
    tmp = (V * np.matrix(x).T).A1
    risk = x * tmp
    var = sum(risk)
    delta_risk = np.sum((risk - var/n)**2)
    return delta_risk

x0 = benchmark_weight
bnds = tuple((0,None) for x in x0)
cons = ({'type':'eq', 'fun': lambda x: sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: x - benchmark_weight + maximum_deviation},
        {'type': 'ineq', 'fun': lambda x: benchmark_weight - x + maximum_deviation})
options={'disp':False, 'maxiter':1000, 'ftol':1e-20}

# Optimization
res = minimize(objfun, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)
print(res)

# %%
#Maximum-diversification portfolio
def objfun(x):
    return -sigma.T.dot(x) / np.sqrt(x.T.dot(V).dot(x))

x0 = benchmark_weight
bnds = tuple((0,None) for x in x0)
cons = ({'type':'eq', 'fun': lambda x: sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: x - benchmark_weight + maximum_deviation},
        {'type': 'ineq', 'fun': lambda x: benchmark_weight - x + maximum_deviation})
options={'disp':False, 'maxiter':1000, 'ftol':1e-20}

# Optimization
res = minimize(objfun, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)
print(res)


# %%
# Hierarchical risk-parity
