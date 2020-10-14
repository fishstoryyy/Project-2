import numpy as np
import cvxpy as cp
import pandas as pd
from scipy.optimize import minimize

def MiniVar(V, n, benchmark_weight, maximum_deviation):
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
    return x.value


def RiskParity(V, n, benchmark_weight, maximum_deviation)
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
        return res.x


def MaxDiverse(V, sigma, n, benchmark_weight, maximum_deviation)
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
    return res.x