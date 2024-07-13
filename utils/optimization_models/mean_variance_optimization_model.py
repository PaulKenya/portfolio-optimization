import traceback

import cvxpy as cp
import numpy as np


def optimize_portfolio_mean_variance(pivoted_data, num_assets):
    returns = pivoted_data.mean().values
    cov_matrix = pivoted_data.cov().values

    num_assets_total = len(returns)

    if num_assets_total == 0:
        print("----------> Error: No assets available for optimization.")
        return None

    x = cp.Variable(num_assets_total)

    # Define the optimization problem
    objective = cp.Minimize(cp.quad_form(x, cov_matrix) - returns @ x)
    constraints = [
        cp.sum(x) == 1,  # Weights sum to 1
        x >= 0          # No short selling
    ]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve()
    except Exception as e:
        traceback.print_exc()
        print(f"----------> Optimization problem encountered an error: {e}")
        return None

    if prob.status != cp.OPTIMAL:
        print("----------> Optimization problem did not converge.")
        return None

    # Select the top `num_assets` assets based on the optimized weights
    sorted_indices = np.argsort(-x.value)  # Sort weights in descending order
    top_indices = sorted_indices[:num_assets]

    optimal_weights = np.zeros(num_assets_total)
    optimal_weights[top_indices] = x.value[top_indices]

    return optimal_weights
