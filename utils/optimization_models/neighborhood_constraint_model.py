import traceback

import cvxpy as cp
import numpy as np
import networkx as nx
import pandas as pd

from utils.performance_calculation import calculate_portfolio_profit
from utils.portfolio_optimization import allocate_funds


def construct_connection_matrix(graph, path_length):
    """
    Construct the connection matrix B_{1,l} which indicates connections between assets through paths up to length l.
    """
    adjacency_matrix = nx.to_numpy_array(graph)
    connection_matrix = np.linalg.matrix_power(adjacency_matrix, path_length)
    return connection_matrix


def optimize_portfolio_with_neighborhood_constraint(pivoted_data, num_assets, graph, path_length):
    returns = pivoted_data.mean().values
    cov_matrix = pivoted_data.cov().values

    num_assets_total = len(returns)

    if num_assets_total == 0:
        print("----------> Error: No assets available for optimization.")
        return None

    x = cp.Variable(num_assets_total)
    y = cp.Variable(num_assets_total, boolean=True)

    connection_matrix = construct_connection_matrix(graph, path_length)
    identity_matrix = np.eye(num_assets_total)

    # Define the optimization problem
    objective = cp.Maximize(returns @ x - cp.quad_form(x, cov_matrix))
    constraints = [
        cp.sum(x) == 1,  # Weights sum to 1
        x >= 0,          # No short selling
        (connection_matrix + identity_matrix) @ y <= 1.5,  # Relaxed Neighborhood constraint
        x <= y,          # Upper bound constraint
        x >= 0.005 * y   # Relaxed Lower bound constraint
    ]

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.ECOS_BB)
    except Exception as e:
        traceback.print_exc()
        print(f"----------> Optimization problem encountered an error: {e}")
        return None

    print("----------> Problem status:", prob.status)

    if prob.status != cp.OPTIMAL:
        print("----------> Optimization problem did not converge.")
        return None

    # Select the top `num_assets` assets based on the optimized weights
    sorted_indices = np.argsort(-x.value)  # Sort weights in descending order
    top_indices = sorted_indices[:num_assets]

    optimal_weights = np.zeros(num_assets_total)
    optimal_weights[top_indices] = x.value[top_indices]

    return optimal_weights


def test_neighborhood_constraint(pivoted_data, mst, path_length, investment_amount, num_assets, results, start_date):
    print("----------> Testing with Neighborhood Constraint")
    optimal_weights = optimize_portfolio_with_neighborhood_constraint(pivoted_data, num_assets, mst, path_length)

    if optimal_weights is None:
        print("----------> Optimization with Neighborhood Constraint did not converge.")
        return

    selected_assets = pivoted_data.columns[optimal_weights > 0]
    allocated_funds, selected_assets, weights = allocate_funds(selected_assets, optimal_weights[optimal_weights > 0], investment_amount)
    profit_percentage = calculate_portfolio_profit(pivoted_data, selected_assets, weights)

    results.append({
        'Timestamp': pd.to_datetime(start_date, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
        'Optimization Type': 'Neighborhood Constraint',
        'Selected Assets': selected_assets.tolist(),
        'Weights': optimal_weights[optimal_weights > 0].tolist(),
        'Allocated Funds': allocated_funds,
        'Profit Percentage': profit_percentage
    })

    print("----------> Optimization with Neighborhood Constraint completed.")
