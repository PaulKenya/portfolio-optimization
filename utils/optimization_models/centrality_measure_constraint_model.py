import cvxpy as cp
import numpy as np
import pandas as pd

from utils.centrality_measures import calculate_centrality_measures
from utils.performance_calculation import calculate_portfolio_profit
from utils.portfolio_optimization import allocate_funds


def optimize_portfolio_with_centrality(pivoted_data, centrality_measure, desired_avg_centrality, num_assets):
    centrality_vector = np.array([centrality_measure[node] for node in pivoted_data.columns])
    returns = pivoted_data.mean().values
    cov_matrix = pivoted_data.cov().values

    num_assets_total = len(returns)

    if num_assets_total == 0:
        print("----------> Error: No assets available for optimization.")
        return None

    x = cp.Variable(num_assets_total)

    # Define the optimization problem
    objective = cp.Maximize(returns @ x - cp.quad_form(x, cov_matrix))
    centrality_constraint_lower = centrality_vector @ x >= desired_avg_centrality * 0.95
    centrality_constraint_upper = centrality_vector @ x <= desired_avg_centrality * 1.05
    constraints = [
        cp.sum(x) == 1,  # Weights sum to 1
        x >= 0,          # No short selling
        centrality_constraint_lower,
        centrality_constraint_upper
    ]
    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception as e:
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


def optimize_portfolio_classical(pivoted_data, num_assets):
    returns = pivoted_data.mean().values
    cov_matrix = pivoted_data.cov().values

    num_assets_total = len(returns)

    if num_assets_total == 0:
        print("----------> Error: No assets available for optimization.")
        return None

    x = cp.Variable(num_assets_total)

    # Define the optimization problem
    objective = cp.Maximize(returns @ x)
    constraints = [
        cp.sum(x) == 1,  # Weights sum to 1
        x >= 0          # No short selling
    ]
    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception as e:
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


def test_different_centralities(pivoted_data, mst, desired_avg_centrality, investment_amount, num_assets, results, start_date):
    degree_centrality, eigenvector_centrality, subgraph_centrality = calculate_centrality_measures(mst)
    centrality_measures = {
        "Degree Centrality": degree_centrality,
        "Eigenvector Centrality": eigenvector_centrality,
        "Subgraph Centrality": subgraph_centrality
    }

    for centrality_name, centrality_measure in centrality_measures.items():
        optimal_weights = optimize_portfolio_with_centrality(pivoted_data, centrality_measure, desired_avg_centrality, num_assets)

        if optimal_weights is None:
            print(f"----------> Optimization with {centrality_name} did not converge.")
            continue

        selected_assets = pivoted_data.columns[optimal_weights > 0]
        allocated_funds, selected_assets, weights = allocate_funds(selected_assets, optimal_weights[optimal_weights > 0], investment_amount)
        profit_percentage = calculate_portfolio_profit(pivoted_data, selected_assets, weights)

        results.append({
            'Timestamp': pd.to_datetime(start_date, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
            'Optimization Type': centrality_name,
            'Selected Assets': selected_assets.tolist(),
            'Weights': optimal_weights[optimal_weights > 0].tolist(),
            'Allocated Funds': allocated_funds,
            'Profit Percentage': profit_percentage
        })

        print(f"----------> Optimization with {centrality_name} completed.")
