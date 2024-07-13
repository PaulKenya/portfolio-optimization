import traceback

import cvxpy as cp
import numpy as np
import pandas as pd

from utils.centrality_measures import calculate_centrality_measures
from utils.performance_calculation import calculate_portfolio_profit
from utils.portfolio_optimization import allocate_funds

def rlvar_measure(portfolio_returns, alpha=0.95):
    sorted_returns = np.sort(portfolio_returns)
    var_index = int((1 - alpha) * len(sorted_returns))

    # Value at Risk (VaR)
    var = sorted_returns[var_index]

    # Expected Shortfall (ES)
    expected_shortfall = sorted_returns[:var_index].mean()

    # Relativistic Value at Risk (RLVaR)
    rlvar = var + expected_shortfall

    return rlvar

def optimize_portfolio_with_rlvar(pivoted_data, centrality_vector, desired_avg_centrality, num_assets):
    returns = pivoted_data.values
    num_assets_total = returns.shape[1]

    if num_assets_total == 0:
        print("Error: No assets available for optimization.")
        return None

    # Ensure centrality_vector is a numpy array
    if isinstance(centrality_vector, dict):
        centrality_vector = np.array(list(centrality_vector.values()))

    # Check for NaN or infinite values in returns and centrality_vector
    if np.isnan(returns).any() or np.isinf(returns).any():
        print("Error: Returns contain NaN or infinite values.")
        return None

    if np.isnan(centrality_vector).any() or np.isinf(centrality_vector).any():
        print("Error: Centrality vector contains NaN or infinite values.")
        return None

    if np.isnan(desired_avg_centrality) or np.isinf(desired_avg_centrality):
        print("Error: Desired average centrality contains NaN or infinite value.")
        return None

    # Define variables
    x = cp.Variable(num_assets_total)
    portfolio_returns = returns @ x

    # Auxiliary variables for RLVaR calculation
    alpha = 0.95
    sorted_returns = cp.Variable(portfolio_returns.shape[0])
    var = cp.Variable()
    expected_shortfall = cp.Variable()
    rlvar = cp.Variable()

    # Check intermediate values before defining the problem
    print("Portfolio returns shape:", portfolio_returns.shape)
    print("Portfolio returns (first 10):", portfolio_returns[:10])
    print("Centrality vector:", centrality_vector)
    print("Desired average centrality:", desired_avg_centrality)

    # Define the optimization problem
    objective = cp.Minimize(rlvar)
    constraints = [
        cp.sum(x) == 1,  # Weights sum to 1
        x >= 0,          # No short selling
        centrality_vector @ x == desired_avg_centrality,  # Centrality constraint
        sorted_returns[:-1] >= sorted_returns[1:],  # Enforce sorting through constraints
        sorted_returns == portfolio_returns,
        var == sorted_returns[int((1 - alpha) * sorted_returns.shape[0])],
        expected_shortfall == cp.sum(sorted_returns[:int((1 - alpha) * sorted_returns.shape[0])]) / int((1 - alpha) * sorted_returns.shape[0]),
        rlvar == var + expected_shortfall
    ]

    # Check constraints for NaN values
    for i, constraint in enumerate(constraints):
        print(f"Constraint {i}: {constraint}")

    print("Objective function and constraints defined.")
    print("Objective function:", objective)

    try:
        prob = cp.Problem(objective, constraints)
        prob.solve()

        print("Problem status:", prob.status)

        if prob.status != cp.OPTIMAL:
            print("Optimization problem did not converge.")
            return None

        # Select the top `num_assets` assets based on the optimized weights
        sorted_indices = np.argsort(-x.value)  # Sort weights in descending order
        top_indices = sorted_indices[:num_assets]

        optimal_weights = np.zeros(num_assets_total)
        optimal_weights[top_indices] = x.value[top_indices]

        return optimal_weights

    except Exception as e:
        traceback.print_exc()
        print(f"Error during optimization: {e}")
        return None

def test_rlvar_optimization(pivoted_data, mst, desired_avg_centrality, investment_amount, num_assets, results, start_date):
    degree_centrality, eigenvector_centrality, subgraph_centrality = calculate_centrality_measures(mst)
    centrality_measures = {
        "Degree Centrality": degree_centrality,
        "Eigenvector Centrality": eigenvector_centrality,
        "Subgraph Centrality": subgraph_centrality
    }

    for centrality_name, centrality_measure in centrality_measures.items():
        print(f"Testing with RLVaR Optimization using {centrality_name}")
        optimal_weights = optimize_portfolio_with_rlvar(pivoted_data, centrality_measure, desired_avg_centrality, num_assets)

        if optimal_weights is None:
            print(f"Optimization with RLVaR using {centrality_name} did not converge.")
            continue

        selected_assets = pivoted_data.columns[optimal_weights > 0]
        allocated_funds, selected_assets, weights = allocate_funds(selected_assets, optimal_weights[optimal_weights > 0], investment_amount)
        profit_percentage = calculate_portfolio_profit(pivoted_data, selected_assets, weights)

        results.append({
            'Timestamp': pd.to_datetime(start_date, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
            'Optimization Type': f'RLVaR using {centrality_name}',
            'Selected Assets': selected_assets.tolist(),
            'Weights': optimal_weights[optimal_weights > 0].tolist(),
            'Allocated Funds': allocated_funds,
            'Profit Percentage': profit_percentage
        })

        print(f"Optimization with RLVaR using {centrality_name} completed.")
