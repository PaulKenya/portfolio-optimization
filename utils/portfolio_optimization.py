import cvxpy as cp
import numpy as np

from utils.centrality_measures import calculate_centrality_measures


def select_top_assets(centrality, num_assets):
    sorted_assets = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    top_assets = [asset for asset, _ in sorted_assets[:num_assets]]
    return top_assets


# def allocate_funds(top_assets, max_investment_per_interval):
#     allocation = {asset: max_investment_per_interval / len(top_assets) for asset in top_assets}
#     return allocation


def allocate_funds(selected_assets, weights, total_investment):
    """
    Allocate funds based on the optimized weights.
    """
    investment_distribution = weights * total_investment
    allocation = {asset: investment for asset, investment in zip(selected_assets, investment_distribution)}
    for asset, investment in allocation.items():
        print(f"----------> Allocating ${investment:.2f} to {asset}")
    return allocation, selected_assets, weights


def optimize_portfolio(returns, covariance_matrix, centrality_vector, target_centrality, adjacency_matrix,
                       max_investment_per_interval):
    n = len(returns)
    x = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(x, covariance_matrix) - returns.T @ x)
    constraints = [
        # This constraint ensures that the total investment allocated across all assets equals the maximum amount
        # that can be invested per interval.
        cp.sum(x) == max_investment_per_interval,

        # This constraint ensures that all asset weights are non-negative, meaning no short selling is allowed.
        x >= 0,

        # This constraint ensures that the average centrality measure of the portfolio is at least 95% of the
        # predefined target.
        cp.sum(cp.multiply(centrality_vector, x)) >= target_centrality * 0.95,

        # This constraint ensures that the average centrality measure of the portfolio does not exceed 105% of the
        # predefined target.
        cp.sum(cp.multiply(centrality_vector, x)) <= target_centrality * 1.05,

        # This constraint ensures that no investment is made in assets that are directly connected in the graph,
        # promoting diversification.
        cp.sum(cp.multiply(adjacency_matrix @ x, x)) == 0
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return x.value


def optimize_portfolio_with_centrality_s(pivoted_data, centrality_measure, desired_avg_centrality, num_assets):
    centrality_vector = np.array([centrality_measure[node] for node in pivoted_data.columns])
    returns = pivoted_data.mean().values
    cov_matrix = pivoted_data.cov().values

    num_assets_total = len(returns)
    x = cp.Variable(num_assets_total)

    # Define the optimization problem
    objective = cp.Maximize(returns @ x)
    centrality_constraint_lower = centrality_vector @ x >= desired_avg_centrality * 0.95
    centrality_constraint_upper = centrality_vector @ x <= desired_avg_centrality * 1.05
    constraints = [
        cp.sum(x) == 1,  # Weights sum to 1
        x >= 0,          # No short selling
        centrality_constraint_lower,
        centrality_constraint_upper
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, max_iter=1000, abstol=1e-8, reltol=1e-8, feastol=1e-8)

    if prob.status != cp.OPTIMAL:
        print("Optimization problem did not converge.")
        return None

    # Select the top `num_assets` assets based on the optimized weights
    sorted_indices = np.argsort(-x.value)  # Sort weights in descending order
    top_indices = sorted_indices[:num_assets]

    optimal_weights = np.zeros(num_assets_total)
    optimal_weights[top_indices] = x.value[top_indices]

    return optimal_weights


def test_different_centralities(pivoted_data, mst, desired_avg_centrality, investment_amount, num_assets, results):
    degree_centrality, eigenvector_centrality, subgraph_centrality = calculate_centrality_measures(mst)
    centrality_measures = {
        "Degree Centrality": degree_centrality,
        "Eigenvector Centrality": eigenvector_centrality,
        "Subgraph Centrality": subgraph_centrality
    }

    for centrality_name, centrality_measure in centrality_measures.items():
        print(f"Testing with {centrality_name}")
        optimal_weights = optimize_portfolio_with_centrality(pivoted_data, centrality_measure, desired_avg_centrality, num_assets)

        if optimal_weights is None:
            print(f"Optimization with {centrality_name} did not converge.")
            continue

        selected_assets = pivoted_data.columns[optimal_weights > 0]
        allocated_funds, profit_percentage = allocate_funds(selected_assets, optimal_weights[optimal_weights > 0], investment_amount, pivoted_data)

        # Save results
        results.append({
            'Optimization Type': centrality_name,
            'Selected Assets': selected_assets.tolist(),
            'Weights': optimal_weights[optimal_weights > 0].tolist(),
            'Allocated Funds': allocated_funds,
            'Profit Percentage': profit_percentage
        })

        print(f"Optimization with {centrality_name} completed.")

