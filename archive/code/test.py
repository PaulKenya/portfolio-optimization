import pandas as pd

from utils.config_data_loader import load_config, load_data, split_time_string, subtract_period, add_period
from utils.graph_construction import calculate_mst, calculate_tmfg, create_tmfg_approx
from utils.optimization_models.centrality_measure_constraint_model import optimize_portfolio_classical, \
    test_different_centralities
from utils.optimization_models.mean_variance_optimization_model import optimize_portfolio_mean_variance
from utils.optimization_models.neighborhood_constraint_model import test_neighborhood_constraint
from utils.optimization_models.rlvar_optimization_model import test_rlvar_optimization
from utils.performance_calculation import calculate_portfolio_profit
from utils.portfolio_optimization import allocate_funds


def main():
    config = load_config()
    returns_df = load_data(config["DATA_FOLDER"])
    returns_df.sort_index(inplace=True)

    start_date = config["START_DATE"]
    end_date = config["END_DATE"]
    investment_amount = config["INVESTMENT_AMOUNT"]
    optimization_interval = config["OPTIMIZATION_INTERVAL"]
    lookback_period = config["LOOK_BACK_PERIOD"]
    num_assets = config["NUM_ASSETS"]
    desired_average_centrality = config["DESIRED_AVERAGE_CENTRALITY"]
    path_length = config["PATH_LENGTH"]
    _, optimization_interval_unit = split_time_string(config["OPTIMIZATION_INTERVAL"])

    skipped_intervals = 0
    classical_results = []
    centrality_results = []
    neighborhood_constraint_results = []
    mean_variance_results = []
    rlvar_optimization_results = []

    while investment_amount > 0 and pd.to_datetime(start_date, utc=True) <= pd.to_datetime(end_date, utc=True):

        lookback_start_date = pd.to_datetime(subtract_period(start_date, lookback_period), utc=True)
        lookback_end_date = pd.to_datetime(subtract_period(start_date, optimization_interval), utc=True)

        historical_data = returns_df.loc[lookback_start_date:lookback_end_date]
        if historical_data.empty:
            skipped_intervals += 1
            print(f"----------> No historical data available from {lookback_start_date} to {lookback_end_date}.")
            print(f"Consider rerunning get_crypto_data_binance.py. Skipping this interval #{skipped_intervals}.")
            start_date = add_period(start_date, optimization_interval)
            continue

        try:
            start_volumes = historical_data.xs(lookback_start_date, level='timestamp', drop_level=False)['volume']
            end_volumes = historical_data.xs(lookback_end_date, level='timestamp', drop_level=False)['volume']
        except KeyError as e:
            skipped_intervals += 1
            print(f"KeyError for volume data at {e}. Skipping this interval #{skipped_intervals}.")
            start_date = add_period(start_date, optimization_interval)
            continue

        active_assets_start = start_volumes[start_volumes > 0].index.get_level_values('symbol')
        active_assets_end = end_volumes[end_volumes > 0].index.get_level_values('symbol')
        active_assets = active_assets_start.intersection(active_assets_end)

        if active_assets.empty:
            skipped_intervals += 1
            print(f"No active assets with non-zero volume from {lookback_start_date} to {lookback_end_date}. Skipping this interval #{skipped_intervals}.")
            start_date = add_period(start_date, optimization_interval)
            continue

        historical_data = historical_data.loc[historical_data.index.get_level_values('symbol').isin(active_assets)]

        # historical_data = historical_data.groupby('symbol').resample(convert_unit_to_pandas_freq(optimization_interval_unit), level=0).last().fillna(method='ffill').dropna()
        # historical_data = historical_data.reset_index()

        pivoted_data = historical_data.pivot_table(index='timestamp', columns='symbol', values='daily_return')
        correlation_matrix = pivoted_data.corr()

        # Classical Optimization
        classical_weights = optimize_portfolio_classical(pivoted_data, num_assets)
        if classical_weights is not None:
            selected_assets = pivoted_data.columns[classical_weights > 0]
            allocated_funds, selected_assets, weights = allocate_funds(selected_assets, classical_weights[classical_weights > 0], investment_amount)
            profit_percentage = calculate_portfolio_profit(pivoted_data, selected_assets, weights)
            classical_results.append({
                'Timestamp': pd.to_datetime(start_date, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
                'Optimization Type': 'Classical',
                'Selected Assets': selected_assets.tolist(),
                'Weights': classical_weights[classical_weights > 0].tolist(),
                'Allocated Funds': allocated_funds,
                'Profit Percentage': profit_percentage
            })

        # Mean-Variance Optimization
        mean_variance_weights = optimize_portfolio_mean_variance(pivoted_data, num_assets)
        if mean_variance_weights is not None:
            selected_assets = pivoted_data.columns[mean_variance_weights > 0]
            allocated_funds, selected_assets, weights = allocate_funds(selected_assets, mean_variance_weights[mean_variance_weights > 0], investment_amount)
            profit_percentage = calculate_portfolio_profit(pivoted_data, selected_assets, weights)
            mean_variance_results.append({
                'Timestamp': pd.to_datetime(start_date, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ"),
                'Optimization Type': 'Mean-Variance',
                'Selected Assets': selected_assets.tolist(),
                'Weights': mean_variance_weights[mean_variance_weights > 0].tolist(),
                'Allocated Funds': allocated_funds,
                'Profit Percentage': profit_percentage
            })

        # Step 1: Graph Construction
        mst = calculate_mst(correlation_matrix)
        # visualize_graph(mst, "Minimum Spanning Tree (MST)")
        tmfg = create_tmfg_approx(correlation_matrix)
        # visualize_graph(tmfg, "Triangulated Maximally Filtered Graph (TMFG) Approximation")

        # Step 2: Centrality Measure Calculation
        test_different_centralities(pivoted_data, mst, desired_average_centrality, investment_amount, num_assets, centrality_results, start_date)
        test_neighborhood_constraint(pivoted_data, mst, path_length, investment_amount, num_assets, neighborhood_constraint_results, start_date)
        # test_rlvar_optimization(pivoted_data, mst, desired_average_centrality, investment_amount, num_assets, rlvar_optimization_results, start_date)

        print("Completed Optimization for", start_date)

        start_date = add_period(start_date, optimization_interval)

    save_results_to_csv(classical_results, 'classical_optimization_results.csv')
    save_results_to_csv(mean_variance_results, 'mean_variance_optimization_results.csv')
    save_results_to_csv(centrality_results, 'centrality_optimization_results.csv')
    save_results_to_csv(neighborhood_constraint_results, 'neighborhood_constraint_optimization_results.csv')
    # save_results_to_csv(rlvar_optimization_results, 'rlvar_optimization_results.csv')


if __name__ == "__main__":
    main()


def save_results_to_csv(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
