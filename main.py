import os
import pandas as pd

from utils.config_data_loader import load_config, split_time_string, subtract_period, load_data, add_period
from utils.optimization_models.CentralityMeasureConstraint import CentralityMeasureConstraint
from utils.optimization_models.Graph import Graph
from utils.optimization_models.MeanVariance import MeanVariance
from utils.optimization_models.NeighbourhoodConstraint import NeighbourhoodConstraintMIP, NeighbourhoodConstraintSDP
from utils.optimization_models.ReturnRLVaR import ReturnRLVaR


def main():
    config = load_config()
    returns_df = load_data(config["DATA_FOLDER"])
    returns_df.sort_index(inplace=True)

    start_date = config["START_DATE"]
    end_date = "2020-01-01T01:20:00"
    optimization_interval = config["OPTIMIZATION_INTERVAL"]
    lookback_period = config["LOOK_BACK_PERIOD"]
    num_assets = config["NUM_ASSETS"]
    desired_average_centrality = config["DESIRED_AVERAGE_CENTRALITY"]
    path_length = config["PATH_LENGTH"]
    _, optimization_interval_unit = split_time_string(config["OPTIMIZATION_INTERVAL"])

    skipped_intervals = 0
    results = []

    while pd.to_datetime(start_date, utc=True) <= pd.to_datetime(end_date, utc=True):
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
            print(f"No active assets with non-zero volume from {lookback_start_date} to {lookback_end_date}. Skipping "
                  f"this interval #{skipped_intervals}.")
            start_date = add_period(start_date, optimization_interval)
            continue

        historical_data = historical_data.loc[historical_data.index.get_level_values('symbol').isin(active_assets)]
        pivoted_data = historical_data.pivot_table(index='timestamp', columns='symbol', values='daily_return')

        # mean_variance = MeanVariance(pivoted_data, num_assets, start_date)
        # results.extend(mean_variance.optimize())

        graph = Graph(pivoted_data)

        # centrality_measure_constraint = CentralityMeasureConstraint(pivoted_data, graph, desired_average_centrality, num_assets, start_date)
        # results.extend(centrality_measure_constraint.optimize())

        return_rlvar = ReturnRLVaR(pivoted_data, num_assets, start_date)
        results.extend(return_rlvar.optimize())

        print("Completed Optimization for", start_date)
        start_date = add_period(start_date, optimization_interval)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(config["DATA_FOLDER"], 'optimization_results.csv'), index=False)
    print("Optimization Completed and results saved to optimization_results.csv.")


if __name__ == "__main__":
    main()
