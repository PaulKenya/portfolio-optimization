import copy
import os
import time

import pandas as pd
import multiprocessing as mp

from utils.config_data_loader import load_config, split_time_string, subtract_period, load_data, add_period, \
    convert_unit_to_pandas_freq
from utils.optimization_models.CentralityMeasureConstraint import CentralityMeasureConstraint
from utils.optimization_models.Graph import Graph
from utils.optimization_models.MeanVariance import MeanVariance
from utils.optimization_models.NeighbourhoodConstraint import NeighbourhoodConstraintMIP, NeighbourhoodConstraintSDP


def perform_optimization(returns_df, start_date, end_date, optimization_interval, lookback_period, num_assets, desired_average_centrality, path_length):
    results = []
    try:
        print(f"Starting process for {start_date} to {end_date}")
        skipped_intervals = 0

        while pd.to_datetime(start_date, utc=True) < pd.to_datetime(end_date, utc=True):
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

            mean_variance = MeanVariance(pivoted_data, num_assets, start_date)
            results.extend(mean_variance.optimize())

            graph = Graph(pivoted_data)

            centrality_measure_constraint = CentralityMeasureConstraint(pivoted_data, graph, desired_average_centrality, num_assets, start_date)
            results.extend(centrality_measure_constraint.optimize())

            neighbourhood_constraint_mip = NeighbourhoodConstraintMIP(pivoted_data, graph, path_length, num_assets, start_date)
            results.extend(neighbourhood_constraint_mip.optimize())

            neighbourhood_constraint_sdp = NeighbourhoodConstraintSDP(pivoted_data, graph, path_length, num_assets, start_date)
            results.extend(neighbourhood_constraint_sdp.optimize())

            print("Completed Optimization for", start_date)
            start_date = add_period(start_date, optimization_interval)

        print(f"Exiting process with start date {start_date}")
        return results
    except Exception as e:
        print(f"Error in perform_optimization: {e}")
        return results


def main():
    start_time = time.time()
    config = load_config()
    returns_df = load_data(config["DATA_FOLDER"])
    returns_df.sort_index(inplace=True)

    start_date = config["START_DATE"]
    end_date = config["END_DATE"]
    optimization_interval = config["OPTIMIZATION_INTERVAL"]
    lookback_period = config["LOOK_BACK_PERIOD"]
    num_assets = config["NUM_ASSETS"]
    desired_average_centrality = config["DESIRED_AVERAGE_CENTRALITY"]
    path_length = config["PATH_LENGTH"]
    _, optimization_interval_unit = split_time_string(config["OPTIMIZATION_INTERVAL"])
    process_count = min(config.get("PROCESS_COUNT", os.cpu_count()), 8)

    date_ranges = pd.date_range(
        start=pd.to_datetime(start_date),
        end=pd.to_datetime(add_period(end_date, optimization_interval)),
        freq=convert_unit_to_pandas_freq(optimization_interval_unit)
    )

    if len(date_ranges) < process_count:
        process_count = len(date_ranges)

    chunk_size = len(date_ranges) // process_count

    chunks = [
        (copy.deepcopy(returns_df), date_ranges[i * chunk_size], date_ranges[min((i + 1) * chunk_size, len(date_ranges) - 1)],
         optimization_interval, lookback_period, num_assets, desired_average_centrality, path_length)
        for i in range(process_count)
        if date_ranges[i * chunk_size] < date_ranges[min((i + 1) * chunk_size, len(date_ranges) - 1)]
    ]

    process_count = len(chunks)

    if process_count == 0:
        print("No valid date ranges to process.")
        exit(0)

    with mp.Pool(process_count) as pool:
        results = pool.starmap(perform_optimization, chunks)

    flat_results = [item for sublist in results for item in sublist]
    results_df = pd.DataFrame(flat_results)
    results_df.to_csv(os.path.join(config["DATA_FOLDER"], 'optimization_results.csv'), index=False)
    print("Optimization Completed and results saved to optimization_results.csv.")

    end_time = time.time()  # Capture the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time

    # Convert elapsed time to hours, minutes, seconds
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total time taken: {int(hours)}h {int(minutes)}m {int(seconds)}s")


if __name__ == "__main__":
    main()
    exit(0)
