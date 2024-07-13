import os
import pandas as pd
import numpy as np
from binance.client import Client
from dotenv import load_dotenv
import json
import sys
import generate_readme
import time

from utils.config_data_loader import subtract_period, parse_date

# Load environment variables
load_dotenv(override=True)

# Fetch configuration from .env file
crypto_list = os.getenv("CRYPTO_LIST").split(",")
start_date = os.getenv("START_DATE")
end_date = os.getenv("END_DATE")
interval = os.getenv("INTERVAL")
lookback_period = os.getenv("LOOKBACK_PERIOD")
api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")
data_folder = os.getenv("DATA_FOLDER")
config_file_path = os.path.join(data_folder, 'config.json')
log_file_path = os.path.join(data_folder, 'fetch_log.json')

# Create a configuration dictionary
current_config = {
    "crypto_list": crypto_list.sort(),
    "start_date": start_date,
    "end_date": end_date,
    "interval": interval,
    "lookback_period": lookback_period,
}

# Initialize Binance client with API credentials
client = Client(api_key, api_secret)

request_count = 0
start_time = time.time()
rate_limit = 600


def fetch_historical_data(symbol, start, end, interval):
    global request_count, start_time
    print(f"fetch_historical_data: Getting data for {symbol}")
    retry_count = 0

    while retry_count < 5:
        try:
            if request_count >= rate_limit:
                elapsed_time = time.time() - start_time
                if elapsed_time < 60:
                    time.sleep(60 - elapsed_time)
                request_count = 0
                start_time = time.time()
            else:
                klines = client.get_historical_klines(symbol, interval, subtract_period(start, lookback_period), end)
                request_count += 1
                break
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(1)
            retry_count += 1
            print(f"fetch_historical_data: Retrying getting data for {symbol}")

    print(f"fetch_historical_data: Data received for {symbol}")
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                         'close_time', 'quote_asset_volume', 'number_of_trades',
                                         'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return data[['open', 'high', 'low', 'close', 'volume']]


def save_log(log):
    with open(log_file_path, 'w') as log_file:
        json.dump(log, log_file, indent=4)


def load_log():
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as log_file:
            return json.load(log_file)
    return {}


def fetch_and_save_data(crypto_list, start_date, end_date, interval):

    # log = load_log()
    for crypto in crypto_list:
        retry_count = 0
        try:
            while retry_count < 1:
                print(f"Processing {crypto}")
                symbol_usdt = f"{crypto}USDT"
                # last_fetched = log.get(symbol, start_date)
                df = fetch_historical_data(symbol_usdt, start_date, end_date, interval)
                df['symbol'] = crypto

                # Save progressively
                csv_file = os.path.join(data_folder, f"{crypto}_data.csv")
                df.to_csv(csv_file)

                # if not df.empty and pd.notna(df.index.max()):
                #     log[symbol] = df.index.max().strftime("%Y-%m-%dT%H:%M:%S")
                # else:
                #     log[symbol] = last_fetched  # Retain the last fetched date if no new data
                #
                # save_log(log)
                print(f"Completed processing for {crypto}")
                break
        except Exception as e:
            print(f"An error occurred on fetch_and_save_data: {e}. Sleeping for 10 seconds before retrying.")
            time.sleep(10)
            retry_count += 1
            print(f"fetch_and_save_data: Retrying getting data for {crypto}")


def pull_from_binance():
    # Fetch and process data
    fetch_and_save_data(crypto_list, start_date, end_date, interval)

    # Combine data
    print("Combining data from all cryptocurrencies")
    return pd.concat([
        df for df in (
            pd.read_csv(os.path.join(data_folder, f"{crypto}_data.csv"), index_col='timestamp', parse_dates=True)
            for crypto in crypto_list
            if os.path.exists(os.path.join(data_folder, f"{crypto}_data.csv"))
        ) if not df.empty
    ])

# Check if the parameters have changed
if os.path.exists(config_file_path):
    with open(config_file_path, 'r') as config_file:
        saved_config = json.load(config_file)
    if saved_config == current_config:
        print("Parameters have not changed. Skipping data pull from Binance.")
        combined_df = pd.read_csv(os.path.join(data_folder, 'crypto_data.csv'), index_col='timestamp', parse_dates=True)
    else:
        print("Parameters have changed. Pulling new data from Binance.")
        combined_df = pull_from_binance()
else:
    print("No existing configuration found. Pulling data from Binance.")
    combined_df = pull_from_binance()


# Calculate additional metrics
print("Calculating additional metrics")
combined_df = combined_df.reset_index(drop=False)
combined_df['timestamp'] = combined_df['timestamp'].apply(lambda x: parse_date(str(x)).strftime("%Y-%m-%dT%H:%M:%S"))
combined_df = combined_df.set_index('timestamp')
combined_df['daily_return'] = combined_df.groupby('symbol')['close'].pct_change()
combined_df['log_return'] = np.log(combined_df['close'] / combined_df['close'].shift(1))
combined_df['market_cap'] = combined_df['close'] * combined_df['volume']

# Calculate technical indicators
print("Calculating technical indicators")


def calculate_sma(data, window):
    return data.rolling(window=window).mean()


def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()


def calculate_rsi(data, window):
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


for symbol in combined_df['symbol'].unique():
    combined_df.loc[combined_df['symbol'] == symbol, 'SMA_20'] = calculate_sma(combined_df[combined_df['symbol'] == symbol]['close'], window=20)
    combined_df.loc[combined_df['symbol'] == symbol, 'EMA_20'] = calculate_ema(combined_df[combined_df['symbol'] == symbol]['close'], window=20)
    combined_df.loc[combined_df['symbol'] == symbol, 'RSI_14'] = calculate_rsi(combined_df[combined_df['symbol'] == symbol]['close'], window=14)

# Calculate correlation matrix and volatility
print("Calculating correlation matrix and volatility")
returns_df = combined_df.pivot_table(index='timestamp', columns='symbol', values='daily_return')
correlation_matrix = returns_df.corr()
volatility = returns_df.std()

# Ensure the data folder exists
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Save to CSV
print("Saving data to CSV files")
combined_df.to_csv(os.path.join(data_folder, 'crypto_data.csv'))
correlation_matrix.to_csv(os.path.join(data_folder, 'correlation_matrix.csv'))
volatility.to_csv(os.path.join(data_folder, 'volatility.csv'))

# Save the current configuration to a temporary file (sorted)
with open(config_file_path, 'w') as config_file:
    json.dump(current_config, config_file, sort_keys=True)

# Display results
print(f"Data collection and processing complete. Files saved in '{data_folder}' folder")

# Generate README.md file
generate_readme.create_readme(start_date, end_date, crypto_list, combined_df.columns, correlation_matrix.columns, volatility.index, data_folder)
exit(0)