import os

def create_readme(start_date, end_date, crypto_list, combined_df_columns, correlation_matrix_columns, volatility_index, data_folder):
    readme_content = f"""
# Cryptocurrency Data Analysis

## Date Range
- Start Date: {start_date}
- End Date: {end_date}

## Cryptocurrencies
- {", ".join(crypto_list)}

## Data Files

### crypto_data.csv
| Column | Description |
| ------ | ----------- |
| timestamp | The date and time of the recorded data |
| open | The opening price of the cryptocurrency for the day |
| high | The highest price of the cryptocurrency for the day |
| low | The lowest price of the cryptocurrency for the day |
| close | The closing price of the cryptocurrency for the day |
| volume | The trading volume of the cryptocurrency for the day |
| symbol | The symbol of the cryptocurrency |
| daily_return | The daily return calculated from the closing prices |
| log_return | The logarithmic return calculated from the closing prices |
| market_cap | The market capitalization calculated as close price * volume |
| SMA_20 | The 20-day simple moving average of the closing price |
| EMA_20 | The 20-day exponential moving average of the closing price |
| RSI_14 | The 14-day relative strength index of the closing price |

### correlation_matrix.csv
Contains the correlation matrix between the returns of different cryptocurrencies.

### volatility.csv
Contains the historical volatility of each cryptocurrency, derived from the returns data.
    """

    with open(os.path.join(data_folder, 'README.md'), 'w') as f:
        f.write(readme_content)

    print("README.md file created in the 'data' folder.")