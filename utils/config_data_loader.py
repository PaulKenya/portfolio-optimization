import os
import pandas as pd
from datetime import datetime, timedelta
import re
from typing import Dict, Any, List

from dotenv import load_dotenv


def load_config() -> Dict[str, Any]:
    _, unit = split_time_string(os.getenv("OPTIMIZATION_INTERVAL", "1m"))
    load_dotenv(override=True)
    config: Dict[str, Any] = {
        "START_DATE": os.getenv("START_DATE", "2020-01-01"),
        "END_DATE": os.getenv("END_DATE", "2021-01-01"),
        "OPTIMIZATION_INTERVAL": os.getenv("INTERVAL", "1m"),
        "LOOK_BACK_PERIOD": os.getenv("LOOKBACK_PERIOD", "5m"),
        "NUM_ASSETS": int(os.getenv("NUM_ASSETS", 5)),
        "PATH_LENGTH": int(os.getenv("PATH_LENGTH", 3)),
        "DESIRED_AVERAGE_CENTRALITY": float(os.getenv("DESIRED_AVERAGE_CENTRALITY", "0.5")),
        "INVESTMENT_AMOUNT": float(os.getenv("INVESTMENT_AMOUNT", "1000")),
        "DATA_FOLDER": os.getenv("DATA_FOLDER", "data"),
    }
    return config


def load_data(data_folder: str) -> pd.DataFrame:
    returns_df = pd.read_csv(os.path.join(data_folder, 'crypto_data.csv'), parse_dates=['timestamp'],
                             index_col=['timestamp', 'symbol'])
    returns_df.index = returns_df.index.set_levels(
        [pd.to_datetime(returns_df.index.levels[0], utc=True), returns_df.index.levels[1]])
    return returns_df


def split_time_string(time_str: str) -> (int, str):
    # Split the input into number and unit
    match = re.match(r"(\d+\.?\d*)\s*([smhdwMy])", time_str.strip())

    if not match:
        raise ValueError(f"Invalid time format. Value: {time_str}")

    value, unit = match.groups()
    return int(value), unit


def convert_unit_to_pandas_freq(unit_str):
    # Map the unit to pandas frequency
    freq_map = {
        's': 's',  # Seconds
        'm': 'min',  # Minutes
        'h': 'h',  # Hours
        'd': 'D',  # Days
        'w': 'W',  # Weeks
        'M': 'ME',  # Months
        'y': 'YE'  # Years
    }

    if unit_str not in freq_map:
        raise ValueError(f"Invalid time unit: {unit_str}")

    return freq_map[unit_str]


def convert_time(time_str: str, target_unit: str) -> int:
    value, unit = split_time_string(time_str)
    # Dictionary to hold conversion factors to seconds for single-character units
    conversion_factors = {
        's': 1,  # seconds
        'm': 60,  # minutes
        'h': 3600,  # hours
        'd': 86400,  # days
        'w': 604800,  # weeks
        'M': 2628000,  # months (average)
        'y': 31536000  # years
    }

    if unit not in conversion_factors:
        raise ValueError(f"Invalid source unit. Value: {unit}")

    if target_unit not in conversion_factors:
        raise ValueError(f"Invalid target unit. Value: {target_unit}")

    # Convert the value to seconds
    seconds = value * conversion_factors[unit]
    # Convert from seconds to the target unit
    return seconds / conversion_factors[target_unit]


def subtract_period(date_str: datetime | str, period_str: str) -> str:
    if isinstance(date_str, datetime):
        date = date_str
    else:
        date = parse_date(date_str)

    match = re.match(r"(\d+)([smhdMy])", period_str)
    if not match:
        raise ValueError("Invalid period format. Example of valid format: '5m', '2h', '1d', '3M', '1y'")

    amount = int(match.group(1))
    unit = match.group(2)

    if unit == 's':
        delta = timedelta(seconds=amount)
    elif unit == 'm':
        delta = timedelta(minutes=amount)
    elif unit == 'h':
        delta = timedelta(hours=amount)
    elif unit == 'd':
        delta = timedelta(days=amount)
    elif unit == 'M':
        month = (date.month - amount - 1) % 12 + 1
        year = date.year + ((date.month - amount - 1) // 12)
        day = min(date.day, (datetime(year, month + 1, 1) - timedelta(days=1)).day)
        new_date = datetime(year, month, day, date.hour, date.minute, date.second)
        delta = new_date - date
    elif unit == 'y':
        year = date.year - amount
        day = min(date.day, (datetime(year, date.month + 1, 1) - timedelta(
            days=1)).day) if date.month == 2 and date.day == 29 else date.day
        new_date = datetime(year, date.month, day, date.hour, date.minute, date.second)
        delta = new_date - date
    else:
        raise ValueError("Invalid unit. Must be one of 's', 'm', 'h', 'd', 'M', 'y'")

    if unit in ['M', 'y']:
        new_date = date + delta
    else:
        new_date = date - delta

    return new_date.strftime("%Y-%m-%dT%H:%M:%SZ")


def add_period(date_str: str, period_str: str) -> str:
    if isinstance(date_str, datetime):
        date = date_str
    else:
        date = parse_date(date_str)

    match = re.match(r"(\d+)([smhdMy])", period_str)
    if not match:
        raise ValueError("Invalid period format. Example of valid format: '5m', '2h', '1d', '3M', '1y'")

    amount: int = int(match.group(1))
    unit: str = match.group(2)

    if unit == 's':
        delta: timedelta = timedelta(seconds=amount)
    elif unit == 'm':
        delta = timedelta(minutes=amount)
    elif unit == 'h':
        delta = timedelta(hours=amount)
    elif unit == 'd':
        delta = timedelta(days=amount)
    elif unit == 'M':
        month = (date.month + amount - 1) % 12 + 1
        year = date.year + ((date.month + amount - 1) // 12)
        day = min(date.day, (datetime(year, month + 1, 1) - timedelta(days=1)).day)
        new_date = datetime(year, month, day, date.hour, date.minute, date.second)
        delta = new_date - date
    elif unit == 'y':
        year = date.year + amount
        day = min(date.day, (datetime(year, date.month + 1, 1) - timedelta(
            days=1)).day) if date.month == 2 and date.day == 29 else date.day
        new_date = datetime(year, date.month, day, date.hour, date.minute, date.second)
        delta = new_date - date
    else:
        raise ValueError("Invalid unit. Must be one of 's', 'm', 'h', 'd', 'M', 'y'")

    new_date = date + delta

    return new_date.strftime("%Y-%m-%dT%H:%M:%S")


def parse_date(date_str: str) -> pd.Timestamp:
    supported_formats: List[str] = [
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d %H:%M:%S',
    ]

    for date_format in supported_formats:
        try:
            parsed_date = datetime.strptime(date_str, date_format)
            return pd.to_datetime(parsed_date)
        except ValueError:
            continue

    raise ValueError(f"Date string '{date_str}' is not in a supported format.")


def convert_datetime_to_str(date: pd.Timestamp | str) -> str:
    if isinstance(date, str):
        return date
    return date.strftime("%Y-%m-%dT%H:%M:%S")
