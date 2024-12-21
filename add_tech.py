import pandas as pd
import os
from ta import add_all_ta_features
from datetime import datetime
import re

# Input files - each file represents data for multiple companies on a given date
# Directory to search
directory_path = "./raw_data"

# Regular expression to match files ending with .csv
csv_regex = re.compile(r".*\.csv$", re.IGNORECASE)

# List to store file names
input_files = []

# Iterate through the files in the directory
for file_name in os.listdir(directory_path):
    if csv_regex.match(file_name):
        input_files.append(f"./raw_data/{file_name}")


# Output directory
output_dir = "./processed_data"
os.makedirs(output_dir, exist_ok=True)

# Combine all files into a single DataFrame
dfs = []
for file_path in input_files:
    # Extract the date from the filename (assuming the format 'YYYY-MM-DD.csv')
    base_name = os.path.basename(file_path)
    date_str = os.path.splitext(base_name)[0]  # e.g. '2024-08-12'
    date = datetime.strptime(date_str, "%Y-%m-%d")

    # Read the CSV
    df = pd.read_csv(file_path)

    # Add the Date column
    df["Date"] = date

    dfs.append(df)

# Concatenate all data into one DataFrame
full_df = pd.concat(dfs, ignore_index=True)


# Ensure all necessary columns exist
required_cols = {"Ticker", "Open", "High", "Low", "Close", "Volume", "Date"}
if not required_cols.issubset(full_df.columns):
    raise ValueError(
        "The DataFrame must have 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', and 'Date' columns."
    )

# Group by company and sort by date
grouped = full_df.groupby("Ticker", group_keys=False)


from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volume import OnBalanceVolumeIndicator


def process_company_data(df):
    df = df.sort_values("Date")

    # Minimum number of rows required (for something like RSI or a 14-day SMA)
    min_required = 14
    if len(df) < min_required:
        # Not enough data to calculate certain indicators reliably
        return df

    # Compute selected technical indicators
    df["SMA_14"] = SMAIndicator(
        close=df["Close"], window=14, fillna=True
    ).sma_indicator()
    df["EMA_14"] = EMAIndicator(
        close=df["Close"], window=14, fillna=True
    ).ema_indicator()
    df["RSI_14"] = RSIIndicator(close=df["Close"], window=14, fillna=True).rsi()
    df["OBV"] = OnBalanceVolumeIndicator(
        close=df["Close"], volume=df["Volume"], fillna=True
    ).on_balance_volume()

    return df


processed_df = grouped.apply(process_company_data, include_groups=True)

# Reset index if needed
processed_df.reset_index(drop=True, inplace=True)

# Save the processed data
output_file = os.path.join(output_dir, "combined_pdata.csv")
processed_df.to_csv(output_file, index=False)

print(f"Processed file has been saved to {output_file}")
