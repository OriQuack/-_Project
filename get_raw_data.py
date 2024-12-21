import pandas as pd
import yfinance as yf
import requests
import os
from datetime import datetime

# Step 1: Get the S&P 500 tickers from Wikipedia
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
response = requests.get(url)
response.raise_for_status()  # Ensure the request was successful

# Parse the HTML tables using pandas
tables = pd.read_html(response.text)
sp500_table = tables[0]

# Extract the ticker symbols
sp500_tickers = sp500_table["Symbol"].tolist()

print(f"Fetched {len(sp500_tickers)} S&P 500 tickers.")

# Step 2: Define date range and directory for saving files
start_date = "2024-07-22"
end_date = "2024-08-24"
output_dir = "sp500_daily_data"
os.makedirs(output_dir, exist_ok=True)

# Step 3: Fetch historical data for each ticker
all_data = {}
for ticker in sp500_tickers:
    print(f"Fetching data for {ticker}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        all_data[ticker] = data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

# Step 4: Process data by date and save as daily CSV files
dates = pd.date_range(start=start_date, end=end_date, freq="B")  # Business days
for date in dates:
    date_str = date.strftime("%Y-%m-%d")
    daily_data = []

    for ticker, data in all_data.items():
        if date_str in data.index:
            row = data.loc[
                date_str, ["Open", "High", "Low", "Close", "Volume"]
            ].to_dict()
            row["Ticker"] = ticker
            daily_data.append(row)

    # Create a DataFrame for the day's data
    if daily_data:
        daily_df = pd.DataFrame(daily_data)
        daily_df.set_index("Ticker", inplace=True)

        # Save to CSV
        output_file = os.path.join(output_dir, f"{date_str}.csv")
        daily_df.to_csv(output_file)
        print(f"Saved data for {date_str} to {output_file}")

print("Data fetching and saving completed.")
