import pandas as pd
import re
import os

# Directory containing the CSV files
input_folder = "data"
output_folder = "processed_data"

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)


# Define the function to process a single file
def process_csv(file_path, output_path):
    # Read the original CSV
    df = pd.read_csv(file_path)

    # Parse columns into tuples (Attribute, Ticker) using a regex
    new_cols = []
    pattern = re.compile(r"^\('([^']+)', '([^']+)'\)$")
    for col in df.columns:
        if col == "Ticker":
            new_cols.append(("Ticker", "Ticker"))
        else:
            match = pattern.match(col)
            if match:
                attribute, ticker_name = match.groups()
                new_cols.append((attribute, ticker_name))
            else:
                new_cols.append((col, col))

    df.columns = pd.MultiIndex.from_tuples(new_cols)

    # Set the index to (Ticker, Ticker)
    df = df.set_index(("Ticker", "Ticker"))

    # Extract data for each ticker
    final_data = []
    for ticker in df.index:
        # Get the row data
        row = df.loc[ticker]

        # For each attribute, try to get the value corresponding to this ticker
        def get_val(attr):
            col = (attr, ticker)
            return row[col] if col in row.index else None

        O = get_val("Open")
        H = get_val("High")
        L = get_val("Low")
        C = get_val("Close")
        V = get_val("Volume")

        final_data.append([ticker, O, H, L, C, V])

    # Create a final DataFrame with the required columns
    final_df = pd.DataFrame(
        final_data, columns=["Ticker", "Open", "High", "Low", "Close", "Volume"]
    )

    # Save to the new output path
    final_df.to_csv(output_path, index=False)


# Process each file in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith(".csv"):
        input_file = os.path.join(input_folder, file_name)
        output_file = os.path.join(output_folder, file_name)
        print(f"Processing {input_file}...")
        process_csv(input_file, output_file)
print(f"All files processed. Check the '{output_folder}' folder.")
