import pandas as pd
import numpy as np
import random
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ----------------------------
# HYPERPARAMETERS
# ----------------------------
ALPHA = 0.01  # Regularization strength for Ridge
TRAIN_SIZE = 400
RANDOM_SEED = 1222  # Seed for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----------------------------
# 1) LOAD DATA
# ----------------------------
data = pd.read_csv("./processed_data/combined_pdata.csv")
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
data.sort_values(by=["Ticker", "Date"], inplace=True)
data.reset_index(drop=True, inplace=True)

# ----------------------------
# 2) FILTER TRAINING DATA
# ----------------------------
train_data = data[(data["Date"] >= "2024-08-12") & (data["Date"] <= "2024-08-16")]


# ----------------------------
# 3) CALCULATE TARGET VARIABLE
# ----------------------------
def calculate_next_week_average(df, start_date, end_date):
    next_week_data = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
    return next_week_data.groupby("Ticker")["Close"].mean()


next_week_avg = calculate_next_week_average(data, "2024-08-19", "2024-08-23")
train_data = train_data.merge(next_week_avg.rename("NextWeekAvg"), on="Ticker")

# ----------------------------
# 4) DEFINE FEATURES
# ----------------------------
FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA_14",
    "EMA_14",
    "RSI_14",
    "OBV",
]
train_data["DaysSince2000"] = (train_data["Date"] - pd.Timestamp("2000-01-01")).dt.days
FEATURES.append("DaysSince2000")

# ----------------------------
# 5) SPLIT DATASET
# ----------------------------
unique_tickers = train_data["Ticker"].unique()
test_tickers = random.sample(list(unique_tickers), 500 - TRAIN_SIZE)
train_set = train_data[~train_data["Ticker"].isin(test_tickers)]
test_set = train_data[train_data["Ticker"].isin(test_tickers)]

X_train, y_train = train_set[FEATURES], train_set["NextWeekAvg"]
X_test, y_test = test_set[FEATURES], test_set["NextWeekAvg"]

# ----------------------------
# 6) STANDARDIZE FEATURES
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 7) TRAIN MODEL
# ----------------------------
ridge_model = Ridge(alpha=ALPHA)
ridge_model.fit(X_train_scaled, y_train)

# ----------------------------
# 8) EVALUATE MODEL
# ----------------------------
y_pred = ridge_model.predict(X_test_scaled)
overall_mse = mean_squared_error(y_test, y_pred)

# Calculate MSE per date
test_set["Predicted"] = y_pred
mse_per_date = test_set.groupby("Date").apply(
    lambda df: mean_squared_error(df["NextWeekAvg"], df["Predicted"])
)
print("\nMean Squared Error per Date:")
print(mse_per_date)
print()
print(f"Overall Mean Squared Error:\n{overall_mse}")

# Visualize MSE per date
plt.figure(figsize=(12, 6))
bars = plt.bar(
    mse_per_date.index,
    mse_per_date.values,
    label="MSE per Date",
    color="green",
    alpha=0.7,
)
plt.axhline(
    y=overall_mse, color="r", linestyle="--", label=f"Overall MSE: {overall_mse:.2f}"
)

# Add value annotations on bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
    )

plt.title("Mean Squared Error (MSE) per Date and Overall")
plt.xlabel("Date")
plt.ylabel("Mean Squared Error")
plt.xticks(mse_per_date.index, ["8/19", "8/20", "8/21", "8/22", "8/23"], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("./model1/mse_per_date.png")
plt.show()


# ----------------------------
# 9) VISUALIZATION
# ----------------------------
# Merge predicted and actual values for visualization
predicted_vs_actual = pd.DataFrame(
    {"Ticker": test_set["Ticker"], "Actual": y_test.values, "Predicted": y_pred}
).reset_index()

# Residuals
predicted_vs_actual["Residual"] = (
    predicted_vs_actual["Actual"] - predicted_vs_actual["Predicted"]
)

# Bar plot for predicted vs actual prices
latest_predictions = predicted_vs_actual.groupby("Ticker").last().reset_index()
x = range(len(latest_predictions))
plt.figure(figsize=(20, 8))
plt.bar(x, latest_predictions["Actual"], alpha=0.6, label="Actual Price")
plt.bar(
    x,
    latest_predictions["Predicted"],
    alpha=0.6,
    label="Predicted Price",
    color="orange",
)
plt.xticks(x, latest_predictions["Ticker"], rotation=90)
plt.xlabel("Companies (Tickers)")
plt.ylabel("Average Price")
plt.title("Comparison of Predicted vs Actual Average Prices")
plt.legend()
plt.tight_layout()
plt.savefig("./model1/model1.png")

# Residual histogram
plt.figure(figsize=(10, 6))
plt.hist(
    predicted_vs_actual["Residual"],
    bins=30,
    color="purple",
    alpha=0.7,
    edgecolor="black",
)
plt.title("Residual Histogram")
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("./model1/model1-residuals.png")

# Best and worst predictions
predicted_vs_actual["Absolute_Error"] = predicted_vs_actual["Residual"].abs()
best_ticker = predicted_vs_actual.groupby("Ticker")["Absolute_Error"].sum().idxmin()
worst_ticker = predicted_vs_actual.groupby("Ticker")["Absolute_Error"].sum().idxmax()

# Plot for best prediction
best_ticker_data = data[
    (data["Ticker"] == best_ticker)
    & (data["Date"] >= "2024-08-19")
    & (data["Date"] <= "2024-08-23")
]
plt.figure(figsize=(12, 6))
plt.plot(
    best_ticker_data["Date"],
    best_ticker_data["Close"],
    label="Actual Price",
    marker="o",
)
plt.plot(
    best_ticker_data["Date"],
    [
        predicted_vs_actual.loc[
            predicted_vs_actual["Ticker"] == best_ticker, "Predicted"
        ].iloc[0]
    ]
    * len(best_ticker_data),
    label="Predicted Price",
    linestyle="--",
    color="r",
)
plt.title(f"Best Prediction: {best_ticker}")
plt.xlabel("Date")
plt.ylabel("Average Price")
plt.legend()
plt.tight_layout()
plt.savefig("./model1/model1-good.png")

# Plot for worst prediction
worst_ticker_data = data[
    (data["Ticker"] == worst_ticker)
    & (data["Date"] >= "2024-08-19")
    & (data["Date"] <= "2024-08-23")
]
plt.figure(figsize=(12, 6))
plt.plot(
    worst_ticker_data["Date"],
    worst_ticker_data["Close"],
    label="Actual Price",
    marker="o",
)
plt.plot(
    worst_ticker_data["Date"],
    [
        predicted_vs_actual.loc[
            predicted_vs_actual["Ticker"] == worst_ticker, "Predicted"
        ].iloc[0]
    ]
    * len(worst_ticker_data),
    label="Predicted Price",
    linestyle="--",
    color="r",
)
plt.title(f"Worst Prediction: {worst_ticker}")
plt.xlabel("Date")
plt.ylabel("Average Price")
plt.legend()
plt.tight_layout()
plt.savefig("./model1/model1-bad.png")
