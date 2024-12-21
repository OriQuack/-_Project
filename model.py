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
ALPHA = 0.001
WEIGHT_EXP = 0.2
TRAIN_SIZE = 400
DATE_FORMAT = "%Y-%m-%d"
RANDOM_SEED = 1222

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----------------------------
# 1) LOAD DATA
# ----------------------------
df = pd.read_csv("./processed_data/combined_pdata.csv")
df["Date"] = pd.to_datetime(df["Date"], format=DATE_FORMAT, errors="coerce")
df = df.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)

# ----------------------------
# 2) SPLIT TICKERS INTO TRAIN & TEST
# ----------------------------
all_tickers = df["Ticker"].unique().tolist()
random.shuffle(all_tickers)
train_tickers = set(all_tickers[:TRAIN_SIZE])
test_tickers = set(all_tickers[TRAIN_SIZE:])


# ----------------------------
# 3) DEFINE HELPER FUNCTIONS
# ----------------------------
def get_features_for_day(subdf, day):
    """
    Return a 1D numpy array of features for the specified date,
    or None if the data is missing.
    """
    row = subdf.loc[subdf["Date"] == day]
    if row.empty:
        return None

    # Drop columns that should not be features
    # Make sure you do not keep columns that are constant or duplicates
    feats = row.drop(columns=["Ticker", "Date", "Close"]).values
    return feats.flatten()


def fit_model(X, y, alpha=ALPHA):
    """
    Fits a Ridge model with X, y.
    Returns the fitted model and the fitted scaler (for feature standardization).
    """
    # 1) Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2) Fit Ridge
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_scaled, y)
    return model, scaler


def predict_with_model(model, scaler, feats):
    """
    Given a model and scaler, transform feats then predict with the model.
    """
    feats_scaled = scaler.transform([feats])
    return model.predict(feats_scaled)[0]


# ----------------------------
# 4) MAIN LOOP TO PREDICT 8/19 to 8/23
# ----------------------------
prediction_days = pd.to_datetime(
    ["2024-08-19", "2024-08-20", "2024-08-21", "2024-08-22", "2024-08-23"]
)

results = []
for target_day in prediction_days:
    # 4a) Build training data and fit 5 models
    models_and_scalers = []
    for i in range(1, 6):
        day_i = target_day - pd.Timedelta(days=i)
        X_train = []
        y_train = []

        for tk in train_tickers:
            subdf = df[df["Ticker"] == tk]
            feats = get_features_for_day(subdf, day_i)
            if feats is None:
                continue

            row_target = subdf.loc[subdf["Date"] == target_day]
            if row_target.empty:
                continue
            close_val = row_target["Close"].values[0]

            X_train.append(feats)
            y_train.append(close_val)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        if len(X_train) > 0:
            model_i, scaler_i = fit_model(X_train, y_train, alpha=ALPHA)
            models_and_scalers.append((model_i, scaler_i))
        else:
            models_and_scalers.append((None, None))

    # 4b) Predict on TEST set with exponential weighting
    for tk in test_tickers:
        subdf = df[df["Ticker"] == tk]

        preds = []
        weights = []
        for i in range(1, 6):
            model_i, scaler_i = models_and_scalers[i - 1]
            if model_i is None:
                continue

            day_i = target_day - pd.Timedelta(days=i)
            feats = get_features_for_day(subdf, day_i)
            if feats is None:
                continue

            pred_val = predict_with_model(model_i, scaler_i, feats)
            w_i = np.exp(-i * WEIGHT_EXP)
            preds.append(pred_val * w_i)
            weights.append(w_i)

        if len(preds) == 0:
            continue

        final_pred = sum(preds) / sum(weights)
        results.append({"Ticker": tk, "Date": target_day, "Pred_Close": final_pred})

# ----------------------------
# 5) COLLECT RESULTS AND CALCULATE MSE
# ----------------------------
pred_df = (
    pd.DataFrame(results).sort_values(by=["Ticker", "Date"]).reset_index(drop=True)
)
merged_df = pred_df.merge(
    df[["Ticker", "Date", "Close"]], how="left", on=["Ticker", "Date"]
)
merged_df.rename(columns={"Close": "Actual_Close"}, inplace=True)

mse_per_date = merged_df.groupby("Date", group_keys=False).apply(
    lambda group: mean_squared_error(group["Actual_Close"], group["Pred_Close"])
)
overall_mse = mean_squared_error(merged_df["Actual_Close"], merged_df["Pred_Close"])

print("MSE per Date:")
print(mse_per_date)
print("\nOverall MSE:")
print(overall_mse)

# Visualize MSE per date for model 2
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

plt.title("Mean Squared Error (MSE) per Date and Overall (Model 2)")
plt.xlabel("Date")
plt.ylabel("Mean Squared Error")
plt.xticks(mse_per_date.index, ["8/19", "8/20", "8/21", "8/22", "8/23"], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("./model2/mse_per_date.png")
plt.show()


# ----------------------------
# 6) VISUALIZATION (Optional)
# ----------------------------
plt.figure(figsize=(20, 8))
latest_predictions = merged_df.groupby("Ticker").last().reset_index()

x = range(len(latest_predictions))
plt.bar(x, latest_predictions["Actual_Close"], alpha=0.6, label="Actual Price")
plt.bar(
    x,
    latest_predictions["Pred_Close"],
    alpha=0.6,
    label="Predicted Price",
    color="orange",
)

plt.xticks(x, latest_predictions["Ticker"], rotation=90)
plt.xlabel("Companies (Tickers)")
plt.ylabel("Stock Price")
plt.title("Comparison of Predicted vs Actual Stock Prices")
plt.legend()
plt.tight_layout()
plt.savefig("./model2/model2.png")

# ----------------------------
# 7) FIND BEST AND WORST PREDICTIONS
# ----------------------------
merged_df["Proportional_Error"] = abs(
    (merged_df["Actual_Close"] - merged_df["Pred_Close"]) / merged_df["Actual_Close"]
)
error_by_ticker = merged_df.groupby("Ticker")["Proportional_Error"].mean()
best_ticker = error_by_ticker.idxmin()
worst_ticker = error_by_ticker.idxmax()

best_ticker_data = merged_df[merged_df["Ticker"] == best_ticker]
worst_ticker_data = merged_df[merged_df["Ticker"] == worst_ticker]

# Plot best ticker
plt.figure(figsize=(12, 6))
plt.plot(
    best_ticker_data["Date"],
    best_ticker_data["Actual_Close"],
    label="Actual Price",
    marker="o",
)
plt.plot(
    best_ticker_data["Date"],
    best_ticker_data["Pred_Close"],
    label="Predicted Price",
    marker="x",
)
plt.title(f"Best Prediction by Proportional Error: {best_ticker}")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.tight_layout()
plt.savefig("model2/model2-good.png")

# Plot worst ticker
plt.figure(figsize=(12, 6))
plt.plot(
    worst_ticker_data["Date"],
    worst_ticker_data["Actual_Close"],
    label="Actual Price",
    marker="o",
)
plt.plot(
    worst_ticker_data["Date"],
    worst_ticker_data["Pred_Close"],
    label="Predicted Price",
    marker="x",
)
plt.title(f"Worst Prediction by Proportional Error: {worst_ticker}")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.tight_layout()
plt.savefig("model2/model2-bad.png")