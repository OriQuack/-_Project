import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
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
    This function looks in the subdf for the row that matches 'day'
    and then drops 'Ticker', 'Date', and 'Close' to yield the features.
    """
    row = subdf.loc[subdf["Date"] == day]
    if row.empty:
        return None

    # Drop columns that should not be features
    feats = row.drop(columns=["Ticker", "Date", "Close"], errors="ignore").values
    return feats.flatten()


def fit_model(X, y, alpha=ALPHA):
    """
    Fits a Ridge model with X, y.
    Returns the fitted model and the fitted scaler (for feature standardization).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

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
# 4) BUILD MODELS FOR 8/12 - 8/16
# ----------------------------
training_days = pd.to_datetime(
    ["2024-08-12", "2024-08-13", "2024-08-14", "2024-08-15", "2024-08-16"]
)

# Dictionary: day -> (model, scaler)
models_and_scalers = {}

for train_day in training_days:
    X_train = []
    y_train = []
    for tk in train_tickers:
        subdf = df[df["Ticker"] == tk]
        feats = get_features_for_day(subdf, train_day)
        if feats is None:
            continue

        target_row = subdf.loc[subdf["Date"] == train_day]
        if target_row.empty:
            continue

        close_val = target_row["Close"].values[0]
        X_train.append(feats)
        y_train.append(close_val)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    if len(X_train) == 0:
        models_and_scalers[train_day] = (None, None)
    else:
        model_i, scaler_i = fit_model(X_train, y_train, alpha=ALPHA)
        models_and_scalers[train_day] = (model_i, scaler_i)

# ----------------------------
# 5) EXPANDING WINDOW PREDICTION FOR 8/19 - 8/23
# ----------------------------
prediction_days = pd.to_datetime(
    ["2024-08-19", "2024-08-20", "2024-08-21", "2024-08-22", "2024-08-23"]
)

# To store final predictions
results = []

# We will store predicted closes in a dictionary: (Ticker, Date) -> predicted close
predicted_closes = {}

for target_day in prediction_days:
    for tk in test_tickers:
        # 5a) Build feature vector for each model day,
        #     INCLUDING previously predicted close if it falls between 8/19 and (target_day - 1).
        weighted_preds = []
        weights = []

        # To simulate an "expanding window," we want to incorporate
        # any predicted close from previous dates as part of the feature set.
        # One simple approach: If you want the predicted close from 8/19 to appear
        # as a "feature" on 8/20, you'd store it in your DataFrame as if it were
        # the actual close. Below is a straightforward demonstration
        # by creating a small copy of the subdf and replacing the close with the predicted ones.
        subdf = df[df["Ticker"] == tk].copy()

        # Incorporate previously predicted closes into subdf
        for day_check in prediction_days:
            if day_check >= target_day:
                break
            if (tk, day_check) in predicted_closes:
                # Replace the actual close with the predicted close
                # so that get_features_for_day(...) will pick it up
                pred_val = predicted_closes[(tk, day_check)]
                subdf.loc[subdf["Date"] == day_check, "Close"] = pred_val

        # Now compute the predictions from the 5 training-day models
        for train_day in training_days:
            model_i, scaler_i = models_and_scalers[train_day]
            if model_i is None:
                continue

            # We get the features for 'target_day' from subdf
            feats = get_features_for_day(subdf, target_day)
            if feats is None:
                continue

            pred_val = predict_with_model(model_i, scaler_i, feats)
            i = (target_day - train_day).days
            w = np.exp(-i * WEIGHT_EXP)
            weighted_preds.append(pred_val * w)
            weights.append(w)

        if len(weighted_preds) == 0:
            continue

        final_pred = sum(weighted_preds) / sum(weights)

        # 5b) Store the predicted close
        predicted_closes[(tk, target_day)] = final_pred
        results.append({"Ticker": tk, "Date": target_day, "Pred_Close": final_pred})

# ----------------------------
# 6) EVALUATE WITH ACTUAL CLOSES
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

# ----------------------------
# 7) VISUALIZE MSE
# ----------------------------
plt.figure(figsize=(12, 6))
bars = plt.bar(
    mse_per_date.index,
    mse_per_date.values,
    label="MSE per Date",
    color="blue",
    alpha=0.7,
)
plt.axhline(
    y=overall_mse, color="r", linestyle="--", label=f"Overall MSE: {overall_mse:.2f}"
)

# Add value annotations
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
    )

plt.title("Mean Squared Error (MSE) per Date (Expanding Window)")
plt.xlabel("Date")
plt.ylabel("Mean Squared Error")
plt.xticks(mse_per_date.index, ["8/19", "8/20", "8/21", "8/22", "8/23"], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("./model2/model2-ew-mse_per_date.png")

# ----------------------------
# 8) COMPARISON PLOT
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
plt.title("Comparison of Predicted vs Actual Prices (Expanding Window)")
plt.legend()
plt.tight_layout()
plt.savefig("./model2/model2-ew-comparison.png")

# ----------------------------
# 9) BEST & WORST PREDICTIONS
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
plt.title(f"Best Prediction by Proportional Error (Expanding Window): {best_ticker}")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.tight_layout()
plt.savefig("./model2/model2-ew-best.png")

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
plt.title(f"Worst Prediction by Proportional Error (Expanding Window): {worst_ticker}")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.tight_layout()
plt.savefig("./model2/model2-ew-worst.png")
