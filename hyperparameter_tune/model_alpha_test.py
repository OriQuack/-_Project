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

    feats = row.drop(columns=["Ticker", "Date", "Close"]).values
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
# 4) TEST WEIGHT_EXP VALUES
# ----------------------------
weight_exp_values = np.arange(1, -1.2, -0.2)
prediction_days = pd.to_datetime([
    "2024-08-19", "2024-08-20", "2024-08-21", "2024-08-22", "2024-08-23"
])

results = []
for weight_exp in weight_exp_values:
    overall_results = []
    for target_day in prediction_days:
        # Build training data and fit 5 models
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

        # Predict on TEST set with exponential weighting
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
                w_i = np.exp(-i * weight_exp)
                preds.append(pred_val * w_i)
                weights.append(w_i)

            if len(preds) == 0:
                continue

            final_pred = sum(preds) / sum(weights)
            overall_results.append({"Ticker": tk, "Date": target_day, "Pred_Close": final_pred})

    # Collect results and calculate MSE
    pred_df = (
        pd.DataFrame(overall_results).sort_values(by=["Ticker", "Date"]).reset_index(drop=True)
    )
    merged_df = pred_df.merge(
        df[["Ticker", "Date", "Close"]], how="left", on=["Ticker", "Date"]
    )
    merged_df.rename(columns={"Close": "Actual_Close"}, inplace=True)

    overall_mse = mean_squared_error(merged_df["Actual_Close"], merged_df["Pred_Close"])
    results.append({"Weight_Exp": weight_exp, "Overall_MSE": overall_mse})

# ----------------------------
# PRINT RESULTS
# ----------------------------
results_df = pd.DataFrame(results)
print(results_df)

# ----------------------------
# VISUALIZE RESULTS
# ----------------------------
plt.figure(figsize=(10, 6))
plt.plot(results_df["Weight_Exp"], results_df["Overall_MSE"], marker="o")
plt.title("Effect of WEIGHT_EXP on Overall MSE")
plt.xlabel("WEIGHT_EXP")
plt.ylabel("Overall MSE")
plt.grid(True)
plt.tight_layout()
plt.savefig("./weight_exp_vs_mse.png")
plt.show()
