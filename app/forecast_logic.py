import numpy as np
import joblib
import xgboost as xgb

# Load XGBoost model and scaler
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("models/xgboost_model.json")
scaler = joblib.load("models/scaler.pkl")

# Signal generator logic
def generate_signal(predicted_price, current_price, threshold=0.002):
    diff = (predicted_price - current_price) / current_price
    if diff > threshold:
        return "BUY"
    elif diff < -threshold:
        return "SELL"
    else:
        return "HOLD"

# Forecast next price and give signal
def forecast_next(close_prices: list):
    if len(close_prices) < 60:
        raise ValueError("Input must have at least 60 closing prices.")

    # Scale the latest 60 prices
    recent_data = np.array(close_prices[-60:]).reshape(-1, 1)
    scaled = scaler.transform(recent_data)

    # Reshape for XGBoost (1 sample, 60 features)
    X_xgb = scaled.reshape(1, 60)
    xgb_pred = xgb_model.predict(X_xgb)

    # Inverse transform to get original price
    predicted_scaled = xgb_pred[0]
    predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]

    current_price = close_prices[-1]
    signal = generate_signal(predicted_price, current_price)

    return {
        "predicted_price": round(predicted_price, 3),
        "current_price": round(current_price, 3),
        "signal": signal
    }
