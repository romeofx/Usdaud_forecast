# forecast_logic.py

import numpy as np
import joblib
import xgboost as xgb
from pathlib import Path

# === Constants ===
SEQUENCE_LENGTH = 350

# === Load model and scaler once ===
MODEL_PATH = Path("models/xgboost_model.json")
SCALER_PATH = Path("models/scaler.pkl")

xgb_model = xgb.XGBRegressor()
if MODEL_PATH.exists():
    xgb_model.load_model(str(MODEL_PATH))
else:
    raise FileNotFoundError("❌ Model file not found at models/xgboost_model.json")

if SCALER_PATH.exists():
    scaler = joblib.load(str(SCALER_PATH))
else:
    raise FileNotFoundError("❌ Scaler file not found at models/scaler.pkl")


def generate_signal(predicted_price: float, current_price: float, threshold: float = 0.002) -> str:
    """Determine buy/sell/hold signal based on predicted and current price."""
    diff_ratio = (predicted_price - current_price) / current_price
    if diff_ratio > threshold:
        return "BUY"
    elif diff_ratio < -threshold:
        return "SELL"
    return "HOLD"


def forecast_next(close_prices: list[float], threshold: float = 0.002) -> dict:
    """
    Forecast the next price and generate trading signal.

    Args:
        close_prices (list): Last 350 closing prices
        threshold (float): Threshold to determine buy/sell/hold

    Returns:
        dict: prediction result
    """
    if len(close_prices) < SEQUENCE_LENGTH:
        raise ValueError(f"⚠️ Input must contain at least {SEQUENCE_LENGTH} closing prices.")

    # Prepare input data
    recent_data = np.array(close_prices[-SEQUENCE_LENGTH:]).reshape(-1, 1)
    scaled = scaler.transform(recent_data)
    X_input = scaled.reshape(1, SEQUENCE_LENGTH)

    # Predict
    predicted_scaled = xgb_model.predict(X_input)[0]
    predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]
    current_price = close_prices[-1]
    signal = generate_signal(predicted_price, current_price, threshold)

    # Calculate TP and SL
    buffer = 0.005
    if signal == "BUY":
        tp = predicted_price * (1 + buffer)
        sl = predicted_price * (1 - buffer)
    elif signal == "SELL":
        tp = predicted_price * (1 - buffer)
        sl = predicted_price * (1 + buffer)
    else:
        tp = sl = predicted_price

    return {
        "predicted_price": round(predicted_price, 3),
        "current_price": round(current_price, 3),
        "signal": signal,
        "take_profit": round(tp, 3),
        "stop_loss": round(sl, 3)
    }
