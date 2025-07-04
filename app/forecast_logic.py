import numpy as np
import joblib
import xgboost as xgb
from pathlib import Path

# === Load model and scaler safely once ===
MODEL_PATH = Path("models/xgboost_model.json")
SCALER_PATH = Path("models/scaler.pkl")

# Load once at the top level
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
    """Determine buy/sell/hold signal."""
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
        close_prices (list): Last 60 closing prices
        threshold (float): Percentage threshold for decision logic

    Returns:
        dict: forecast result
    """
    if len(close_prices) < 60:
        raise ValueError("⚠️ Input must contain at least 60 closing prices.")

    # Prepare and scale the input
    recent_data = np.array(close_prices[-60:]).reshape(-1, 1)
    scaled = scaler.transform(recent_data)
    X_xgb = scaled.reshape(1, 60)

    # Predict next scaled value
    predicted_scaled = xgb_model.predict(X_xgb)[0]
    predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]
    current_price = close_prices[-1]

    # Get signal
    signal = generate_signal(predicted_price, current_price, threshold)

    # TP/SL logic based on signal
    buffer = 0.005  # ±0.5%
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
