import numpy as np
import joblib
import xgboost as xgb

# Load XGBoost model and scaler
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("models/xgboost_model.json")
scaler = joblib.load("models/scaler.pkl")

# Signal generator with adjustable threshold
def generate_signal(predicted_price, current_price, threshold=0.002):
    diff_ratio = (predicted_price - current_price) / current_price
    if diff_ratio > threshold:
        return "BUY"
    elif diff_ratio < -threshold:
        return "SELL"
    else:
        return "HOLD"

# Main forecast function
def forecast_next(close_prices: list, threshold: float = 0.002):
    if len(close_prices) < 60:
        raise ValueError("Input must have at least 60 closing prices.")

    # Prepare the last 60 prices
    recent_data = np.array(close_prices[-60:]).reshape(-1, 1)

    # Scale and reshape
    scaled = scaler.transform(recent_data)
    X_xgb = scaled.reshape(1, 60)

    # Predict next scaled value
    xgb_pred = xgb_model.predict(X_xgb)
    predicted_scaled = xgb_pred[0]

    # Inverse scale to get real price
    predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]

    # Extract current price
    current_price = close_prices[-1]

    # Generate buy/sell/hold signal
    signal = generate_signal(predicted_price, current_price, threshold)

    # TP & SL logic (buffer = 0.5%)
    buffer = 0.005
    if signal == "BUY":
        tp = predicted_price * (1 + buffer)
        sl = predicted_price * (1 - buffer)
    elif signal == "SELL":
        tp = predicted_price * (1 - buffer)
        sl = predicted_price * (1 + buffer)
    else:  # HOLD
        tp = sl = predicted_price

    # Final response
    return {
        "predicted_price": round(predicted_price, 3),
        "current_price": round(current_price, 3),
        "signal": signal,
        "take_profit": round(tp, 3),
        "stop_loss": round(sl, 3)
    }
