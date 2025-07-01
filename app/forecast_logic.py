import numpy as np
import joblib
import xgboost as xgb

# Load model and scaler
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("models/xgboost_model.json")
scaler = joblib.load("models/scaler.pkl")

# Signal generator
def generate_signal(predicted_price, current_price, threshold=0.002):
    diff = (predicted_price - current_price) / current_price
    if diff > threshold:
        return "BUY"
    elif diff < -threshold:
        return "SELL"
    else:
        return "HOLD"

# Forecast function
def forecast_next(close_prices: list):
    if len(close_prices) < 60:
        raise ValueError("Input must have at least 60 closing prices.")

    recent_data = np.array(close_prices[-60:]).reshape(-1, 1)
    scaled = scaler.transform(recent_data)
    X_xgb = scaled.reshape(1, 60)
    xgb_pred = xgb_model.predict(X_xgb)
    predicted_scaled = xgb_pred[0]
    predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]

    current_price = close_prices[-1]
    signal = generate_signal(predicted_price, current_price)

    # TP & SL logic (0.5% distance)
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
