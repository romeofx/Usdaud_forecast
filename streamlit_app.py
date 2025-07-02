import streamlit as st
import numpy as np
import joblib
import xgboost as xgb

# Load model and scaler
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("models/xgboost_model.json")
scaler = joblib.load("models/scaler.pkl")

# Signal generator
def generate_signal(predicted_price, current_price, threshold=0.002):
    diff_ratio = (predicted_price - current_price) / current_price
    if diff_ratio > threshold:
        return "BUY"
    elif diff_ratio < -threshold:
        return "SELL"
    else:
        return "HOLD"

# Forecast function
def forecast_next(close_prices: list, threshold: float = 0.002):
    if len(close_prices) < 60:
        raise ValueError("Input must have at least 60 closing prices.")
    
    recent_data = np.array(close_prices[-60:]).reshape(-1, 1)
    scaled = scaler.transform(recent_data)
    X_xgb = scaled.reshape(1, 60)
    predicted_scaled = xgb_model.predict(X_xgb)[0]
    predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]
    current_price = close_prices[-1]
    signal = generate_signal(predicted_price, current_price, threshold)

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

# Streamlit UI
st.set_page_config(page_title="XAUUSD Forecast", layout="centered")
st.title("📊 XAUUSD Forecast & Trade Signal")

st.markdown("Enter the last 60 close prices of gold (XAUUSD) to forecast the next price and receive a trading signal.")

input_prices = st.text_area("Close Prices (comma-separated):", height=150)
threshold = st.slider("Signal Sensitivity Threshold", 0.0, 0.01, 0.002, step=0.001)

if st.button("Predict"):
    try:
        prices = [float(p.strip()) for p in input_prices.split(",") if p.strip()]
        result = forecast_next(prices, threshold)

        st.success("✅ Prediction successful!")
        st.write(f"**Predicted Price:** ${result['predicted_price']}")
        st.write(f"**Current Price:** ${result['current_price']}")
        st.write(f"**Signal:** 🟢 {result['signal']}")
        st.write(f"**Take Profit (TP):** ${result['take_profit']}")
        st.write(f"**Stop Loss (SL):** ${result['stop_loss']}")
    except ValueError as ve:
        st.error(f"⚠️ Input Error: {ve}")
    except Exception as e:
        st.error(f"❌ Server Error: {e}")
