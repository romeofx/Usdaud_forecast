import streamlit as st
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

# === Step 1: Load and explore the dataset
try:
    df = pd.read_csv("xauusd_data.csv")
    if "close" not in df.columns:
        st.warning("⚠️ 'close' column not found in CSV.")
        df = None
except Exception as e:
    st.warning(f"⚠️ Failed to load dataset: {e}")
    df = None

# === Load model and scaler
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("models/xgboost_model.json")
scaler = joblib.load("models/scaler.pkl")

# === Apply Custom CSS for dark theme
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #ffffff;
    }
    .main {
        background-color: #1e1e1e;
        padding: 2rem;
        border-radius: 10px;
    }
    .stTextInput, .stNumberInput, .stButton {
        background-color: #333333 !important;
        color: #ffffff !important;
    }
    h1, h2, h3 {
        color: #f39c12;
    }
    </style>
""", unsafe_allow_html=True)

# === App Title and Intro
st.title("📈 Gold Price Forecast (XAUUSD)")
st.markdown("AI-powered prediction & trading signal generator using XGBoost")

# === Extract last 60 closing prices if data available
if df is not None:
    last_60_prices = ",".join([str(int(val)) for val in df["close"].tail(60)])
    with st.expander("📥 Auto-filled Last 60 Close Prices"):
        st.code(last_60_prices, language="text")
        st.download_button(
            label="⬇️ Download Prices",
            data=last_60_prices,
            file_name="last_60_prices.txt",
            mime="text/plain"
        )

# === User Input Section
st.subheader("📝 Enter the last 60 closing prices:")
user_input = st.text_area("Closing prices (comma-separated)", value=last_60_prices if df is not None else "", height=150)

threshold = st.slider("🎚️ Signal Threshold", 0.0, 0.05, 0.002, step=0.001)

# === Signal Generator
def generate_signal(predicted_price, current_price, threshold=0.002):
    diff_ratio = (predicted_price - current_price) / current_price
    if diff_ratio > threshold:
        return "BUY"
    elif diff_ratio < -threshold:
        return "SELL"
    else:
        return "HOLD"

# === Forecast Logic
def forecast_next(close_prices: list, threshold: float = 0.002):
    if len(close_prices) < 60:
        raise ValueError("Input must have at least 60 closing prices.")

    recent_data = np.array(close_prices[-60:]).reshape(-1, 1)
    scaled = scaler.transform(recent_data)
    X_xgb = scaled.reshape(1, 60)

    xgb_pred = xgb_model.predict(X_xgb)
    predicted_scaled = xgb_pred[0]
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

# === Predict Button
if st.button("🚀 Predict"):
    try:
        prices = [float(x.strip()) for x in user_input.split(",") if x.strip()]
        result = forecast_next(prices, threshold)

        st.success(f"📊 Predicted Price: ${result['predicted_price']}")
        st.info(f"💡 Signal: {result['signal']}")
        st.markdown(f"✅ Take Profit: **${result['take_profit']}**")
        st.markdown(f"❌ Stop Loss: **${result['stop_loss']}**")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
