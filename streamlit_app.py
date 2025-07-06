import streamlit as st
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import requests

# === Configuration ===
SEQUENCE_LENGTH = 600  # Changed from 350 to 600

# === Load Model and Scaler ===
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("models/xgboost_model.json")
scaler = joblib.load("models/scaler.pkl")

# === Custom CSS for Dark Theme ===
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
    .stTextInput, .stTextArea, .stNumberInput, .stButton, .stSlider {
        background-color: #333333 !important;
        color: #ffffff !important;
    }
    h1, h2, h3 {
        color: #f39c12;
    }
    </style>
""", unsafe_allow_html=True)

# === App Title ===
st.title("📈 Gold Price Forecast (XAUUSD)")
st.markdown("AI-powered prediction & trading signal generator using XGBoost")

# === Load Latest 600 Prices from CSV ===
def load_last_prices(n=SEQUENCE_LENGTH):
    try:
        df = pd.read_csv("xauusd_data.csv")
        if "close" not in df.columns:
            st.warning("⚠️ 'close' column not found in CSV.")
            return ""
        return ",".join([str(int(round(val))) for val in df["close"].tail(n)])
    except Exception as e:
        st.warning(f"⚠️ Failed to load dataset: {e}")
        return ""

# === Signal Generation Logic ===
def generate_signal(predicted_price, current_price, threshold=0.002):
    diff_ratio = (predicted_price - current_price) / current_price
    if diff_ratio > threshold:
        return "BUY"
    elif diff_ratio < -threshold:
        return "SELL"
    return "HOLD"

# === Forecast Function ===
def forecast_next(close_prices: list, threshold: float = 0.002):
    if len(close_prices) < SEQUENCE_LENGTH:
        raise ValueError(f"Input must have at least {SEQUENCE_LENGTH} closing prices.")
    recent_data = np.array(close_prices[-SEQUENCE_LENGTH:]).reshape(-1, 1)
    scaled = scaler.transform(recent_data)
    X_input = scaled.reshape(1, SEQUENCE_LENGTH)
    predicted_scaled = xgb_model.predict(X_input)[0]
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

# === Load & Display Prices ===
last_600_prices = load_last_prices()

with st.expander(f"📥 Auto-filled Last {SEQUENCE_LENGTH} Close Prices"):
    st.code(last_600_prices, language="text")
    st.download_button(
        label="⬇️ Download Prices",
        data=last_600_prices,
        file_name=f"last_{SEQUENCE_LENGTH}_prices.txt",
        mime="text/plain"
    )

# === Input Section ===
st.subheader(f"Enter {SEQUENCE_LENGTH} closing prices:")
user_input = st.text_area("Closing prices (comma-separated)", value=last_600_prices, height=200)
threshold = st.slider("Signal Threshold", 0.0, 0.05, 0.002, step=0.001)

# === Prediction Trigger ===
if st.button("🔮 Predict"):
    try:
        prices = [float(x.strip()) for x in user_input.split(",") if x.strip()]
        result = forecast_next(prices, threshold)

        st.success("📊 Prediction Result:")
        st.markdown(f"**Predicted Price:** ${result['predicted_price']}")
        st.markdown(f"**Current Price:** ${result['current_price']}")
        st.markdown(f"**Signal:** 🚩 `{result['signal']}`")
        st.markdown(f"✅ **Take Profit:** ${result['take_profit']}")
        st.markdown(f"❌ **Stop Loss:** ${result['stop_loss']}")

        result_text = (
            f"📊 Prediction Result:\n"
            f"Predicted Price: {result['predicted_price']}\n"
            f"Current Price: {result['current_price']}\n"
            f"Signal: {result['signal']}\n"
            f"Take Profit: {result['take_profit']}\n"
            f"Stop Loss: {result['stop_loss']}"
        )
        st.download_button(
            label="📄 Download Result",
            data=result_text,
            file_name="prediction_result.txt",
            mime="text/plain"
        )
    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# === Contact Form ===
st.markdown("---")
st.subheader("📬 Stay Updated with Trading Signals")

with st.form("contact_form", clear_on_submit=True):
    name = st.text_input("Name")
    email = st.text_input("Email")
    message = st.text_area("Message")
    submit = st.form_submit_button("Send")

    if submit:
        if not name or not email or not message:
            st.warning("⚠️ Please fill out all fields.")
        else:
            form_data = {
                "name": name,
                "email": email,
                "message": message
            }
            formspree_url = "https://formspree.io/f/mpwrnoqv"
            try:
                response = requests.post(formspree_url, data=form_data)
                if response.status_code in [200, 202]:
                    st.success("✅ Message sent successfully!")
                else:
                    st.error(f"❌ Failed to send message. (Status: {response.status_code})")
            except Exception as e:
                st.error(f"⚠️ Error sending message: {e}")
