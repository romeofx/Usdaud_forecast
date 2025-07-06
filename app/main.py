from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import requests
import logging

from app.forecast_logic import forecast_next

# === Constants ===
SEQUENCE_LENGTH = 600
CSV_PATH = "xauusd_data.csv"
FORMSPREE_URL = "https://formspree.io/f/mpwrnoqv"

# === App initialization ===
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# === Logging setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CORS setup ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# === Utility: Load last 600 close prices ===
def load_last_600_prices() -> str:
    try:
        df = pd.read_csv(CSV_PATH)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        last_prices = df["close"].tail(SEQUENCE_LENGTH)
        return ",".join([str(int(round(val))) for val in last_prices])
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return ""

# === Web Page ===
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    last_600_prices = load_last_600_prices()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "last_600_prices": last_600_prices
    })

# === API Endpoint: Prediction ===
@app.post("/predict")
async def predict_api(request: Request, threshold: Optional[float] = 0.002):
    try:
        data = await request.json()
        close_prices: List[float] = data.get("close_prices", [])
        if len(close_prices) < SEQUENCE_LENGTH:
            return JSONResponse(
                content={"error": f"❌ You must provide at least {SEQUENCE_LENGTH} closing prices."},
                status_code=400
            )
        result = forecast_next(close_prices, threshold)
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception("Prediction failed")
        return JSONResponse(content={"error": f"❌ Prediction failed: {e}"}, status_code=500)

# === API Endpoint: Download prices ===
@app.get("/download-prices")
def download_prices():
    prices = load_last_600_prices()
    return HTMLResponse(
        content=prices,
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename=last_{SEQUENCE_LENGTH}_prices.txt"}
    )

# === API Endpoint: Subscribe via Formspree ===
@app.post("/subscribe")
def subscribe(name: str = Form(...), email: str = Form(...), message: str = Form("")):
    try:
        form_data = {"name": name, "email": email, "message": message}
        res = requests.post(FORMSPREE_URL, data=form_data)
        if res.status_code in [200, 202]:
            logger.info(f"Subscription successful for {email}")
            return RedirectResponse("/", status_code=303)
        else:
            logger.warning(f"Subscription failed: {res.status_code}")
            return JSONResponse({"error": "Subscription failed"}, status_code=500)
    except Exception as e:
        logger.exception("Subscription error")
        return JSONResponse({"error": f"Subscription error: {e}"}, status_code=500)
