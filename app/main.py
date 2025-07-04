# main.py

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import json
import requests

from app.forecast_logic import forecast_next


app = FastAPI()

templates = Jinja2Templates(directory="templates")

# === CORS Setup ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

SEQUENCE_LENGTH = 350

# === Helper to Load CSV Data ===
def load_last_350_prices() -> str:
    try:
        df = pd.read_csv("xauusd_data.csv")
        if "close" not in df.columns:
            return ""
        return ",".join([str(int(val)) for val in df["close"].tail(SEQUENCE_LENGTH)])
    except Exception:
        return ""

# === HTML Page ===
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    last_350_prices = load_last_350_prices()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "last_350_prices": last_350_prices
    })

# === JSON Prediction API ===
@app.post("/predict")
async def predict_api(request: Request, threshold: Optional[float] = 0.002):
    data = await request.json()
    close_prices = data.get("close_prices", [])
    try:
        result = forecast_next(close_prices, threshold)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

# === Download prices ===
@app.get("/download-prices")
def download_prices():
    prices = load_last_350_prices()
    return HTMLResponse(
        content=prices,
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=last_350_prices.txt"}
    )

# === Subscribe (Formspree handler) ===
@app.post("/subscribe")
def subscribe(name: str = Form(...), email: str = Form(...), message: str = Form("")):
    form_data = {"name": name, "email": email, "message": message}
    res = requests.post("https://formspree.io/f/mpwrnoqv", data=form_data)
    if res.status_code in [200, 202]:
        return RedirectResponse("/", status_code=303)
    return JSONResponse({"error": "Subscription failed"}, status_code=500)
