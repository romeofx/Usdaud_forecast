from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os

#from app.forecast_logic import forecast_next

app = FastAPI(
    title="XAUUSD Forecast API",
    description="Forecast & trading signal generator using XGBoost.",
    version="1.0.0"
)

# Template directory
templates = Jinja2Templates(directory="templates")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# === Load latest 60 prices from CSV ===
def load_last_60_prices() -> str:
    try:
        df = pd.read_csv("xauusd_data.csv")
        if "close" not in df.columns:
            return ""
        return ",".join([str(int(val)) for val in df["close"].tail(60)])
    except Exception:
        return ""


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    last_60_prices = load_last_60_prices()
    return templates.TemplateResponse("index.html", {"request": request, "last_60_prices": last_60_prices})


class PredictionInput(BaseModel):
    close_prices: List[float]


@app.post("/predict")
async def predict(payload: PredictionInput, threshold: Optional[float] = 0.002):
    try:
        # Clamp threshold between 0.00 and 0.05
        threshold = max(0.00, min(threshold, 0.05))
        result = forecast_next(payload.close_prices, threshold)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.post("/subscribe")
async def subscribe(name: str = Form("Anonymous"), email: str = Form(...), message: str = Form("")):
    import requests
    form_data = {
        "name": name,
        "email": email,
        "message": message
    }
    res = requests.post("https://formspree.io/f/mpwrnoqv", data=form_data)
    if res.status_code in [200, 202]:
        return RedirectResponse("/", status_code=303)
    return JSONResponse({"error": "Subscription failed"}, status_code=500)


@app.get("/ping")
def ping():
    return {"status": "ok"}
