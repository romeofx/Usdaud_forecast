from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import requests

from app.forecast_logic import forecast_next

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

SEQUENCE_LENGTH = 600

def load_last_600_prices() -> str:
    try:
        df = pd.read_csv("xauusd_data.csv")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        return ",".join([str(int(round(val))) for val in df["close"].tail(SEQUENCE_LENGTH)])
    except Exception:
        return ""

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    last_600_prices = load_last_600_prices()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "last_600_prices": last_600_prices
    })

@app.post("/predict")
async def predict_api(request: Request, threshold: Optional[float] = 0.002):
    data = await request.json()
    close_prices = data.get("close_prices", [])
    try:
        result = forecast_next(close_prices, threshold)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.get("/download-prices")
def download_prices():
    prices = load_last_600_prices()
    return HTMLResponse(
        content=prices,
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=last_600_prices.txt"}
    )

@app.post("/subscribe")
def subscribe(name: str = Form(...), email: str = Form(...), message: str = Form("")):
    form_data = {"name": name, "email": email, "message": message}
    res = requests.post("https://formspree.io/f/mpwrnoqv", data=form_data)
    if res.status_code in [200, 202]:
        return RedirectResponse("/", status_code=303)
    return JSONResponse({"error": "Subscription failed"}, status_code=500)