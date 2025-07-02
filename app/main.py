from fastapi import FastAPI, Request, Query
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.forecast_logic import forecast_next
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(
    title="XAUUSD Forecast API",
    description="AI-powered forecast and trading signal generator for gold (XAUUSD) using XGBoost.",
    version="1.0.0"
)

# HTML template directory
templates = Jinja2Templates(directory="templates")

# CORS middleware setup (Allow all origins for dev; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["https://yourdomain.com"] in production
    allow_methods=["*"],
    allow_headers=["*"]
)

# Input schema for closing prices
class PriceInput(BaseModel):
    close_prices: List[float]

# Root route for rendering frontend UI
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Health check route
@app.get("/ping")
def ping():
    return {"status": "ok"}

# Prediction route with optional threshold
@app.post("/predict")
def predict(input: PriceInput, threshold: Optional[float] = Query(0.002, ge=0, le=1)):
    """
    Forecast next gold price and trading signal.
    Accepts last 60 close prices and an optional threshold for signal sensitivity.
    """
    try:
        result = forecast_next(input.close_prices, threshold)
        return JSONResponse(content=result)
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=422)
    except Exception as e:
        return JSONResponse(content={"error": "Server error", "details": str(e)}, status_code=500)
