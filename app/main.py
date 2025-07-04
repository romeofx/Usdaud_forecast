from fastapi import FastAPI, Request, Query, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from app.forecast_logic import forecast_next

app = FastAPI(
    title="XAUUSD Forecast API",
    description="Forecast & trading signal generator using XGBoost.",
    version="1.0.0"
)

# Setup CORS (for frontend POSTs if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

templates = Jinja2Templates(directory="app/templates")

# === Routes ===

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(close_prices: List[float] = Form(...), threshold: Optional[float] = Form(0.002)):
    try:
        result = forecast_next(close_prices, threshold)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/subscribe")
async def subscribe(email: str = Form(...)):
    import requests
    response = requests.post("https://formspree.io/f/mpwrnoqv", data={"email": email})
    if response.status_code in [200, 202]:
        return RedirectResponse("/", status_code=303)
    return JSONResponse({"error": "Subscription failed"}, status_code=500)

@app.get("/ping")
def ping():
    return {"status": "ok"}
