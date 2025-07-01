from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from app.forecast_logic import forecast_next
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="XAUUSD Forecast API",
    description="AI-powered forecast and trading signal generator for gold (XAUUSD) using XGBoost.",
    version="1.0.0"
)

##templates = Jinja2Templates(directory="app/templates")
templates = Jinja2Templates(directory="templates")


# Enable CORS for external requests (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class PriceInput(BaseModel):
    close_prices: list[float]

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    """Home page with input form."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(input: PriceInput):
    """API endpoint for price forecasting."""
    try:
        result = forecast_next(input.close_prices)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
