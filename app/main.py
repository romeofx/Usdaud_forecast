from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.forecast_logic import forecast_next
from pydantic import BaseModel

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

class PriceInput(BaseModel):
    close_prices: list

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(input: PriceInput):
    return forecast_next(input.close_prices)
