from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.pipelines.prediction_pipeline import PredictionPipeline

app = FastAPI(title="Tail Risk Modeling App")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Templates directory
templates = Jinja2Templates(directory="templates")

# Routes

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.get("/problem", response_class=HTMLResponse)
async def problem_statement(request: Request):
    return templates.TemplateResponse(
        "problem.html",
        {"request": request}
    )

@app.get("/model", response_class=HTMLResponse)
async def model_page(request: Request):
    return templates.TemplateResponse(
        "model.html",
        {"request": request}
    )

@app.get("/explorer", response_class=HTMLResponse)
async def explorer_page(request: Request):
    return templates.TemplateResponse(
        "explorer.html",
        {"request": request}
    )
