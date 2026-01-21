from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import random

from src.pipelines.prediction_pipeline import PredictionPipeline
from src.utils.model_utils import get_test_df, generate_tail_plot

app = FastAPI(title="Tail Risk Modeling App")

# -------------------------------
# Static & Templates
# -------------------------------

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

templates = Jinja2Templates(directory="templates")

# -------------------------------
# Initialize pipeline (ONCE)
# -------------------------------

prediction_pipeline = PredictionPipeline()

# -------------------------------
# Page Routes
# -------------------------------

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

# -------------------------------
# Explorer API
# -------------------------------

X_test, y_test = get_test_df()

@app.get("/explore/sample")
async def sample_claim():
    # Randomly sample one test observation
    idx = random.randint(0, len(X_test) - 1)

    X_row = X_test.iloc[[idx]]
    y_true = float(y_test.iloc[idx])

    # Predict quantile
    y_pred = float(prediction_pipeline.predict(X_row)[0])

    # Quantile level
    tau = 0.90

    # Pinball loss (single observation)
    if y_true >= y_pred:
        pinball_loss = tau * (y_true - y_pred)
    else:
        pinball_loss = (1 - tau) * (y_pred - y_true)

    # Generate matplotlib distribution plot (base64)
    plot_base64 = generate_tail_plot(
        y_test=y_test,
        actual=y_true,
        predicted=y_pred
    )

    return JSONResponse({
        "predicted_loss": y_pred,
        "actual_loss": y_true,
        "pinball_loss": float(pinball_loss),
        "tail_plot": plot_base64
    })
