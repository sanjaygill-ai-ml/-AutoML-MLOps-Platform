from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import PlainTextResponse, HTMLResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
import joblib
import pandas as pd
import time
import logging
import os
import json
import shutil
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="AutoML Prediction API", version="2.0")

PREDICTION_COUNTER = Counter("predictions_total", "Total predictions made")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")
ERROR_COUNTER = Counter("prediction_errors_total", "Total errors")

model = None
preprocessor = None
label_encoder = None


@app.on_event("startup")
def load_artifacts():
    global model, preprocessor, label_encoder
    model = joblib.load("artifacts/best_model.pkl")
    preprocessor = joblib.load("artifacts/preprocessor.pkl")
    label_encoder = joblib.load("artifacts/label_encoder.pkl")
    logger.info("All artifacts loaded!")


class PredictRequest(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float


class PredictResponse(BaseModel):
    prediction: str
    confidence: float


@app.get("/", response_class=HTMLResponse)
def home():
    return dashboard()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start = time.time()
    try:
        df = pd.DataFrame([request.dict()])
        X = preprocessor.transform(df)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        confidence = float(max(proba))
        species = label_encoder.inverse_transform([pred])[0]
        PREDICTION_COUNTER.inc()
        PREDICTION_LATENCY.observe(time.time() - start)
        return PredictResponse(prediction=species, confidence=confidence)
    except Exception as e:
        ERROR_COUNTER.inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    import subprocess
    os.makedirs("data/raw", exist_ok=True)
    save_path = f"data/raw/dataset.csv"

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    df = pd.read_csv(save_path)

    # Retrain pipeline automatically
    try:
        subprocess.run(
            [sys.executable, "pipeline/training_pipeline.py"],
            check=True
            )
        # Reload model artifacts
        global model, preprocessor, label_encoder
        model = joblib.load("artifacts/best_model.pkl")
        preprocessor = joblib.load("artifacts/preprocessor.pkl")
        label_encoder = joblib.load("artifacts/label_encoder.pkl")
        retrained = True
        message = "File uploaded and model retrained successfully!"
    except Exception as e:
        retrained = False
        message = f"File uploaded but retraining failed: {str(e)}"

    return {
        "message": message,
        "filename": file.filename,
        "rows": len(df),
        "columns": df.columns.tolist(),
        "retrained": retrained
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    metrics = {}
    if os.path.exists("artifacts/metrics.json"):
        with open("artifacts/metrics.json") as f:
            metrics = json.load(f)

    report = metrics.get("classification_report", {})
    rows = ""
    for cls, vals in report.items():
        if isinstance(vals, dict):
            rows += f"""
            <tr>
                <td>{cls}</td>
                <td>{vals.get('precision', 0):.2f}</td>
                <td>{vals.get('recall', 0):.2f}</td>
                <td>{vals.get('f1-score', 0):.2f}</td>
                <td>{int(vals.get('support', 0))}</td>
            </tr>"""

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AutoML Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px;
                   background: #0f172a; color: #e2e8f0; }}
            h1 {{ color: #38bdf8; }}
            h2 {{ color: #94a3b8; border-bottom: 1px solid #334155;
                  padding-bottom: 8px; }}
            .card {{ background: #1e293b; border-radius: 12px;
                     padding: 24px; margin: 20px 0; }}
            .metric {{ display: inline-block; background: #0f172a;
                       border-radius: 8px; padding: 16px 32px;
                       margin: 8px; text-align: center; }}
            .metric .value {{ font-size: 2em; color: #38bdf8;
                               font-weight: bold; }}
            .metric .label {{ color: #94a3b8; font-size: 0.9em; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th {{ background: #334155; padding: 12px; text-align: left; }}
            td {{ padding: 10px 12px; border-bottom: 1px solid #1e293b; }}
            tr:hover td {{ background: #1e293b; }}
            .upload-form {{ background: #1e293b; border-radius: 12px;
                            padding: 24px; margin: 20px 0; }}
            input[type=file] {{ color: #e2e8f0; margin: 10px 0; }}
            button {{ background: #38bdf8; color: #0f172a; border: none;
                      padding: 10px 24px; border-radius: 6px;
                      cursor: pointer; font-weight: bold; }}
            button:hover {{ background: #0ea5e9; }}
            #result {{ margin-top: 12px; color: #4ade80; }}
        </style>
    </head>
    <body>
        <h1>AutoML Platform Dashboard</h1>

        <div class="card">
            <h2>Model Performance</h2>
            <div class="metric">
                <div class="value">{metrics.get('accuracy', 0):.2%}</div>
                <div class="label">Accuracy</div>
            </div>
            <div class="metric">
                <div class="value">{metrics.get('f1_score', 0):.2%}</div>
                <div class="label">F1 Score</div>
            </div>
            <div class="metric">
                <div class="value">{metrics.get('model_type', 'N/A')}</div>
                <div class="label">Best Model</div>
            </div>
        </div>

        <div class="card">
            <h2>Classification Report</h2>
            <table>
                <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>Support</th>
                </tr>
                {rows}
            </table>
        </div>

        <div class="upload-form">
            <h2>Upload New Dataset</h2>
            <input type="file" id="fileInput" accept=".csv"/>
            <br/>
            <button onclick="uploadFile()">Upload Dataset</button>
            <div id="result"></div>
        </div>

        <script>
        async function uploadFile() {{
            const file = document.getElementById('fileInput').files[0];
            if (!file) {{ alert('Please select a file!'); return; }}

            document.getElementById('result').innerHTML = 'Uploading and retraining... please wait...';
            document.getElementById('result').style.color = '#facc15';

            const form = new FormData();
            form.append('file', file);

            const res = await fetch('/upload', {{method:'POST', body:form}});
            const data = await res.json();

            if (data.retrained) {{
                document.getElementById('result').innerHTML =
                    'Done! Rows: ' + data.rows +
                    ' | Model retrained! Refreshing...';
                document.getElementById('result').style.color = '#4ade80';
                setTimeout(() => location.reload(), 2000);
            }} else {{
                document.getElementById('result').innerHTML = data.message;
                document.getElementById('result').style.color = '#f87171';
            }}
        }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest()