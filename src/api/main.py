"""
FastAPI application for SentinAir — Air Quality Classification

Endpoints:
  GET  /            — root info
  GET  /health      — liveness check
  GET  /metrics     — Prometheus metrics
  POST /predict     — single-row inference
  POST /predict/batch — bulk CSV-style inference
  POST /feedback    — label submission (stores to DB)
  GET  /feedback/stats — feedback count + error rate (for verification)
"""

import os
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import uvicorn
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="SentinAir — Air Quality API", version="2.0.0")

# ── Prometheus metrics ─────────────────────────────────────────────────────────
PREDICTION_COUNTER  = Counter('sentinair_predictions_total', 'Total predictions made', ['class_label'])
FEEDBACK_COUNTER    = Counter('sentinair_feedback_total', 'Total feedback submissions')
ERROR_RATE_GAUGE    = Gauge('sentinair_error_rate', 'Current model error rate from feedback')
INFERENCE_LATENCY   = Histogram('sentinair_inference_latency_seconds', 'Inference latency in seconds')
BATCH_SIZE_HIST     = Histogram('sentinair_batch_size', 'Rows per batch prediction request', buckets=[1,5,10,25,50,100,250,500])

# ── Database ───────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://airflow:airflow@postgres/airflow")
engine       = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()

LABEL_MAP  = {0: "Good", 1: "Moderate", 2: "Poor"}

class Feedback(Base):
    __tablename__ = "feedback"
    id         = Column(Integer, primary_key=True, index=True)
    prediction = Column(Integer)
    actual     = Column(Integer)
    features   = Column(String)   # JSON string
    source     = Column(String, default="single")   # "single" | "batch"
    timestamp  = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ── Model loading ──────────────────────────────────────────────────────────────
MODEL_PATH       = os.getenv("MODEL_PATH",       "/app/artifacts/best_model.pkl")
PREPROCESSOR_PATH= os.getenv("PREPROCESSOR_PATH","/app/artifacts/fitted_preprocessor.pkl")

def _load_artifact(path: str, label: str):
    p = Path(path)
    if not p.exists():
        logger.error(f"{label} file not found at {path}")
        return None
    try:
        obj = joblib.load(path)
        logger.info(f"{label} loaded from {path}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load {label}: {e}")
        return None

model        = _load_artifact(MODEL_PATH,        "Model")
preprocessor = _load_artifact(PREPROCESSOR_PATH, "Preprocessor")

# ── Pydantic schemas ───────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: List[float]

class BatchPredictionRequest(BaseModel):
    rows: List[Dict[str, float]]

class FeedbackRequest(BaseModel):
    prediction: int
    actual: int
    features: Dict[str, float]
    source: str = "single"      # "single" or "batch"

class BatchFeedbackRequest(BaseModel):
    """Submit feedback for an entire batch at once."""
    items: List[FeedbackRequest]

# ── Helper ─────────────────────────────────────────────────────────────────────
def _run_inference(features: Dict[str, float]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. Map UI display names to your EXACT dataset column names
    mapping = {
        "Temperature (T)": "T",
        "Relative Humidity (RH)": "RH",
        "Absolute Humidity (AH)": "AH"
    }
    
    # Create cleaned dict with the correct headers
    cleaned_features = {mapping.get(k, k): v for k, v in features.items()}

    # 2. THE MASTER LIST (Exactly matching your test dataset, NO CO(GT)!)
    expected_features = [
        "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)", 
        "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", 
        "PT08.S5(O3)", "T", "RH", "AH"
    ]

    df = pd.DataFrame([cleaned_features])

    # 3. Handle missing columns (like NMHC(GT))
    for col in expected_features:
        if col not in df.columns:
            df[col] = -200.0  # Default for the air quality dataset
            
    # 4. Force exact order AND drop any rogue columns (like CO(GT) if it sneaks in)
    df = df[expected_features]

    # 5. Transform and Predict
    if preprocessor:
        X = preprocessor.transform(df)
    else:
        X = df.values

    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0].tolist()
    return pred, proba

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "SentinAir API", "version": "2.0.0", "model_loaded": model is not None}

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. "
                f"Expected at: {MODEL_PATH}. "
                "Run the air_quality_dag in Airflow first."
            )
        )
    return {"status": "healthy", "model_path": MODEL_PATH}

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus scrape endpoint."""
    # Refresh error-rate gauge from DB
    try:
        db = SessionLocal()
        result = db.execute(text("SELECT prediction, actual FROM feedback WHERE actual IS NOT NULL")).fetchall()
        db.close()
        if result:
            total   = len(result)
            correct = sum(1 for r in result if r[0] == r[1])
            ERROR_RATE_GAUGE.set(1.0 - correct / total)
    except Exception as e:
        logger.warning(f"Could not refresh error rate gauge: {e}")
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Single-row inference."""
    import time
    start = time.time()
    try:
        pred, proba = _run_inference(request.features)
        PREDICTION_COUNTER.labels(class_label=LABEL_MAP.get(pred, str(pred))).inc()
        INFERENCE_LATENCY.observe(time.time() - start)
        return PredictionResponse(
            prediction=pred,
            label=LABEL_MAP.get(pred, str(pred)),
            probability=proba,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    """Bulk inference — returns prediction for each row."""
    import time
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    BATCH_SIZE_HIST.observe(len(request.rows))
    results = []
    for features in request.rows:
        try:
            start = time.time()
            pred, proba = _run_inference(features)
            INFERENCE_LATENCY.observe(time.time() - start)
            PREDICTION_COUNTER.labels(class_label=LABEL_MAP.get(pred, str(pred))).inc()
            results.append({"prediction": pred, "label": LABEL_MAP.get(pred, str(pred)), "probability": proba})
        except Exception as e:
            results.append({"prediction": None, "label": "error", "error": str(e)})
    return {"results": results, "total": len(results)}

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    """Submit a single feedback record."""
    db = SessionLocal()
    try:
        fb = Feedback(
            prediction=request.prediction,
            actual=request.actual,
            features=json.dumps(request.features),
            source=request.source,
        )
        db.add(fb)
        db.commit()
        db.refresh(fb)
        FEEDBACK_COUNTER.inc()
        return {"message": "Feedback stored", "id": fb.id}
    except Exception as e:
        db.rollback()
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail="Feedback submission failed")
    finally:
        db.close()

@app.post("/feedback/batch")
def submit_feedback_batch(request: BatchFeedbackRequest):
    """Submit feedback for a whole batch at once."""
    db = SessionLocal()
    stored = 0
    try:
        for item in request.items:
            fb = Feedback(
                prediction=item.prediction,
                actual=item.actual,
                features=json.dumps(item.features),
                source="batch",
            )
            db.add(fb)
            stored += 1
        db.commit()
        FEEDBACK_COUNTER.inc(stored)
        return {"message": f"Stored {stored} feedback records"}
    except Exception as e:
        db.rollback()
        logger.error(f"Batch feedback failed: {e}")
        raise HTTPException(status_code=500, detail="Batch feedback failed")
    finally:
        db.close()

@app.get("/feedback/stats")
def feedback_stats():
    """
    Returns feedback count, error rate, and per-class breakdown.
    Use this to verify feedback is being collected.
    """
    db = SessionLocal()
    try:
        rows = db.execute(text(
            "SELECT prediction, actual, source, timestamp FROM feedback ORDER BY timestamp DESC LIMIT 500"
        )).fetchall()

        total   = len(rows)
        correct = sum(1 for r in rows if r[0] == r[1])
        error_r = round(1.0 - correct / total, 4) if total else 0.0

        class_breakdown = {}
        for r in rows:
            lbl = LABEL_MAP.get(r[1], str(r[1]))
            class_breakdown[lbl] = class_breakdown.get(lbl, 0) + 1

        source_breakdown = {}
        for r in rows:
            src = r[2] or "unknown"
            source_breakdown[src] = source_breakdown.get(src, 0) + 1

        recent = [
            {"prediction": r[0], "actual": r[1], "source": r[2],
             "timestamp": str(r[3])}
            for r in rows[:10]
        ]

        return {
            "total_feedback_rows": total,
            "error_rate": error_r,
            "correct": correct,
            "class_breakdown": class_breakdown,
            "source_breakdown": source_breakdown,
            "recent_10": recent,
        }
    finally:
        db.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)