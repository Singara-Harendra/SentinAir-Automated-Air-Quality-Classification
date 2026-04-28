# SentinAir — High-Level Design (HLD) & Architecture Document

## 1. System Overview

SentinAir is an end-to-end MLOps application that classifies urban air quality into three categories — **Good**, **Moderate**, and **Poor** — using multivariate chemical sensor data. The system incorporates automated data pipelines, experiment tracking, model monitoring, and a self-improving feedback-driven retraining loop.

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                         │
│                                                                             │
│  [Raw CSV] ──DVC versioned──► [data/raw/data.csv]                           │
│       └──────────────────────► [data/processed/intermediate/]               │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER (Apache Airflow)                     │
│                                                                             │
│  ┌─────────────────────────────────────────────────┐                       │
│  │  air_quality_dag                                │                       │
│  │  data_ingestion ──► preprocessing ──► model_training                    │
│  └─────────────────────────────────────────────────┘                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────┐                       │
│  │  retraining_pipeline (runs hourly)              │                       │
│  │  check_feedback_volume                          │                       │
│  │  check_error_rate     ──► decide_retraining ──►│                       │
│  │  detect_data_drift                              │                       │
│  │     └──► merge_feedback ──► preprocess ──►     │                       │
│  │          train_retrain_models ──► validate      │                       │
│  └─────────────────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
                    │ artifacts/best_model.pkl
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 EXPERIMENT TRACKING LAYER (MLflow)                         │
│                                                                             │
│  MLflow Server (:5000) ◄──── Model Training Logs                           │
│       ├── Experiment: Air_Quality_Classification                            │
│       │     └── Runs: RF_train0, LR_train0, SVM_train0, RF_retrain1, ...   │
│       └── Model Registry: AirQualityModel (versioned)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INFERENCE LAYER (FastAPI :8000)                         │
│                                                                             │
│  POST /predict          ── single-row classification                        │
│  POST /predict/batch    ── bulk CSV-based classification                    │
│  POST /feedback         ── user label correction                            │
│  POST /feedback/batch   ── bulk feedback from batch predictions             │
│  GET  /feedback/stats   ── error rate + breakdown                           │
│  GET  /metrics          ── Prometheus scrape endpoint                       │
│  GET  /health           ── liveness probe                                   │
└─────────────────────────────────────────────────────────────────────────────┘
          │                         │
          ▼                         ▼
┌─────────────────┐      ┌─────────────────────────────────────────────────┐
│ FRONTEND LAYER  │      │           STORAGE LAYER                         │
│ Streamlit :8501 │      │  PostgreSQL (:5432)                             │
│                 │      │    ├── airflow metadata DB                      │
│ Tab: Single     │◄────►│    └── feedback table (predictions + actuals)   │
│ Tab: Batch      │      └─────────────────────────────────────────────────┘
│ Tab: Monitor    │
└─────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   MONITORING LAYER                                          │
│                                                                             │
│  Prometheus (:9090) ──scrapes /metrics──► FastAPI                          │
│       └──► Grafana (:3000) [NRT dashboards]                                │
│  Alertmanager (:9093) ──triggers on── sentinair_error_rate > 0.1           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Descriptions

### 3.1 Data Layer
- **Raw Data**: `data/raw/data.csv` — UCI Air Quality dataset (9358 rows × 13 sensors).
- **DVC Tracking**: `data/raw/data.csv.dvc` tracks the raw file via DVC. Processed outputs are generated by the Airflow pipeline.
- **Intermediate Outputs**: `data/processed/intermediate/` holds train/test CSVs and encoded datasets produced during pipeline runs.

### 3.2 Orchestration Layer (Apache Airflow 2.9.3)
Two DAGs are deployed:

| DAG | Schedule | Purpose |
|-----|----------|---------|
| `air_quality_dag` | Daily | Full data ingestion → preprocessing → model training |
| `retraining_pipeline` | Hourly | Drift detection → conditional retraining on feedback data |

The Airflow stack uses **LocalExecutor** backed by **PostgreSQL**, deployed via Docker Compose.

### 3.3 Experiment Tracking (MLflow)
- Tracking server runs at `:5000` with a PostgreSQL backend.
- Each model training run logs: accuracy, precision, recall, F1, confusion matrices, learning curves, and classification reports.
- Best model is registered in the **MLflow Model Registry** as `AirQualityModel`.
- Run naming convention: `ModelName_train0`, `ModelName_retrain1`, etc. for full lineage.

### 3.4 Inference Layer (FastAPI)
- Stateless REST API; model loaded from `artifacts/best_model.pkl` at startup.
- Preprocessing via `artifacts/fitted_preprocessor.pkl` (scikit-learn Pipeline with imputation + scaling).
- Prometheus metrics exported at `/metrics` (counters, histograms, gauges).
- Feedback stored to PostgreSQL `feedback` table.

### 3.5 Frontend Layer (Streamlit)
- Communicates with backend exclusively via REST API calls — strict loose coupling.
- Three tabs: Single Prediction, Batch Prediction (CSV upload), Feedback Monitor.
- No direct database access; no model loading.

### 3.6 Monitoring Layer
- **Prometheus**: Scrapes `api:8000/metrics` every 15s. Tracks prediction counts, error rate, inference latency, batch size.
- **Grafana**: Pre-provisioned dashboard (`sentinair.json`) visualises NRT metrics.
- **Alertmanager**: Fires email alerts when `sentinair_error_rate > 0.1`.

---

## 4. Design Principles

| Principle | Implementation |
|-----------|---------------|
| Loose coupling | Frontend ↔ Backend communicate only via REST API |
| Separation of concerns | Airflow orchestrates; FastAPI serves; Streamlit presents |
| Reproducibility | DVC tracks data; MLflow tracks experiments; Docker fixes environments |
| Automated retraining | Feedback loop → drift detection → conditional retraining |
| Observability | Prometheus + Grafana for NRT monitoring |

---

## 5. Technology Stack

| Component | Technology | Port |
|-----------|------------|------|
| Orchestration | Apache Airflow 2.9.3 | 8080 |
| Experiment Tracking | MLflow | 5000 |
| Inference API | FastAPI + Uvicorn | 8000 |
| Frontend | Streamlit | 8501 |
| Database | PostgreSQL 13 | 5432 |
| Monitoring | Prometheus | 9090 |
| Dashboards | Grafana | 3000 |
| Alerting | Alertmanager | 9093 |
| Containerisation | Docker + Docker Compose | — |
| Data Versioning | DVC | — |
| ML Framework | scikit-learn | — |

---

## 6. Data Flow Summary

```
Raw CSV
  └─► data_ingestion.py    → full_data.csv
  └─► preprocessing.py     → train_encoded.csv, test_encoded.csv,
                              fitted_preprocessor.pkl, label_mapping.csv
  └─► model_training.py    → best_model.pkl  (+ MLflow run artifacts)
  └─► FastAPI              → /predict endpoint
  └─► Streamlit            → user sees prediction
  └─► Feedback table       → actual labels
  └─► retraining_dag       → merge → retrain → new best_model.pkl
```