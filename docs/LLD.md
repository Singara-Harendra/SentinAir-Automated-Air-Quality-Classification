# SentinAir — Low-Level Design (LLD) Document

## 1. API Endpoint Definitions

Base URL: `http://localhost:8000`

---

### 1.1 `GET /`

**Purpose**: Root info — confirms the service is alive and reports model load status.

**Request**: No parameters.

**Response** `200 OK`:
```json
{
  "service": "SentinAir API",
  "version": "2.0.0",
  "model_loaded": true
}
```

---

### 1.2 `GET /health`

**Purpose**: Liveness probe. Returns 503 if model has not been loaded.

**Request**: No parameters.

**Response** `200 OK`:
```json
{
  "status": "healthy",
  "model_path": "/app/artifacts/best_model.pkl"
}
```

**Response** `503 Service Unavailable`:
```json
{
  "detail": "Model not loaded. Expected at: /app/artifacts/best_model.pkl. Run the air_quality_dag in Airflow first."
}
```

---

### 1.3 `GET /metrics`

**Purpose**: Prometheus scrape endpoint. Returns plain-text metrics in OpenMetrics format.

**Request**: No parameters.

**Response** `200 OK` (Content-Type: `text/plain`):
```
# HELP sentinair_predictions_total Total predictions made
# TYPE sentinair_predictions_total counter
sentinair_predictions_total{class_label="Good"} 42.0
sentinair_predictions_total{class_label="Moderate"} 18.0
sentinair_predictions_total{class_label="Poor"} 5.0
# HELP sentinair_feedback_total Total feedback submissions
...
```

**Metrics exported**:

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|--------|
| `sentinair_predictions_total` | Counter | Total predictions | `class_label` |
| `sentinair_feedback_total` | Counter | Total feedback rows | — |
| `sentinair_error_rate` | Gauge | Model error rate from feedback | — |
| `sentinair_inference_latency_seconds` | Histogram | Per-request inference time | — |
| `sentinair_batch_size` | Histogram | Rows per batch request | — |

---

### 1.4 `POST /predict`

**Purpose**: Single-row air quality classification.

**Request Body** (`application/json`):
```json
{
  "features": {
    "PT08.S1(CO)": 1200.0,
    "NMHC(GT)": -200.0,
    "C6H6(GT)": 5.0,
    "PT08.S2(NMHC)": 900.0,
    "NOx(GT)": 100.0,
    "PT08.S3(NOx)": 1000.0,
    "NO2(GT)": 80.0,
    "PT08.S4(NO2)": 1500.0,
    "PT08.S5(O3)": 1000.0,
    "T": 15.0,
    "RH": 50.0,
    "AH": 0.75
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `features` | `Dict[str, float]` | Yes | Key-value map of sensor readings. Missing columns are filled with `-200.0`. |

**Response** `200 OK`:
```json
{
  "prediction": 0,
  "label": "Good",
  "probability": [0.85, 0.10, 0.05]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | `int` | Class index: 0=Good, 1=Moderate, 2=Poor |
| `label` | `str` | Human-readable label |
| `probability` | `List[float]` | Class probabilities (sums to 1.0) |

**Response** `503`: Model not loaded.  
**Response** `500`: Internal inference error.

---

### 1.5 `POST /predict/batch`

**Purpose**: Bulk classification of multiple rows.

**Request Body** (`application/json`):
```json
{
  "rows": [
    {
      "PT08.S1(CO)": 1200.0,
      "C6H6(GT)": 5.0,
      "PT08.S2(NMHC)": 900.0,
      "NOx(GT)": 100.0,
      "PT08.S3(NOx)": 1000.0,
      "NO2(GT)": 80.0,
      "PT08.S4(NO2)": 1500.0,
      "PT08.S5(O3)": 1000.0,
      "T": 15.0,
      "RH": 50.0,
      "AH": 0.75
    }
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `rows` | `List[Dict[str, float]]` | Yes | List of sensor-reading dicts. |

**Response** `200 OK`:
```json
{
  "results": [
    {
      "prediction": 0,
      "label": "Good",
      "probability": [0.85, 0.10, 0.05]
    },
    {
      "prediction": null,
      "label": "error",
      "error": "Preprocessing failed: ..."
    }
  ],
  "total": 2
}
```

---

### 1.6 `POST /feedback`

**Purpose**: Submit a single corrected label for a prediction.

**Request Body**:
```json
{
  "prediction": 0,
  "actual": 1,
  "features": {
    "PT08.S1(CO)": 1200.0,
    "T": 15.0,
    "RH": 50.0,
    "AH": 0.75
  },
  "source": "single"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prediction` | `int` | Yes | Model's predicted class (0/1/2) |
| `actual` | `int` | Yes | Correct class label |
| `features` | `Dict[str, float]` | Yes | Feature values used for prediction |
| `source` | `str` | No | `"single"` or `"batch"`. Default: `"single"` |

**Response** `200 OK`:
```json
{
  "message": "Feedback stored",
  "id": 42
}
```

---

### 1.7 `POST /feedback/batch`

**Purpose**: Submit feedback for an entire batch at once.

**Request Body**:
```json
{
  "items": [
    {
      "prediction": 0,
      "actual": 1,
      "features": {"PT08.S1(CO)": 1200.0, "T": 15.0},
      "source": "batch"
    }
  ]
}
```

**Response** `200 OK`:
```json
{
  "message": "Stored 5 feedback records"
}
```

---

### 1.8 `GET /feedback/stats`

**Purpose**: Returns feedback collection stats for monitoring and retraining decisions.

**Response** `200 OK`:
```json
{
  "total_feedback_rows": 120,
  "error_rate": 0.1250,
  "correct": 105,
  "class_breakdown": {
    "Good": 60,
    "Moderate": 40,
    "Poor": 20
  },
  "source_breakdown": {
    "single": 30,
    "batch": 90
  },
  "recent_10": [
    {
      "prediction": 0,
      "actual": 1,
      "source": "single",
      "timestamp": "2026-04-27 10:00:00"
    }
  ]
}
```

---

## 2. Database Schema

### Table: `feedback`

```sql
CREATE TABLE feedback (
    id          SERIAL PRIMARY KEY,
    prediction  INTEGER NOT NULL,
    actual      INTEGER NOT NULL,
    features    TEXT,           -- JSON string of feature dict
    source      VARCHAR(10) DEFAULT 'single',  -- 'single' | 'batch'
    timestamp   TIMESTAMP DEFAULT NOW()
);
```

---

## 3. Preprocessing Pipeline (Internal)

The preprocessor (`fitted_preprocessor.pkl`) is a scikit-learn `Pipeline`:

```
Input: Dict[str, float]  (11 sensor features)
  └─► pd.DataFrame (column ordering enforced)
  └─► Imputer (median strategy)  ── handles -200 sentinel values
  └─► StandardScaler             ── zero-mean, unit-variance scaling
Output: numpy array (shape: [1, 11])
```

Feature column order enforced internally (12 features, NMHC(GT) filled with -200 if absent):
```
PT08.S1(CO), NMHC(GT), C6H6(GT), PT08.S2(NMHC),
NOx(GT), PT08.S3(NOx), NO2(GT), PT08.S4(NO2),
PT08.S5(O3), T, RH, AH
```

---

## 4. Module Descriptions

### `scripts/data_ingestion.py`
- `ingest_data(input_path, output_path)` — loads raw CSV, validates schema, writes to intermediate directory.

### `scripts/preprocessing.py`
- `split_train_test(input_path, train_path, test_path)` — 80/20 chronological split.
- `feature_selection(input_path, output_path, features_to_drop)` — drops specified columns.
- `label_encoding(input_path, output_path, label_mapping_path)` — converts AQI buckets to 0/1/2.
- `create_preprocessor(input_path, output_path)` — fits and saves the sklearn Pipeline.

### `scripts/model_training.py`
- `train_and_log_models(train_path, test_path, experiment_name, round_num)` — trains RF, LR, SVM; logs all to MLflow; saves `best_model.pkl`.

### `scripts/retraining.py`
- Drift detection, feedback merging, re-running `train_and_log_models` with incremented `round_num`.

### `src/api/main.py`
- FastAPI application. All endpoints described in Section 1.

### `src/frontend/app.py`
- Streamlit application. Communicates with `src/api/main.py` via HTTP only.

---

## 5. Error Codes Reference

| HTTP Code | Meaning |
|-----------|---------|
| 200 | Success |
| 422 | Validation error (malformed request body) |
| 500 | Internal server error (inference or DB failure) |
| 503 | Model not loaded (run Airflow DAG first) |