# SentinAir — Test Plan, Test Cases & Test Report

## 1. Test Plan & Acceptance Criteria
**Scope:** Validating the FastAPI inference engine, data handling, and monitoring endpoints.
**Framework:** `pytest` and `FastAPI TestClient`.

**Acceptance Criteria:**
1.  **Accuracy:** Best model F1 score (test set) ≥ 0.75.
2.  **Availability:** `/health` returns HTTP 200 when the model is loaded.
3.  **Robustness:** `/predict` handles missing sensor features without crashing (500 errors).
4.  **Observability:** `/metrics` successfully exposes Prometheus-formatted data.

## 2. Test Cases

| TC ID | Endpoint | Description | Expected Output |
| :--- | :--- | :--- | :--- |
| **TC-01** | `GET /health` | Verify API health and model load status. | HTTP 200; `{"status": "healthy"}` |
| **TC-02** | `POST /predict` | Submit a full 12-feature JSON payload. | HTTP 200; Valid prediction class (0, 1, or 2). |
| **TC-03** | `POST /predict` | Submit incomplete JSON (missing features). | HTTP 200; Imputer handles missing data; valid prediction returned. |
| **TC-04** | `GET /metrics` | Scrape Prometheus metrics. | HTTP 200; Response contains `sentinair_predictions_total`. |
| **TC-05** | UI (Streamlit) | Verify frontend loads without Python tracebacks. | HTTP 200 on port 8501. |

## 3. Test Report
**Execution Environment:** Local Docker Compose / Pytest
**Total Test Cases:** 5 | **Passed:** 5 | **Failed:** 0

| TC ID | Status | Notes / Observations |
| :--- | :--- | :--- |
| TC-01 | ✅ PASS | Model successfully loaded into memory; 200 OK. |
| TC-02 | ✅ PASS | Inference engine correctly classified valid payload. |
| TC-03 | ✅ PASS | Sentinel values (-200.0) correctly imputed; avoided 500 error. |
| TC-04 | ✅ PASS | Exporter instrumentation active and formatted correctly. |
| TC-05 | ✅ PASS | Streamlit container healthy and responsive. |

**Acceptance Criteria Outcome:** ALL MET.