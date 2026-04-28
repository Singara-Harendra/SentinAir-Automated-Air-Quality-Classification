# SentinAir — User Manual
### Automated Air Quality Classification System

---

## What Does This Application Do?

SentinAir classifies urban air quality into three levels using data from chemical sensor arrays:

| Label | Meaning | What to do |
|---|---|---|
| 🟢 **Good** | Air quality is healthy | No action needed |
| 🟡 **Moderate** | Air quality has mild pollutants | Sensitive groups should limit outdoor activity |
| 🔴 **Poor** | Air quality is hazardous | Avoid outdoor exposure |

---

## How to Access the Application

Open your web browser and go to:

| Service | URL | Who uses it |
|---|---|---|
| **Main App (Streamlit)** | `http://localhost:8501` | End users — you |
| **API Documentation** | `http://localhost:8000/docs` | Developers |
| **Pipeline Monitor (Airflow)** | `http://localhost:8080` | Admin (admin/admin) |
| **Experiment Tracker (MLflow)** | `http://localhost:5000` | Data scientists |
| **Metrics Dashboard (Grafana)** | `http://localhost:3000` | Admin (admin/admin) |

---

## Part 1 — Making a Single Prediction

**Step 1.** Open `http://localhost:8501` in your browser.

**Step 2.** Check the sidebar on the left. It should show **✅ API Online**. If it shows ❌, wait 30 seconds and refresh.

**Step 3.** You are on the **📡 Single Prediction** tab. Enter the sensor readings:

| Field | What it measures | Typical range |
|---|---|---|
| PT08.S1(CO) | Tin oxide CO sensor response | 700 – 2000 |
| C6H6(GT) | Benzene concentration (µg/m³) | 0.1 – 30 |
| PT08.S2(NMHC) | Titania NMHC sensor response | 500 – 1800 |
| NOx(GT) | NOx concentration (ppb) | 1 – 1000 |
| PT08.S3(NOx) | Tungsten oxide NOx sensor | 500 – 2000 |
| NO2(GT) | Nitrogen dioxide (µg/m³) | 1 – 350 |
| PT08.S4(NO2) | Tungsten oxide NO2 sensor | 500 – 2500 |
| PT08.S5(O3) | Indium oxide O3 sensor | 500 – 2000 |
| Temperature (T) | Ambient temperature (°C) | -5 – 45 |
| Relative Humidity (RH) | Humidity (%) | 10 – 100 |
| Absolute Humidity (AH) | Water vapour (g/m³) | 0.1 – 2.5 |

> **Tip:** If a sensor is faulty or missing, the system will handle it automatically using median imputation.

**Step 4.** Click the **🔍 Predict Air Quality** button.

**Step 5.** The result appears with a colour bar chart showing the probability for each class.

---

## Part 2 — Providing Feedback (Helps Improve the Model)

After making a prediction, a feedback section appears below the result.

**Step 1.** Select the **actual air quality** you observed at that location.

**Step 2.** Click **Submit Feedback**.

**Step 3.** You will see ✅ Feedback stored! This data is saved and used to automatically retrain the model when enough corrections accumulate.

---

## Part 3 — Batch Prediction (Upload a CSV File)

Use this when you have many sensor readings to classify at once.

**Step 1.** Click the **📂 Batch Prediction** tab.

**Step 2.** Prepare a CSV file with these column headers (exactly as shown):

```
PT08.S1(CO),C6H6(GT),PT08.S2(NMHC),NOx(GT),PT08.S3(NOx),NO2(GT),PT08.S4(NO2),PT08.S5(O3),T,RH,AH
```

Optionally add a `Target` column (Good / Moderate / Poor) if you know the true labels — this enables automatic feedback.

**Step 3.** Click **Choose a CSV file** and select your file.

**Step 4.** Tick **Auto-submit feedback after prediction** if your CSV has a `Target` column.

**Step 5.** Click **▶️ Run Batch Prediction**. A progress bar shows the status.

**Step 6.** Download the results by clicking **⬇️ Download Results CSV**. The file includes two extra columns: `Predicted_Class` (0/1/2) and `Predicted_Label` (Good/Moderate/Poor).

---

## Part 4 — Monitoring Feedback Collection

**Step 1.** Click the **📊 Feedback Monitor** tab.

**Step 2.** Click **🔃 Refresh Stats**.

**Step 3.** You will see:
- Total feedback rows collected
- Current model error rate
- Whether retraining will be triggered (shown if error rate > 5%)
- Breakdown by feedback source (single vs batch)
- The last 10 feedback records

> If error rate exceeds 5%, the system will automatically retrain the model the next time the retraining pipeline runs (every hour).

---

## Part 5 — Checking the System Health Sidebar

The sidebar always shows the live API status:

- **✅ API Online** — System is working normally
- **⚠️ API returned 503** — Model not loaded yet. Trigger the training DAG in Airflow.
- **❌ Cannot reach API** — The API container is not running. Run `docker compose up -d`.

Click **🔄 Check Feedback Stats** in the sidebar for a quick summary without switching tabs.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Sidebar shows ❌ Cannot reach API | Run `docker compose up -d` in your terminal |
| Sidebar shows ⚠️ 503 Model not loaded | Go to Airflow UI → Trigger `air_quality_dag` → Wait for all 3 tasks to turn green |
| Batch prediction says "CSV missing columns" | Check that your CSV header exactly matches the 11 column names listed above |
| Feedback tab shows 0 rows | You haven't submitted any feedback yet — make a prediction and submit |
| Grafana shows no data | Make a few predictions first to generate metrics, then wait 15 seconds |

---

*SentinAir — DA5402 MLOps Project*