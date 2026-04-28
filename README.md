
# SentinAir: Automated Air Quality Classification MLOps Pipeline

A complete MLOps pipeline for air quality classification using Apache Airflow, MLflow, FastAPI, and monitoring.

## 🏗️ Architecture Overview

```
Raw Data (DVC) → Airflow Pipeline → Multiple Models → MLflow Registry → FastAPI
                     ↓                                                       ↓
             Feedback Loop ← Streamlit ← PostgreSQL ← Prometheus → Grafana & cAdvisor
                               ↓
                       Retraining DAG
```

**Key Components:**
- **Data Versioning**: Raw and processed datasets tracked with DVC
- **Orchestration**: Airflow DAGs for pipeline automation
- **Model Training**: Multiple ML models (RF, XGBoost, LR, SVM) with MLflow tracking
- **Model Registry**: Best model registered for deployment
- **API Service**: FastAPI with prediction and feedback endpoints
- **Frontend**: Streamlit app with feedback collection
- **Database**: PostgreSQL for feedback storage and MLflow backend
- **Monitoring**: Prometheus metrics, Grafana dashboards, Alertmanager notifications, and cAdvisor resource tracking
- **Retraining**: Automated pipeline combining raw + feedback data

## 📁 Project Structure

```
├── data/
│   ├── raw/           # Raw datasets (versioned with DVC)
│   ├── processed/     # Processed datasets (auto-tracked with DVC)
│   └── intermediate/  # Intermediate processing files
├── artifacts/         # Model artifacts, preprocessors, mappings
├── dags/             # Apache Airflow DAGs
├── scripts/          # Pipeline processing scripts
├── src/
│   ├── api/          # FastAPI application with feedback endpoint
│   ├── frontend/     # Streamlit web interface with feedback
│   └── monitoring/   # Prometheus, Grafana, Alertmanager, cAdvisor configs
├── logs/             # Pipeline execution logs
├── mlruns/           # MLflow experiment tracking
├── requirements.txt  # Python dependencies
├── docker-compose.yml # Multi-service orchestration
├── Dockerfile.api    # API service container
├── Dockerfile.frontend # Frontend service container
└── README.md         # This file
```

## 🚀 Setup and Run Instructions

### Prerequisites
- Docker and Docker Compose installed
- Git and Git LFS for data versioning
- Python 3.9+ (for local development)

### 1. Clone and Initialize Repository
```bash
git clone <your-repo-url>
cd SentinAir-Automated-Air-Quality-Classification

# Initialize Git LFS for large files
git lfs install
git lfs track "*.csv"

# Initialize DVC and fix cache link types for Docker volume compatibility
dvc init --no-scm
dvc config cache.type copy
dvc remote add -d myremote /path/to/dvc/storage  # Or cloud storage
```

### 2. Data Versioning with DVC
```bash
# Track raw dataset
dvc add data/raw/data.csv
git add data/raw/data.csv.dvc
git commit -m "Track raw dataset with DVC"

# For processed datasets, they are automatically tracked during pipeline runs
```

### 3. Environment Configuration
Create `.env` file for sensitive configurations:
```env
DATABASE_URL=postgresql://airflow:airflow@postgres/airflow
MLFLOW_TRACKING_URI=http://localhost:5000
ALERT_SMTP_SERVER=smtp.gmail.com
ALERT_SMTP_PORT=587
ALERT_SMTP_USERNAME=your-email@gmail.com
ALERT_SMTP_PASSWORD=your-app-password
ALERT_EMAIL_FROM=your-email@gmail.com
ALERT_EMAIL_TO=recipient@example.com
```

### 4. Build and Start Services
```bash
# Build all services
docker compose build

# Start all services
docker compose up -d

# Initialize Airflow database
docker compose exec airflow-webserver airflow db init

# Create Airflow admin user
docker compose exec airflow-webserver airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

### 5. Access Services
- **Airflow UI**: http://localhost:8080 (admin/admin)
- **MLflow UI**: http://localhost:5000
- **FastAPI**: http://localhost:8000/docs (API documentation)
- **Streamlit App**: http://localhost:8501
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Alertmanager**: http://localhost:9093 (View active alerts)
- **cAdvisor**: http://localhost:8081 (Container resource monitoring)

### 6. Run the Pipeline
```bash
# Trigger DAG manually in Airflow UI or via CLI
docker compose exec airflow-webserver airflow dags unpause air_quality_pipeline
docker compose exec airflow-webserver airflow dags trigger air_quality_pipeline

# Or run individual scripts locally (after pip install -r requirements.txt)
python scripts/data_ingestion.py
python scripts/preprocessing.py
python scripts/model_training.py
```

### 7. Model Deployment and Feedback
- Models are automatically registered in MLflow Model Registry
- Best model is saved as `artifacts/best_model.pkl`
- Use the Streamlit app to make predictions and provide feedback
- Feedback is stored in PostgreSQL and used for retraining

### 8. Retraining Pipeline
- Runs hourly via `retraining_dag.py`
- Detects drift using stored baseline feature statistics and feedback distribution
- Uses deduplicated feedback so the same feedback row is not included twice
- Retrains only when drift or error rate triggers are met
- Logs runs as `modelname_train0` for first training and `modelname_retrain1`, `modelname_retrain2`, etc.
- Registers the best model version in MLflow and saves model artifacts, loss curves, and confusion matrices

### 9. Monitoring and Alerting
- Prometheus scrapes metrics from API
- cAdvisor tracks Docker container resources
- Grafana dashboards for visualization
- Alertmanager sends emails (if configured) or displays alerts for high error rates

### 10. Scaling and Production
- Use Kubernetes for production deployment
- Configure DVC remote for cloud storage (S3, GCS, etc.)
- Set up CI/CD pipelines for automated testing and deployment

## 📊 Pipeline Tasks

### Task 1-9: Data Processing Pipeline
- **Data Ingestion**: Load and validate raw dataset
- **Train-Test Split**: Time-series aware splitting (80/20)
- **Feature Selection**: Remove high-missing features (NMHC(GT))
- **Label Encoding**: Convert classes to numeric (Good→0, Moderate→1, Poor→2)
- **Preprocessing Pipeline**: Create imputation + scaling pipeline
- **Fit Preprocessor**: Learn parameters on training data
- **Transform Data**: Apply preprocessing to training set
- **Feature Engineering**: Optional temporal features
- **Artifact Persistence**: Save all preprocessing artifacts

### Task 10: Model Training
- Train multiple models: RandomForest, XGBoost, LogisticRegression
- Hyperparameter tuning with cross-validation
- MLflow experiment tracking
- Model registry integration

### Task 11: Model Deployment
- FastAPI service with preprocessing pipeline
- Input validation and error handling
- REST API endpoints for predictions

### Task 12: Drift Detection
- Statistical drift detection (KS test, PSI)
- Automated retraining triggers
- MLflow logging of drift metrics

### Task 13-14: Frontend & Monitoring
- Streamlit web interface
- Prometheus metrics collection
- Grafana dashboards for monitoring

## 🔧 Configuration

### Environment Variables

```bash
# Airflow
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=air_quality_classification

# API
MODEL_PATH=artifacts/best_model.pkl
PREPROCESSOR_PATH=artifacts/fitted_preprocessor.pkl
API_PORT=8000
```

### Pipeline Parameters

Edit `dags/air_quality_pipeline.py` to modify:
- Test size (default: 0.2)
- Random state (default: 42)
- Drift threshold (default: 0.1)
- Model hyperparameters
- Feature selection criteria

## 📈 Monitoring & Metrics

### Key Metrics Tracked
- Model prediction latency
- Request success/failure rates
- Data drift scores
- Model performance over time
- System resource usage

### Accessing Dashboards
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **MLflow**: http://localhost:5000

## 🧪 Testing the Pipeline

### Unit Tests
```bash
# Run preprocessing tests locally
python -m pytest tests/test_preprocessing.py

# Run API tests directly inside the container (Recommended)
docker exec -it air_quality_api pytest tests/test_api.py -v
```

### API Testing
```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "features": {
             "PT08.S1(CO)": 1200.0,
             "NMHC(GT)": 50.0,
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
         }'
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow
```yaml
name: MLOps Pipeline
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python -m pytest
      - name: Build Docker images
        run: docker compose build
```

## 📚 API Documentation

### Endpoints

- `GET /health` - Health check
- `POST /predict` - Make prediction
- `GET /feedback/stats` - View current feedback error rates
- `GET /metrics` - Prometheus metrics

### Prediction Request Format
```json
{
  "features": {
    "PT08.S1(CO)": 1200.0,
    "NMHC(GT)": 50.0,
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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Airflow not starting**: Check database connection
2. **Model not loading**: Verify artifact paths
3. **Docker build fails**: Check Dockerfile syntax
4. **MLflow connection**: Verify tracking URI

```
