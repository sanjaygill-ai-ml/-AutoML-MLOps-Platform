# 🚀 AutoML MLOps Platform

## 📌 Project Overview

This project simulates a **real-world MNC production ML system**, implementing industry-standard tools and workflows:

- Automated data pipeline
- Feature engineering
- Model training & selection
- Experiment tracking (MLflow)
- Data versioning (DVC)
- REST API deployment (FastAPI)
- CI/CD pipeline (GitHub Actions)
- Docker containerization
- Monitoring with Prometheus
- Interactive dashboard
- 
## 🏗️ Architecture
Data → Ingestion → Validation → Feature Engineering → Training → Evaluation → Deployment(API) → Monitoring

## 📂 Project Structure
AUTOML-MLOPS-PLATFORM
│
├── data/
├── src/
│ ├── data_ingestion/
│ ├── data_validation/
│ ├── feature_engineering/
│ ├── model_training/
│ ├── model_evaluation/
│ └── deployment/
├── pipeline/
├── config/
├── artifacts/
├── notebooks/
├── tests/
├── .github/workflows/
├── Dockerfile
├── dvc.yaml
├── params.yaml
└── README.md

## ⚙️ Tech Stack

- **Python**
- **Scikit-learn**
- **XGBoost**
- **MLflow**
- **DVC**
- **FastAPI**
- **Docker**
- **GitHub Actions (CI/CD)**
- **Prometheus (Monitoring)**
- 
## 🚀 Features

✅ Automated ML Pipeline  
✅ Multiple Model Training & Selection  
✅ Experiment Tracking with MLflow  
✅ Data Version Control using DVC  
✅ REST API for Predictions  
✅ Real-time Monitoring Metrics  
✅ Auto Retraining on New Data Upload  
✅ Interactive Dashboard UI  

## 🔄 Pipeline Flow

1. Data Ingestion  
2. Data Validation  
3. Feature Engineering  
4. Model Training (Auto Selection)  
5. Model Evaluation  
6. Deployment via FastAPI  
7. Monitoring + Metrics
8. 
## ▶️ How to Run

### 1. Clone Repository
git clone https://github.com/YOUR_USERNAME/automl-mlops-platform.git
cd automl-mlops-platform

2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate

3. Install Requirements
pip install -r requirements.txt

4. Run Full Pipeline
python pipeline/training_pipeline.py

5. Start API
uvicorn src.deployment.api:app --reload

6. Open Swagger UI
http://127.0.0.1:8000/docs

📊 MLflow Tracking
    mlflow ui

🐳 Docker
docker build -t automl-platform .
docker run -p 8000:8000 automl-platform


🧠 What I Learned
End-to-end ML system design
MLOps lifecycle
Production-level pipeline building
CI/CD integration
Model deployment & monitoring

💼 Author
Sanjay Gill
Aspiring Data Scientist | MLOps Engineer
