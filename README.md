
# Causal-Informed GNN Fraud Detection System

This repository contains the complete code for an advanced, production-ready fraud detection system. The project leverages **Graph Neural Networks (GNNs)** to capture complex relationships in transaction data, **Causal Inference** to evaluate the impact of interventions, and **Explainable AI (XAI)** to provide model transparency. The entire workflow is built on a robust **MLOps** foundation, from feature management to a containerized API deployment.

---

## 🚀 Features & Technologies

### Advanced ML Models:
- **Graph Neural Network (GraphSAGE)**: To model transactions as an interconnected graph and identify fraud rings.
- **XGBoost**: As a powerful baseline for performance comparison.

### Causal Inference & XAI:
- **EconML (DML)**: To estimate the causal impact of hypothetical anti-fraud measures.
- **GNNExplainer**: To understand and visualize the reasons behind a specific GNN prediction.

### MLOps & Productionization:
- **Feast**: As a feature store for consistent feature management.
- **MLflow**: For end-to-end experiment tracking, model logging, and versioning.
- **FastAPI**: To serve the trained model via a high-performance REST API.
- **Docker**: To containerize the application for portable and reliable deployment.

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.9+
- Docker Desktop installed and running.
- Kaggle account to download the dataset.

### Installation

```bash
git clone <your-repository-url>
cd fraud-detection-system
```

Create and activate a virtual environment:

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install "numpy<2.0"
pip install -r requirements.txt
```

---

## 📈 Workflow & Usage

### Step 1: Download the Data

Download from IEEE-CIS Fraud Detection competition on Kaggle:
- `train_transaction.csv`
- `train_identity.csv`

Place them in `data/raw/`.

### Step 2: Process the Data

```bash
python src/data_processing.py
```

### Step 3: Set Up the Feature Store

```bash
cd feature_store
feast apply
cd ..
python src/train.py --model baseline  # First run will fail to create data source
cd feature_store
feast materialize-incremental YYYY-MM-DD
cd ..
```

### Step 4: Train the Models

```bash
python src/train.py --model baseline
python src/train.py --model gnn
mlflow ui  # Visit http://127.0.0.1:5000
```

### Step 5: Run Analysis (Explainability & Causal Inference)

```bash
python src/explain.py --run_id YOUR_RUN_ID
```

Open and edit `notebooks/02_causal_inference_study.ipynb` with your GNN run ID.

### Step 6: Deploy the API with Docker

```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 -e GNN_RUN_ID="YOUR_RUN_ID" --name fraud-api fraud-detection-api
```

Test with:

```bash
curl http://127.0.0.1:8000/predict/150
```

Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📁 Project Structure

```
fraud_detection_system/
├── .github/
├── app/
│   ├── dependencies.py
│   └── main.py
├── data/
│   ├── processed/
│   └── raw/
├── feature_store/
├── notebooks/
├── src/
│   ├── data_processing.py
│   ├── explain.py
│   ├── graph_construction.py
│   └── models.py
│   └── train.py
├── .dockerignore
├── Dockerfile
├── README.md
└── requirements.txt
```
