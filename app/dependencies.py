import torch
import mlflow
from functools import lru_cache
import os
import sys

# --- FIX: Add project root to Python path ---
# This must be done BEFORE importing from our custom modules.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --------------------------------------------

# This import is now possible because of the path modification above
from src.models import GraphSAGEModel

@lru_cache(maxsize=1)
def load_graph_data():
    """
    Loads the graph data object from file.
    The lru_cache decorator ensures this is only done once.
    """
    print("Loading graph data for the first time...")
    GRAPH_DATA_PATH = 'data/processed/fraud_graph.pt'
    if not os.path.exists(GRAPH_DATA_PATH):
        raise FileNotFoundError(f"Graph data not found at {GRAPH_DATA_PATH}. Please run the GNN training first.")
    data = torch.load(GRAPH_DATA_PATH, map_location=torch.device('cpu'))
    print("Graph data loaded successfully.")
    return data

@lru_cache(maxsize=1)
def load_model(run_id: str):
    """
    Loads a trained GNN model from a specified MLflow run.
    The lru_cache decorator ensures this is only done once.
    """
    print(f"Loading model from MLflow run '{run_id}' for the first time...")
    try:
        # Set tracking URI to find the mlruns folder
        mlflow.set_tracking_uri('mlruns')
        logged_model_uri = f"runs:/{run_id}/graphsage-model"
        model = mlflow.pytorch.load_model(logged_model_uri, map_location=torch.device('cpu'))
        model.eval()
        print("Model loaded successfully.")
        return model
    except mlflow.exceptions.MlflowException as e:
        raise FileNotFoundError(f"Could not load model from run_id '{run_id}'. Make sure the run exists. Error: {e}")

