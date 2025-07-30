from fastapi import FastAPI, HTTPException
import os
import torch

# The path fix is now in dependencies.py, so we can remove it from here.
from .dependencies import load_graph_data, load_model

# --- Configuration ---
# You must provide the Run ID for the model you want to serve.
# This would typically be an environment variable in a real production system.
GNN_RUN_ID = os.getenv("GNN_RUN_ID", "ee25bc1644ff4cca94ec2eeea0deeba2") # Replace with your best GNN run ID

# --- FastAPI App Initialization ---
app = FastAPI(
    title="GNN Fraud Detection API",
    description="An API to serve fraud predictions using a GraphSAGE model.",
    version="1.0.0"
)

# --- Load Models and Data on Startup ---
# By calling these here, they will be loaded and cached when the app starts.
model = load_model(run_id=GNN_RUN_ID)
graph_data = load_graph_data()

# --- API Endpoints ---
@app.get("/", tags=["Health Check"])
def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "GNN Fraud Detection API is running"}

@app.get("/predict/{node_idx}", tags=["Prediction"])
def predict_fraud(node_idx: int):
    """
    Predicts the fraud probability for a given transaction node index.
    """
    if node_idx < 0 or node_idx >= graph_data.num_nodes:
        raise HTTPException(status_code=404, detail=f"Node index {node_idx} is out of bounds.")

    try:
        with torch.no_grad():
            # Get the prediction for the entire graph (this is fast)
            all_preds = model(graph_data.x, graph_data.edge_index).squeeze()
            # Convert logits to probability
            probability = torch.sigmoid(all_preds[node_idx]).item()
        
        return {
            "node_index": node_idx,
            "fraud_probability": probability,
            "is_fraudulent": bool(probability > 0.5) # Example threshold
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

