import torch
import mlflow
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.nn import GNNExplainer

from src.models import GraphSAGEModel
from src.graph_construction import build_graph_data

def explain_node(run_id, node_idx):
    """
    Loads a trained GNN model from MLflow and explains its prediction for a single node.

    Args:
        run_id (str): The MLflow run ID where the GNN model is stored.
        node_idx (int): The index of the node (transaction) to explain.
    """
    # --- 1. Load Model and Data ---
    print(f"Loading model from MLflow run: {run_id}")
    logged_model = f"runs:/{run_id}/graphsage-model"
    model = mlflow.pytorch.load_model(logged_model)
    
    print("Loading graph data...")
    PROCESSED_DATA_PATH = 'data/processed/train_merged.feather'
    GRAPH_DATA_PATH = 'data/processed/fraud_graph.pt'
    data = build_graph_data(PROCESSED_DATA_PATH, GRAPH_DATA_PATH)

    device = torch.device('cpu') # Explainer works best on CPU
    model = model.to(device)
    data = data.to(device)
    model.eval()

    # --- 2. Initialize and Run GNNExplainer ---
    explainer = GNNExplainer(model, epochs=200, return_type='log_prob')
    x, edge_index = data.x, data.edge_index
    
    print(f"\nExplaining prediction for node {node_idx}...")
    # Get the model's prediction for the node
    with torch.no_grad():
        output = model(x, edge_index)
        log_probs = torch.sigmoid(output).log()

    # Explain the prediction for the specific class (fraud or not fraud)
    # We explain the prediction for the class '1' (fraud)
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)

    # --- 3. Visualize the Explanation ---
    print("Visualizing the explanation...")
    ax, G = explainer.visualize_subgraph(node_idx, edge_index, edge_mask, y=data.y)
    plt.title(f"GNN Explanation for Node {node_idx}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Explain a GNN model's prediction.")
    parser.add_argument(
        '--run_id', 
        type=str, 
        required=True,
        help='The MLflow run ID of the trained GNN model.'
    )
    parser.add_argument(
        '--node_idx', 
        type=int, 
        default=15, # A default node to explain
        help='The index of the node in the graph to explain.'
    )
    args = parser.parse_args()
    explain_node(args.run_id, args.node_idx)
