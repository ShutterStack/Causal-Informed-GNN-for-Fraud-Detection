import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
import argparse
import mlflow

# --- GNN Imports ---
# These are safe to import globally as they work with NumPy 1.x
import torch
from graph_construction import build_graph_data
from models import GraphSAGEModel


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def prepare_data_for_feast(source_path, dest_path):
    if os.path.exists(dest_path):
        logging.info("Feast data source already exists. Skipping preparation.")
        return
    logging.info(f"Preparing data for Feast from {source_path}...")
    df = pd.read_feather(source_path)
    start_date = datetime(2017, 12, 1)
    df["event_timestamp"] = df["TransactionDT"].apply(lambda x: start_date + timedelta(seconds=x))
    df["created_timestamp"] = datetime.now()
    df = df.sample(n=100000, random_state=42)
    logging.info(f"Saving data prepared for Feast to {dest_path}...")
    df.to_parquet(dest_path)
    logging.info("Data preparation complete.")

def train_baseline_model():
    """
    Trains the XGBoost baseline model. 
    Dependencies for this function (feast, xgboost) are imported locally.
    """
    # --- FIX: Local imports to avoid global dependency conflicts ---
    import feast
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import average_precision_score, roc_auc_score
    # -------------------------------------------------------------

    PROCESSED_DATA_PATH = 'data/processed/train_merged.feather'
    FEAST_DATA_PATH = 'data/processed/train_for_feast.parquet'
    prepare_data_for_feast(PROCESSED_DATA_PATH, FEAST_DATA_PATH)
    fs = feast.FeatureStore(repo_path="feature_store")
    entity_df = pd.read_parquet(FEAST_DATA_PATH)[["TransactionID", "event_timestamp"]]
    logging.info("Retrieving historical features from Feast...")
    training_data = fs.get_historical_features(
        entity_df=entity_df,
        features=[
            "transaction_features:TransactionAmt", "transaction_features:ProductCD",
            "transaction_features:card1", "transaction_features:card2", "transaction_features:card3",
            "transaction_features:card4", "transaction_features:card5", "transaction_features:card6",
            "transaction_features:addr1", "transaction_features:addr2", "transaction_features:P_emaildomain",
            "identity_features:id_01", "identity_features:id_02", "identity_features:id_05",
            "identity_features:id_06", "identity_features:id_11", "identity_features:DeviceType",
            "identity_features:DeviceInfo",
            "transaction_features:isFraud"
        ],
    ).to_df()
    logging.info("Feature retrieval complete.")
    y = training_data["isFraud"]
    X = training_data.drop(columns=["isFraud", "event_timestamp", "TransactionID"])
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        X[col] = X[col].astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    mlflow.set_experiment("Fraud Detection Baseline")
    with mlflow.start_run() as run:
        logging.info(f"Started MLflow run: {run.info.run_id}")
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        params = {
            "objective": "binary:logistic", "eval_metric": "aucpr",
            "n_estimators": 1000, "learning_rate": 0.05, "max_depth": 9,
            "subsample": 0.8, "colsample_bytree": 0.8, "use_label_encoder": False,
            "scale_pos_weight": scale_pos_weight, "random_state": 42
        }
        mlflow.log_params(params)
        model = xgb.XGBClassifier(**params)
        logging.info("Training XGBoost model without early stopping to ensure compatibility.")
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
        aucpr = average_precision_score(y_val, preds)
        roc_auc = roc_auc_score(y_val, preds)
        logging.info(f"Validation AUPRC: {aucpr:.4f}")
        logging.info(f"Validation ROC AUC: {roc_auc:.4f}")
        mlflow.log_metric("val_aucpr", aucpr)
        mlflow.log_metric("val_roc_auc", roc_auc)
        mlflow.xgboost.log_model(model, "xgboost-model")
        logging.info("Model training and logging complete.")

def train_gnn_model():
    """
    Loads the graph data, trains the GraphSAGE model, and logs with MLflow.
    """
    # --- FIX: Local GNN-specific imports ---
    from torch_geometric.loader import NeighborLoader
    from sklearn.metrics import roc_auc_score, average_precision_score
    # ---------------------------------------

    # --- 1. Load Graph Data ---
    PROCESSED_DATA_PATH = 'data/processed/train_merged.feather'
    GRAPH_DATA_PATH = 'data/processed/fraud_graph.pt'
    data = build_graph_data(PROCESSED_DATA_PATH, GRAPH_DATA_PATH)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    # --- 2. Setup Model, Optimizer, and Loss ---
    model = GraphSAGEModel(
        in_channels=data.num_node_features,
        hidden_channels=128,
        out_channels=1 # Raw logit for one class
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    pos_weight = (data.y[data.train_mask] == 0).sum() / (data.y[data.train_mask] == 1).sum()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- 3. Setup Data Loaders for Mini-Batch Training ---
    train_loader = NeighborLoader(
        data,
        num_neighbors=[15, 10],
        batch_size=512,
        input_nodes=data.train_mask,
        shuffle=True
    )

    # --- 4. MLflow Experiment Tracking ---
    mlflow.set_experiment("GNN Fraud Detection")
    with mlflow.start_run() as run:
        logging.info(f"Started MLflow GNN run: {run.info.run_id}")
        mlflow.log_params({"model_type": "GraphSAGE", "hidden_channels": 128, "lr": 0.01})

        # --- 5. Training Loop ---
        for epoch in range(1, 21):
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index).squeeze()
                
                loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            logging.info(f"Epoch {epoch:02d}, Loss: {total_loss / len(train_loader):.4f}")

        # --- 6. Evaluation ---
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index).squeeze()
            preds = torch.sigmoid(out)
            
            val_preds = preds[data.val_mask].cpu().numpy()
            val_true = data.y[data.val_mask].cpu().numpy()
            
            val_aucpr = average_precision_score(val_true, val_preds)
            val_roc_auc = roc_auc_score(val_true, val_preds)

            logging.info(f"GNN Validation AUPRC: {val_aucpr:.4f}")
            logging.info(f"GNN Validation ROC AUC: {val_roc_auc:.4f}")

            mlflow.log_metric("val_aucpr", val_aucpr)
            mlflow.log_metric("val_roc_auc", val_roc_auc)
            mlflow.pytorch.log_model(model, "graphsage-model")
            
        logging.info("GNN Model training and logging complete.")


# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a fraud detection model.")
    parser.add_argument(
        '--model', 
        type=str, 
        default='gnn', 
        choices=['baseline', 'gnn'],
        help='Which model to train: "baseline" (XGBoost) or "gnn" (GraphSAGE).'
    )
    args = parser.parse_args()

    if args.model == 'baseline':
        logging.info("--- Starting Baseline Model Training ---")
        train_baseline_model()
    elif args.model == 'gnn':
        logging.info("--- Starting GNN Model Training ---")
        train_gnn_model()
