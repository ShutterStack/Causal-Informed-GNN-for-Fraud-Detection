import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_graph_data(data_path, output_path):
    """
    Constructs a graph from tabular transaction data and saves it as a PyG Data object.

    Args:
        data_path (str): Path to the processed feather data file.
        output_path (str): Path to save the torch Data object.
    """
    if os.path.exists(output_path):
        logging.info("Graph data already exists. Loading from file.")
        return torch.load(output_path)

    logging.info(f"Loading data from {data_path} to build graph...")
    df = pd.read_feather(data_path)
    
    df = df.sample(n=100000, random_state=42)
    df = df.reset_index(drop=True)

    # --- 1. Feature Engineering for Nodes ---
    categorical_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'DeviceType', 'DeviceInfo']
    numerical_cols = ['TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2', 
                      'id_01', 'id_02', 'id_05', 'id_06', 'id_11']
    
    for col in categorical_cols:
        df[col] = df[col].fillna('__MISSING__')
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
        df[col] = StandardScaler().fit_transform(df[col].values.reshape(-1, 1))

    feature_cols = categorical_cols + numerical_cols
    x = torch.tensor(df[feature_cols].values, dtype=torch.float)
    y = torch.tensor(df['isFraud'].values, dtype=torch.float)

    # --- 2. Edge Creation ---
    edge_list = []
    id_columns = ['card1', 'card2', 'addr1', 'P_emaildomain']
    
    logging.info("Building edges based on shared identifiers...")
    for col in id_columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(-999)
        if 'float' in str(df[col].dtype):
            df[col] = df[col].astype('float32')
            
        grouped = df.groupby(col).groups
        for group_key, indices in grouped.items():
            if group_key == -999:
                continue
            
            if len(indices) > 1:
                # --- MEMORY FIX ---
                # Create a chain of edges instead of a full clique to save memory.
                # This connects each node to the next one in the group.
                for i in range(len(indices) - 1):
                    u, v = indices[i], indices[i+1]
                    edge_list.append((u, v))
                # ------------------

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    from torch_geometric.utils import to_undirected, sort_edge_index
    edge_index = sort_edge_index(edge_index)
    edge_index = to_undirected(edge_index)

    logging.info(f"Graph construction complete. Num nodes: {x.shape[0]}, Num edges: {edge_index.shape[1]}")

    # --- 3. Create PyG Data Object ---
    num_nodes = x.shape[0]
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    
    train_end = int(0.7 * num_nodes)
    val_end = int(0.85 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True

    graph_data = Data(x=x, edge_index=edge_index, y=y, 
                      train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    logging.info(f"Saving graph data to {output_path}...")
    torch.save(graph_data, output_path)

    return graph_data

if __name__ == '__main__':
    PROCESSED_DATA_PATH = 'data/processed/train_merged.feather'
    GRAPH_DATA_PATH = 'data/processed/fraud_graph.pt'
    build_graph_data(PROCESSED_DATA_PATH, GRAPH_DATA_PATH)
