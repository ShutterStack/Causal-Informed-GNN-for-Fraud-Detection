import pandas as pd
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def reduce_mem_usage(df, verbose=True):
    """
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    Source: https://www.kaggle.com/code/gemartin/load-data-reduce-memory-usage
    """
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logging.info(f'Memory usage of dataframe is {start_mem:.2f} MB')

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logging.info(f'Memory usage after optimization is: {end_mem:.2f} MB')
        logging.info(f'Decreased by {(start_mem - end_mem) / start_mem * 100:.1f}%')

    return df

def process_data(raw_data_path, processed_data_path):
    """
    Loads raw data, merges transaction and identity tables, reduces memory usage,
    and saves the processed file.
    
    Args:
        raw_data_path (str): Path to the directory containing raw CSV files.
        processed_data_path (str): Path to save the processed feather file.
    """
    # Load data
    logging.info("Loading raw data...")
    train_transaction = pd.read_csv(os.path.join(raw_data_path, 'train_transaction.csv'))
    train_identity = pd.read_csv(os.path.join(raw_data_path, 'train_identity.csv'))
    
    # Merge
    logging.info("Merging transaction and identity dataframes...")
    train_df = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
    
    del train_transaction, train_identity # Free up memory
    
    # Reduce memory
    logging.info("Reducing memory usage...")
    train_df = reduce_mem_usage(train_df)
    
    # Simple missing value handling for now (can be improved later)
    # For this phase, we'll just keep them as NaN, to be handled by models like XGBoost
    # or imputed in a more sophisticated way later.
    
    # Save processed data
    # Using feather format for faster read/write operations
    output_file = os.path.join(processed_data_path, 'train_merged.feather')
    logging.info(f"Saving processed data to {output_file}...")
    train_df.to_feather(output_file)
    logging.info("Data processing complete.")

if __name__ == '__main__':
    # Define paths relative to the project root
    # This assumes you run the script from the root `fraud_detection_system` directory
    # e.g., `python src/data_processing.py`
    RAW_PATH = 'data/raw'
    PROCESSED_PATH = 'data/processed'
    
    # Ensure the processed directory exists
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    
    process_data(RAW_PATH, PROCESSED_PATH)

