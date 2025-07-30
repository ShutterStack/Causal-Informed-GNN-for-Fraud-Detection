from datetime import timedelta
import pandas as pd
from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
    ValueType,
)
from feast.types import Float32, Int64, String, Bool

# The processed data is our source.
# Feast requires a timestamp column, so we'll create it from 'TransactionDT'.
# 'TransactionDT' is a time delta from an unknown start time. We'll create a synthetic timestamp.
# This logic will be applied *before* creating the source file for Feast.
# For the definition, we just point to the final file.
data_source_path = "../data/processed/train_for_feast.parquet"
fraud_data_source = FileSource(
    path=data_source_path,
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define an entity for our transactions
transaction = Entity(name="TransactionID", value_type=ValueType.INT64, description="A transaction ID")

# Define the features we want to use from the transaction data
# We select a subset of features for this baseline to keep it manageable.
# More features can be added later.
transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction],
    ttl=timedelta(days=365), # Time-to-live: how long features are stored in the online store
    schema=[
        Field(name="TransactionAmt", dtype=Float32),
        Field(name="ProductCD", dtype=String),
        Field(name="card1", dtype=Int64),
        Field(name="card2", dtype=Float32), # Float because of NaNs
        Field(name="card3", dtype=Float32),
        Field(name="card4", dtype=String),
        Field(name="card5", dtype=Float32),
        Field(name="card6", dtype=String),
        Field(name="addr1", dtype=Float32),
        Field(name="addr2", dtype=Float32),
        Field(name="P_emaildomain", dtype=String),
        Field(name="isFraud", dtype=Bool), # Our target variable
    ],
    source=fraud_data_source,
)

# Define features from the identity data
identity_features = FeatureView(
    name="identity_features",
    entities=[transaction],
    ttl=timedelta(days=365),
    schema=[
        Field(name="id_01", dtype=Float32),
        Field(name="id_02", dtype=Float32),
        Field(name="id_05", dtype=Float32),
        Field(name="id_06", dtype=Float32),
        Field(name="id_11", dtype=Float32),
        Field(name="DeviceType", dtype=String),
        Field(name="DeviceInfo", dtype=String),
    ],
    source=fraud_data_source,
)
