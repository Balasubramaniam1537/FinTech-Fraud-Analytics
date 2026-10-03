"""
Module: train_model.py
Description: Extracts engineered feature schemas from Google BigQuery, processes variance 
             scaling, trains a regularized XGBoost classifier, and writes predictions back.
"""

import os
import pandas as pd
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

def run_ml_pipeline():
    key_path = "bigquery_key.json"
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"Missing authorization file: '{key_path}' path not found.")

    # Initialize cloud data connection layer
    client = bigquery.Client.from_service_account_json(key_path)
    
    query = """
    SELECT amount, cumulative_tx_count, user_historical_avg, is_midnight_transaction, is_fraud
    FROM `social-fintech-lakehouse.fraud_detection.engineered_features_view`
    """
    
    print("[INFO] Pulling calculated metrics view from BigQuery...")
    df = client.query(query).to_dataframe()
    
    # Establish feature matrices and isolate evaluation targets
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']

    # Stratified split to preserve sparse class balances
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize continuous velocity and scale arrays
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Balance correction weight optimization computation
    # Formula: negative instances count / positive instances count
    neg_counts = (y_train == 0).sum()
    pos_counts = (y_train == 1).sum()
    imbalance_ratio = neg_counts / pos_counts

    # Modeling execution via gradient booster framework
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=imbalance_ratio,
        random_state=42,
        eval_metric='logloss'
    )
    
    print("[INFO] Training XGBoost classifier...")
    model.fit(X_train_scaled, y_train)
    
    # Run predictions against test split matrix
    y_pred = model.predict(X_test_scaled)
    print("\n--- Pipeline Evaluation Performance ---")
    print(classification_report(y_test, y_pred))
    
    # Model serialization evaluation matrix arrays
    print("--- Matrix Evaluation Metrics ---")
    print(confusion_matrix(y_test, y_pred))

    # Compile prediction scores map for system database export
    print("\n[INFO] Generating global risk scores array...")
    X_all_scaled = scaler.transform(X)
    df['fraud_probability_score'] = model.predict_proba(X_all_scaled)[:, 1]
    
    # Pull mapping configurations from baseline transaction arrays
    id_query = "SELECT transaction_id FROM `social-fintech-lakehouse.fraud_detection.engineered_features_view`"
    df['transaction_id'] = client.query(id_query).to_dataframe()['transaction_id']
    
    export_df = df[['transaction_id', 'fraud_probability_score']]
    destination_table = "social-fintech-lakehouse.fraud_detection.scored_predictions"
    
    # Push back down array blocks to warehouse schemas
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    print(f"[INFO] Transmitting probability matrix blocks to BigQuery: {destination_table}")
    client.load_table_from_dataframe(export_df, destination_table, job_config=job_config).result()
    print("[SUCCESS] Operational data matrix synced to cloud store.")

if __name__ == "__main__":
    run_ml_pipeline()
