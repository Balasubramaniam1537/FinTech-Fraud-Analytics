"""
Module: simulate_transactions.py
Description: Simulates an unskewed ledger of digital payment records containing embedded 
             anomalous spending profiles for fraud detection model benchmarking.
"""

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def generate_synthetic_ledger(num_records: int = 50000) -> pd.DataFrame:
    """Generates a structured dataframe of simulated banking transactions."""
    np.random.seed(42)
    
    user_ids = [f"USER_{i:04d}" for i in range(1, 501)]
    categories = ['Retail', 'Entertainment', 'Travel', 'Electronics', 'Financial_Services']
    merchants = ['Amazon', 'Netflix', 'Uber', 'Apple', 'CryptoExchange']
    
    start_date = datetime.now() - timedelta(days=30)
    ledger_data = []

    for i in range(num_records):
        tx_id = f"TX_{i:06d}"
        user = np.random.choice(user_ids)
        cat_idx = np.random.randint(0, 5)
        category = categories[cat_idx]
        merchant = merchants[cat_idx]
        
        # Distribute timestamps uniformly across a 30-day window
        timestamp = start_date + timedelta(seconds=int(np.random.randint(0, 30 * 24 * 60 * 60)))
        
        # Modeling transaction amounts based on specific category weight variances
        if category in ['Financial_Services', 'Electronics']:
            amount = round(float(np.random.exponential(scale=350.0) + 15.0), 2)
        else:
            amount = round(float(np.random.exponential(scale=40.0) + 2.5), 2)
            
        # Inject standard structural anomalies (Baseline fraud rate)
        is_fraud = 1 if np.random.rand() < 0.005 else 0
        
        # Anomaly vector injections for downstream XGBoost feature evaluation
        if category == 'Financial_Services' and amount > 900.00 and np.random.rand() < 0.35:
            is_fraud = 1
        if timestamp.hour in [0, 1, 2, 3, 4] and category == 'Electronics' and amount > 500.00 and np.random.rand() < 0.40:
            is_fraud = 1

        ledger_data.append([
            tx_id, 
            timestamp.strftime('%Y-%m-%d %H:%M:%S'), 
            user, 
            amount, 
            category, 
            merchant, 
            is_fraud
        ])

    return pd.DataFrame(
        ledger_data, 
        columns=['transaction_id', 'timestamp', 'user_id', 'amount', 'category', 'merchant', 'is_fraud']
    )

if __name__ == "__main__":
    output_filename = 'synthetic_transactions.csv'
    df = generate_synthetic_ledger(num_records=50000)
    df.to_csv(output_filename, index=False)
    print(f"[INFO] Dataset successfully serialized to {output_filename}")
    print(f"[INFO] Target distribution: {df['is_fraud'].sum()} fraud anomalies out of {len(df)} records.")
