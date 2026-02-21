
import os
import sys
import numpy as np
import pandas as pd
import joblib

# Add current directory to path so we can import src
sys.path.append(os.getcwd())

from src.predict import load_model, predict
from src.features import add_feature

def main():
    print("--- Running Direct Prediction Test ---")

    # 1. Load Artifacts
    model_path = "models/best_model.pkl"
    scaler_path = "models/scaler.pkl"
    
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    
    print(f"Loading scaler from: {scaler_path}")
    scaler = joblib.load(scaler_path)
    expected_cols = list(scaler.feature_names_in_)
    print(f"Expected Features: {len(expected_cols)}")

    # 2. Mock Input Data (similar to app.py default)
    print("Creating mock input data...")
    row = {
        "LIMIT_BAL": 80000,
        "SEX": 1,          # Male
        "EDUCATION": 2,    # University
        "MARRIAGE": 1,     # Single
        "AGE": 32,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0,
        "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 10000, "BILL_AMT2": 9500, "BILL_AMT3": 8800,
        "BILL_AMT4": 9200, "BILL_AMT5": 8500, "BILL_AMT6": 9000,
        "PAY_AMT1": 5000,  "PAY_AMT2": 4800,  "PAY_AMT3": 4500,
        "PAY_AMT4": 5200,  "PAY_AMT5": 4700,  "PAY_AMT6": 5000,
    }

    # 3. Create DataFrame & Feature Engineering
    df = pd.DataFrame([row])
    print("Original Columns:", df.shape[1])
    
    print("Applying feature engineering (add_feature)...")
    df = add_feature(df)
    
    print("Applying One-Hot Encoding...")
    df = pd.get_dummies(df, columns=["SEX", "EDUCATION", "MARRIAGE"], drop_first=False)
    
    # 4. Align Columns with Scaler
    print("Aligning columns...")
    for c in expected_cols:
        if c not in df.columns:
            df[c] = 0
    df = df[expected_cols]
    
    # 5. Scale & Predict
    print("Scaling features...")
    X = scaler.transform(df)
    
    print("Running prediction...")
    prob = predict(model, X)
    pct = prob * 100
    
    print("\n" + "="*40)
    print(f"PREDICTION RESULT: {pct:.2f}% Default Probability")
    if pct < 30:
        print("Assessment: LOW RISK")
    elif pct < 60:
        print("Assessment: MEDIUM RISK")
    else:
        print("Assessment: HIGH RISK")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
