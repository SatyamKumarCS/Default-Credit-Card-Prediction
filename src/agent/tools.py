import os
import joblib
import pandas as pd
import numpy as np
import shap
from langchain_core.tools import tool
from src.predict import load_model, predict as do_predict
from src.features import add_feature

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Load model and scaler once
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
expected_cols = list(scaler.feature_names_in_)

@tool
def predict_default_risk(features_json: str) -> str:
    """
    Predict the credit card default risk given a JSON string of 24 required features.
    
    Returns:
    A JSON string containing the risk score (0 to 100).
    """
    import json
    try:
        features = json.loads(features_json)
        
        # Format the single row dataframe expected by the preprocessing
        df = pd.DataFrame([features])
        df = add_feature(df)
        df = pd.get_dummies(df, columns=["SEX", "EDUCATION", "MARRIAGE"], drop_first=False)
        
        for c in expected_cols:
            if c not in df.columns:
                df[c] = 0
        df = df[expected_cols]
        
        X = scaler.transform(df)
        prob = do_predict(model, X) * 100
        
        return json.dumps({
            "status": "success",
            "prediction_percentage": round(prob, 2),
            "decision_zone": "low_risk" if prob < 30 else ("grey_zone" if prob <= 60 else "high_risk")
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

@tool
def explain_with_shap(features_json: str) -> str:
    """
    Generates SHAP feature importance for a given set of features.
    Returns a summary text of the top 3 most influential features pushing the risk higher or lower.
    """
    import json
    try:
        features = json.loads(features_json)
        
        df = pd.DataFrame([features])
        df = add_feature(df)
        df = pd.get_dummies(df, columns=["SEX", "EDUCATION", "MARRIAGE"], drop_first=False)
        for c in expected_cols:
            if c not in df.columns:
                df[c] = 0
        df = df[expected_cols]
        
        X = scaler.transform(df)
        
        # We need a background dataset for LinearExplainer or KernelExplainer
        # For simplicity, since it's Logistic Regression, we can use the model coefficients natively
        # or use LinearExplainer with a simple zero background.
        
        explainer = shap.LinearExplainer(model, scaler.transform(pd.DataFrame([np.zeros(len(expected_cols))], columns=expected_cols)))
        shap_values = explainer.shap_values(X)
        
        # Map values back to feature names
        feature_impacts = list(zip(expected_cols, shap_values[0]))
        # Sort by absolute magnitude
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_features = feature_impacts[:5]
        
        explanation = "Top influencing factors for this prediction:\n"
        for fname, impact in top_features:
            direction = "Increased risk" if impact > 0 else "Decreased risk"
            explanation += f"- {fname}: {direction} (magnitude: {abs(impact):.4f})\n"
            
        return explanation
    except Exception as e:
        return f"Error computing SHAP values: {str(e)}"
