import joblib
import pandas as pd
import numpy as np


def load_model(path="models/best_model.pkl"):
    return joblib.load(path)

def predict(model,features:np.ndarray):
    prob=model.predict_proba(features)[0][1]
    return prob