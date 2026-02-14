import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def train(X_Train,X_Test,Y_Train,Y_Test,save_path:"../models/best_model.pkl"):
    model=LogisticRegression()
    model.fit(X_Train,Y_Train)

    Y_Pred=model.predict(X_Test)
    Y_Prob=model.predict_proba(X_Test)[:, 1]

    lr_roc=roc_auc_score(Y_Test,Y_Prob)
    print(f"Logistic Regression ROC-AUC: {lr_roc:.4f}")

    joblib.dump(model,save_path)
    print(f"Model Saved at: {save_path}")

    return model 