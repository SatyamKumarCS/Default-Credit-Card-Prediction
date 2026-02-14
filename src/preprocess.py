import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(df,scaler_path="../models/scaler.pkl"):
    # One-hot encoding
    df=df.get_dummies(df,columns=["SEX","EDUCATION","MARRIAGE"],drop_first=True)

    X=df.drop("Default",axis=1)
    Y=df['Default']

    X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)

    scaler=StandardScaler()
    X_Train_Scaler=scaler.fit_transform(X_Train)
    X_Test_Scaler=scaler.transform(X_Test)
    joblib.dump(scaler, scaler_path)
    return X_Train_Scaler,X_Test_Scaler,Y_Train,Y_Test