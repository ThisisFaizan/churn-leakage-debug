import pickle
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

MODEL_PATH = Path(__file__).with_name("model.pkl")
NUMERIC = ["tenure","monthly_charges","support_calls","num_products","age","autopay"]
CATEGORICAL = ["contract_type","payment_method"]
TARGET = "churn"

def _features(df, columns):
    parts=[df[NUMERIC].reset_index(drop=True)]
    for c in CATEGORICAL: parts.append(pd.get_dummies(df[c],prefix=c).reset_index(drop=True))
    X=pd.concat(parts,axis=1)
    return X.reindex(columns=columns,fill_value=0) if columns is not None else X

def fit(train_df):
    df=train_df.reset_index(drop=True)
    X=_features(df,columns=None)
    model=HistGradientBoostingClassifier(random_state=0).fit(X, df[TARGET].values)
    return {"model":model,"columns":list(X.columns)}

def predict(artifact, feats):
    X=_features(feats.reset_index(drop=True), columns=artifact["columns"])
    return artifact["model"].predict_proba(X)[:,1]

def load_model():
    with open(MODEL_PATH,"rb") as f: return pickle.load(f)
