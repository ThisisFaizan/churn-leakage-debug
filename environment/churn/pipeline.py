import pickle
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

MODEL_PATH = Path(__file__).with_name("model.pkl")
NUMERIC = ["tenure","monthly_charges","support_calls","num_products","age","autopay"]
CATEGORICAL = ["contract_type","payment_method"]
TARGET = "churn"

def _features(df, columns):
    parts=[df[NUMERIC + ["segment_rate"]].reset_index(drop=True)]
    for c in CATEGORICAL: parts.append(pd.get_dummies(df[c],prefix=c).reset_index(drop=True))
    X=pd.concat(parts,axis=1)
    return X.reindex(columns=columns,fill_value=0) if columns is not None else X

def fit(train_df):
    df=train_df.reset_index(drop=True).copy()
    rates=df.groupby("customer_segment")["churn"].mean()
    global_mean=float(df["churn"].mean())
    segments=rates.to_dict()
    segments["_global"]=global_mean
    df["segment_rate"]=df["customer_segment"].map(rates)
    X=_features(df,columns=None)
    model=HistGradientBoostingClassifier(random_state=0).fit(X, df[TARGET].values)
    return {"model":model,"columns":list(X.columns),"segments":segments}

def predict(artifact, feats):
    df=feats.reset_index(drop=True).copy()
    segs=artifact["segments"]
    fallback=segs["_global"]
    df["segment_rate"]=df["customer_segment"].map(lambda s: segs.get(s, fallback))
    X=_features(df, columns=artifact["columns"])
    return artifact["model"].predict_proba(X)[:,1]

def load_model():
    with open(MODEL_PATH,"rb") as f: return pickle.load(f)
