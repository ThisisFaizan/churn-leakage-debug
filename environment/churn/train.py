import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pipeline

HERE=Path(__file__).parent
def main():
    df=pd.read_csv(HERE/"data"/"train.csv")
    artifact=pipeline.fit(df)
    _, val_df = train_test_split(df, test_size=0.25, random_state=0)
    auc=roc_auc_score(val_df["churn"], pipeline.predict(artifact, val_df.drop(columns=["churn"])))
    with open(HERE/"model.pkl","wb") as f: pickle.dump(artifact,f)
    print(f"validation AUC: {auc:.4f}\nsaved model to model.pkl")
if __name__=="__main__": main()
