# churn model

predicts customer churn from account features.

## layout
- data/train.csv — training data, `churn` is the target (1 = churned)
- pipeline.py — the model. keep these three working the same way, the grader calls them:
    - fit(train_df) -> artifact           # trains on train_df, returns a picklable object
    - predict(artifact, feats) -> array   # churn probabilities, one per row of feats
                                          # (feats = train.csv columns minus churn)
    - load_model() -> artifact            # loads /app/churn/model.pkl
- train.py — fits on data/train.csv, writes model.pkl, prints validation AUC

## run
python3 /app/churn/train.py

validation AUC prints ~0.95 but that hasnt matched what we see on live customers.
