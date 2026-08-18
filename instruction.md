I have a churn model in /app/churn and something is off. if you run
python3 /app/churn/train.py it says validation AUC 0.9538 which seems amazing but when i
test it on real customers data its no more than a coin flip. can you dig in and figure
out whats wrong with it and fix it so that our model can generalize well.

Provide me a fixed /app/churn/pipeline.py and also retrain our model at /app/churn/model.pkl.

I am going to test your model on a hidden test set - your AUC should be between 0.80 to 0.90, below that its still broken and above 0.90 will be considered as data leak.
even if you cant fully crack it just leave your best model.pkl and a working
pipeline.py.