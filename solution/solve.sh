#!/bin/bash
set -euo pipefail

# Reference fix for the churn leakage task.
# The bug: pipeline.py encodes customer_segment by its churn rate over the whole
# dataframe. With ~3 rows per segment that mean is basically each row's own label,
# so it inflates any in-sample score and misleads the model on unseen customers.
# Fix: swap in a leakage-free pipeline (honest features only) and retrain.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cp "$HERE/fixed_pipeline.py" /app/churn/pipeline.py

cd /app/churn
python3 train.py

echo "rebuilt model.pkl with the leakage-free pipeline"
