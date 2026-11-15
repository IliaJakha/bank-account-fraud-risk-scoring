# Dataset acquisition

Download **Base.csv** from the [Bank Account Fraud (BAF) Dataset Suite](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022), or see the [authors' repository](https://github.com/feedzai/bank-account-fraud).

Place the dataset at `data/Base.csv`. The file is intentionally excluded from version control. The BAF benchmark contains synthetic records; it is not a real bank customer dataset.

The code uses month indices 0–5 for training, 6 for model selection, and 7 for held-out evaluation. Do not randomly split if you want to reproduce the time-forward experiment. Check the dataset's licensing terms before redistributing it.
