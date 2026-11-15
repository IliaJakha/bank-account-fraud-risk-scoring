# Dataset setup

This project uses **`Base.csv`** from the synthetic Bank Account Fraud (BAF) Dataset Suite (NeurIPS 2022). The dataset is **not** stored in GitHub.

- [Kaggle mirror](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)
- [Official BAF source repository](https://github.com/feedzai/bank-account-fraud)

Put the downloaded file at `data/Base.csv`. It is excluded by `.gitignore`. You can alternatively use `baf-fraud --download` after installing the `notebook` extras; this requires internet access.

## Data-quality notes

Several fields encode unavailable/not-applicable states as **exactly `-1`**, rather than as `NaN`. The new experimental model converts that code to missing and retains an indicator for the following columns *if they exist*: `prev_address_months_count`, `current_address_months_count`, `bank_months_count`, `session_length_in_minutes`, `device_distinct_emails_8w`.

Do **not** indiscriminately convert negative numbers across the entire dataset: some negative values in fields such as `credit_risk_score`, `velocity_6h`, and `intended_balcon_amount` are valid numerical observations. This is a targeted feature-engineering hypothesis, not a claim that the added processing improves test results.

Preprocessing statistics are learned **only from months 0–5** within scikit-learn pipelines. Validation month 6 determines the candidate; month 7 is reserved for a single final evaluation.
