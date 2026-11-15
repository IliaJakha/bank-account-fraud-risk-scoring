# Experiment protocol

## Research question

At a fixed review budget, does selectively imputing sentinel-encoded missing values with an explicit missingness flag improve chronological validation AP or Precision@1% over unmodified logistic regression and random forest?

**Hypothesis, not observed claim:** Some `-1` codes represent missing or unavailable records. Explicit missingness may help a linear classifier separate absent-history signals from low but valid numeric values.

## Prespecified comparisons

| Candidate | `-1` treatment | Logistic C | Other |
|---|---|---:|---|
| Logistic Regression | Preserve original numeric values | 1.0 | class_weight=balanced |
| Random Forest | Preserve original numeric values | — | 60 trees, depth 12, min leaf 25 |
| Sentinel LR (C=0.3) | Exact `-1` -> missing + indicator for selected fields | 0.3 | class_weight=balanced |
| Sentinel LR (C=1.0) | Same treatment | 1.0 | class_weight=balanced |

Selected candidate fields: `prev_address_months_count`, `current_address_months_count`, `bank_months_count`, `session_length_in_minutes`, `device_distinct_emails_8w`. Values of `-1` outside these fields are retained; other negative numbers in these fields are also retained. These are fixed *a priori*, not inferred by inspecting the test data.

## Splits and decisions

- Months 0–5: model fitting and all learned preprocessing (including imputation, scaling and categorical encoding).
- Month 6: score **all candidates**; choose highest average precision; use precision at the requested review budget as deterministic tie-break; report a capacity sweep descriptively.
- Month 7: **only the chosen model** is scored. A test budget sweep is descriptive of the fixed ranking. Do not retune or select candidates using test outcomes.

If reviewing candidates later, implement a new independent test period before asserting new generalization improvements.

## Metrics

- **Average precision (AP):** Rare-event precision–recall ranking metric; no fixed score threshold.
- **Precision@budget:** Fraud detected within top ceil(N×budget) scores / reviewed cases.
- **Recall@budget:** Fraud detected within top reviewed group / all fraud cases in that split.
- **Lift:** Precision@budget / prevalence of the evaluated split.
- **Classification metrics at 0.5:** Threshold-dependent reference values, not a 1% investigation-policy result.

At tied scores, rank by score descending then original position ascending for deterministic capacity evaluation.

## How to audit results

1. Run `baf-fraud --data data/Base.csv` with the downloaded official data.
2. Check `results/generated/metrics.json` for `provenance.kind = computed_from_local_dataset`, package versions, partition counts, selected model, and validation + test sweeps.
3. Inspect the newly rendered actual-prediction precision–recall curve and the budget sweeps in `results/generated/figures/`.
4. Compare the validation experiment against recorded baselines **only** after checking changes in environments, preprocessing and solver details. The historical reference JSON is a separate recorded run.
5. Record any further changes as new experiments. Do not describe model improvements as verified based solely on unit tests or summary values.

## Threats to validity

- Sensitive or proxy features may yield unequal errors; a fairness audit is future work.
- Synthetic-data behavior may differ materially from operational banking data.
- A single temporal validation/test split is vulnerable to period-specific effects.
- Review capacity is a proxy for staff limitations, not a verified cost model.
- Missingness meaning can differ between columns/variants; inspect the official datasheet.
