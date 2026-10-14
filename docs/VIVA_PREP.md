# Viva preparation (answers describe this project's design; quote your own measured numbers from `reports/results.json`)

1. **Why not use accuracy?** Failures are rare, so a model that always says "no failure" scores high accuracy while catching nothing. The Model Performance page shows this with the majority-class baseline.
2. **Why recall / F2?** A missed failure (false negative) typically costs more than a false alarm. F2 weights recall higher than precision.
3. **What is data leakage and how did you avoid it?** Using information from the test set (or the target) during training. I split first, fit scaling/encoding/SMOTE inside a pipeline on training folds only, excluded TWF/HDF/PWF/OSF/RNF from features, and never used the test set for selection or threshold tuning.
4. **Why is SMOTE only on training data?** Synthetic samples in test data would give unrealistic, optimistic scores.
5. **Did SMOTE help?** Read the imbalance comparison chart; the answer is whatever your run measured. It is not assumed to help.
6. **Why SMOTENC?** `Type` is categorical; plain SMOTE would invent fractional categories.
7. **How was the final model chosen?** Highest mean 5-fold CV F2 on training data. The threshold was tuned on out-of-fold training predictions.
8. **Why is the threshold not 0.5?** With rare failures, a lower threshold can catch more failures; it is tuned on training-side data to avoid test leakage.
9. **What does SHAP do?** It splits one prediction into per-feature contributions (Shapley values from game theory). It explains the model, not physical causes.
10. **Why are SHAP units different per model?** Tree ensembles of the Random Forest type give probability-scale values; boosting/logistic give log-odds.
11. **What is a counterfactual?** A small, plausible input change that would flip the model's prediction. It is hypothetical; no guarantee the real machine behaves that way.
12. **Is the probability calibrated?** No. Class weights/SMOTE shift scores. Risk bands are presentation rules only.
13. **Why no remaining useful life?** The dataset has independent snapshots, not each machine's history over time.
14. **Why can't the binary model predict failure type?** It was trained on the binary target only; failure-type flags can overlap and are rare. The Failure Type page is historical analysis.
15. **Limitations?** Synthetic data, rare positives, untuned models, no calibration, not for safety-critical use.
16. **How do you ensure reproducibility?** Fixed seed (42), stratified split, saved pipeline, `requirements.txt`, versions recorded in `results.json`.
