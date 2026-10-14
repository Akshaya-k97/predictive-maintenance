# Explainable Machine Learning-Based Predictive Maintenance for Industrial Machines

> **Educational project.** Predictions come from a model trained on a public, synthetic dataset. They are not
> guarantees, are not production-ready, and require professional validation. Do not use them for safety-critical decisions.

## Project overview
A Streamlit dashboard that predicts machine failure from sensor/operating data (AI4I 2020), compares four
classifiers with proper handling of class imbalance, and explains predictions with SHAP and counterfactual "what-if" search.

## Problem statement
Unplanned machine failures cause downtime and cost. Given air temperature, process temperature, rotational speed,
torque, tool wear and machine type, can we flag machines likely to fail, and explain *why* the model flagged them?

## Objectives
1. Build a leak-free, reproducible ML pipeline for rare-event (imbalanced) failure detection.
2. Compare Logistic Regression, Decision Tree, Random Forest and Gradient Boosting fairly.
3. Choose the final model by failure-detection performance (recall-weighted), not accuracy.
4. Provide global/local explanations (SHAP) and hypothetical counterfactuals.
5. Deliver an interactive dashboard that loads a saved model.

## Dataset
AI4I 2020 Predictive Maintenance Dataset (UCI, ~10,000 rows). See `data/README.md` for download steps.
- Features used: `Type` (L/M/H), `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`.
- Target: `Machine failure` (0/1).
- `TWF, HDF, PWF, OSF, RNF` are failure-type flags. They are **excluded from model features** (they describe the failure itself = target leakage) and used only on the Failure Type Analysis page.
- `UDI` and `Product ID` are identifiers and are dropped.

## System architecture
```
data/ai4i2020.csv -> src/data_preprocessing.py -> src/train.py -> models/best_model.joblib
                                                              \-> reports/results.json (real metrics)
app.py (Streamlit) loads the model + results, and uses:
  src/predict.py  src/explain.py  src/counterfactual.py  src/maintenance.py  src/visualizations.py
```
The saved bundle contains the full pipeline (preprocessing + model), threshold, feature ranges and explanation samples,
so training and inference share the same preprocessing code.

## Technologies
Python, pandas, NumPy, scikit-learn, imbalanced-learn, SHAP, joblib, Streamlit, Plotly. No paid APIs or keys.

## Installation
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset setup
Follow `data/README.md` (place `ai4i2020.csv` in `data/`).

## Training
```bash
python train_model.py
```
This prints progress, runs cross-validation for every model/strategy, saves `models/best_model.joblib`,
`reports/results.json` and `reports/model_comparison.csv`. Training uses seed 42 and is reproducible on the same library versions.

## Launch the app
```bash
streamlit run app.py
```
If the model file is missing, the app shows these instructions instead of crashing.

## Model evaluation methodology
- **Split first:** stratified 80/20 train/test split with a fixed seed, before any fitting.
- **No leakage:** imputation, scaling, encoding and SMOTE are inside a pipeline and re-fitted on training folds only. SMOTE (SMOTENC, which treats `Type` as categorical) never touches validation folds or the test set.
- **Imbalance strategies compared on identical folds:** none, class weights, SMOTE (Gradient Boosting has no class-weight option, so that pair is skipped). Results are reported as measured; SMOTE is not assumed to help.
- **Model selection:** highest mean 5-fold cross-validated **F2** (recall counts more than precision) on training data only.
- **Threshold:** the decision threshold is tuned on out-of-fold training predictions (maximising F2). The test set is never used for selection or tuning.
- **Test set:** used once per model for reporting precision, recall, F1, ROC-AUC, PR curves, confusion matrices; accuracy is secondary. A majority-class baseline is reported to show why accuracy misleads.
- **Cost trade-off:** a false negative (missed failure) usually costs more than a false positive (extra inspection), hence the recall-weighted choice.

## Explainability methodology
- **Global:** model-native feature importance and a SHAP summary (beeswarm) on a sample of test rows.
- **Local:** SHAP contributions for one machine (bar plot). Random Forest/Decision Tree SHAP values are in probability units; boosting and logistic regression use log-odds. The app labels the units.
- **Counterfactual:** a grid search over small changes (default max 30% of a feature's observed range, up to 2-3 features at once), keeping Type fixed, values inside the training range, and a realistic process-air temperature gap. Every candidate is scored by the real trained model; the smallest changes that flip the class are shown. If none is found, the app says so. Results are hypothetical, not validated actions.
- SHAP and counterfactuals describe the **model**, not physical causation.

## Dashboard pages
Overview - Machine Risk Assessment - Model Performance - Explainability - Failure Type Analysis (historical only) - Maintenance Insights (rule-based, separate from the ML model).
Risk bands (Low/Medium/High) are adjustable presentation rules, not calibrated probabilities or safety limits.

## Limitations
- Synthetic dataset; no real time-series, so **no remaining-useful-life** estimates are offered.
- Rare failures (few positives in the test set) make metrics uncertain; no confidence intervals are computed.
- Class weighting/SMOTE shifts scores, so scores are not calibrated probabilities.
- Hyperparameters are modest defaults, not tuned. Failure-type prediction is not implemented.
- Maintenance reference rules come from the dataset's documentation (verify on the UCI page) and are not manufacturer limits.

## Future enhancements
Hyperparameter search, probability calibration, confidence intervals via repeated CV, multi-label failure-type classifier,
time-series data with RUL models, drift monitoring.

## Citation
Cite the UCI AI4I 2020 dataset as instructed on its repository page.
