# Breast Cancer Detection Using Machine Learning

An educational binary-classification project using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

> **Medical disclaimer:** This is an educational research prototype, not a clinically validated medical device. It must not be used to diagnose, rule out, or guide treatment for breast cancer. The Streamlit form requires 30 dataset-specific numerical measurements; it does not accept medical images.

## Dataset
- 569 observations
- 30 numeric features
- Target: `diagnosis` (`B` = Benign / class 0, `M` = Malignant / class 1)
- The supplied CSV is in `data/data.csv`.
- Original dataset source: [UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

## Project workflow
1. Dataset inspection and cleaning
2. Exploratory data analysis
3. Train/test split (80/20, stratified)
4. Standardization (fit on training data only)
5. Compare six classifiers and tune hyperparameters (see notebook)
6. Evaluate with precision, recall, F1 and confusion matrix
7. Explainability with SHAP (see notebook)
8. Streamlit demonstration app

## Model included
The bundled model is a Logistic Regression classifier (`C=1`, `solver=liblinear`) trained with a fixed 80/20 stratified split (`random_state=42`). The scaler was fit on training data only. Class 0 is Benign; class 1 is Malignant.

Reproduced metrics from the bundled training run are in `models/training_metrics.json`. Metrics can vary if the dataset or software versions change. Do not treat the reported metrics as evidence of clinical performance.

## Run locally
Python 3.10–3.12 is recommended.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Re-train the bundled model
Ensure `data/data.csv` is present, then run:

```bash
python train_model.py
```

## Files
- `app.py` — Streamlit demonstration app
- `train_model.py` — reproducible model training script
- `notebooks/Breast_Cancer_Detection_ML.ipynb` — end-to-end Colab/Jupyter notebook
- `data/data.csv` — dataset CSV supplied for this project
- `models/` — model, scaler, feature schema, defaults, and metrics

## Limitations
- The model predicts from precomputed cell-nuclei measurements, not images.
- Input values must follow the WDBC feature definitions and units.
- The dataset is relatively small and is not a substitute for external clinical validation.
- Model probability outputs are not clinical risk estimates.
