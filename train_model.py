import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).parent
df = pd.read_csv(ROOT / "data" / "data.csv")
df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})
if df["diagnosis"].isna().any():
    raise ValueError("Diagnosis must contain only B and M labels.")

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"].astype(int)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

model = LogisticRegression(C=1, solver="liblinear", max_iter=3000)
model.fit(X_train, y_train)
pred = model.predict(X_test)

out = ROOT / "models"
out.mkdir(exist_ok=True)
joblib.dump(model, out / "breast_cancer_model.pkl")
joblib.dump(scaler, out / "scaler.pkl")
(out / "feature_names.json").write_text(json.dumps(X.columns.tolist(), indent=2))
(out / "feature_medians.json").write_text(json.dumps(
    {k: float(v) for k, v in X_train_raw.median().to_dict().items()}, indent=2
))
metrics = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision_malignant": float(precision_score(y_test, pred, zero_division=0)),
    "recall_malignant": float(recall_score(y_test, pred, zero_division=0)),
    "f1_malignant": float(f1_score(y_test, pred, zero_division=0)),
    "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist()
}
(out / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
