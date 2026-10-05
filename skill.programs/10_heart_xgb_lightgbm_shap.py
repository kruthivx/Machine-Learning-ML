import pandas as pd
from pathlib import Path

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
DATA = Path("datasets/heart_disease.csv")
DATA.parent.mkdir(exist_ok=True)

columns = [
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach",
    "exang","oldpeak","slope","ca","thal","target"
]

if not DATA.exists():
    df = pd.read_csv(URL, header=None, names=columns, na_values="?")
    df.to_csv(DATA, index=False)
else:
    df = pd.read_csv(DATA)

# UCI target: 0 = no disease; 1-4 = disease levels.
df["target"] = (df["target"] > 0).astype(int)
df = df.dropna().reset_index(drop=True)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {}

xgb = XGBClassifier(
    n_estimators=250, max_depth=4, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85,
    eval_metric="logloss", random_state=42
)
models["XGBoost"] = xgb

try:
    from lightgbm import LGBMClassifier
    lgbm = LGBMClassifier(
        n_estimators=250, learning_rate=0.05,
        max_depth=-1, random_state=42, verbosity=-1
    )
    models["LightGBM"] = lgbm
except Exception as e:
    print("LightGBM unavailable:", e)

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\n{name} Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))

# SHAP explanation for XGBoost
try:
    import shap
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)
    print("\nSHAP computed successfully.")
    print("Mean absolute SHAP importance:")
    if isinstance(shap_values, list):
        vals = np.abs(shap_values[1]).mean(axis=0)
    else:
        vals = np.abs(shap_values).mean(axis=0)
    print(
        pd.Series(vals, index=X.columns)
        .sort_values(ascending=False)
    )
except Exception as e:
    print("\nSHAP explanation could not be computed:", e)
    print("Install/upgrade shap if necessary: pip install -U shap")
