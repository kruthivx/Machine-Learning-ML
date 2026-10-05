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

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))
print("\nFeature importance:")
for name, value in zip(X.columns, model.feature_importances_):
    print(f"{name:10s} {value:.4f}")
