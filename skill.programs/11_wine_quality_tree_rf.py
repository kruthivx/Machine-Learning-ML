import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
DATA = Path("datasets/winequality-red.csv")
DATA.parent.mkdir(exist_ok=True)

if not DATA.exists():
    df = pd.read_csv(URL, sep=";")
    df.to_csv(DATA, index=False)
else:
    df = pd.read_csv(DATA)

# Convert quality score to binary quality class.
# 1 = good (quality >= 7), 0 = not good.
df["good_quality"] = (df["quality"] >= 7).astype(int)

X = df.drop(columns=["quality","good_quality"])
y = df["good_quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

print("\nDataset shape:", df.shape)
print("Saved datasets/winequality-red.csv")
