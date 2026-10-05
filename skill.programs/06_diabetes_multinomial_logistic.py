import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = load_diabetes(as_frame=True)
df = data.frame.copy()

# Educational target engineering: 3 severity groups from continuous progression score.
q1 = df["target"].quantile(1/3)
q2 = df["target"].quantile(2/3)

df["severity"] = pd.cut(
    df["target"],
    bins=[-np.inf, q1, q2, np.inf],
    labels=["Low", "Medium", "High"]
)

df.to_csv("datasets/diabetes_severity.csv", index=False)

X = df.drop(columns=["target","severity"])
y = df["severity"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("scale", StandardScaler()),
    ("logistic", LogisticRegression(
        multi_class="multinomial", max_iter=2000
    ))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))

sample = X_test.iloc[[0]]
print("\nSample actual:", y_test.iloc[0])
print("Sample predicted:", model.predict(sample)[0])
print("Probabilities:", model.predict_proba(sample)[0])
print("\nSeverity cutoffs:", q1, q2)
print("Saved datasets/diabetes_severity.csv")
