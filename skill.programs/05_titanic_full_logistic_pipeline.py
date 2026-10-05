import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
DATA = Path("datasets/titanic.csv")
DATA.parent.mkdir(exist_ok=True)

if not DATA.exists():
    df = pd.read_csv(URL)
    df.to_csv(DATA, index=False)
else:
    df = pd.read_csv(DATA)

X = df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
y = df["Survived"]

numeric = ["Pclass","Age","SibSp","Parch","Fare"]
categorical = ["Sex","Embarked"]

prep = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

model = Pipeline([
    ("preprocessing", prep),
    ("classifier", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall   :", recall_score(y_test, pred))
print("F1       :", f1_score(y_test, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))

# Example prediction
sample = pd.DataFrame([{
    "Pclass": 1, "Sex": "female", "Age": 25,
    "SibSp": 0, "Parch": 0, "Fare": 80, "Embarked": "C"
}])
print("\nExample prediction:", model.predict(sample)[0])
print("Survival probability:", round(model.predict_proba(sample)[0,1], 4))
