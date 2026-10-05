import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

DATA = Path("datasets/titanic.csv")
CLEAN = Path("datasets/titanic_cleaned.csv")
URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

if not DATA.exists():
    df = pd.read_csv(URL)
    DATA.parent.mkdir(exist_ok=True)
    df.to_csv(DATA, index=False)
else:
    df = pd.read_csv(DATA)

# Keep useful predictive columns.
X = df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]].copy()
y = df["Survived"]

numeric = ["Pclass","Age","SibSp","Parch","Fare"]
categorical = ["Sex","Embarked"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_clean = preprocessor.fit_transform(X_train)
X_test_clean = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()
train_clean = pd.DataFrame(X_train_clean, columns=feature_names)
train_clean["Survived"] = y_train.to_numpy()

test_clean = pd.DataFrame(X_test_clean, columns=feature_names)
test_clean["Survived"] = y_test.to_numpy()

train_clean.to_csv("datasets/titanic_train_cleaned.csv", index=False)
test_clean.to_csv("datasets/titanic_test_cleaned.csv", index=False)

# A complete cleaned non-scaled CSV is also useful for inspection.
inspection = X.copy()
inspection["Age"] = inspection["Age"].fillna(inspection["Age"].median())
inspection["Fare"] = inspection["Fare"].fillna(inspection["Fare"].median())
inspection["Embarked"] = inspection["Embarked"].fillna(inspection["Embarked"].mode()[0])
inspection.to_csv(CLEAN, index=False)

print("Original shape:", df.shape)
print("Training cleaned shape:", train_clean.shape)
print("Test cleaned shape:", test_clean.shape)
print("\nSaved:")
print("datasets/titanic_cleaned.csv")
print("datasets/titanic_train_cleaned.csv")
print("datasets/titanic_test_cleaned.csv")
print("\nColumns:")
print(feature_names.tolist())
