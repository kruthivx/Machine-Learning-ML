import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
DATA = Path("datasets/auto_mpg.csv")
DATA.parent.mkdir(exist_ok=True)

columns = [
    "mpg","cylinders","displacement","horsepower","weight",
    "acceleration","model_year","origin","car_name"
]

if not DATA.exists():
    df = pd.read_csv(
        URL, sep=r"\s+", header=None, names=columns,
        na_values="?", engine="python"
    )
    df.to_csv(DATA, index=False)
else:
    df = pd.read_csv(DATA)

# car_name is an identifier/high-cardinality text field; drop it.
df = df.drop(columns=["car_name"])

X = df.drop(columns=["mpg"])
y = df["mpg"]

# Treat origin as categorical; remaining variables as numeric.
X["origin"] = X["origin"].astype(str)

numeric = ["cylinders","displacement","horsepower","weight","acceleration","model_year"]
categorical = ["origin"]

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

model = Pipeline([
    ("preprocess", preprocess),
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("MAE :", round(mean_absolute_error(y_test, pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 4))
print("R2  :", round(r2_score(y_test, pred), 4))
print("Dataset shape:", df.shape)
print("Saved datasets/auto_mpg.csv")
