import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

housing = fetch_california_housing(as_frame=True)
df = housing.frame
df.to_csv("datasets/california_housing.csv", index=False)

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": Pipeline([
        ("scale", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Ridge": Pipeline([
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),
    "Lasso": Pipeline([
        ("scale", StandardScaler()),
        ("model", Lasso(alpha=0.001, max_iter=10000))
    ])
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\n{name}")
    print("MAE :", round(mean_absolute_error(y_test, pred), 4))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 4))
    print("R2  :", round(r2_score(y_test, pred), 4))

print("\nDataset:", df.shape)
print("Saved datasets/california_housing.csv")
