import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
DATA = Path("datasets/adult.csv")
DATA.parent.mkdir(exist_ok=True)

columns = [
    "age","workclass","fnlwgt","education","education_num","marital_status",
    "occupation","relationship","race","sex","capital_gain","capital_loss",
    "hours_per_week","native_country","income"
]

if not DATA.exists():
    df = pd.read_csv(URL, names=columns, skipinitialspace=True, na_values="?")
    df.to_csv(DATA, index=False)
else:
    df = pd.read_csv(DATA)

print(df.head())
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# Feature engineering
df["capital_net"] = df["capital_gain"] - df["capital_loss"]
df["is_married"] = df["marital_status"].str.contains("Married", na=False).astype(int)
df["age_group"] = pd.cut(
    df["age"], bins=[0,25,35,50,65,100],
    labels=["<=25","26-35","36-50","51-65","66+"]
)
df["work_hours_group"] = pd.cut(
    df["hours_per_week"], bins=[0,20,40,60,100],
    labels=["Low","Normal","High","Very High"]
)

print("\nIncome distribution:\n", df["income"].value_counts(normalize=True))
print("\nAverage hours by income:\n", df.groupby("income")["hours_per_week"].mean())

plt.figure(figsize=(8,5))
sns.countplot(data=df, x="education", hue="income")
plt.xticks(rotation=70)
plt.title("Education vs Income")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
sns.boxplot(data=df, x="income", y="age")
plt.title("Age vs Income")
plt.tight_layout()
plt.show()

df.to_csv("datasets/adult_feature_engineered.csv", index=False)
print("\nSaved datasets/adult_feature_engineered.csv")
