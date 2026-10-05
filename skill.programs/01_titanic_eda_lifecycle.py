import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA = Path("datasets/titanic.csv")
URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
DATA.parent.mkdir(exist_ok=True)

if not DATA.exists():
    df = pd.read_csv(URL)
    df.to_csv(DATA, index=False)
else:
    df = pd.read_csv(DATA)

print("\n=== 1. DATA UNDERSTANDING ===")
print(df.head())
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nStatistics:\n", df.describe(include="all").T)

print("\n=== 2. EDA ===")
print("\nSurvival rate:\n", df["Survived"].value_counts(normalize=True))
print("\nSurvival by sex:\n", df.groupby("Sex")["Survived"].mean())
print("\nSurvival by class:\n", df.groupby("Pclass")["Survived"].mean())

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Survived")
plt.title("Titanic Survival Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
sns.barplot(data=df, x="Sex", y="Survived", hue="Pclass")
plt.title("Survival Rate by Sex and Passenger Class")
plt.tight_layout()
plt.show()

print("""
=== 3. ML LIFECYCLE MAP ===
Problem -> Data Collection -> Data Understanding -> Cleaning ->
EDA -> Feature Engineering -> Train/Test Split -> Preprocessing ->
Model Training -> Evaluation -> Interpretation -> Deployment/Monitoring

For Titanic, target = Survived.
""")
