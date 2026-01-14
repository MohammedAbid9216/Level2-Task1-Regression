# Level 2 - Task 1: Regression
# Using company-provided dataset: house Prediction Data Set.csv

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1) Load dataset (space-separated, no header)
df = pd.read_csv("house Prediction Data Set.csv", sep=r"\s+", header=None)

# 2) Assign column names (Boston Housing format)
df.columns = [
    "CRIM","ZN","INDUS","CHAS","NOX","RM","AGE",
    "DIS","RAD","TAX","PTRATIO","B","LSTAT","PRICE"
]

print("Dataset loaded. Columns:")
print(df.columns.tolist())
print(df.head())

# 3) Split features & target
X = df.drop("PRICE", axis=1)
y = df["PRICE"]

# 4) Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5) Models to compare
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# 6) Train & evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\nModel: {name}")
    print("MSE:", mse)
    print("R2:", r2)
