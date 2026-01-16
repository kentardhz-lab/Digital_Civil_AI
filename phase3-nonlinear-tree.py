import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("data/elements.csv")

# Target
y = df["Load_kN"]

# Features (non-linear friendly)
X = df[["Length_m"]].copy()
X["inv_length"] = 1 / df["Length_m"]

# Train / test split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# Non-linear model
model = DecisionTreeRegressor(
    max_depth=3,
    random_state=42
)

model.fit(X_tr, y_tr)

# Predictions
yp = model.predict(X_te)

# Metrics
print("Decision Tree")
print("MAE :", mean_absolute_error(y_te, yp))
print("RMSE:", mean_squared_error(y_te, yp) ** 0.5)
print("R2  :", r2_score(y_te, yp))
