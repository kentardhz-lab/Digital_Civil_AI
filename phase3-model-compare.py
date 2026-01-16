# Phase 3 — Model comparison (baseline vs engineered features)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("data/elements.csv")

# Target
y = df["Load_kN"]

# --- Baseline features ---
X_base = df[["Length_m"]]

# --- Engineered features ---
X_feat = df[["Length_m"]].copy()
X_feat["inv_length"] = 1 / df["Length_m"]

# Train / test split (same for both)
Xb_tr, Xb_te, y_tr, y_te = train_test_split(X_base, y, test_size=0.4, random_state=42)
Xf_tr, Xf_te, _, _ = train_test_split(X_feat, y, test_size=0.4, random_state=42)

# Models
m_base = LinearRegression()
m_feat = LinearRegression()

m_base.fit(Xb_tr, y_tr)
m_feat.fit(Xf_tr, y_tr)

# Predictions
pb = m_base.predict(Xb_te)
pf = m_feat.predict(Xf_te)

def report(name, y_true, y_pred):
    print(f"\n{name}")
    print("MAE :", mean_absolute_error(y_true, y_pred))
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    print("RMSE:", rmse)
    print("R2  :", r2_score(y_true, y_pred))

report("Baseline", y_te, pb)
report("With engineered features", y_te, pf)
