import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path


# Load data
df = pd.read_csv("data/elements.csv")

# Target and features
X = df[["Length_m"]]
y = df["Load_kN"]

# Cross-validation strategy
kf = KFold(n_splits=2, shuffle=True, random_state=42)
print("DEBUG: n_splits =", kf.get_n_splits())

models = {
    "LinearRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "DecisionTree": DecisionTreeRegressor(
        max_depth=3,
        random_state=42
    )
}

print("Cross-validation results (R2):\n")

for name, model in models.items():
    scores = cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring="r2"
    )

    print(f"{name}:")
    print(f"  Mean R2: {scores.mean():.3f}")
    print(f"  Std  R2: {scores.std():.3f}\n")

    out_path = Path("outputs/phase4_cv_metrics.txt")
out_path.parent.mkdir(parents=True, exist_ok=True)

lines = []
lines.append("Phase 4 — Cross-Validation (R2)\n")
lines.append("CV strategy: KFold(n_splits=2, shuffle=True, random_state=42)\n")
lines.append("Features: Length_m\n")
lines.append("Target: Load_kN\n\n")

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
    lines.append(f"{name}:\n")
    lines.append(f"- Mean R2: {scores.mean():.3f}\n")
    lines.append(f"- Std  R2: {scores.std():.3f}\n\n")

lines.append("Notes:\n")
lines.append("- Dataset is extremely small; R2 is unstable and can be strongly negative.\n")
lines.append("- n_splits was chosen to keep at least 2 samples in each test fold.\n")

out_path.write_text("".join(lines), encoding="utf-8")
print(f"\nSaved: {out_path}")

