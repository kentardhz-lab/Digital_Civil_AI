# phase2-ml-baseline.py
# Phase 2 — ML Baseline (Regression)
# Goal: Predict Load_kN from Element (categorical) + Length_m (numeric)

import warnings
import pandas as pd

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


# Keep console output clean (R2 warning happens with extremely small test sets)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def main():
    # 1) Load data
    df = pd.read_csv("data/elements.csv")

    # 2) Basic validation
    required_cols = ["Element", "Length_m", "Load_kN"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    # Drop rows with missing values in required columns (defensive)
    df = df.dropna(subset=required_cols).copy()

    # 3) Features / Target
    X = df[["Element", "Length_m"]]
    y = df["Load_kN"]

    # Defensive check: need at least 3 rows to split into train/test
    if len(df) < 3:
        raise ValueError(
            f"Dataset too small for train/test split. Need >= 3 rows, got {len(df)}."
        )

    # 4) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # 5) Preprocess + Model
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Element"]),
            ("num", "passthrough", ["Length_m"]),
        ]
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("reg", LinearRegression()),
        ]
    )

    # 6) Train
    model.fit(X_train, y_train)

    # 7) Predict
    y_pred = model.predict(X_test)

    # 8) Metrics (always safe)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    # R2 is only meaningful with >= 2 test samples
    r2_text = "skipped (need >= 2 test samples)"
    if len(y_test) >= 2:
        r2 = r2_score(y_test, y_pred)
        r2_text = f"{r2:.3f}"

    # 9) Report
    print("=== Phase 2: ML Baseline (Linear Regression) ===")
    print(f"Rows: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"MAE : {mae:.3f} kN")
    print(f"RMSE: {rmse:.3f} kN")
    print(f"R2  : {r2_text}")

    # 10) Quick sanity table (worst errors first)
    out = X_test.copy()
    out["Load_kN_true"] = y_test.values
    out["Load_kN_pred"] = y_pred
    out["abs_error"] = (out["Load_kN_true"] - out["Load_kN_pred"]).abs()

    print("\nSample predictions (worst 10 errors):")
    print(out.sort_values("abs_error", ascending=False).head(10))


if __name__ == "__main__":
    main()
