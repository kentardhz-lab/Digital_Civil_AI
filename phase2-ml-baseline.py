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
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


# Keep console output clean (R2 warning happens with extremely small test sets)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def safe_r2_text(y_true, y_pred) -> str:
    """Return R2 formatted text only if it is statistically defined."""
    if len(y_true) >= 2:
        return f"{r2_score(y_true, y_pred):.3f}"
    return "skipped (need >= 2 test samples)"


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

    # Defensive check: need at least 3 rows to split into train/test
    if len(df) < 3:
        raise ValueError(
            f"Dataset too small for train/test split. Need >= 3 rows, got {len(df)}."
        )

    # 3) Features / Target
    X = df[["Element", "Length_m"]]
    y = df["Load_kN"]

    # 4) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # 5) Preprocessing (shared)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Element"]),
            ("num", "passthrough", ["Length_m"]),
        ]
    )

    # ----------------------------
    # Model 1: Linear Regression
    # ----------------------------
    lin_model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("reg", LinearRegression()),
        ]
    )

    lin_model.fit(X_train, y_train)
    y_pred_lin = lin_model.predict(X_test)

    mae_lin = mean_absolute_error(y_test, y_pred_lin)
    rmse_lin = root_mean_squared_error(y_test, y_pred_lin)
    r2_lin_text = safe_r2_text(y_test, y_pred_lin)

    # ----------------------------
    # Model 2: Decision Tree
    # ----------------------------
    tree_model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("reg", DecisionTreeRegressor(random_state=42, max_depth=3)),
        ]
    )

    tree_model.fit(X_train, y_train)
    y_pred_tree = tree_model.predict(X_test)

    mae_tree = mean_absolute_error(y_test, y_pred_tree)
    rmse_tree = root_mean_squared_error(y_test, y_pred_tree)
    r2_tree_text = safe_r2_text(y_test, y_pred_tree)

    # 6) Report
    print("=== Phase 2: ML Models (Regression) ===")
    print(f"Rows: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")

    print("\n--- Linear Regression ---")
    print(f"MAE : {mae_lin:.3f} kN")
    print(f"RMSE: {rmse_lin:.3f} kN")
    print(f"R2  : {r2_lin_text}")

    print("\n--- Decision Tree (max_depth=3) ---")
    print(f"MAE : {mae_tree:.3f} kN")
    print(f"RMSE: {rmse_tree:.3f} kN")
    print(f"R2  : {r2_tree_text}")

    # 7) Sanity table (based on Linear predictions)
    out = X_test.copy()
    out["Load_kN_true"] = y_test.values
    out["Load_kN_pred_lin"] = y_pred_lin
    out["abs_error_lin"] = (out["Load_kN_true"] - out["Load_kN_pred_lin"]).abs()

    print("\nSample predictions (Linear, worst errors first):")
    print(out.sort_values("abs_error_lin", ascending=False).head(10))


if __name__ == "__main__":
    main()
