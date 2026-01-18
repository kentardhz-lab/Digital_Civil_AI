# phase6-feature-perturbation.py
# Purpose: Feature perturbation test (sensitivity / dependency) WITHOUT CV metrics.
# Output: outputs/phase6_feature_perturbation.txt

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


# -----------------------------
# 1) Config
# -----------------------------
DATA_PATH_CANDIDATES = [
    os.path.join("data", "elements.csv"),
    os.path.join("data_generated", "elements_combined.csv"),
    "elements_combined.csv",
    "elements.csv",
]

TARGET_CANDIDATES = ["Load_kN", "Load", "load", "Target", "target"]

PERTURB_LEVELS = [-0.10, -0.05, 0.00, 0.05, 0.10]  # -10%, -5%, 0, +5%, +10%
RANDOM_SEED = 42

OUTPUT_PATH = os.path.join("outputs", "phase6_feature_perturbation.txt")


# -----------------------------
# 2) Helpers
# -----------------------------
def find_data_path() -> str:
    for p in DATA_PATH_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No dataset found. Tried:\n" + "\n".join(DATA_PATH_CANDIDATES))


def find_target_column(df: pd.DataFrame) -> str:
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(
        f"Target column not found. Tried: {TARGET_CANDIDATES}. "
        f"Available columns: {list(df.columns)}"
    )


def build_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", num_pipe, numeric_features)],
        remainder="drop",
    )


def build_models(pre: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "LinearRegression": Pipeline(steps=[("pre", pre), ("model", LinearRegression())]),
        "DecisionTree(max_depth=3)": Pipeline(
            steps=[("pre", pre), ("model", DecisionTreeRegressor(max_depth=3, random_state=RANDOM_SEED))]
        ),
    }


def perturb_feature(X: pd.DataFrame, feature: str, factor: float) -> pd.DataFrame:
    Xp = X.copy()
    Xp[feature] = Xp[feature].astype(float) * (1.0 + factor)
    return Xp


def pct_change(new: np.ndarray, base: np.ndarray) -> np.ndarray:
    # percent change, safe for zeros (if base is 0 -> NaN)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (new - base) / base * 100.0


# -----------------------------
# 3) Main
# -----------------------------
def main() -> None:
    os.makedirs("outputs", exist_ok=True)

    data_path = find_data_path()
    df = pd.read_csv(data_path)

    target_col = find_target_column(df)
    y = df[target_col].astype(float)

    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [c for c in numeric_features if c != target_col]

    if len(numeric_features) == 0:
        raise ValueError("No numeric feature columns found after excluding target.")

    X = df[numeric_features].copy()

    # If only one numeric feature exists, test that. Otherwise test the first one (or you can choose later).
    feature_to_test = numeric_features[0]

    pre = build_preprocessor(numeric_features)
    models = build_models(pre)

    lines = []
    lines.append("PHASE 6.2 — FEATURE PERTURBATION TEST (NO CV, NO METRICS)\n")
    lines.append(f"Dataset: {data_path}")
    lines.append(f"Rows (n): {len(df)}")
    lines.append(f"Target: {target_col}")
    lines.append(f"Numeric features: {numeric_features}")
    lines.append(f"Feature under test: {feature_to_test}")
    lines.append(f"Perturb levels: {PERTURB_LEVELS}")
    lines.append("-" * 70)

    for name, pipe in models.items():
        # Fit once on full data (engineering-style behavior check, not generalization claim)
        pipe.fit(X, y)

        base_pred = pipe.predict(X)

        lines.append(f"\nModel: {name}")
        lines.append("Baseline predictions (first 10): " + np.array2string(base_pred[:10], precision=4, separator=", "))

        for p in PERTURB_LEVELS:
            Xp = perturb_feature(X, feature_to_test, p)
            pred = pipe.predict(Xp)

            delta = pred - base_pred
            delta_pct = pct_change(pred, base_pred)

            # Summaries
            lines.append(f"  Perturb {int(p*100)}%:")
            lines.append(f"    abs_delta: mean={np.nanmean(delta):.4f}, std={np.nanstd(delta):.4f}, min={np.nanmin(delta):.4f}, max={np.nanmax(delta):.4f}")
            lines.append(f"    pct_delta: mean={np.nanmean(delta_pct):.2f}%, std={np.nanstd(delta_pct):.2f}%")

    lines.append("\nEngineering interpretation guide:")
    lines.append("  - If +/-10% in input causes extreme output swings, the model is highly sensitive (risk).")
    lines.append("  - Trees may show step-like changes; linear models tend to change smoothly.")
    lines.append("  - This is a behavior test only; it does NOT claim real-world generalization with n=4.")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
