# phase6-robustness-noise.py
# Purpose: Robustness test via controlled noise injection on numeric features.
# Output: outputs/phase6_robustness_noise.txt

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
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

NOISE_LEVELS = [0.00, 0.05, 0.10]  # 0%, 5%, 10%
RANDOM_SEED = 42

OUTPUT_PATH = os.path.join("outputs", "phase6_robustness_noise.txt")

# For tiny datasets, keep folds small but valid (>=2)
MAX_FOLDS = 5


# -----------------------------
# 2) Helpers
# -----------------------------
def find_data_path() -> str:
    for p in DATA_PATH_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "No dataset found. Tried:\n" + "\n".join(DATA_PATH_CANDIDATES)
    )


def find_target_column(df: pd.DataFrame) -> str:
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(
        f"Target column not found. Tried: {TARGET_CANDIDATES}. "
        f"Available columns: {list(df.columns)}"
    )


def build_models(numeric_features: list[str]) -> dict[str, Pipeline]:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    pre = ColumnTransformer(
        transformers=[("num", num_pipe, numeric_features)],
        remainder="drop",
    )

    models = {
        "LinearRegression": Pipeline(
            steps=[("pre", pre), ("model", LinearRegression())]
        ),
        # Keep tree constrained to reduce overfitting on tiny datasets
        "DecisionTree(max_depth=3)": Pipeline(
            steps=[("pre", pre), ("model", DecisionTreeRegressor(max_depth=3, random_state=RANDOM_SEED))]
        ),
    }
    return models


def inject_noise(X: pd.DataFrame, numeric_features: list[str], noise_level: float, rng: np.random.Generator) -> pd.DataFrame:
    """
    Multiplicative Gaussian noise:
        X_noisy = X * (1 + N(0, noise_level))
    Only applied to numeric features; non-numeric columns are ignored by design.
    """
    Xn = X.copy()
    if noise_level <= 0:
        return Xn

    for col in numeric_features:
        col_values = Xn[col].astype(float).to_numpy()
        noise = rng.normal(loc=0.0, scale=noise_level, size=col_values.shape)
        Xn[col] = col_values * (1.0 + noise)

    return Xn


def safe_r2_summary(scores: np.ndarray) -> str:
    # cross_val_score returns array; can contain nan if R2 undefined
    mean = np.nanmean(scores)
    std = np.nanstd(scores)
    return f"mean={mean:.4f}, std={std:.4f}, raw={np.array2string(scores, precision=4, separator=', ')}"


# -----------------------------
# 3) Main
# -----------------------------
def main() -> None:
    os.makedirs("outputs", exist_ok=True)

    data_path = find_data_path()
    df = pd.read_csv(data_path)

    target_col = find_target_column(df)
    y = df[target_col].astype(float)

    # Numeric feature selection: all numeric columns except target
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [c for c in numeric_features if c != target_col]

    if len(numeric_features) == 0:
        raise ValueError("No numeric feature columns found after excluding target.")

    X = df[numeric_features].copy()

    n = len(df)
    n_splits = min(MAX_FOLDS, n)  # cannot exceed n
    if n_splits < 2:
        raise ValueError(f"Dataset too small for CV. n={n} must be >= 2.")

    # For very small n, ensure each fold has at least 1 sample
    # KFold requires n_splits <= n
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    models = build_models(numeric_features)

    rng = np.random.default_rng(RANDOM_SEED)

    lines = []
    lines.append("PHASE 6.1 — ROBUSTNESS TEST (NOISE INJECTION)\n")
    lines.append(f"Dataset: {data_path}")
    lines.append(f"Rows (n): {n}")
    lines.append(f"Target: {target_col}")
    lines.append(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    lines.append(f"CV: KFold(n_splits={n_splits}, shuffle=True, seed={RANDOM_SEED})")
    lines.append("Metric: R2 (note: can be unstable/NaN for tiny datasets)")
    lines.append("-" * 70)

    for noise_level in NOISE_LEVELS:
        Xn = inject_noise(X, numeric_features, noise_level, rng)

        lines.append(f"\nNoise level: {int(noise_level*100)}%")
        for name, pipe in models.items():
            scores = cross_val_score(pipe, Xn, y, cv=kf, scoring="r2")
            lines.append(f"  {name}: {safe_r2_summary(scores)}")

    lines.append("\nEngineering note:")
    lines.append(
        "  If small noise causes large swings in mean/std, the pipeline is not robust.\n"
        "  Robustness matters more than a single high score on tiny datasets."
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
