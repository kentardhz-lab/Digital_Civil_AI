# phase6-scenarios.py
# Purpose: Scenario framing (Best / Nominal / Worst) using simple feature sweeps.
# Output: outputs/phase6_scenarios.txt

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

RANDOM_SEED = 42
OUTPUT_PATH = os.path.join("outputs", "phase6_scenarios.txt")

# Percentile-based scenario points (robust for small datasets)
SCENARIOS = {
    "BestCase(P10)": 10,
    "Nominal(P50)": 50,
    "WorstCase(P90)": 90,
}


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


def make_scenario_inputs(
    X: pd.DataFrame,
    feature: str,
    scenario_percentiles: dict[str, int],
) -> dict[str, pd.DataFrame]:
    """
    Creates scenario input tables by setting 'feature' to percentile values,
    while keeping all other features at their median (engineering 'nominal' baseline).
    """
    med = X.median(numeric_only=True)
    base_row = med.to_frame().T  # single-row DataFrame

    values = np.percentile(X[feature].astype(float).to_numpy(), list(scenario_percentiles.values()))
    scenario_inputs = {}

    for (scenario_name, _), v in zip(scenario_percentiles.items(), values):
        row = base_row.copy()
        row[feature] = float(v)
        scenario_inputs[scenario_name] = row

    return scenario_inputs


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

    # Choose the main driver feature: if Length_m exists, prefer it; otherwise take the first numeric feature.
    feature_main = "Length_m" if "Length_m" in numeric_features else numeric_features[0]

    pre = build_preprocessor(numeric_features)
    models = build_models(pre)

    scenario_inputs = make_scenario_inputs(X, feature_main, SCENARIOS)

    lines = []
    lines.append("PHASE 6.3 — SCENARIO FRAMING (BEST / NOMINAL / WORST)\n")
    lines.append(f"Dataset: {data_path}")
    lines.append(f"Rows (n): {len(df)}")
    lines.append(f"Target: {target_col}")
    lines.append(f"Numeric features: {numeric_features}")
    lines.append(f"Main scenario driver: {feature_main}")
    lines.append(f"Scenario definition (percentiles): {SCENARIOS}")
    lines.append("-" * 70)

    # Fit once on full data (behavioral decision-support view, not a generalization claim)
    for name, pipe in models.items():
        pipe.fit(X, y)

        lines.append(f"\nModel: {name}")
        lines.append("Scenario inputs (single-row; other features fixed at median):")
        for sc_name, sc_X in scenario_inputs.items():
            lines.append(f"  {sc_name}: {sc_X.to_dict(orient='records')[0]}")

        lines.append("Scenario predictions:")
        preds = {}
        for sc_name, sc_X in scenario_inputs.items():
            pred = float(pipe.predict(sc_X)[0])
            preds[sc_name] = pred
            lines.append(f"  {sc_name}: pred={pred:.4f}")

        # Spread summary
        pred_vals = np.array(list(preds.values()), dtype=float)
        spread = float(np.max(pred_vals) - np.min(pred_vals))
        lines.append(f"Spread (max - min): {spread:.4f}")

    lines.append("\nEngineering interpretation guide:")
    lines.append("  - You now have a range, not a single point estimate.")
    lines.append("  - Compare spreads between models: bigger spread => higher sensitivity/uncertainty.")
    lines.append("  - Use this to support decisions (risk buffers, contingencies), not as a 'truth' number.")
    lines.append("  - With n=4, treat results as illustrative decision-support logic only.")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
