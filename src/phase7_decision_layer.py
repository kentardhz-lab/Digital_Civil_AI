from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------
DATA_PATH_CANDIDATES = [
    os.path.join("data_generated", "elements_combined.csv"),
    os.path.join("data", "elements.csv"),
]

OUTPUT_DIR = "outputs"
OUT_CSV = os.path.join(OUTPUT_DIR, "phase7_decision_table.csv")
OUT_TXT = os.path.join(OUTPUT_DIR, "phase7_engineering_verdict.txt")


# -----------------------------
# Utilities
# -----------------------------
def _first_existing_path(paths: List[str]) -> str:
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "No input CSV found. Tried:\n- " + "\n- ".join(paths)
    )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


@dataclass
class DecisionRule:
    name: str
    # thresholds are expressed as fractions (e.g., 0.10 = 10%)
    risk_worst_over_nominal: float = 0.15
    border_worst_over_nominal: float = 0.08
    risk_sensitivity_score: float = 0.60
    border_sensitivity_score: float = 0.35


def _pick_target_column(df: pd.DataFrame) -> str:
    # Prefer common phase-2/3 target naming
    candidates = [
        "Load_per_meter",
        "Load_per_m",
        "Load_kN_per_m",
        "Load_kN",
        "load",
        "target",
        "y",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    # fallback: last numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found to use as target.")
    return numeric_cols[-1]


def _pick_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c != target_col]
    if not features:
        raise ValueError("No numeric feature columns found.")
    return features


def _scenario_ranges(y: np.ndarray) -> Tuple[float, float, float]:
    # robust percentile-based ranges (best/nominal/worst)
    y = y[~np.isnan(y)]
    if len(y) == 0:
        return np.nan, np.nan, np.nan
    best = float(np.percentile(y, 10))
    nominal = float(np.percentile(y, 50))
    worst = float(np.percentile(y, 90))
    return best, nominal, worst


def _sensitivity_score(df: pd.DataFrame, features: List[str], target: str) -> Tuple[float, Dict[str, float]]:
    """
    Simple, model-free sensitivity proxy:
      - compute absolute Pearson correlation between each feature and target
      - normalize into 0..1 and aggregate via mean
    """
    scores: Dict[str, float] = {}
    y = df[target].astype(float)
    for f in features:
        x = df[f].astype(float)
        if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
            scores[f] = np.nan
            continue
        corr = np.corrcoef(x.fillna(x.median()), y.fillna(y.median()))[0, 1]
        scores[f] = float(abs(corr))

    valid = [v for v in scores.values() if not np.isnan(v)]
    if not valid:
        return np.nan, scores

    # normalize (cap at 1.0 already), aggregate
    agg = float(np.mean(valid))
    return agg, scores


def _confidence_level(n_rows: int, worst_over_nominal: float, sens_score: float) -> str:
    # Conservative: tiny data => low confidence
    if n_rows <= 10:
        return "Low"
    if worst_over_nominal >= 0.15 or (not np.isnan(sens_score) and sens_score >= 0.60):
        return "Low"
    if worst_over_nominal >= 0.08 or (not np.isnan(sens_score) and sens_score >= 0.35):
        return "Medium"
    return "High"


def _verdict(rule: DecisionRule, worst_over_nominal: float, sens_score: float) -> str:
    # Decision based on volatility and sensitivity
    if np.isnan(worst_over_nominal):
        return "Borderline"

    if (worst_over_nominal >= rule.risk_worst_over_nominal) or (
        not np.isnan(sens_score) and sens_score >= rule.risk_sensitivity_score
    ):
        return "Risky"

    if (worst_over_nominal >= rule.border_worst_over_nominal) or (
        not np.isnan(sens_score) and sens_score >= rule.border_sensitivity_score
    ):
        return "Borderline"

    return "Safe"


def _recommended_action(v: str) -> str:
    if v == "Safe":
        return "Proceed with standard checks; monitor within expected tolerances."
    if v == "Borderline":
        return "Request additional data / checks; use conservative assumptions; run scenario review."
    return "Do not rely on point estimates; require redesign / mitigation; prioritize worst-case planning."


def main() -> None:
    in_csv = _first_existing_path(DATA_PATH_CANDIDATES)
    _ensure_dir(OUTPUT_DIR)

    df = pd.read_csv(in_csv)

    # Ensure numeric conversion where possible
    for c in df.columns:
        if df[c].dtype == object:
            # attempt numeric parse for objects that look numeric
            parsed = pd.to_numeric(df[c], errors="ignore")
            df[c] = parsed

    target = _pick_target_column(df)
    features = _pick_feature_columns(df, target)

    y = df[target].astype(float).to_numpy()
    best, nominal, worst = _scenario_ranges(y)

    worst_over_nominal = np.nan
    if not np.isnan(worst) and not np.isnan(nominal) and nominal != 0:
        worst_over_nominal = float((worst - nominal) / abs(nominal))

    sens_score, sens_map = _sensitivity_score(df, features, target)

    rule = DecisionRule(name="phase7_default")
    verdict = _verdict(rule, worst_over_nominal, sens_score)
    confidence = _confidence_level(len(df), worst_over_nominal, sens_score)
    action = _recommended_action(verdict)

    # Build decision table (single-row, portfolio-friendly)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_csv": in_csv,
        "n_rows": int(len(df)),
        "target": target,
        "features_used": ", ".join(features),
        "best_p10": best,
        "nominal_p50": nominal,
        "worst_p90": worst,
        "worst_over_nominal_frac": worst_over_nominal,
        "sensitivity_score_mean_abs_corr": sens_score,
        "verdict": verdict,
        "confidence": confidence,
        "recommended_action": action,
    }
    out_df = pd.DataFrame([row])
    out_df.to_csv(OUT_CSV, index=False)

    # Write verdict text
    lines = [
        "Phase 7 — Engineering Decision Verdict",
        "=====================================",
        f"Timestamp: {row['timestamp']}",
        f"Input: {in_csv}",
        "",
        f"Data size: n={row['n_rows']}",
        f"Target: {target}",
        f"Features: {', '.join(features)}",
        "",
        "Scenario ranges (percentiles):",
        f"  Best (P10):    {best:.6g}" if not np.isnan(best) else "  Best (P10):    NaN",
        f"  Nominal (P50): {nominal:.6g}" if not np.isnan(nominal) else "  Nominal (P50): NaN",
        f"  Worst (P90):   {worst:.6g}" if not np.isnan(worst) else "  Worst (P90):   NaN",
        "",
        f"Worst-over-nominal (fraction): {worst_over_nominal:.6g}" if not np.isnan(worst_over_nominal) else "Worst-over-nominal (fraction): NaN",
        f"Sensitivity score (mean abs corr): {sens_score:.6g}" if not np.isnan(sens_score) else "Sensitivity score (mean abs corr): NaN",
        "",
        f"Verdict: {verdict}",
        f"Confidence: {confidence}",
        f"Recommended action: {action}",
        "",
        "Per-feature sensitivity (abs corr):",
    ]
    for k, v in sorted(sens_map.items(), key=lambda kv: (np.nan_to_num(kv[1], nan=-1.0)), reverse=True):
        if np.isnan(v):
            lines.append(f"  - {k}: NaN")
        else:
            lines.append(f"  - {k}: {v:.6g}")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Wrote: {OUT_CSV}")
    print(f"[OK] Wrote: {OUT_TXT}")


if __name__ == "__main__":
    main()

