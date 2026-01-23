# src/qc/missing_values.py
from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd


def check_missing_values(
    df: pd.DataFrame,
    required_columns: List[str] | None = None,
    max_missing_ratio: float = 0.0,
) -> Dict[str, Any]:
    """
    QC pack: checks missing values overall and for required columns.
    max_missing_ratio=0.0 means no missing allowed for required columns.
    Returns a dict result (runner-compatible).
    """
    required_columns = required_columns or []

    issues: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {}

    missing_per_col = df.isna().sum().to_dict()
    metrics["rows"] = int(df.shape[0])
    metrics["cols"] = int(df.shape[1])
    metrics["missing_per_col"] = {k: int(v) for k, v in missing_per_col.items()}
    metrics["missing_total"] = int(df.isna().sum().sum())

    passed = True

    for col in required_columns:
        if col not in df.columns:
            passed = False
            issues.append(
                {
                    "code": "MISSING_REQUIRED_COLUMN",
                    "message": f"Required column '{col}' is missing from input.",
                    "column": col,
                }
            )
            continue

        miss = int(df[col].isna().sum())
        ratio = miss / max(1, int(df.shape[0]))

        if ratio > max_missing_ratio:
            passed = False
            issues.append(
                {
                    "code": "TOO_MANY_MISSING_VALUES",
                    "message": f"Column '{col}' has {miss} missing values (ratio={ratio:.3f}) > allowed {max_missing_ratio:.3f}.",
                    "column": col,
                    "details": {"missing": miss, "ratio": ratio, "max_missing_ratio": max_missing_ratio},
                }
            )

    return {
        "check_id": "missing_values",
        "title": "Missing values check",
        "passed": passed,
        "severity": "error" if not passed else "info",
        "metrics": metrics,
        "issues": issues,
        "artifacts": [],
    }
