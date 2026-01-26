from __future__ import annotations

from typing import List
import pandas as pd

from src.qc.models import QCCheckResult, QCIssue


def check_missing_required_columns(df: pd.DataFrame, required: List[str]) -> QCCheckResult:
    missing = [c for c in required if c not in df.columns]

    passed = len(missing) == 0
    issues = []
    for col in missing:
        issues.append(
            QCIssue(
                code="MISSING_REQUIRED_COLUMN",
                message=f"Required column '{col}' is missing from input.",
                column=col,
            )
        )

    return QCCheckResult(
        check_id="missing_required_columns",
        title="Missing required columns check",
        passed=passed,
        severity="error",
        metrics={"missing_columns": missing, "required_columns": required},
        issues=issues,
        artifacts=[],
    )
