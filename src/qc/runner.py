# src/qc/runner.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.qc.missing_values import check_missing_values


def run_qc_packs(
    df,
    out_dir: Path,
    required_columns: Optional[List[str]] = None,
    max_missing_ratio: float = 0.0,
    enabled: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """
    Runs enabled QC packs and writes qc_report.json into out_dir.
    Returns qc_report as dict.

    This version is dict-based (no src.qc.base dataclasses required).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    enabled = enabled or {"missing_values": True}

    results: List[Dict[str, Any]] = []

    if enabled.get("missing_values", True):
        results.append(
            check_missing_values(
                df=df,
                required_columns=required_columns or [],
                max_missing_ratio=max_missing_ratio,
            )
        )

    # summary / gate
    n_failed = sum(1 for r in results if not r.get("passed", True))
    status = "PASS" if n_failed == 0 else "FAIL"

    report: Dict[str, Any] = {
        "qc_version": "0.1",
        "status": status,
        "summary": {
            "checks_total": len(results),
            "failed": n_failed,
        },
        "checks": results,
    }

    qc_path = out_dir / "qc_report.json"
    qc_path.write_text(__import__("json").dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return report
