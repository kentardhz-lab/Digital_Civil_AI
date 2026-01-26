# src/qc/runner.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.qc.missing_values import check_missing_values


def write_qc_report(report: Dict[str, Any], out_dir: Path) -> Path:
    """
    Writes qc_report.json into out_dir and returns its path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_path = out_dir / "qc_report.json"
    qc_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return qc_path


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

    Dict-based design (no dataclasses) for simplicity and portability.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    enabled = enabled or {"missing_values": True}

    results: List[Dict[str, Any]] = []

    # Pack: missing values (+ required columns)
    if enabled.get("missing_values", True):
        results.append(
            check_missing_values(
                df=df,
                required_columns=required_columns or [],
                max_missing_ratio=max_missing_ratio,
            )
        )

    # Summary / gate
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

    write_qc_report(report, out_dir)
    return report
