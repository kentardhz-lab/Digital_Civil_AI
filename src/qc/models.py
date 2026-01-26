from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Literal, Optional


Severity = Literal["info", "warn", "error"]


@dataclass
class QCIssue:
    code: str
    message: str
    column: Optional[str] = None
    row: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QCCheckResult:
    check_id: str
    title: str
    passed: bool
    severity: Severity
    metrics: Dict[str, Any] = field(default_factory=dict)
    issues: List[QCIssue] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # dataclasses -> dict OK
        return d


@dataclass
class QCReport:
    qc_version: str
    status: Literal["PASS", "FAIL"]
    summary: Dict[str, Any]
    checks: List[QCCheckResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qc_version": self.qc_version,
            "status": self.status,
            "summary": self.summary,
            "checks": [c.to_dict() for c in self.checks],
        }
