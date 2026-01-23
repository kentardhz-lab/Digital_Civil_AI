from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ManifestItem:
    path: str
    sha256: str


def build_run_manifest(
    *,
    run_id: str,
    config_name: str,
    input_source: str,
    run_dir: Path,
    outputs: List[Path],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)

    items: List[Dict[str, str]] = []
    for p in outputs:
        p = Path(p)
        if p.exists():
            items.append({"path": str(p), "sha256": file_sha256(p)})

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": _utc_now_iso(),
        "config_name": config_name,
        "input_source": input_source,
        "output_directory": str(run_dir),
        "outputs": items,
    }

    if extra:
        manifest["extra"] = extra

    return manifest


def write_manifest(run_dir: Path, manifest: Dict[str, Any]) -> Path:
    run_dir = Path(run_dir)
    out_path = run_dir / "run_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
