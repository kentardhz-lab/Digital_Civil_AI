from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional

import yaml  # pyyaml


# =========================
# Utilities
# =========================

def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_hash() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def pip_freeze() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["python", "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode()
    except Exception:
        return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================
# Main runner
# =========================

def run(config_path: str) -> Path:
    config_file = Path(config_path)
    cfg = load_yaml(config_file)

    project_name = cfg["project"]["name"]
    output_root = Path(cfg["run"]["output_dir"])
    run_id = cfg["run"].get("run_id") or utc_run_id()

    run_dir = output_root / project_name / run_id
    ensure_dir(run_dir)

    # ---- inputs
    elements_csv = Path(cfg["inputs"]["elements_csv"])
    if not elements_csv.exists():
        raise FileNotFoundError(f"Input file not found: {elements_csv}")

    # ---- copy config for traceability
    config_copy = run_dir / "config_used.yaml"
    config_copy.write_text(config_file.read_text(encoding="utf-8"), encoding="utf-8")

    # ---- manifest
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "project": {
            "name": project_name,
            "units": cfg["project"].get("units", "metric"),
        },
        "paths": {
            "config_file": str(config_file.as_posix()),
            "config_used": str(config_copy.as_posix()),
            "run_dir": str(run_dir.as_posix()),
            "elements_csv": str(elements_csv.as_posix()),
        },
        "checksums": {
            "config_sha256": file_sha256(config_file),
            "elements_csv_sha256": file_sha256(elements_csv),
        },
        "env": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "git_commit": git_commit_hash(),
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scenarios": cfg["run"].get("scenarios", ["base", "degraded", "extreme"]),
    }

    freeze = pip_freeze()
    if freeze:
        freeze_path = run_dir / "pip_freeze.txt"
        freeze_path.write_text(freeze, encoding="utf-8")
        manifest["env"]["pip_freeze_path"] = str(freeze_path.as_posix())

    write_json(run_dir / "run_manifest.json", manifest)

    # ---- run full pipeline
    env = os.environ.copy()
    env["CIVIL_AI_RUN_DIR"] = str(run_dir.resolve())

    subprocess.check_call(
    ["python", "-m", "src.pipeline.run_full_pipeline"],
    env=env,
    )

    return run_dir


# =========================
# CLI
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Digital Civil AI from YAML config")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (e.g. configs/demo_project.yaml)",
    )
    args = parser.parse_args()

    out_dir = run(args.config)
    print(f"[OK] Run completed. Outputs at: {out_dir}")


