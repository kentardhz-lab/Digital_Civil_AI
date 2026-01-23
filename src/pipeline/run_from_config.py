from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml

from src.pipeline.run_full_pipeline import run_full_pipeline


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_yaml(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_text(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _pip_freeze() -> str:
    # بدون subprocess برای ساده نگه‌داشتن؛ در صورت نیاز فردا بهترش می‌کنیم
    # اینجا fallback: اگر pip freeze ممکن نبود، فایل خالی نشه.
    try:
        import subprocess
        r = subprocess.run(["python", "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception as e:
        return f"# pip freeze failed: {e}"


@dataclass(frozen=True)
class RunPaths:
    root: Path
    project_dir: Path
    run_dir: Path


def resolve_run_paths(cfg: Dict[str, Any]) -> RunPaths:
    project_name = (cfg.get("project") or {}).get("name", "demo_project")

    output_root = (cfg.get("output") or {}).get("root_dir", "outputs")
    root = Path(output_root)

    run_id = (cfg.get("run") or {}).get("id") or _utc_stamp()

    project_dir = root / project_name
    run_dir = project_dir / run_id

    return RunPaths(root=root, project_dir=project_dir, run_dir=run_dir)


def build_run_manifest(*, cfg: Dict[str, Any], config_path: Path, run_dir: Path, outputs: list[str]) -> Dict[str, Any]:
    return {
        "run_id": run_dir.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_name": config_path.name,
        "config_path": str(config_path.as_posix()),
        "input_source": "CSV",
        "output_directory": str(run_dir.as_posix()),
        "outputs": outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config, e.g. configs/demo_project.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = _load_yaml(config_path)

    paths = resolve_run_paths(cfg)
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    elements_csv = Path((cfg.get("inputs") or {}).get("elements_csv", "data/elements.csv"))
    scenarios_cfg = cfg.get("scenarios") or {}
    scenarios = {
        "base": bool(scenarios_cfg.get("base", True)),
        "degraded": bool(scenarios_cfg.get("degraded", True)),
        "extreme": bool(scenarios_cfg.get("extreme", True)),
    }

    # 1) copy config_used.yaml (for traceability)
    if bool((cfg.get("output") or {}).get("keep_config_copy", True)):
        cfg_used_path = paths.run_dir / "config_used.yaml"
        _write_text(cfg_used_path, yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    # 2) run pipeline
    run_full_pipeline(
        elements_csv=elements_csv,
        run_dir=paths.run_dir,
        scenarios=scenarios,
    )

    # 3) pip freeze
    if bool((cfg.get("output") or {}).get("write_pip_freeze", True)):
        _write_text(paths.run_dir / "pip_freeze.txt", _pip_freeze())

    # 4) run_manifest.json
    outputs = [
        "final_engineering_report.csv",
        "qc_report.json",
        "pip_freeze.txt",
        "config_used.yaml",
        "run_manifest.json",
    ]
    manifest = build_run_manifest(
        cfg=cfg,
        config_path=config_path,
        run_dir=paths.run_dir,
        outputs=outputs,
    )
    _write_json(paths.run_dir / "run_manifest.json", manifest)

    print(f"[OK] Run completed. Outputs at: {paths.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
