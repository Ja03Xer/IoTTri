# logging_utils.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_hash(config_snapshot: Dict[str, Any]) -> str:
    return sha256_text(stable_json(config_snapshot))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_manifest(
    *,
    experiment_id: str,
    code_version: str,
    prompt_version: str,
    schema_version: str,
    config_snapshot: Dict[str, Any],
    runtime_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "code_version": code_version,
        "prompt_version": prompt_version,
        "schema_version": schema_version,
        "config_snapshot": config_snapshot,
        "config_hash": config_hash(config_snapshot),
        "runtime_metadata": runtime_metadata,
    }