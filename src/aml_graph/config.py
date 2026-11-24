"""Configuration loading, hashing, and runtime provenance logging."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file into a dictionary."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def load_all_configs(config_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all standard experiment YAML files from config_dir."""
    names = ("data", "sampling", "benchmark", "tuning", "extensions", "phase2")
    loaded: dict[str, dict[str, Any]] = {}
    for name in names:
        p = config_dir / f"{name}.yaml"
        if p.exists():
            loaded[name] = load_yaml(p)
    return loaded


def config_hashes(config_dir: Path) -> dict[str, str]:
    """Compute per-file and combined SHA-256 digests over YAML files in config_dir."""
    digests: dict[str, str] = {}
    combined = hashlib.sha256()
    for path in sorted(config_dir.glob("*.yaml")):
        raw = path.read_bytes()
        file_hex = hashlib.sha256(raw).hexdigest()
        digests[path.name] = file_hex
        combined.update(path.name.encode("utf-8") + b":" + raw + b"\n")
    digests["combined_sha256"] = combined.hexdigest()
    return digests


def collect_run_metadata(config_dir: Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build reproducible run provenance (versions, platform, config hashes)."""
    meta: dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "networkx": nx.__version__,
            "matplotlib": matplotlib.__version__,
            "pyyaml": yaml.__version__,
        },
        "config_hashes": config_hashes(config_dir),
    }
    if extra:
        meta.update(extra)
    return meta


def save_run_metadata(
    output_dir: Path, stage: str, config_dir: Path, extra: dict[str, Any] | None = None
) -> Path:
    """Persist stage provenance JSON alongside output files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = collect_run_metadata(config_dir, extra={"stage": stage, **(extra or {})})
    out_path = output_dir / f"provenance_{stage}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    return out_path
