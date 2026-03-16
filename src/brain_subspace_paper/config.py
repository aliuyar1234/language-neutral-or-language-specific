from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_path(name: str) -> Path:
    return project_root() / "configs" / name


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


@lru_cache(maxsize=None)
def project_config() -> dict[str, Any]:
    return load_yaml(config_path("project.yaml"))


@lru_cache(maxsize=None)
def pipeline_config() -> dict[str, Any]:
    return load_yaml(config_path("pipeline.yaml"))


@lru_cache(maxsize=None)
def output_config() -> dict[str, Any]:
    return load_yaml(config_path("outputs.yaml"))


@lru_cache(maxsize=None)
def model_config() -> dict[str, Any]:
    return load_yaml(config_path("models.yaml"))
