"""Smoke tests: the package layout matches the declared architecture."""

import importlib
from pathlib import Path

import pytest

import mmlab

MODULES = (
    "backtesting",
    "data_ingestion",
    "dynamics",
    "evaluation",
    "execution_simulator",
    "feature_engineering",
    "hypothesis_registry",
    "market_data",
    "microstructure",
    "orderbook",
    "reporting",
    "risk",
    "stochastic_models",
    "topology",
)


def test_layout_matches_declared_modules() -> None:
    root = Path(mmlab.__file__).parent
    found = {p.parent.name for p in root.glob("*/__init__.py")}
    assert found == set(MODULES)


@pytest.mark.parametrize("name", MODULES)
def test_module_importable(name: str) -> None:
    importlib.import_module(f"mmlab.{name}")
