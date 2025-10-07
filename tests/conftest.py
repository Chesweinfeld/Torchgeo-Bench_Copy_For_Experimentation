"""Pytest configuration and fixtures for torchgeo-bench tests."""

import os
from pathlib import Path

import pytest

# GeoBench data root - can override with environment variable
GEOBENCH_ROOT = os.getenv("GEOBENCH_ROOT", "data/classification_v1.0")

# Set GEO_BENCH_DIR for reference implementation (geobench library)
# This needs to be set BEFORE the geobench library is imported
# Reference implementation expects parent directory (without classification_v1.0)
if "GEO_BENCH_DIR" not in os.environ:
    os.environ["GEO_BENCH_DIR"] = str(Path(GEOBENCH_ROOT).parent)


@pytest.fixture
def geobench_root():
    """Fixture providing GeoBench data root path."""
    root = Path(GEOBENCH_ROOT)
    if not root.exists():
        pytest.skip(f"GeoBench data not found at {root}")
    return str(root)


@pytest.fixture
def all_datasets():
    """Fixture providing list of all available dataset names."""
    return [
        "m-eurosat",
        "m-forestnet",
        "m-so2sat",
        "m-pv4ger",
        "m-brick-kiln",
    ]


@pytest.fixture
def small_partition():
    """Fixture providing a small partition name for fast tests."""
    return "0.01x_train"


@pytest.fixture
def all_splits():
    """Fixture providing all split names."""
    return ["train", "valid", "test"]
