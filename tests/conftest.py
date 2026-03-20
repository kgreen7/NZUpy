"""Shared test fixtures for NZUpy test suite."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Path to frozen test fixture data
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_dir():
    """Path to the test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def test_years():
    """Standard test year range (shorter than production for speed)."""
    return list(range(2024, 2036))


@pytest.fixture
def extended_years():
    """Extended years beyond model end for forward price calculations."""
    return list(range(2036, 2061))


@pytest.fixture
def test_data_dir(tmp_path, fixture_dir):
    """
    Create a temporary data directory structure mirroring data/inputs/,
    populated with frozen fixture files.

    This allows DataHandler to load from fixtures without touching real data.
    """
    # Create directory structure
    inputs_dir = tmp_path / "inputs"
    (inputs_dir / "supply").mkdir(parents=True)
    (inputs_dir / "demand").mkdir(parents=True)
    (inputs_dir / "parameters").mkdir(parents=True)
    (inputs_dir / "stockpile").mkdir(parents=True)
    (inputs_dir / "forestry").mkdir(parents=True)
    (inputs_dir / "economic").mkdir(parents=True)
    (inputs_dir / "historical").mkdir(parents=True)

    # Copy fixture files to appropriate subdirectories
    import shutil

    file_mapping = {
        "auctions.csv": "supply",
        "removals.csv": "supply",
        "historical_removals.csv": "supply",
        "industrial_allocation.csv": "supply",
        "emissions_baselines.csv": "demand",
        "demand_models.csv": "demand",
        "model_parameters.csv": "parameters",
        "price.csv": "parameters",
        "price_control.csv": "parameters",
        "stockpile_balance.csv": "stockpile",
        "yield_tables.csv": "forestry",
        "afforestation_projections.csv": "forestry",
        "manley_parameters.csv": "forestry",
        "CPI.csv": "economic",
    }

    for filename, subdir in file_mapping.items():
        src = fixture_dir / filename
        dst = inputs_dir / subdir / filename
        if src.exists():
            shutil.copy2(src, dst)

    # Copy price.csv to historical directory as carbon_price.csv
    # (DataHandler discovers files in historical/ by variable name)
    price_src = fixture_dir / "price.csv"
    if price_src.exists():
        price_df = pd.read_csv(price_src)
        hist_price = price_df[price_df["Config"] == "central"][["Year", "Value"]].copy()
        hist_price.to_csv(inputs_dir / "historical" / "carbon_price.csv", index=False)

    return tmp_path


@pytest.fixture
def data_handler(test_data_dir):
    """DataHandler initialised with frozen fixture data."""
    from model.utils.data_handler import DataHandler
    return DataHandler(test_data_dir)


@pytest.fixture
def basic_model(test_data_dir):
    """
    A fully configured NZUpy model instance ready to run.
    Uses frozen fixture data with central defaults.
    """
    from model.core.base_model import NZUpy

    nzu = NZUpy(data_dir=test_data_dir)
    nzu.define_time(2024, 2035)
    nzu.define_scenarios(["Test"])
    nzu.allocate()
    nzu.fill_defaults()
    return nzu


@pytest.fixture
def run_model(basic_model):
    """
    A model that has been run with central defaults.
    Returns the model instance with results populated.
    """
    basic_model.run()
    return basic_model
