"""
Tests for DataHandler — Tier 1 component isolation.

Verifies that fixtures load correctly and DataHandler returns
properly structured data for each component.
"""

import pytest
import numpy as np
import pandas as pd


class TestAuctionData:
    def test_get_auction_data_central(self, data_handler):
        """Returns DataFrame with expected columns for 'central' config."""
        df = data_handler.get_auction_data('central')
        assert isinstance(df, pd.DataFrame)
        required_cols = ['base_volume', 'ccr_volume_1', 'ccr_volume_2',
                         'ccr_trigger_price_1', 'ccr_trigger_price_2', 'auction_reserve_price']
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_get_auction_data_invalid_config_raises(self, data_handler):
        """KeyError raised for a nonexistent config."""
        with pytest.raises(KeyError):
            data_handler.get_auction_data('nonexistent_config')


class TestForestryData:
    def test_get_forestry_data_central(self, data_handler):
        """Returns DataFrame with forestry_tradeable column for 'central' config."""
        df = data_handler.get_forestry_data('central')
        assert isinstance(df, pd.DataFrame)
        assert 'forestry_tradeable' in df.columns


class TestStockpileParameters:
    def test_get_stockpile_parameters_central(self, data_handler):
        """Returns dict with all required keys for 'central' config."""
        params = data_handler.get_stockpile_parameters('central', model_start_year=2024)
        assert isinstance(params, dict)
        required_keys = ['initial_stockpile', 'initial_surplus', 'liquidity_factor',
                         'payback_period', 'discount_rate', 'stockpile_usage_start_year',
                         'stockpile_reference_year']
        for key in required_keys:
            assert key in params, f"Missing key: {key}"


class TestDemandModel:
    def test_get_demand_model_central(self, data_handler):
        """Returns dict with constant, reduction_to_t1, price keys."""
        params = data_handler.get_demand_model('central', model_number=2)
        assert isinstance(params, dict)
        assert 'constant' in params
        assert 'reduction_to_t1' in params
        assert 'price' in params


class TestHistoricalRemovals:
    def test_get_historical_removals_units_converted(self, data_handler):
        """Values are in kt (divided by 1000 from raw CSV values in t CO2-e)."""
        df = data_handler.get_historical_removals('central')
        assert isinstance(df, pd.DataFrame)
        assert 'historic_forestry_tradeable' in df.columns
        # Raw fixture values are ~3,500,000 t; divided by 1000 = ~3,500 kt
        assert df['historic_forestry_tradeable'].max() < 10000, \
            "Expected kt-scale values (< 10000), got raw t values"
        assert df['historic_forestry_tradeable'].min() > 0


class TestYieldIncrements:
    def test_get_yield_increments_structure(self, data_handler):
        """Returns dict with three forest type keys, each a numpy array."""
        increments = data_handler.get_yield_increments()
        assert isinstance(increments, dict)
        expected_keys = ['permanent_exotic', 'production_exotic', 'natural_forest']
        for key in expected_keys:
            assert key in increments, f"Missing key: {key}"
            assert isinstance(increments[key], np.ndarray)
            assert len(increments[key]) > 0


class TestListAvailableConfigs:
    def test_list_available_configs_returns_non_empty(self, data_handler):
        """Returns non-empty list for each supported component type."""
        for component in ['emissions', 'auction', 'industrial', 'forestry', 'demand_model']:
            configs = data_handler.list_available_configs(component)
            assert isinstance(configs, list)
            assert len(configs) > 0, f"No configs found for component '{component}'"
            assert 'central' in configs, f"'central' config missing for '{component}'"
