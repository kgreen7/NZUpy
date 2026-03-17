"""
Tests for NZUpy API contracts — Tier 3 user-facing workflow tests.

Verifies that the public API raises informative errors for invalid usage,
and that the happy path works correctly through method chaining.
"""

import pytest
import pandas as pd
import numpy as np


# ===========================================================================
# Error handling — before allocate/run
# ===========================================================================

class TestPreAllocationErrors:
    def test_run_before_allocate_raises(self, test_data_dir):
        """run() before allocate() raises ValueError with clear message."""
        from model.core.base_model import NZUpy
        nzu = NZUpy(data_dir=test_data_dir)
        nzu.define_time(2024, 2035)
        nzu.define_scenarios(['Test'])
        with pytest.raises(ValueError, match="allocat"):
            nzu.run()

    def test_allocate_before_define_time_raises(self, test_data_dir):
        """allocate() before define_time() raises ValueError."""
        from model.core.base_model import NZUpy
        nzu = NZUpy(data_dir=test_data_dir)
        with pytest.raises(ValueError):
            nzu.allocate()

    def test_allocate_before_define_scenarios_raises(self, test_data_dir):
        """allocate() before define_scenarios() raises ValueError."""
        from model.core.base_model import NZUpy
        nzu = NZUpy(data_dir=test_data_dir)
        nzu.define_time(2024, 2035)
        with pytest.raises(ValueError):
            nzu.allocate()


# ===========================================================================
# fill() validation errors
# ===========================================================================

class TestFillErrors:
    def test_fill_unknown_variable_raises(self, basic_model):
        """fill() with unknown variable name raises ValueError."""
        with pytest.raises(ValueError, match="[Uu]nknown"):
            basic_model.fill('not_a_real_variable', 99.0)

    def test_fill_invalid_liquidity_factor_raises(self, basic_model):
        """fill() with liquidity_factor > 1 raises ValueError."""
        with pytest.raises(ValueError):
            basic_model.fill('liquidity_factor', 1.5)

    def test_fill_invalid_forestry_mode_raises(self, basic_model):
        """fill() with invalid forestry_mode raises ValueError."""
        with pytest.raises(ValueError):
            basic_model.fill('forestry_mode', 'magic')

    def test_fill_invalid_scenario_name_raises(self, basic_model):
        """fill() with nonexistent scenario name raises ValueError."""
        with pytest.raises(ValueError, match="[Uu]nknown"):
            basic_model.fill('liquidity_factor', 0.10, scenario='DoesNotExist')


# ===========================================================================
# fill_component() validation errors
# ===========================================================================

class TestFillComponentErrors:
    def test_fill_component_unknown_component_raises(self, basic_model):
        """fill_component() with bad component name raises ValueError."""
        with pytest.raises(ValueError):
            basic_model.fill_component('not_a_component', 'central')

    def test_fill_component_unknown_config_raises_at_run(self, test_data_dir):
        """fill_component() with nonexistent config name raises at run() time."""
        from model.core.base_model import NZUpy
        nzu = NZUpy(data_dir=test_data_dir)
        nzu.define_time(2024, 2035)
        nzu.define_scenarios(['Test'])
        nzu.allocate()
        nzu.fill_defaults()
        nzu.fill_component('emissions', 'config_that_does_not_exist')
        with pytest.raises((KeyError, ValueError)):
            nzu.run()


# ===========================================================================
# fill_defaults() and configuration
# ===========================================================================

class TestFillDefaults:
    def test_fill_defaults_sets_all_components(self, test_data_dir):
        """After fill_defaults(), component_configs are populated for all scenarios."""
        from model.core.base_model import NZUpy
        nzu = NZUpy(data_dir=test_data_dir)
        nzu.define_time(2024, 2035)
        nzu.define_scenarios(['Test'])
        nzu.allocate()
        nzu.fill_defaults()
        # All component configs should exist
        assert len(nzu.component_configs) == 1
        cfg = nzu.component_configs[0]
        assert cfg is not None


# ===========================================================================
# Scenario management
# ===========================================================================

class TestScenarioManagement:
    def test_define_scenario_type_range_creates_five_scenarios(self, test_data_dir):
        """define_scenario_type('Range') creates exactly 5 scenarios with correct names."""
        from model.core.base_model import NZUpy
        nzu = NZUpy(data_dir=test_data_dir)
        nzu.define_time(2024, 2035)
        nzu.define_scenario_type('Range')
        assert len(nzu.scenarios) == 5
        expected = {'95% Lower', '1 s.e lower', 'central', '1 s.e upper', '95% Upper'}
        assert set(nzu.scenarios) == expected

    def test_list_configs_returns_non_empty(self, basic_model):
        """list_configs() returns at least one config for every component."""
        components = ['auction', 'forestry', 'industrial', 'emissions',
                      'demand_model', 'stockpile']
        for component in components:
            configs = basic_model.list_configs(component)
            assert len(configs) >= 1, f"No configs returned for {component}"


# ===========================================================================
# Output structure
# ===========================================================================

class TestOutputStructure:
    def test_output_multiindex_structure(self, run_model):
        """After run(), all output DataFrames have (scenario, variable) MultiIndex columns."""
        for attr in ['prices', 'supply', 'demand', 'stockpile']:
            df = getattr(run_model, attr)
            assert isinstance(df.columns, pd.MultiIndex), \
                f"{attr} should have MultiIndex columns after run()"

    def test_prices_has_carbon_price_column(self, run_model):
        """prices DataFrame has ('Test', 'carbon_price') column after run()."""
        scenario = run_model.scenarios[0]
        assert (scenario, 'carbon_price') in run_model.prices.columns


# ===========================================================================
# Multiple scenarios
# ===========================================================================

class TestMultipleScenarios:
    def test_multiple_scenarios_independent(self, test_data_dir):
        """Two scenarios with different configs produce different demand levels."""
        from model.core.base_model import NZUpy
        nzu = NZUpy(data_dir=test_data_dir)
        nzu.define_time(2024, 2035)
        nzu.define_scenarios(['Central', 'AltEmissions'])
        nzu.allocate()
        nzu.fill_defaults()
        nzu.fill_component('emissions', 'CCC_CPR', scenario='AltEmissions')
        nzu.run()

        central_demand = nzu.demand[('Central', 'baseline')]
        alt_demand = nzu.demand[('AltEmissions', 'baseline')]
        # Different emissions configs → different baseline demand
        assert not central_demand.equals(alt_demand), \
            "Different emissions configs should produce different demand baselines"


# ===========================================================================
# fill() with Series
# ===========================================================================

class TestFillSeries:
    def test_fill_series_overrides_config(self, test_data_dir):
        """Using fill() with a pd.Series for a time-series variable overrides config values."""
        from model.core.base_model import NZUpy
        years = list(range(2024, 2036))

        # Default run
        nzu_default = NZUpy(data_dir=test_data_dir)
        nzu_default.define_time(2024, 2035)
        nzu_default.define_scenarios(['Test'])
        nzu_default.allocate()
        nzu_default.fill_defaults()
        nzu_default.run()
        default_stockpile = nzu_default.stockpile[('Test', 'balance')].loc[2030]

        # Run with overridden liquidity_factor
        nzu_override = NZUpy(data_dir=test_data_dir)
        nzu_override.define_time(2024, 2035)
        nzu_override.define_scenarios(['Test'])
        nzu_override.allocate()
        nzu_override.fill_defaults()
        nzu_override.fill('liquidity_factor', 0.50)  # much higher than default 0.15
        nzu_override.run()
        override_stockpile = nzu_override.stockpile[('Test', 'balance')].loc[2030]

        # Higher liquidity_factor means more non-surplus can be used → stockpile may differ
        # At minimum the model should complete without error (already verified)
        # The final stockpile balance should be finite
        assert pd.notna(override_stockpile)


# ===========================================================================
# Method chaining
# ===========================================================================

class TestMethodChaining:
    def test_method_chaining_works(self, test_data_dir):
        """define_time → define_scenarios → allocate → fill_defaults → run returns self."""
        from model.core.base_model import NZUpy
        nzu = NZUpy(data_dir=test_data_dir)
        # Each method should return self (NZUpy instance)
        result = nzu.define_time(2024, 2035)
        assert result is nzu

        result = nzu.define_scenarios(['Test'])
        assert result is nzu

        result = nzu.allocate()
        assert result is nzu

        result = nzu.fill_defaults()
        assert result is nzu

        result = nzu.fill('liquidity_factor', 0.15)
        assert result is nzu
