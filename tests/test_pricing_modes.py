"""
Tests for set_mode(), fixed_path pricing, and fixed_rate pricing.

Test categories:
  - set_mode() validation
  - fixed_path mode: prices injected, no optimisation
  - fixed_rate mode: user-supplied rate, no optimisation
  - Default (optimised) mode unchanged
  - fill() prints a note for mode variables
"""

import pytest
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def allocated_model(test_data_dir):
    """Allocated but not yet filled — used to test set_mode validation."""
    from model.core.base_model import NZUpy
    nzu = NZUpy(data_dir=test_data_dir)
    nzu.define_time(2024, 2035)
    nzu.define_scenarios(["S1", "S2"])
    nzu.allocate()
    return nzu


@pytest.fixture
def filled_model(test_data_dir):
    """Fully filled model ready to run — 3 scenarios for mode comparison."""
    from model.core.base_model import NZUpy
    nzu = NZUpy(data_dir=test_data_dir)
    nzu.define_time(2024, 2035)
    nzu.define_scenarios(["Optimised", "FixedPath", "FixedRate"])
    nzu.allocate()
    nzu.fill_defaults()
    return nzu


@pytest.fixture
def price_path_series():
    """A simple fixed price path for 2024–2035."""
    return pd.Series({y: 50.0 + (y - 2024) * 2 for y in range(2024, 2036)})


# ---------------------------------------------------------------------------
# set_mode() validation
# ---------------------------------------------------------------------------

class TestSetModeValidation:
    def test_unknown_mode_raises(self, allocated_model):
        with pytest.raises(ValueError, match="Unknown mode"):
            allocated_model.set_mode("nonexistent_mode", "value")

    def test_invalid_value_raises(self, allocated_model):
        with pytest.raises(ValueError, match="Invalid value"):
            allocated_model.set_mode("pricing_mode", "banana")

    def test_valid_pricing_mode_accepted(self, allocated_model):
        allocated_model.set_mode("pricing_mode", "fixed_path", scenario="S1")
        assert allocated_model.component_configs[0].pricing_mode == "fixed_path"

    def test_valid_forestry_mode_accepted(self, allocated_model):
        allocated_model.set_mode("forestry_mode", "endogenous", scenario="S1")
        assert allocated_model.component_configs[0].forestry_mode == "endogenous"

    def test_all_scenarios_default(self, allocated_model):
        allocated_model.set_mode("pricing_mode", "fixed_rate")
        assert all(c.pricing_mode == "fixed_rate" for c in allocated_model.component_configs)

    def test_single_scenario_by_name(self, allocated_model):
        allocated_model.set_mode("pricing_mode", "fixed_path", scenario="S2")
        assert allocated_model.component_configs[0].pricing_mode == "optimised"
        assert allocated_model.component_configs[1].pricing_mode == "fixed_path"

    def test_single_scenario_by_index(self, allocated_model):
        allocated_model.set_mode("pricing_mode", "fixed_path", scenario=0)
        assert allocated_model.component_configs[0].pricing_mode == "fixed_path"
        assert allocated_model.component_configs[1].pricing_mode == "optimised"

    def test_invalid_scenario_name_raises(self, allocated_model):
        with pytest.raises(ValueError, match="Unknown scenario"):
            allocated_model.set_mode("pricing_mode", "fixed_path", scenario="NoSuch")

    def test_invalid_scenario_index_raises(self, allocated_model):
        with pytest.raises(ValueError):
            allocated_model.set_mode("pricing_mode", "fixed_path", scenario=99)

    def test_before_allocate_raises(self, test_data_dir):
        from model.core.base_model import NZUpy
        nzu = NZUpy(data_dir=test_data_dir)
        nzu.define_time(2024, 2035)
        nzu.define_scenarios(["S1"])
        with pytest.raises(ValueError, match="allocate"):
            nzu.set_mode("pricing_mode", "optimised")

    def test_returns_self_for_chaining(self, allocated_model):
        result = allocated_model.set_mode("pricing_mode", "optimised")
        assert result is allocated_model


# ---------------------------------------------------------------------------
# fixed_path mode
# ---------------------------------------------------------------------------

class TestFixedPathMode:
    def test_fixed_path_runs_without_error(self, filled_model, price_path_series):
        filled_model.set_mode("pricing_mode", "fixed_path", scenario="FixedPath")
        filled_model.fill("price_path", price_path_series, scenario="FixedPath")
        filled_model.run()

    def test_fixed_path_prices_match_supplied_series(self, filled_model, price_path_series):
        filled_model.set_mode("pricing_mode", "fixed_path", scenario="FixedPath")
        filled_model.fill("price_path", price_path_series, scenario="FixedPath")
        filled_model.run()

        result_prices = filled_model.results["FixedPath"]["prices"]
        for year in range(2024, 2036):
            assert abs(result_prices[year] - price_path_series[year]) < 0.01, (
                f"Year {year}: expected {price_path_series[year]}, got {result_prices[year]}"
            )

    def test_fixed_path_result_has_expected_keys(self, filled_model, price_path_series):
        filled_model.set_mode("pricing_mode", "fixed_path", scenario="FixedPath")
        filled_model.fill("price_path", price_path_series, scenario="FixedPath")
        filled_model.run()
        result = filled_model.results["FixedPath"]
        assert "prices" in result
        assert "supply" in result
        assert "demand" in result
        assert result["total_gap"] is None
        assert result["price_change_rate"] is None

    def test_fixed_path_missing_series_raises(self, filled_model):
        filled_model.set_mode("pricing_mode", "fixed_path", scenario="FixedPath")
        # Don't supply price_path
        with pytest.raises(ValueError, match="price_path"):
            filled_model.run()

    def test_fixed_path_does_not_affect_other_scenarios(self, filled_model, price_path_series):
        """Optimised scenario should still run normally."""
        filled_model.set_mode("pricing_mode", "fixed_path", scenario="FixedPath")
        filled_model.fill("price_path", price_path_series, scenario="FixedPath")
        filled_model.run()
        opt_result = filled_model.results["Optimised"]
        assert opt_result["price_change_rate"] is not None


# ---------------------------------------------------------------------------
# fixed_rate mode
# ---------------------------------------------------------------------------

class TestFixedRateMode:
    def test_fixed_rate_runs_without_error(self, filled_model):
        filled_model.set_mode("pricing_mode", "fixed_rate", scenario="FixedRate")
        filled_model.fill("price_change_rate", 0.05, scenario="FixedRate")
        filled_model.run()

    def test_fixed_rate_result_has_expected_keys(self, filled_model):
        filled_model.set_mode("pricing_mode", "fixed_rate", scenario="FixedRate")
        filled_model.fill("price_change_rate", 0.05, scenario="FixedRate")
        filled_model.run()
        result = filled_model.results["FixedRate"]
        assert "prices" in result
        assert "supply" in result
        assert "demand" in result
        assert result["total_gap"] is None
        assert result["price_change_rate"] == pytest.approx(0.05)

    def test_fixed_rate_missing_rate_raises(self, filled_model):
        filled_model.set_mode("pricing_mode", "fixed_rate", scenario="FixedRate")
        with pytest.raises(ValueError, match="price_change_rate"):
            filled_model.run()

    def test_fixed_rate_zero_gives_flat_prices(self, filled_model):
        filled_model.set_mode("pricing_mode", "fixed_rate", scenario="FixedRate")
        filled_model.fill("price_change_rate", 0.0, scenario="FixedRate")
        filled_model.run()
        result_prices = filled_model.results["FixedRate"]["prices"]
        model_years = filled_model.years
        first = result_prices[model_years[0]]
        for year in model_years[1:]:
            assert abs(result_prices[year] - first) < 0.01, (
                f"Expected flat prices but year {year} differs from {model_years[0]}"
            )

    def test_fixed_rate_different_rates_give_different_prices(self, filled_model):
        """Two runs with different rates produce different end prices."""
        from model.core.base_model import NZUpy

        def run_with_rate(data_dir, rate):
            nzu = NZUpy(data_dir=data_dir)
            nzu.define_time(2024, 2035)
            nzu.define_scenarios(["S"])
            nzu.allocate()
            nzu.fill_defaults()
            nzu.set_mode("pricing_mode", "fixed_rate")
            nzu.fill("price_change_rate", rate)
            nzu.run()
            return nzu.results["S"]["prices"][2035]

        data_dir = filled_model.data_handler.data_dir
        price_low = run_with_rate(data_dir, 0.01)
        price_high = run_with_rate(data_dir, 0.10)
        assert price_high > price_low


# ---------------------------------------------------------------------------
# Default (optimised) mode unchanged
# ---------------------------------------------------------------------------

class TestOptimisedModeUnchanged:
    def test_default_pricing_mode_is_optimised(self, allocated_model):
        for cfg in allocated_model.component_configs:
            assert cfg.pricing_mode == "optimised"

    def test_optimised_run_produces_non_none_price_change_rate(self, basic_model):
        basic_model.run()
        result = basic_model.results["Test"]
        assert result["price_change_rate"] is not None

    def test_optimised_run_prices_positive(self, basic_model):
        basic_model.run()
        result = basic_model.results["Test"]
        assert all(result["prices"][y] > 0 for y in basic_model.years)


# ---------------------------------------------------------------------------
# fill() note for mode variables
# ---------------------------------------------------------------------------

class TestFillModeNote:
    def test_fill_pricing_mode_prints_note(self, allocated_model, capsys):
        allocated_model.fill("pricing_mode", "optimised", scenario="S1")
        captured = capsys.readouterr()
        assert "set_mode" in captured.out

    def test_fill_forestry_mode_prints_note(self, allocated_model, capsys):
        allocated_model.fill("forestry_mode", "exogenous", scenario="S1")
        captured = capsys.readouterr()
        assert "set_mode" in captured.out

    def test_fill_pricing_mode_still_sets_value(self, allocated_model):
        allocated_model.fill("pricing_mode", "fixed_rate", scenario="S1")
        assert allocated_model.component_configs[0].pricing_mode == "fixed_rate"
