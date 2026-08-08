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


# ---------------------------------------------------------------------------
# Smooth peak price control mode
# ---------------------------------------------------------------------------

class TestSmoothPeakPriceControl:
    """Tests for smooth_peak and smooth_peak_search price control modes."""

    def test_smootherstep_formula_correctness(self, test_data_dir):
        """Test quintic smootherstep formula produces correct values."""
        from model.core.base_model import NZUpy

        nzu = NZUpy(data_dir=test_data_dir)
        nzu.define_time(2024, 2040)
        nzu.define_scenarios(["Test"])
        nzu.allocate()
        nzu.fill_defaults()

        # Set smooth_peak mode with peak_year=2032, width=6
        # Transition: 2029 (2032-3) to 2035 (2032+3)
        nzu.set_mode('price_control_mode', 'smooth_peak', scenario='Test')
        nzu.fill('price_control_peak_year', 2032, scenario='Test')
        nzu.fill('price_control_width', 6.0, scenario='Test')
        nzu.fill('price_control_before', -1.0, scenario='Test')
        nzu.fill('price_control_after', 0.5, scenario='Test')

        # Manually call the smooth control calculation
        cfg = nzu.component_configs[0]
        nzu._active_scenario_index = 0
        engine = nzu.calculation_engine

        # Test boundary regions
        assert engine._calculate_smooth_control_value(2028, cfg) == -1.0  # Before
        assert engine._calculate_smooth_control_value(2036, cfg) == 0.5   # After

        # Test midpoint (should be halfway between -1.0 and 0.5 = -0.25)
        mid_value = engine._calculate_smooth_control_value(2032, cfg)
        assert -0.26 < mid_value < -0.24  # Approximately -0.25

        # Test monotonicity: values should increase from before to after
        v1 = engine._calculate_smooth_control_value(2030, cfg)
        v2 = engine._calculate_smooth_control_value(2031, cfg)
        v3 = engine._calculate_smooth_control_value(2033, cfg)
        v4 = engine._calculate_smooth_control_value(2034, cfg)
        assert -1.0 < v1 < v2 < mid_value < v3 < v4 < 0.5

    def test_exogenous_mode_unchanged(self, test_data_dir):
        """Regression test: exogenous mode still works with CSV configs."""
        from model.core.base_model import NZUpy

        nzu = NZUpy(data_dir=test_data_dir)
        nzu.define_time(2024, 2035)
        nzu.define_scenarios(["Central"])
        nzu.allocate()
        nzu.fill_defaults()
        nzu.fill_component('price', config='central', scenario='Central')

        # Run with exogenous mode (default)
        results = nzu.run()

        # Should complete successfully
        assert 'Central' in results
        assert 'prices' in results['Central']
        assert results['Central']['convergence_success']

    def test_smooth_peak_mode_validation(self, allocated_model):
        """Test that smooth_peak mode requires peak_year to be set."""
        allocated_model.fill_defaults()
        allocated_model.set_mode('price_control_mode', 'smooth_peak', scenario='S1')

        # Should raise if peak_year not set
        with pytest.raises(ValueError, match="price_control_peak_year must be set"):
            allocated_model.run()

    def test_smooth_peak_mode_peak_year_in_range(self, allocated_model):
        """Test that peak_year must be within model year range."""
        allocated_model.fill_defaults()
        allocated_model.set_mode('price_control_mode', 'smooth_peak', scenario='S1')
        allocated_model.fill('price_control_peak_year', 2050, scenario='S1')  # Outside 2024-2035

        # Should raise during run
        with pytest.raises(ValueError, match="must fall within the model's year range"):
            allocated_model.run()

    def test_smooth_peak_mode_default_parameters(self, test_data_dir):
        """Test that smooth_peak mode uses correct defaults."""
        from model.core.base_model import NZUpy

        nzu = NZUpy(data_dir=test_data_dir)
        nzu.define_time(2024, 2035)
        nzu.define_scenarios(["Test"])
        nzu.allocate()
        nzu.fill_defaults()

        cfg = nzu.component_configs[0]
        # Check defaults from ComponentConfig
        assert cfg.price_control_before == -1.0
        assert cfg.price_control_after == 0.5
        assert cfg.price_control_width == 5.0

    def test_fill_smooth_peak_parameters(self, allocated_model):
        """Test that smooth peak parameters can be set via fill()."""
        allocated_model.fill('price_control_peak_year', 2030, scenario='S1')
        allocated_model.fill('price_control_before', -0.8, scenario='S1')
        allocated_model.fill('price_control_after', 0.7, scenario='S1')
        allocated_model.fill('price_control_width', 4.0, scenario='S1')

        cfg = allocated_model.component_configs[0]
        assert cfg.price_control_peak_year == 2030
        assert cfg.price_control_before == -0.8
        assert cfg.price_control_after == 0.7
        assert cfg.price_control_width == 4.0

    def test_smooth_peak_search_requires_range(self, allocated_model):
        """Test that smooth_peak_search mode requires peak_year_range."""
        allocated_model.fill_defaults()
        allocated_model.set_mode('price_control_mode', 'smooth_peak_search', scenario='S1')

        # Should raise if range not set
        with pytest.raises(ValueError, match="peak_year_range to be set"):
            allocated_model.run()

    def test_fill_peak_year_range_validation(self, allocated_model):
        """Test validation of peak_year_range parameter."""
        # Should accept valid tuple
        allocated_model.fill('price_control_peak_year_range', (2028, 2032), scenario='S1')
        assert allocated_model.component_configs[0].price_control_peak_year_range == (2028, 2032)

        # Should reject non-tuple
        with pytest.raises(ValueError, match="must be a tuple"):
            allocated_model.fill('price_control_peak_year_range', [2028, 2032], scenario='S1')

        # Should reject wrong size
        with pytest.raises(ValueError, match="must be a tuple"):
            allocated_model.fill('price_control_peak_year_range', (2028, 2030, 2032), scenario='S1')

        # Should reject non-integers
        with pytest.raises(ValueError, match="must be integers"):
            allocated_model.fill('price_control_peak_year_range', (2028.5, 2032.5), scenario='S1')

        # Should reject start > end
        with pytest.raises(ValueError, match="start must be <= end"):
            allocated_model.fill('price_control_peak_year_range', (2032, 2028), scenario='S1')


    def test_exogenous_price_control_continues_beyond_csv_range(self, test_data_dir):
        """Test that exogenous price control continues last value beyond CSV range (no kink post-2050)."""
        from model.core.base_model import NZUpy

        # Create model with extended calculation years
        nzu = NZUpy(data_dir=test_data_dir)
        nzu.define_time(2024, 2050)  # projection_horizon defaults to 25, so calculation_years extend to 2075
        nzu.define_scenarios(["Test"])
        nzu.allocate()

        # Get price control values for years beyond CSV range
        control_2050 = nzu.data_handler.get_price_control(2050, config='scarcity_then_surplus')
        control_2051 = nzu.data_handler.get_price_control(2051, config='scarcity_then_surplus')
        control_2075 = nzu.data_handler.get_price_control(2075, config='scarcity_then_surplus')

        # All should be 0.5 (the 2050 value), not 1.0
        assert control_2050 == 0.5
        assert control_2051 == 0.5  # Continues last value
        assert control_2075 == 0.5  # Continues last value
        assert control_2051 != 1.0  # Should NOT default to neutral


class TestSmoothPeakSearchMode:
    """Tests for smooth_peak_search optimization."""

    def test_peak_search_selects_lowest_gap(self, test_data_dir):
        """Test that peak search selects the candidate with lowest gap."""
        from model.core.base_model import NZUpy

        nzu = NZUpy(data_dir=test_data_dir)
        nzu.define_time(2024, 2035)
        nzu.define_scenarios(["Search"])
        nzu.allocate()
        nzu.fill_defaults()

        # Configure for search with narrow range (3 candidates to keep test fast)
        nzu.set_mode('price_control_mode', 'smooth_peak_search', scenario='Search')
        nzu.fill('price_control_peak_year_range', (2029, 2031), scenario='Search')
        nzu.fill('price_control_width', 4.0, scenario='Search')

        # Run the search
        results = nzu.run()

        # Verify search diagnostics exist
        assert 'peak_year_search' in results['Search']
        search = results['Search']['peak_year_search']
        assert 'candidates' in search
        assert 'selected_peak_year' in search

        # Verify all candidates were tested
        assert len(search['candidates']) == 3
        tested_years = [c['peak_year'] for c in search['candidates']]
        assert set(tested_years) == {2029, 2030, 2031}

        # Verify selected peak year has lowest gap
        selected = search['selected_peak_year']
        selected_gap = next(c['min_gap'] for c in search['candidates'] if c['peak_year'] == selected)
        all_gaps = [c['min_gap'] for c in search['candidates']]
        assert selected_gap == min(all_gaps)

    def test_peak_search_stores_all_candidates(self, test_data_dir):
        """Test that search results store all candidate information."""
        from model.core.base_model import NZUpy

        nzu = NZUpy(data_dir=test_data_dir)
        nzu.define_time(2024, 2035)
        nzu.define_scenarios(["Test"])
        nzu.allocate()
        nzu.fill_defaults()

        nzu.set_mode('price_control_mode', 'smooth_peak_search', scenario='Test')
        nzu.fill('price_control_peak_year_range', (2028, 2030), scenario='Test')

        results = nzu.run()
        candidates = results['Test']['peak_year_search']['candidates']

        # Each candidate should have required fields
        for candidate in candidates:
            assert 'peak_year' in candidate
            assert 'optimal_rate' in candidate
            assert 'min_gap' in candidate
            assert isinstance(candidate['optimal_rate'], (int, float))
            assert isinstance(candidate['min_gap'], (int, float))
