"""
Tests for full model runs — Tier 2 integration tests.

Uses run_model fixture (full model run against frozen fixture data).
Structural tests check invariants that must hold for any valid run.
Regression tests lock in specific numeric outputs computed from fixture data.
"""

import pytest
import pandas as pd
import numpy as np


MODEL_YEARS = list(range(2024, 2036))
SCENARIO = 'Test'


def supply_col(run_model, col):
    return run_model.supply[(SCENARIO, col)]


def demand_col(run_model, col):
    return run_model.demand[(SCENARIO, col)]


def price_col(run_model):
    return run_model.prices[(SCENARIO, 'carbon_price')].loc[MODEL_YEARS]


def sp_col(run_model, col):
    return run_model.stockpile[(SCENARIO, col)]


# ===========================================================================
# Structural tests
# ===========================================================================

class TestModelRunsClean:
    def test_model_runs_without_error(self, basic_model):
        """run() completes without raising an exception."""
        basic_model.run()


class TestSupplyStructure:
    def test_supply_total_equals_sum_of_components(self, run_model):
        """For every year: total = auction + industrial + forestry + stockpile."""
        for year in MODEL_YEARS:
            total = supply_col(run_model, 'total').loc[year]
            components = sum(
                supply_col(run_model, c).loc[year]
                for c in ['auction', 'industrial', 'forestry', 'stockpile']
            )
            assert total == pytest.approx(components, abs=0.1), \
                f"Year {year}: total={total:.2f} != components sum={components:.2f}"

    def test_surplus_used_never_exceeds_balance(self, run_model):
        """surplus_used ≤ starting surplus_balance for every year."""
        for year in MODEL_YEARS:
            surplus_used = sp_col(run_model, 'surplus_used').loc[year]
            surplus_balance = sp_col(run_model, 'surplus_balance').loc[year]
            # surplus_balance is end-of-year; used ≤ start balance = balance + used
            assert surplus_used <= surplus_balance + surplus_used + 1e-9


class TestOutputStructure:
    def test_output_dataframes_have_correct_columns(self, run_model):
        """prices, supply, demand, stockpile all have MultiIndex (scenario, variable) columns."""
        for attr in ['prices', 'supply', 'demand', 'stockpile']:
            df = getattr(run_model, attr)
            assert isinstance(df.columns, pd.MultiIndex), \
                f"{attr} columns should be MultiIndex"
            # All columns should start with the scenario name
            for col in df.columns:
                assert col[0] == SCENARIO, f"Unexpected scenario in {attr}: {col[0]}"

    def test_all_model_years_present_in_output(self, run_model):
        """Every model year appears in every output DataFrame index."""
        for attr in ['prices', 'supply', 'demand', 'stockpile']:
            df = getattr(run_model, attr)
            for year in MODEL_YEARS:
                assert year in df.index, f"Year {year} missing from {attr}"


class TestPriceConstraints:
    def test_prices_positive_for_all_years(self, run_model):
        """Carbon price > 0 for all model years."""
        prices = price_col(run_model)
        for year in MODEL_YEARS:
            assert prices[year] > 0, f"Non-positive price in {year}: {prices[year]}"

    def test_prices_at_or_above_floor(self, run_model):
        """Carbon price >= long_term_min (50 in fixture) for all model years."""
        prices = price_col(run_model)
        floor = 50.0
        for year in MODEL_YEARS:
            assert prices[year] >= floor - 0.01, \
                f"Price {prices[year]:.2f} in {year} below floor {floor}"


class TestDemandConstraints:
    def test_demand_less_than_or_equal_to_baseline(self, run_model):
        """Emissions ≤ baseline for every year (price response only reduces demand)."""
        for year in MODEL_YEARS:
            emissions = demand_col(run_model, 'emissions').loc[year]
            baseline = demand_col(run_model, 'baseline').loc[year]
            assert emissions <= baseline + 0.01, \
                f"Year {year}: emissions={emissions:.0f} > baseline={baseline:.0f}"


# ===========================================================================
# Regression tests — values computed from fixture data 2026-03-17
# ===========================================================================

class TestRegressionValues:
    def test_central_run_price_2035(self, run_model):
        """Price in final year matches known-good value.
        # Computed from fixture data on 2026-03-17 — if this changes, a calculation has been altered
        """
        prices = price_col(run_model)
        assert prices[2035] == pytest.approx(50.0, abs=0.5)

    def test_central_run_stockpile_declines(self, run_model):
        """Stockpile balance declines over the model period.
        # Computed from fixture data on 2026-03-17 — supply slightly exceeds demand each year
        """
        balance_start = sp_col(run_model, 'balance').loc[2024]
        balance_end = sp_col(run_model, 'balance').loc[2035]
        assert balance_end < balance_start, \
            f"Expected stockpile to decline: start={balance_start:.0f}, end={balance_end:.0f}"

    def test_range_run_produces_five_scenarios(self, test_data_dir):
        """Range run creates results for all 5 sensitivity scenarios."""
        from model.core.base_model import NZUpy
        nzu = NZUpy(data_dir=test_data_dir)
        nzu.define_time(2024, 2035)
        nzu.define_scenario_type('Range')
        nzu.allocate()
        nzu.fill_defaults()
        nzu.fill_range_configs()
        nzu.run()

        assert len(nzu.scenarios) == 5
        expected = {'95% Lower', '1 s.e lower', 'central', '1 s.e upper', '95% Upper'}
        assert set(nzu.scenarios) == expected

    def test_range_central_equals_single_central(self, test_data_dir):
        """Central scenario in a range run produces same prices as a standalone central run.
        # Both use the same fixture data with the same config — results must match.
        """
        from model.core.base_model import NZUpy

        # Single central run
        nzu_single = NZUpy(data_dir=test_data_dir)
        nzu_single.define_time(2024, 2035)
        nzu_single.define_scenarios(['central'])
        nzu_single.allocate()
        nzu_single.fill_defaults()
        nzu_single.run()
        single_prices = nzu_single.prices[('central', 'carbon_price')].loc[MODEL_YEARS]

        # Range run (central scenario)
        nzu_range = NZUpy(data_dir=test_data_dir)
        nzu_range.define_time(2024, 2035)
        nzu_range.define_scenario_type('Range')
        nzu_range.allocate()
        nzu_range.fill_defaults()
        nzu_range.fill_range_configs()
        nzu_range.run()
        range_prices = nzu_range.prices[('central', 'carbon_price')].loc[MODEL_YEARS]

        for year in MODEL_YEARS:
            assert single_prices[year] == pytest.approx(range_prices[year], abs=0.1), \
                f"Year {year}: single={single_prices[year]:.2f}, range={range_prices[year]:.2f}"


# ===========================================================================
# Endogenous forestry integration tests
# ===========================================================================

class TestEndogenousForestry:
    def test_endogenous_run_completes(self, test_data_dir):
        """Model runs in endogenous forestry mode without error."""
        from model.core.base_model import NZUpy
        nzu = NZUpy(data_dir=test_data_dir)
        nzu.define_time(2024, 2035)
        nzu.define_scenarios(['Test'])
        nzu.allocate()
        nzu.fill_defaults()
        nzu.fill('forestry_mode', 'endogenous')
        nzu.run()
        # Verify output is populated
        assert nzu.supply is not None
        assert ('Test', 'forestry') in nzu.supply.columns

    def test_endogenous_forestry_supply_non_negative(self, test_data_dir):
        """Endogenous forestry supply is non-negative for all model years."""
        from model.core.base_model import NZUpy
        nzu = NZUpy(data_dir=test_data_dir)
        nzu.define_time(2024, 2035)
        nzu.define_scenarios(['Test'])
        nzu.allocate()
        nzu.fill_defaults()
        nzu.fill('forestry_mode', 'endogenous')
        nzu.run()
        forestry = nzu.supply[('Test', 'forestry')].loc[MODEL_YEARS]
        assert (forestry >= 0).all(), "Endogenous forestry supply has negative values"

    def test_endogenous_differs_from_exogenous(self, test_data_dir):
        """Endogenous forestry produces a different supply path than exogenous.

        Exogenous uses the flat fixture forestry_tradeable series.
        Endogenous computes supply from Manley equation + historical removals.
        """
        from model.core.base_model import NZUpy

        # Exogenous run
        nzu_exog = NZUpy(data_dir=test_data_dir)
        nzu_exog.define_time(2024, 2035)
        nzu_exog.define_scenarios(['Exog'])
        nzu_exog.allocate()
        nzu_exog.fill_defaults()
        nzu_exog.run()
        exog_fs = nzu_exog.supply[('Exog', 'forestry')].loc[MODEL_YEARS]

        # Endogenous run
        nzu_endo = NZUpy(data_dir=test_data_dir)
        nzu_endo.define_time(2024, 2035)
        nzu_endo.define_scenarios(['Endo'])
        nzu_endo.allocate()
        nzu_endo.fill_defaults()
        nzu_endo.fill('forestry_mode', 'endogenous')
        nzu_endo.run()
        endo_fs = nzu_endo.supply[('Endo', 'forestry')].loc[MODEL_YEARS]

        # The two paths must differ (endogenous computes fresh; exogenous uses flat fixture values)
        assert not exog_fs.equals(endo_fs), \
            "Endogenous and exogenous forestry supply should differ"
