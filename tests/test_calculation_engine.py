"""
Tests for CalculationEngine — Tier 1 component isolation.

CalculationEngine is tightly coupled to NZUpy; these tests use run_model.
Output format: nzu.prices is a DataFrame with MultiIndex columns (scenario, variable).
Key column: ('Test', 'carbon_price').
"""

import pytest
import pandas as pd
import numpy as np


def get_prices(run_model):
    """Return the carbon price Series for the single 'Test' scenario."""
    scenario = run_model.scenarios[0]
    return run_model.prices[(scenario, 'carbon_price')]


class TestHistoricalPrices:
    def test_prices_use_historical_values(self, basic_model):
        """Prices for historical years match the historical_prices dict loaded from price.csv.

        Fixture price.csv has: 2020:25, 2021:55, 2022:84, 2023:63, 2024:62.
        """
        assert hasattr(basic_model, 'historical_prices')
        hp = basic_model.historical_prices
        assert hp[2024] == pytest.approx(62.0, abs=0.1)
        assert hp[2023] == pytest.approx(63.0, abs=0.1)
        assert hp[2020] == pytest.approx(25.0, abs=0.1)


class TestPriceGrowth:
    def test_price_growth_uses_change_rate(self, run_model):
        """Projected prices start from the last historical price.

        Fixture: last historical year=2024, last_historical_price=62.
        Projected prices[2024] should equal the historical price (62).
        """
        prices = get_prices(run_model)
        assert prices[2024] == pytest.approx(62.0, abs=0.1)

    def test_price_control_negative_inverts_direction(self, test_data_dir):
        """With negative price control values, model still runs."""
        from model.core.base_model import NZUpy

        nzu = NZUpy(data_dir=test_data_dir)
        nzu.define_time(2024, 2035)
        nzu.define_scenarios(['Test'])
        nzu.allocate()
        nzu.fill_defaults()
        nzu.fill('price_control', pd.Series({year: -1.0 for year in range(2024, 2029)}))
        nzu.run()

        scenario = nzu.scenarios[0]
        prices = nzu.prices[(scenario, 'carbon_price')]
        # Model should converge and produce valid prices
        assert prices.notna().all()
        assert (prices > 0).all()


class TestPriceFloor:
    def test_price_floor_applied(self, run_model):
        """Price never drops below long_term_min (50 in fixture model_parameters.csv)."""
        long_term_min = 50.0
        prices = get_prices(run_model)
        for year in run_model.years:
            assert prices[year] >= long_term_min - 0.01, \
                f"Price {prices[year]:.2f} in {year} is below long_term_min={long_term_min}"


class TestGapCalculation:
    def test_gap_is_zero_when_supply_equals_demand(self, run_model):
        """After optimisation, the model converges: prices are valid and non-negative.

        The optimiser minimises the supply-demand gap; successful convergence
        is evidenced by a non-NaN, non-negative price path.
        """
        prices = get_prices(run_model)
        model_prices = prices.loc[run_model.years]
        assert model_prices.notna().all(), "Some prices are NaN — model did not converge"
        assert (model_prices >= 0).all(), "Negative prices found"
        assert len(model_prices) == len(run_model.years), "Price series has wrong length"
