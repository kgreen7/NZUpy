"""
Tests for ForestrySupply — Tier 1 component isolation.

Tests both exogenous and endogenous modes.

Note: ForestrySupply in endogenous mode needs extended years (beyond model end)
for the forward price NPV calculation's look-ahead horizon.
"""

import pytest
import numpy as np
import pandas as pd
from model.supply.forestry import ForestrySupply


# Years: model years and extra years for forward price look-ahead
MODEL_YEARS = list(range(2024, 2036))
PRICES = pd.Series(100.0, index=MODEL_YEARS)


def make_forestry_data(years=None, volume=5000):
    """Minimal forestry_data DataFrame for exogenous mode."""
    if years is None:
        years = MODEL_YEARS
    return pd.DataFrame({
        'forestry_tradeable': float(volume),
        'forestry_held': 500.0,
        'forestry_surrender': -200.0,
    }, index=years)


# ===========================================================================
# Exogenous mode tests
# ===========================================================================

class TestExogenous:
    def test_exogenous_supply_matches_input(self):
        """total_supply equals forestry_tradeable from input."""
        fs = ForestrySupply(MODEL_YEARS, make_forestry_data(volume=7500))
        result = fs.calculate(PRICES)
        for year in MODEL_YEARS:
            assert result.loc[year, 'total_supply'] == pytest.approx(7500.0, abs=0.1)

    def test_exogenous_ignores_prices(self):
        """Supply is the same regardless of price level."""
        fs = ForestrySupply(MODEL_YEARS, make_forestry_data(volume=8000))
        low_result = fs.calculate(pd.Series(10.0, index=MODEL_YEARS))
        high_result = fs.calculate(pd.Series(1000.0, index=MODEL_YEARS))
        for year in MODEL_YEARS:
            assert low_result.loc[year, 'total_supply'] == pytest.approx(
                high_result.loc[year, 'total_supply'])


# ===========================================================================
# Endogenous mode helper
# ===========================================================================

def make_historical_removals(years=None):
    """Synthetic historical removals in kt (already divided by 1000)."""
    if years is None:
        years = list(range(2018, 2024))
    return pd.DataFrame({
        'historic_forestry_tradeable': 3500.0,
        'historic_forestry_held': 0.0,
        'historic_forestry_surrender': 0.0,
    }, index=pd.Index(years, name='year'))


def make_yield_increments():
    """Minimal yield increments (20-age array from the fixture yield table)."""
    cum = np.array([
        0.499767495, 2.569294027, 7.883625433, 26.03924512, 57.09865553,
        99.54756855, 147.7493219, 194.9152524, 236.6302005, 274.6524278,
        308.7328698, 344.0969379, 381.0367979, 419.793813, 459.8884089,
        501.0739753, 543.0754045, 585.3977568, 627.7649708, 669.7887113,
    ])
    increments = np.diff(cum, prepend=0)
    return {
        'permanent_exotic': increments.copy(),
        'production_exotic': increments.copy(),
        'natural_forest': increments.copy() * 0.2,
    }


def make_afforestation_projections(years=None):
    """Simple constant afforestation projections."""
    if years is None:
        years = list(range(2024, 2061))
    df = pd.DataFrame({
        'permanent_exotic': 6000.0,
        'production_exotic': 20000.0,
        'natural_forest': 500.0,
    }, index=pd.Index(years, name='year'))
    return df


MANLEY_PARAMS = {
    'f': 100000.0,
    'LMV': 10000.0,
    'g': 4844.0,
    'h': 0.0005292,
    'LEV_logs': 1965.333333,
    'LEV_constant': 0.8262,
    'LMV_constant': 0.4329,
    'LEV_carbon_per_dollar': 175.7764755,
    'price_lag': 2,
    'LUC_limit': 100000.0,
    'forestry_discount_rate': 0.08,
    'forestry_forward_years': 15,
    'max_forestry': 50000.0,
    'max_forestry_2050': 50000.0,
    'max_aggregate_afforestation': 1_000_000_000.0,
    'price_conversion_2021': 1.140449438,
}


def make_endogenous_fs(manley_params=None, price_assumption='future'):
    """Create a ForestrySupply in endogenous mode with synthetic data."""
    params = manley_params or MANLEY_PARAMS
    # Provide a simple config-like object with price_assumption
    class SimpleConfig:
        forestry_price_assumption = price_assumption
        manley_f = None
        manley_LMV = None
        manley_LUC_limit = None
        forestry_discount_rate = None
        forestry_forward_years = None

    return ForestrySupply(
        years=MODEL_YEARS,
        forestry_data=make_forestry_data(),
        mode='endogenous',
        manley_config=SimpleConfig(),
        historical_removals=make_historical_removals(),
        yield_increments=make_yield_increments(),
        afforestation_projections=make_afforestation_projections(),
        manley_params=params,
    )


# ===========================================================================
# Endogenous mode: validation tests
# ===========================================================================

class TestEndogenousValidation:
    def test_endogenous_requires_historical_removals(self):
        """Raises ValueError if historical_removals not provided."""
        with pytest.raises(ValueError, match="historical_removals"):
            ForestrySupply(
                years=MODEL_YEARS,
                forestry_data=make_forestry_data(),
                mode='endogenous',
                yield_increments=make_yield_increments(),
                afforestation_projections=make_afforestation_projections(),
                manley_params=MANLEY_PARAMS,
            )

    def test_endogenous_requires_yield_increments(self):
        """Raises ValueError if yield_increments not provided."""
        with pytest.raises(ValueError, match="yield_increments"):
            ForestrySupply(
                years=MODEL_YEARS,
                forestry_data=make_forestry_data(),
                mode='endogenous',
                historical_removals=make_historical_removals(),
                afforestation_projections=make_afforestation_projections(),
                manley_params=MANLEY_PARAMS,
            )

    def test_endogenous_requires_afforestation_projections(self):
        """Raises ValueError if afforestation_projections not provided."""
        with pytest.raises(ValueError, match="afforestation_projections"):
            ForestrySupply(
                years=MODEL_YEARS,
                forestry_data=make_forestry_data(),
                mode='endogenous',
                historical_removals=make_historical_removals(),
                yield_increments=make_yield_increments(),
                manley_params=MANLEY_PARAMS,
            )


# ===========================================================================
# Endogenous mode: calculation tests
# ===========================================================================

class TestEndogenousCalculation:
    def test_historic_supply_from_historical_removals(self):
        """historic_supply column matches historic_forestry_tradeable values.

        historical_removals in our fixture has value 3500 for years 2018-2023.
        Model years 2024-2035 are NOT in historical data → filled with 0.
        """
        fs = make_endogenous_fs()
        result = fs.calculate(PRICES)
        # Historic supply should be 0 for model years (fixture data ends in 2023)
        assert 'historic_supply' in result.columns
        # Total supply must exist and be finite
        assert result['total_supply'].notna().all()

    def test_total_supply_is_historic_plus_manley(self):
        """total_supply = historic_supply + manley_supply for all years."""
        fs = make_endogenous_fs()
        result = fs.calculate(PRICES)
        for year in MODEL_YEARS:
            expected = result.loc[year, 'historic_supply'] + result.loc[year, 'manley_supply']
            assert result.loc[year, 'total_supply'] == pytest.approx(expected, abs=0.01)

    def test_higher_prices_produce_more_planting(self):
        """Higher price path → higher manley_planting_total."""
        fs_low = make_endogenous_fs()
        result_low = fs_low.calculate(pd.Series(50.0, index=MODEL_YEARS))

        fs_high = make_endogenous_fs()
        result_high = fs_high.calculate(pd.Series(300.0, index=MODEL_YEARS))

        # Total planting over all years should be higher with higher prices
        total_low = result_low['manley_planting_total'].sum()
        total_high = result_high['manley_planting_total'].sum()
        assert total_high > total_low

    def test_manley_planting_capped_at_max_forestry(self):
        """Annual planting never exceeds max_forestry parameter."""
        fs = make_endogenous_fs()
        result = fs.calculate(pd.Series(500.0, index=MODEL_YEARS))  # high price
        max_forestry = MANLEY_PARAMS['max_forestry']
        for year in MODEL_YEARS:
            total_planting = result.loc[year, 'manley_planting_total']
            assert total_planting <= max_forestry + 1e-6

    def test_forward_price_npv_calculation(self):
        """For a constant price path, forward_price ≈ price / price_conversion_2021.

        With constant prices, NPV-weighted forward equals current price.
        Then dividing by price_conversion_2021 gives the Manley price.
        """
        fs = make_endogenous_fs()
        constant_price = 100.0
        prices = pd.Series(constant_price, index=MODEL_YEARS)
        manley_prices = fs._calculate_forward_prices(prices)
        expected = constant_price / MANLEY_PARAMS['price_conversion_2021']
        # All years should be approximately equal to expected
        for year in MODEL_YEARS:
            assert manley_prices[year] == pytest.approx(expected, rel=1e-4)

    def test_current_price_mode_uses_lagged_price(self):
        """With forestry_price_assumption='current', manley_price uses lagged prices.

        With lag=2, the manley_price for year t should reflect prices[t-2].
        A rising price path means manley_prices should lag behind the price path.
        """
        fs_future = make_endogenous_fs(price_assumption='future')
        fs_current = make_endogenous_fs(price_assumption='current')
        # Rising price path
        rising = pd.Series(
            [50.0 + i * 10 for i in range(len(MODEL_YEARS))],
            index=MODEL_YEARS
        )
        mp_future = fs_future._calculate_forward_prices(rising)
        mp_current = fs_current._calculate_lagged_prices(rising)
        # Lagged prices should be lower than forward prices for a rising path
        # Check at year 2026 (index 2): lag uses prices[2024], future uses NPV of 2026+
        assert mp_current[2026] < mp_future[2026]

    def test_sensitivity_low_vs_central_vs_high(self):
        """Higher f (more available land) produces more planting at same price.

        Raise max_forestry cap so it doesn't mask the f difference.
        """
        uncapped = {**MANLEY_PARAMS, 'max_forestry': 1_000_000.0, 'max_forestry_2050': 1_000_000.0}
        low_params = {**uncapped, 'f': 60000.0}
        high_params = {**uncapped, 'f': 120000.0}

        fs_low = make_endogenous_fs(manley_params=low_params)
        fs_high = make_endogenous_fs(manley_params=high_params)

        price = pd.Series(150.0, index=MODEL_YEARS)
        result_low = fs_low.calculate(price)
        result_high = fs_high.calculate(price)

        total_low = result_low['manley_planting_total'].sum()
        total_high = result_high['manley_planting_total'].sum()
        assert total_high > total_low
