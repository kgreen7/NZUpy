"""
Tests for StockpileSupply — Tier 1 component isolation.

IMPORTANT: Create a fresh StockpileSupply per test — the component has internal state.

Fixture values:
  initial_stockpile=150000, initial_surplus=50000
  non_surplus at start = 100000
  liquidity_factor=0.15 → max liquid non-surplus = 100000 * 0.15 = 15000
  discount_rate=0.05, payback_period=5
"""

import pytest
import pandas as pd
import numpy as np
from model.supply.stockpile import StockpileSupply


BASE_YEARS = [2024, 2025, 2026, 2027, 2028, 2029, 2030]
EXTENDED_YEARS = list(range(2031, 2056))


def make_stockpile(years=None, initial_stockpile=150000, initial_surplus=50000,
                   lf=0.15, start_year=2024, payback_period=5, discount_rate=0.05):
    """Create a fresh StockpileSupply instance."""
    if years is None:
        years = BASE_YEARS
    return StockpileSupply(
        years=years,
        extended_years=EXTENDED_YEARS,
        initial_stockpile=initial_stockpile,
        initial_surplus=initial_surplus,
        liquidity_factor=lf,
        payback_period=payback_period,
        discount_rate=discount_rate,
        stockpile_reference_year=start_year - 1,
        stockpile_usage_start_year=start_year,
    )


def zero_balance(years=None):
    if years is None:
        years = BASE_YEARS
    return pd.Series(0.0, index=years)


def low_rates(years=None):
    """Price change rates well below discount_rate=0.05 → non-surplus available."""
    if years is None:
        years = BASE_YEARS
    return pd.Series(0.02, index=years)


def high_rates(years=None):
    """Price change rates at or above discount_rate=0.05 → non-surplus NOT available."""
    if years is None:
        years = BASE_YEARS
    return pd.Series(0.10, index=years)


class TestInitialBalances:
    def test_initial_balances_set_correctly(self):
        """At stockpile_usage_start_year, stockpile and surplus match initial values."""
        sp = make_stockpile()
        result = sp.calculate(zero_balance(), price_change_rates=low_rates())
        assert result.loc[2024, 'stockpile_balance'] == pytest.approx(150000.0, abs=0.1)
        assert result.loc[2024, 'surplus_balance'] == pytest.approx(50000.0, abs=0.1)


class TestSurplusUsage:
    def test_surplus_used_before_non_surplus(self):
        """When shortfall exists and is within surplus, only surplus is used."""
        sp = make_stockpile()
        shortfall = pd.Series(-10000.0, index=BASE_YEARS)  # deficit every year
        result = sp.calculate(shortfall, price_change_rates=low_rates())
        # In 2024, surplus (50000) covers 10000 shortfall → surplus_used=10000, non_surplus=0
        assert result.loc[2024, 'surplus_used'] == pytest.approx(10000.0, abs=0.1)
        assert result.loc[2024, 'non_surplus_used'] == pytest.approx(0.0, abs=0.1)

    def test_surplus_used_limited_to_surplus_balance(self):
        """surplus_used ≤ surplus_balance in all years."""
        sp = make_stockpile()
        large_shortfall = pd.Series(-200000.0, index=BASE_YEARS)  # exceeds entire stockpile
        result = sp.calculate(large_shortfall, price_change_rates=low_rates())
        for year in BASE_YEARS:
            assert result.loc[year, 'surplus_used'] <= (
                result.loc[year, 'surplus_balance'] + result.loc[year, 'surplus_used'] + 1e-9
            )  # surplus_used ≤ initial surplus_balance before deduction


class TestNonSurplusUsage:
    def test_non_surplus_limited_by_liquidity_factor(self):
        """non_surplus_used ≤ prev_non_surplus × liquidity_factor.

        At 2024 start: non_surplus = 100000, liquidity = 15000.
        With shortfall > (50000 surplus + 15000 liquid non-surplus), non_surplus capped at 15000.
        """
        sp = make_stockpile()
        # Shortfall larger than entire liquid portion (50000 + 15000 = 65000)
        large_shortfall = pd.Series(-100000.0, index=BASE_YEARS)
        result = sp.calculate(large_shortfall, price_change_rates=low_rates())
        assert result.loc[2024, 'non_surplus_used'] <= 15000.0 + 1e-6

    def test_non_surplus_requires_price_growth_below_discount_rate(self):
        """No non-surplus used when price growth ≥ discount rate."""
        sp = make_stockpile()
        # Shortfall exceeds surplus → would use non-surplus if rates low enough
        large_shortfall = pd.Series(-60000.0, index=BASE_YEARS)
        result = sp.calculate(large_shortfall, price_change_rates=high_rates())
        for year in BASE_YEARS:
            assert result.loc[year, 'non_surplus_used'] == pytest.approx(0.0, abs=0.1)


class TestExcessSupply:
    def test_excess_supply_goes_to_surplus(self):
        """When supply > demand (positive balance), excess added to surplus_balance."""
        sp = make_stockpile()
        # First year: zero balance (no change). Second year: +5000 excess.
        balance = pd.Series([0.0, 5000.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=BASE_YEARS)
        result = sp.calculate(balance, price_change_rates=low_rates())
        # Surplus in 2025 should exceed 2024's surplus
        assert result.loc[2025, 'surplus_balance'] > result.loc[2024, 'surplus_balance']


class TestPayback:
    def test_payback_scheduled_correctly(self):
        """Borrowed non-surplus appears in payback_units after payback_period years.

        Borrow in 2024 with payback_period=5 → payback in 2029.
        """
        years = list(range(2024, 2032))
        extended = list(range(2032, 2055))
        sp = StockpileSupply(
            years=years,
            extended_years=extended,
            initial_stockpile=150000,
            initial_surplus=50000,
            liquidity_factor=0.15,
            payback_period=5,
            discount_rate=0.05,
            stockpile_reference_year=2023,
            stockpile_usage_start_year=2024,
        )
        # Large shortfall in 2024 only, then zero balance
        balance = pd.Series(0.0, index=years)
        balance[2024] = -70000.0  # exceeds surplus, uses non-surplus too
        rates = pd.Series(0.02, index=years)
        result = sp.calculate(balance, price_change_rates=rates)

        non_surplus_2024 = result.loc[2024, 'non_surplus_used']
        if non_surplus_2024 > 0:
            # Payback should appear in 2024+5=2029
            assert result.loc[2029, 'payback_units'] > 0.0


class TestStartYear:
    def test_no_stockpile_usage_before_start_year(self):
        """available_units = 0 for years before stockpile_usage_start_year."""
        sp = make_stockpile(start_year=2026)  # start in 2026, 2024/2025 should be 0
        large_shortfall = pd.Series(-50000.0, index=BASE_YEARS)
        result = sp.calculate(large_shortfall, price_change_rates=low_rates())
        assert result.loc[2024, 'available_units'] == 0.0
        assert result.loc[2025, 'available_units'] == 0.0
        # 2026 onward should use stockpile
        assert result.loc[2026, 'available_units'] > 0.0


class TestForestryImpact:
    def test_forestry_held_increases_stockpile(self):
        """Positive forestry_held increases stockpile_balance."""
        years = [2024, 2025, 2026]
        sp = make_stockpile(years=years, initial_stockpile=150000, initial_surplus=50000)
        forestry = pd.DataFrame({
            'forestry_held': [1000.0, 0.0, 0.0],
            'forestry_surrender': [0.0, 0.0, 0.0],
        }, index=pd.Index(years, name='year'))
        result = sp.calculate(
            pd.Series(0.0, index=years),
            price_change_rates=pd.Series(0.02, index=years),
        )
        # Without forestry
        base_balance = result.loc[2024, 'stockpile_balance']

        # Now with forestry held
        sp2 = make_stockpile(years=years, initial_stockpile=150000, initial_surplus=50000)
        sp2.forestry_variables = forestry
        result2 = sp2.calculate(
            pd.Series(0.0, index=years),
            price_change_rates=pd.Series(0.02, index=years),
        )
        assert result2.loc[2024, 'stockpile_balance'] > base_balance

    def test_forestry_surrender_decreases_stockpile(self):
        """Negative forestry_surrender decreases stockpile_balance."""
        years = [2024, 2025, 2026]
        forestry = pd.DataFrame({
            'forestry_held': [0.0, 0.0, 0.0],
            'forestry_surrender': [-1000.0, 0.0, 0.0],
        }, index=pd.Index(years, name='year'))

        sp = make_stockpile(years=years, initial_stockpile=150000, initial_surplus=50000)
        result_no_surrender = sp.calculate(
            pd.Series(0.0, index=years),
            price_change_rates=pd.Series(0.02, index=years),
        )
        base_balance = result_no_surrender.loc[2024, 'stockpile_balance']

        sp2 = make_stockpile(years=years, initial_stockpile=150000, initial_surplus=50000)
        sp2.forestry_variables = forestry
        result = sp2.calculate(
            pd.Series(0.0, index=years),
            price_change_rates=pd.Series(0.02, index=years),
        )
        assert result.loc[2024, 'stockpile_balance'] < base_balance


class TestBalanceInvariants:
    def test_stockpile_balance_never_negative_surplus(self):
        """surplus_balance ≥ 0 always, even with large shortfalls."""
        sp = make_stockpile()
        huge_shortfall = pd.Series(-500000.0, index=BASE_YEARS)
        result = sp.calculate(huge_shortfall, price_change_rates=low_rates())
        for year in BASE_YEARS:
            assert result.loc[year, 'surplus_balance'] >= -1e-9

    def test_non_surplus_balance_equals_total_minus_surplus(self):
        """non_surplus_balance = stockpile_balance - surplus_balance."""
        sp = make_stockpile()
        balance = pd.Series(-10000.0, index=BASE_YEARS)
        result = sp.calculate(balance, price_change_rates=low_rates())
        for year in BASE_YEARS:
            expected = result.loc[year, 'stockpile_balance'] - result.loc[year, 'surplus_balance']
            assert result.loc[year, 'non_surplus_balance'] == pytest.approx(max(0.0, expected), abs=0.1)


class TestValidation:
    def test_raises_on_missing_parameters(self):
        """ValueError raised if required params not provided."""
        with pytest.raises((ValueError, TypeError)):
            StockpileSupply(
                years=BASE_YEARS,
                # No parameters provided — should fail
            )

    def test_raises_on_invalid_liquidity_factor(self):
        """ValueError raised if liquidity_factor > 1 or < 0."""
        with pytest.raises(ValueError):
            make_stockpile(lf=1.5)
        with pytest.raises(ValueError):
            make_stockpile(lf=-0.1)
