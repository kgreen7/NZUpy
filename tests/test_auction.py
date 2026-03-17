"""
Tests for AuctionSupply — Tier 1 component isolation.

Instantiates AuctionSupply directly with synthetic data.
Fixture values:
  reserve price = 50, CCR1 trigger = 150, CCR2 trigger = 200
  base_volume = 10000, ccr_volume_1 = 1000, ccr_volume_2 = 500
"""

import pytest
import pandas as pd
import numpy as np
from model.supply.auction import AuctionSupply


YEARS = [2024, 2025, 2026]


def make_auction_data(base_volume=10000, reserve=50, ccr1=150, ccr2=200,
                      ccr_vol1=1000, ccr_vol2=500):
    """Build a minimal auction_data DataFrame for testing."""
    return pd.DataFrame({
        'base_volume': base_volume,
        'auction_reserve_price': reserve,
        'ccr_trigger_price_1': ccr1,
        'ccr_trigger_price_2': ccr2,
        'ccr_volume_1': ccr_vol1,
        'ccr_volume_2': ccr_vol2,
    }, index=YEARS)


def make_prices(value):
    """Uniform price across all test years."""
    return pd.Series(value, index=YEARS)


class TestBaseVolumes:
    def test_base_auction_volumes_loaded(self):
        """Base volumes match input data for all years."""
        auction = AuctionSupply(YEARS, make_auction_data(base_volume=10000))
        result = auction.calculate(make_prices(100))  # price above reserve, below CCR1
        assert list(result['base_auction']) == [10000, 10000, 10000]


class TestReservePrice:
    def test_no_auction_below_reserve_price(self):
        """When price < reserve price, total_auction = 0 for that year."""
        auction = AuctionSupply(YEARS, make_auction_data())
        prices = pd.Series([40, 100, 100], index=YEARS)  # 2024 below reserve
        result = auction.calculate(prices)
        assert result.loc[2024, 'total_auction'] == 0.0
        assert result.loc[2025, 'total_auction'] > 0

    def test_revenue_zero_when_below_reserve(self):
        """Revenue = 0 when price < reserve price."""
        auction = AuctionSupply(YEARS, make_auction_data())
        prices = pd.Series([40, 100, 100], index=YEARS)
        result = auction.calculate(prices)
        assert result.loc[2024, 'revenue'] == 0.0


class TestCCR1:
    def test_ccr1_triggers_at_threshold(self):
        """When price >= CCR1 trigger, ccr_auction_1 > 0."""
        auction = AuctionSupply(YEARS, make_auction_data())
        result = auction.calculate(make_prices(150))  # exactly at CCR1
        assert result.loc[2024, 'ccr_auction_1'] == 1000

    def test_ccr1_does_not_trigger_below_threshold(self):
        """When price < CCR1 trigger, ccr_auction_1 = 0."""
        auction = AuctionSupply(YEARS, make_auction_data())
        result = auction.calculate(make_prices(100))  # below CCR1
        for year in YEARS:
            assert result.loc[year, 'ccr_auction_1'] == 0.0


class TestCCR2:
    def test_ccr2_triggers_at_threshold(self):
        """When price >= CCR2 trigger, ccr_auction_2 > 0."""
        auction = AuctionSupply(YEARS, make_auction_data())
        result = auction.calculate(make_prices(200))  # exactly at CCR2
        assert result.loc[2024, 'ccr_auction_2'] == 500

    def test_ccr2_requires_ccr1(self):
        """CCR2 only triggers if price also exceeds CCR1.

        With CCR1=150 and CCR2=200, a price of exactly 200 triggers both.
        A price of 150-199 triggers CCR1 only.
        """
        auction = AuctionSupply(YEARS, make_auction_data())
        # Price between CCR1 and CCR2 triggers CCR1 but not CCR2
        result = auction.calculate(make_prices(175))
        for year in YEARS:
            assert result.loc[year, 'ccr_auction_1'] == 1000
            assert result.loc[year, 'ccr_auction_2'] == 0.0


class TestTotals:
    def test_total_auction_is_sum_of_components(self):
        """total = base + ccr1 + ccr2 for every year."""
        auction = AuctionSupply(YEARS, make_auction_data())
        result = auction.calculate(make_prices(200))  # both CCRs trigger
        for year in YEARS:
            expected = (result.loc[year, 'base_auction'] +
                        result.loc[year, 'ccr_auction_1'] +
                        result.loc[year, 'ccr_auction_2'])
            assert result.loc[year, 'total_auction'] == pytest.approx(expected)


class TestRevenue:
    def test_revenue_calculation(self):
        """Revenue = price × total_auction × 1000 (converting kt to t)."""
        auction = AuctionSupply(YEARS, make_auction_data())
        price = 100.0
        result = auction.calculate(make_prices(price))  # above reserve, below CCR1
        for year in YEARS:
            expected_revenue = price * result.loc[year, 'total_auction'] * 1000
            assert result.loc[year, 'revenue'] == pytest.approx(expected_revenue)
