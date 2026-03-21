"""
Tests for PriceResponse — Tier 1 component isolation.

Uses simple custom parameters where the formulas can be verified by hand.
"""

import pytest
import numpy as np
import pandas as pd
from model.demand.price_response import PriceResponse


# Enough years to test multi-year behaviour; extended so NPV can look forward
YEARS = list(range(2024, 2035))
BASE_EMISSIONS = pd.Series(30000.0, index=YEARS)

# Minimal params: price_conversion_factor=1.0 so adjusted_price = forward_price
SIMPLE_PARAMS_M2 = {
    'constant': 0.0,
    'reduction_to_t1': 0.0,  # ignore previous reduction
    'price': 1.0,             # coef on ln(price)
    'model_number': 2,
    'discount_rate': 0.05,
    'forward_years': 3,
    'price_conversion_factor': 1.0,
}

SIMPLE_PARAMS_M1 = {
    'constant': 100.0,
    'reduction_to_t1': 0.0,
    'price': 0.0,
    'model_number': 1,
    'discount_rate': 0.05,
    'forward_years': 3,
    'price_conversion_factor': 1.0,
}


def constant_prices(value=100.0):
    return pd.Series(value, index=YEARS)


class TestModel2Formula:
    def test_model2_log_linear_formula(self):
        """For constant prices and simple params, first-year reduction = exp(ln(forward_price)).

        With constant=0, reduction_to_t1=0, price_coef=1, price_conversion_factor=1:
        new_reduction = exp(ln(1.0)*0 + ln(forward_price)*1 + 0) = forward_price.
        For constant price path, forward_price == current_price == 100.
        """
        pr = PriceResponse(YEARS, SIMPLE_PARAMS_M2, base_price=100.0)
        result = pr.calculate(constant_prices(100.0), BASE_EMISSIONS)
        # First year: prev_reduction=1.0 (ignored), forward=100 → reduction=100
        assert result.loc[2024, 'emissions_reduction'] == pytest.approx(100.0, rel=1e-4)


class TestModel1Formula:
    def test_model1_linear_formula(self):
        """With constant=100, reduction_to_t1=0, price=0, result = 100 every year."""
        pr = PriceResponse(YEARS, SIMPLE_PARAMS_M1, base_price=100.0)
        result = pr.calculate(constant_prices(100.0), BASE_EMISSIONS)
        # reduction = max(0, 1.0*0 + 100/1*0 + 100) = 100
        # For years 2024-2029 (below 2030), no extra additions
        assert result.loc[2024, 'emissions_reduction'] == pytest.approx(100.0, abs=0.1)


class TestPriceEffect:
    def test_higher_prices_produce_more_reduction(self):
        """Higher prices → larger emissions_reduction (model 2)."""
        params = {
            'constant': 0.0,
            'reduction_to_t1': 0.0,
            'price': 1.0,
            'model_number': 2,
            'discount_rate': 0.05,
            'forward_years': 3,
            'price_conversion_factor': 1.0,
        }
        pr_low = PriceResponse(YEARS, params, base_price=50.0)
        result_low = pr_low.calculate(constant_prices(50.0), BASE_EMISSIONS)

        pr_high = PriceResponse(YEARS, params, base_price=200.0)
        result_high = pr_high.calculate(constant_prices(200.0), BASE_EMISSIONS)

        # Higher price → higher reduction (exp(ln(200)) > exp(ln(50)))
        assert result_high.loc[2024, 'emissions_reduction'] > result_low.loc[2024, 'emissions_reduction']


class TestForwardPriceNPV:
    def test_forward_price_npv_calculation(self):
        """For a constant price path, NPV forward price == current price."""
        pr = PriceResponse(YEARS, SIMPLE_PARAMS_M2, base_price=100.0)
        forward = pr.calculate_forward_price_with_npv(2024, constant_prices(100.0))
        assert forward == pytest.approx(100.0, rel=1e-6)


class TestReductionBounds:
    def test_reduction_never_exceeds_baseline(self):
        """emissions_reduction ≤ base_emissions for every year."""
        pr = PriceResponse(YEARS, SIMPLE_PARAMS_M2, base_price=100.0)
        result = pr.calculate(constant_prices(100.0), BASE_EMISSIONS)
        for year in YEARS:
            assert result.loc[year, 'emissions_reduction'] <= BASE_EMISSIONS[year] + 1e-9

    def test_reduction_non_negative(self):
        """emissions_reduction ≥ 0 for every year."""
        pr = PriceResponse(YEARS, SIMPLE_PARAMS_M2, base_price=0.01)
        prices = constant_prices(0.01)  # very low price → small reduction
        result = pr.calculate(prices, BASE_EMISSIONS)
        for year in YEARS:
            assert result.loc[year, 'emissions_reduction'] >= 0.0
