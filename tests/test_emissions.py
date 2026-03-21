"""
Tests for EmissionsDemand — Tier 1 component isolation.
"""

import pytest
import pandas as pd
import numpy as np
from model.demand.emissions import EmissionsDemand


YEARS = [2024, 2025, 2026]

# Fixture-style emissions data with two configs
EMISSIONS_DATA = pd.DataFrame({
    'Year': YEARS * 2,
    'Config': ['central'] * 3 + ['CCC_CPR'] * 3,
    'Value': [30000, 29500, 29000, 28000, 27500, 27000],
    'Metadata': [''] * 6,
})


class TestBaselineEmissions:
    def test_baseline_emissions_loaded_from_config(self):
        """Emissions match the selected config's values."""
        em = EmissionsDemand(YEARS, EMISSIONS_DATA, config_name='central')
        result = em.calculate()
        assert list(result['baseline_emissions']) == [30000, 29500, 29000]

    def test_price_response_reduces_emissions(self):
        """When price_response provided, total_demand < baseline_emissions."""
        em = EmissionsDemand(YEARS, EMISSIONS_DATA, config_name='central')
        response = pd.Series(500.0, index=YEARS)
        result = em.calculate(price_response=response)
        for year in YEARS:
            assert result.loc[year, 'total_demand'] < result.loc[year, 'baseline_emissions']

    def test_emissions_never_negative(self):
        """Even with a very large price response, emissions ≥ 0."""
        em = EmissionsDemand(YEARS, EMISSIONS_DATA, config_name='central')
        huge_response = pd.Series(1_000_000.0, index=YEARS)
        result = em.calculate(price_response=huge_response)
        for year in YEARS:
            assert result.loc[year, 'total_demand'] >= 0.0

    def test_invalid_config_raises(self):
        """Requesting a nonexistent config raises ValueError."""
        em = EmissionsDemand(YEARS, EMISSIONS_DATA, config_name='central')
        with pytest.raises(ValueError):
            em.set_config('not_a_config')
