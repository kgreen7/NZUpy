"""
Tests for IndustrialAllocation — Tier 1 component isolation.

IndustrialAllocation is a simple passthrough: adjusted_allocation == baseline_allocation.
"""

import pytest
import pandas as pd
from model.supply.industrial import IndustrialAllocation


YEARS = [2024, 2025, 2026]


def make_ia_data(values=None):
    """Build a minimal ia_data DataFrame."""
    if values is None:
        values = [5000, 4800, 4600]
    return pd.DataFrame({'baseline_allocation': values}, index=YEARS)


class TestIndustrialAllocation:
    def test_allocation_matches_input(self):
        """adjusted_allocation equals baseline_allocation for all years."""
        ia = IndustrialAllocation(YEARS, make_ia_data())
        result = ia.calculate()
        assert list(result['baseline_allocation']) == [5000, 4800, 4600]
        assert list(result['adjusted_allocation']) == [5000, 4800, 4600]

    def test_handles_zero_allocation(self):
        """Years with 0 allocation produce 0 output."""
        ia = IndustrialAllocation(YEARS, make_ia_data([0, 4800, 0]))
        result = ia.calculate()
        assert result.loc[2024, 'adjusted_allocation'] == 0.0
        assert result.loc[2025, 'adjusted_allocation'] == 4800
        assert result.loc[2026, 'adjusted_allocation'] == 0.0
