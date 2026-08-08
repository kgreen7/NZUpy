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


class TestIndustrialAllocationCalculatedMode:
    """Tests for calculated mode using sector-level assumptions."""

    def make_calculated_inputs(self):
        """Create minimal test data for calculated mode."""
        base_output = pd.DataFrame({
            'Sector': ['Sector-A', 'Sector-B'],
            'Category': ['High', 'Moderate'],
            'BaseOutput': [100.0, 50.0]
        })

        phase_down = pd.DataFrame({
            'Year': [2024, 2024, 2025, 2025, 2026, 2026],
            'Category': ['High', 'Moderate', 'High', 'Moderate', 'High', 'Moderate'],
            'Rate': [0.86, 0.56, 0.85, 0.55, 0.84, 0.54]
        })

        output_red = pd.DataFrame({
            'Year': [2024, 2024, 2025, 2025, 2026, 2026],
            'Sector': ['Sector-A', 'Sector-B', 'Sector-A', 'Sector-B', 'Sector-A', 'Sector-B'],
            'Factor': [1.0, 1.0, 1.0, 1.0, 0.6, 1.0]  # Sector-A reduces to 60% in 2026
        })

        other_adj = pd.DataFrame({
            'Year': [2024, 2024, 2025, 2025, 2026, 2026],
            'Sector': ['Sector-A', 'Sector-B', 'Sector-A', 'Sector-B', 'Sector-A', 'Sector-B'],
            'Amount': [0.0, 0.0, 10.0, 0.0, 10.0, 0.0]  # Sector-A gets +10 from 2025
        })

        return base_output, phase_down, output_red, other_adj

    def test_calculated_mode_basic_allocation(self):
        """Calculated mode computes allocation from phase-down rates."""
        base_output, phase_down, output_red, other_adj = self.make_calculated_inputs()

        ia = IndustrialAllocation(
            years=YEARS,
            mode='calculated',
            base_output=base_output,
            phase_down_rates=phase_down,
            output_reduction=output_red,
            other_adjustments=other_adj
        )

        result = ia.calculate()

        # 2024: Sector-A: (100/0.86)*0.86 + 0 = 100.0
        #       Sector-B: (50/0.56)*0.56 + 0 = 50.0
        #       Total: 150.0
        assert result.loc[2024, 'baseline_allocation'] == pytest.approx(150.0, abs=0.01)

        # 2025: Sector-A: (100/0.86)*0.85 + 10 = 108.837...
        #       Sector-B: (50/0.56)*0.55 = 49.107...
        #       Total: ~157.94
        assert result.loc[2025, 'baseline_allocation'] == pytest.approx(157.94, abs=0.1)

    def test_calculated_mode_output_reduction(self):
        """Output reduction (closure) factors are applied correctly."""
        base_output, phase_down, output_red, other_adj = self.make_calculated_inputs()

        ia = IndustrialAllocation(
            years=YEARS,
            mode='calculated',
            base_output=base_output,
            phase_down_rates=phase_down,
            output_reduction=output_red,
            other_adjustments=other_adj
        )

        result = ia.calculate()

        # 2026: Sector-A has 0.6 reduction factor
        # Sector-A: ((100/0.86)*0.84 + 10) * 0.6 = (97.67 + 10) * 0.6 = 64.60
        # Sector-B: (50/0.56)*0.54 * 1.0 = 48.21
        # Total: ~112.81
        assert result.loc[2026, 'baseline_allocation'] == pytest.approx(112.81, abs=0.1)

    def test_calculated_mode_validates_output_reduction_range(self):
        """Output reduction factors must be between 0 and 1."""
        base_output, phase_down, output_red, other_adj = self.make_calculated_inputs()

        # Invalid: factor > 1
        output_red_invalid = output_red.copy()
        output_red_invalid.loc[0, 'Factor'] = 1.5

        with pytest.raises(ValueError, match="output_reduction Factor must be in"):
            IndustrialAllocation(
                years=YEARS,
                mode='calculated',
                base_output=base_output,
                phase_down_rates=phase_down,
                output_reduction=output_red_invalid,
                other_adjustments=other_adj
            )

    def test_calculated_mode_requires_inputs(self):
        """Calculated mode raises clear errors when required inputs missing."""
        base_output, phase_down, output_red, other_adj = self.make_calculated_inputs()

        with pytest.raises(ValueError, match="base_output is required"):
            IndustrialAllocation(
                years=YEARS,
                mode='calculated',
                base_output=None,
                phase_down_rates=phase_down,
                output_reduction=output_red
            )

        with pytest.raises(ValueError, match="phase_down_rates is required"):
            IndustrialAllocation(
                years=YEARS,
                mode='calculated',
                base_output=base_output,
                phase_down_rates=None,
                output_reduction=output_red
            )

    def test_backward_compatibility_old_signature(self):
        """Old signature IndustrialAllocation(years, ia_data) still works."""
        # This should work exactly as before (auto-detected as fixed mode)
        ia = IndustrialAllocation(YEARS, make_ia_data())
        result = ia.calculate()
        assert ia.mode == 'fixed'
        assert list(result['baseline_allocation']) == [5000, 4800, 4600]
