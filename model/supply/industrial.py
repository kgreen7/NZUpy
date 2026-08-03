"""
Industrial allocation supply component for the NZ ETS model.

Supports two modes:
- 'fixed': a single pre-computed baseline_allocation series is used directly (default
  prior to May 2026 update; still fully supported for users who want to swap in an
  alternative series without engaging with the underlying assumptions).
- 'calculated' (new default): baseline_allocation is derived from a phase-down rate
  schedule, sector-level output-reduction/closure assumptions, base-year sector
  output, and other one-off adjustments. Computed once at setup — this does not
  respond to price and is not part of the solver iteration loop.

Methodology follows the Climate Change Commission (CCC) approach, which is the same
underlying methodology used in the May 2026 MfE Excel model but laid out more cleanly.
"""

import pandas as pd
from typing import Dict, Optional, List, Any


class IndustrialAllocation:
    """
    Industrial allocation supply component for NZ ETS model.

    In 'fixed' mode, passes through a single pre-computed baseline_allocation series.

    In 'calculated' mode, computes allocation from:
    - Base output per sector (2024 base year)
    - Phase-down rates by trade-exposure category (high/moderate)
    - Output reduction/closure assumptions by sector
    - Other one-off adjustments (e.g., Aluminium's contract adjustment)

    Formula (calculated mode):
        allocation[s, year] = (base_output[s] / base_rate[c]) * phase_down_rate[c, year]
        total_before_reduction[s, year] = allocation[s, year] + other_adjustments[s, year]
        total[s, year] = total_before_reduction[s, year] * output_reduction[s, year]
        baseline_allocation[year] = sum over all sectors s of total[s, year]

    Where:
        - base_output[s]: 2024 base allocation for sector s
        - base_rate[c]: 2024 allocation rate for category c (0.86 for high, 0.56 for moderate)
        - phase_down_rate[c, year]: allocation rate schedule for category c
        - output_reduction[s, year]: 0-1 factor for sector s (e.g., 0.6 = 40% closure)
        - other_adjustments[s, year]: flat NZU adjustment (e.g., Aluminium contract)

    Fixed sector list (11 sectors, 2 categories):
        High (0.86):     Aluminium, Cement and lime, Iron and steel, Methanol,
                         Other-High, Pulp and paper products, Urea
        Moderate (0.56): Dairy products, Horticulture, Meat products, Other-Moderate
    """

    # Base rates for 2024 (from CCC workbook row 4-5, column C)
    BASE_RATES = {
        'High': 0.86,
        'Moderate': 0.56,
    }

    def __init__(
        self,
        years: List[int],
        mode: str = 'calculated',
        # 'fixed' mode parameter
        ia_data: Optional[pd.DataFrame] = None,
        # 'calculated' mode parameters
        base_output: Optional[pd.DataFrame] = None,
        phase_down_rates: Optional[pd.DataFrame] = None,
        output_reduction: Optional[pd.DataFrame] = None,
        other_adjustments: Optional[pd.DataFrame] = None,
    ):
        """
        Initialise the industrial allocation component.

        Args:
            years: List of years for the model.
            mode: 'calculated' (default) or 'fixed'.
                  NOTE: For backward compatibility, if a DataFrame is passed as the
                  second positional argument, it's treated as ia_data in fixed mode.

            For 'fixed' mode:
                ia_data: DataFrame with industrial allocation data indexed by year.
                    Required column: 'baseline_allocation'

            For 'calculated' mode:
                base_output: DataFrame with columns: Sector, Category, BaseOutput
                phase_down_rates: DataFrame with columns: Year, Category, Rate
                output_reduction: DataFrame with columns: Year, Sector, Factor (0-1)
                other_adjustments: DataFrame with columns: Year, Sector, Amount
        """
        self.years = years

        # Backward compatibility: if mode is a DataFrame, treat it as old signature
        # Old: IndustrialAllocation(years, ia_data)
        # New: IndustrialAllocation(years, mode='calculated', ia_data=...)
        if isinstance(mode, pd.DataFrame):
            # Old signature detected: second arg is ia_data
            ia_data = mode
            mode = 'fixed'

        self.mode = mode

        if mode == 'fixed':
            self._init_fixed_mode(ia_data)
        elif mode == 'calculated':
            self._init_calculated_mode(base_output, phase_down_rates, output_reduction, other_adjustments)
        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'fixed' or 'calculated'."
            )

        # Initialise results DataFrame
        self.results = pd.DataFrame(
            index=self.years,
            columns=['baseline_allocation', 'adjusted_allocation'],
            data=0.0
        )

    def _init_fixed_mode(self, ia_data: Optional[pd.DataFrame]):
        """Initialise fixed mode with pre-computed allocation series."""
        if ia_data is None:
            raise ValueError(
                "ia_data is required for 'fixed' mode industrial allocation. "
                "Provide a DataFrame with 'baseline_allocation' column."
            )

        if 'baseline_allocation' not in ia_data.columns:
            raise ValueError(
                "ia_data must contain 'baseline_allocation' column for 'fixed' mode."
            )

        self.baseline_allocation = ia_data['baseline_allocation'].astype(float)

    def _init_calculated_mode(
        self,
        base_output: Optional[pd.DataFrame],
        phase_down_rates: Optional[pd.DataFrame],
        output_reduction: Optional[pd.DataFrame],
        other_adjustments: Optional[pd.DataFrame],
    ):
        """
        Initialise calculated mode with sector-level assumptions.

        Validates inputs and computes baseline_allocation series once at init.
        """
        # Validate required inputs
        if base_output is None:
            raise ValueError(
                "base_output is required for 'calculated' mode. "
                "Provide a DataFrame with columns: Sector, Category, BaseOutput."
            )
        if phase_down_rates is None:
            raise ValueError(
                "phase_down_rates is required for 'calculated' mode. "
                "Provide a DataFrame with columns: Year, Category, Rate."
            )
        if output_reduction is None:
            raise ValueError(
                "output_reduction is required for 'calculated' mode. "
                "Provide a DataFrame with columns: Year, Sector, Factor."
            )

        # other_adjustments defaults to all zeros if not provided
        if other_adjustments is None:
            # Create empty other adjustments DataFrame
            sectors = base_output['Sector'].unique()
            other_adjustments = pd.DataFrame({
                'Year': [y for y in self.years for _ in sectors],
                'Sector': list(sectors) * len(self.years),
                'Amount': 0.0,
            })

        # Validate output_reduction factors are in [0, 1]
        if not output_reduction['Factor'].between(0, 1, inclusive='both').all():
            invalid = output_reduction[~output_reduction['Factor'].between(0, 1, inclusive='both')]
            raise ValueError(
                f"output_reduction Factor must be in [0, 1]. Found invalid values:\n{invalid}"
            )

        # Store data
        self.base_output = base_output.set_index('Sector')
        self.phase_down_rates = phase_down_rates
        self.output_reduction = output_reduction
        self.other_adjustments = other_adjustments

        # Compute baseline allocation series
        self.baseline_allocation = self._compute_baseline_allocation()

    def _compute_baseline_allocation(self) -> pd.Series:
        """
        Compute baseline allocation series for all years in 'calculated' mode.

        Implements the CCC methodology:
        For each sector s:
            allocation[s, year] = (base_output[s] / base_rate[category]) * phase_rate[category, year]
            before_reduction[s, year] = allocation[s, year] + other_adj[s, year]
            total[s, year] = before_reduction[s, year] * output_reduction[s, year]

        baseline_allocation[year] = sum over sectors of total[s, year]

        Returns:
            pd.Series indexed by year with total allocation across all sectors.
        """
        # Pivot phase-down rates for easy lookup
        phase_rates = self.phase_down_rates.pivot(
            index='Year', columns='Category', values='Rate'
        )

        # Pivot output reductions and other adjustments
        output_red = self.output_reduction.pivot(
            index='Year', columns='Sector', values='Factor'
        ).fillna(1.0)  # Default to no reduction if missing

        other_adj = self.other_adjustments.pivot(
            index='Year', columns='Sector', values='Amount'
        ).fillna(0.0)  # Default to no adjustment if missing

        # Initialize total allocation series
        total_allocation = pd.Series(0.0, index=self.years)

        # Calculate for each sector
        for sector in self.base_output.index:
            base = self.base_output.loc[sector, 'BaseOutput']
            category = self.base_output.loc[sector, 'Category']
            base_rate = self.BASE_RATES[category]

            for year in self.years:
                if year not in phase_rates.index:
                    # Year not in phase-down schedule, skip
                    continue

                # Step 1: Calculate base allocation with phase-down rate
                # Matches CCC Excel formula: =(base / base_rate) * phase_rate
                phase_rate = phase_rates.loc[year, category]
                allocation = (base / base_rate) * phase_rate

                # Step 2: Add other adjustments
                adj = other_adj.loc[year, sector] if sector in other_adj.columns and year in other_adj.index else 0.0
                before_reduction = allocation + adj

                # Step 3: Apply output reduction/closure factor
                reduction = output_red.loc[year, sector] if sector in output_red.columns and year in output_red.index else 1.0
                sector_total = before_reduction * reduction

                total_allocation[year] += sector_total

        return total_allocation

    def calculate(self, prices: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Calculate industrial allocation based on mode.

        Args:
            prices: Optional series of NZU prices indexed by year.
                   Not used in either mode (industrial allocation is price-independent)
                   but kept for API compatibility.

        Returns:
            DataFrame containing industrial allocation volumes by year.
        """
        # Set both baseline and adjusted allocation to baseline_allocation
        # (no price response or activity adjustment applied beyond what's
        # already in the calculated/fixed series)
        self.results['baseline_allocation'] = self.baseline_allocation
        self.results['adjusted_allocation'] = self.baseline_allocation

        return self.results

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the industrial allocation component to a dictionary.

        Returns:
            Dictionary representation of the industrial allocation component.
        """
        return {
            'mode': self.mode,
            'baseline_allocation': self.baseline_allocation.to_dict(),
            'results': self.results.to_dict()
        }
