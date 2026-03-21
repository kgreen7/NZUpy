"""
Forestry supply component for the NZ ETS model.

Supports two modes:
- 'exogenous': static supply from removals.csv (default)
- 'endogenous': price-responsive Manley logistic equation for new-forest planting,
  combined with historic (old-forest) exogenous data from historical_removals.csv.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any


class ForestrySupply:
    """
    Forestry supply component for the NZ ETS model.

    In exogenous mode (default), supply is read directly from removals.csv.

    In endogenous mode (Manley), new-forest supply is calculated each solver
    iteration via the Manley logistic equation → vintage convolution chain.
    Historic (old-forest) supply, held, and surrender come from
    historical_removals.csv and are price-independent.
    """

    def __init__(
        self,
        years: List[int],
        forestry_data: pd.DataFrame,
        mode: str = 'exogenous',
        manley_config: Optional[Any] = None,  # ComponentConfig dataclass
        historical_removals: Optional[pd.DataFrame] = None,
        yield_increments: Optional[Dict[str, np.ndarray]] = None,
        afforestation_projections: Optional[pd.DataFrame] = None,
        manley_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialise the forestry supply component.

        Args:
            years: Model years.
            forestry_data: DataFrame with exogenous forestry data (from removals.csv).
                           Required column: 'forestry_tradeable'. Used in exogenous mode only.
            mode: 'exogenous' (default) or 'endogenous'.
            manley_config: ComponentConfig fields that control Manley behaviour
                           (forestry_price_assumption, manley_f, manley_LMV, etc.).
            historical_removals: DataFrame indexed by year with columns
                               historic_forestry_tradeable, historic_forestry_held,
                               historic_forestry_surrender. Required in endogenous mode.
            yield_increments: Pre-computed annual yield increments per forest type
                              {type: np.ndarray of tCO2/ha increments by age}.
            afforestation_projections: MPI planting projections DataFrame (year × forest type).
                                       Used for proportioning Manley total by forest type.
            manley_params: Dict of Manley equation parameters from manley_parameters.csv.
        """
        self.years = years
        self.forestry_data = forestry_data
        self.mode = mode

        # Exogenous-mode removals (tradeable units from removals.csv)
        self.removals = forestry_data['forestry_tradeable']

        # ---- Endogenous-mode data ----
        self.historical_removals = historical_removals
        self.yield_increments = yield_increments
        self.afforestation_projections = afforestation_projections
        self.manley_params = manley_params or {}
        self.manley_config = manley_config or {}

        if mode == 'endogenous':
            self._init_endogenous()

        # Results DataFrame (all columns defined upfront; unused ones stay 0)
        self.results = pd.DataFrame(
            index=self.years,
            columns=[
                'static_supply',
                'historic_supply',
                'manley_supply',
                'total_supply',
                'manley_price',
                'manley_planting_total',
                'manley_planting_permanent',
                'manley_planting_production',
                'manley_planting_natural',
            ],
            data=0.0,
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_endogenous(self):
        """Validate and pre-process data for endogenous mode."""
        if self.historical_removals is None:
            raise ValueError(
                "historical_removals required for endogenous forestry mode. "
                "Ensure historical_removals.csv is loaded."
            )
        if self.yield_increments is None:
            raise ValueError(
                "yield_increments required for endogenous forestry mode. "
                "Ensure yield_tables.csv is loaded."
            )
        if self.afforestation_projections is None:
            raise ValueError(
                "afforestation_projections required for endogenous forestry mode. "
                "Ensure afforestation_projections.csv is loaded."
            )

        # Align historical_removals to model years (fill missing years with 0)
        self.historic_tradeable = (
            self.historical_removals['historic_forestry_tradeable']
            .reindex(self.years, fill_value=0.0)
        )
        self.historic_held = (
            self.historical_removals['historic_forestry_held']
            .reindex(self.years, fill_value=0.0)
        )
        self.historic_surrender = (
            self.historical_removals['historic_forestry_surrender']
            .reindex(self.years, fill_value=0.0)
        )

        # Extract Manley parameters with ComponentConfig overrides.
        # manley_config is a ComponentConfig dataclass; use getattr() throughout.
        p = self.manley_params
        cfg = self.manley_config  # ComponentConfig dataclass or None

        def _cfg(attr, csv_key, default):
            """Return ComponentConfig override if set, else CSV param, else default."""
            override = getattr(cfg, attr, None) if cfg is not None else None
            return override if override is not None else p.get(csv_key, default)

        self.price_assumption = (
            getattr(cfg, 'forestry_price_assumption', 'future')
            if cfg is not None else 'future'
        )
        self.discount_rate = _cfg('forestry_discount_rate', 'forestry_discount_rate', 0.08)
        self.forward_years = int(_cfg('forestry_forward_years', 'forestry_forward_years', 15))
        self.price_conversion_2021 = p.get('price_conversion_2021', 1.1404494382022472)
        self.price_lag = int(p.get('price_lag', 2))

        self.g = p.get('g', 4844.0)
        self.h = p.get('h', 0.0005292)
        self.LEV_logs = p.get('LEV_logs', 1965.333333)
        self.LEV_constant = p.get('LEV_constant', 0.8262)
        self.LEV_carbon_per_dollar = p.get('LEV_carbon_per_dollar', 175.77647551273383)
        self.LMV_constant = p.get('LMV_constant', 0.4329)

        # f (available land, '000 ha): varies by manley_sensitivity, overridable
        self.manley_f = _cfg('manley_f', 'f', 100000.0)
        # LMV (land market value): varies by manley_sensitivity, overridable
        self.LMV = _cfg('manley_LMV', 'LMV', 10000.0)
        # LUC limit (max annual new planting in ha)
        self.LUC_limit = _cfg('manley_LUC_limit', 'LUC_limit', 100000.0)
        self.max_forestry = p.get('max_forestry', 50000.0)
        self.max_forestry_2050 = p.get('max_forestry_2050', 50000.0)
        self.max_aggregate_afforestation = p.get('max_aggregate_afforestation', 1_000_000_000.0)

        n = len(self.years)

        # f[t]: logistic asymptote in '000 ha — always sensitivity-specific (low/central/high)
        self._f_series = pd.Series(self.manley_f / 1000, index=self.years)

        # max_forestry cap[t]: separate hard cap on resultant annual planting (in ha),
        # linearly interpolated from max_forestry (model start) to max_forestry_2050.
        # Both default to 50,000 ha in manley_parameters.csv.
        self._max_forestry_cap = np.linspace(
            self.max_forestry, self.max_forestry_2050, n
        )

        # Align afforestation projections to cover at least the model years
        # (forward-fill if projections don't reach end year)
        self._aff_proj = self.afforestation_projections.reindex(
            self.years, method='ffill'
        ).bfill().fillna(0.0)

        # Compute MPI proportions by year (summing the three forest types)
        types = ['permanent_exotic', 'production_exotic', 'natural_forest']
        self._mpi_totals = self._aff_proj[types].sum(axis=1).replace(0, np.nan)
        self._mpi_proportions = self._aff_proj[types].div(self._mpi_totals, axis=0).fillna(
            1 / 3  # equal split if no projection data
        )

    # ------------------------------------------------------------------
    # Main entry point (called on every solver iteration)
    # ------------------------------------------------------------------

    def calculate(self, prices: pd.Series) -> pd.DataFrame:
        """
        Calculate forestry supply. Called on every solver iteration.

        Args:
            prices: NZU price series indexed by year.

        Returns:
            Results DataFrame with supply columns.
        """
        if self.mode == 'exogenous':
            return self._calculate_exogenous(prices)
        else:
            return self._calculate_endogenous(prices)

    # ------------------------------------------------------------------
    # Exogenous mode (unchanged from Phase 1/2)
    # ------------------------------------------------------------------

    def _calculate_exogenous(self, prices: pd.Series) -> pd.DataFrame:
        self.results['static_supply'] = self.removals
        self.results['total_supply'] = self.results['static_supply']
        return self.results

    # ------------------------------------------------------------------
    # Endogenous mode: Manley calculation chain
    # ------------------------------------------------------------------

    def _calculate_endogenous(self, prices: pd.Series) -> pd.DataFrame:
        # Step 2: Forward/lagged price → Manley price (2021$)
        manley_prices = self._calculate_manley_prices(prices)

        # Step 3+4: Logistic equation → cumulative → annual planting → proportioning
        planting_by_type = self._calculate_manley_planting(manley_prices)

        # Step 5+6: Vintage convolution → new-forest absorption (kt CO2-e)
        new_forest_absorption = self._vintage_convolution(planting_by_type)

        # Step 7: Combine with historic data
        self.results['historic_supply'] = self.historic_tradeable
        self.results['manley_supply'] = new_forest_absorption
        self.results['total_supply'] = (
            self.results['historic_supply'] + self.results['manley_supply']
        )

        # Diagnostics
        self.results['manley_price'] = manley_prices
        self.results['manley_planting_total'] = (
            planting_by_type['permanent_exotic']
            + planting_by_type['production_exotic']
            + planting_by_type['natural_forest']
        )
        self.results['manley_planting_permanent'] = planting_by_type['permanent_exotic']
        self.results['manley_planting_production'] = planting_by_type['production_exotic']
        self.results['manley_planting_natural'] = planting_by_type['natural_forest']

        return self.results

    # ------------------------------------------------------------------
    # Step 2: Forward / lagged price calculation
    # ------------------------------------------------------------------

    def _calculate_manley_prices(self, prices: pd.Series) -> pd.Series:
        """Convert model prices to the price input for the Manley equation (2021$)."""
        prices_aligned = prices.reindex(self.years)

        if self.price_assumption == 'current':
            return self._calculate_lagged_prices(prices_aligned)
        else:
            return self._calculate_forward_prices(prices_aligned)

    def _calculate_forward_prices(self, prices: pd.Series) -> pd.Series:
        """
        NPV-weighted forward price matching Excel column AG:
          forward_price[t] = (price[t] + NPV(r, price[t+1..t+N])) / annuity_factor(r, N)
          annuity_factor(r, N) = 1 + (1 - (1+r)^-N) / r
        Then divide by price_conversion_2021 to get 2021$.
        """
        r = self.discount_rate
        N = self.forward_years
        price_arr = prices.values.astype(float)
        n = len(price_arr)

        annuity_factor = 1.0 + (1.0 - (1.0 + r) ** (-N)) / r

        forward = np.empty(n)
        for t in range(n):
            npv = 0.0
            for i in range(1, N + 1):
                idx = t + i
                # Use last available price for years beyond the model horizon
                p = price_arr[idx] if idx < n else price_arr[-1]
                npv += p / (1.0 + r) ** i
            forward[t] = (price_arr[t] + npv) / annuity_factor

        return pd.Series(forward / self.price_conversion_2021, index=self.years)

    def _calculate_lagged_prices(self, prices: pd.Series) -> pd.Series:
        """Use price from lag years ago (current mode)."""
        lag = self.price_lag
        price_arr = prices.values.astype(float)
        n = len(price_arr)
        lagged = np.empty(n)
        for t in range(n):
            src = t - lag
            lagged[t] = price_arr[src] if src >= 0 else price_arr[0]
        return pd.Series(lagged / self.price_conversion_2021, index=self.years)

    # ------------------------------------------------------------------
    # Step 3+4: Manley logistic → cumulative → planting by forest type
    # ------------------------------------------------------------------

    def _calculate_manley_planting(
        self, manley_prices: pd.Series
    ) -> Dict[str, pd.Series]:
        """
        Apply the Manley logistic equation, compute cumulative afforestation,
        apply the LUC limit, then proportion by forest type.

        Returns dict {forest_type: pd.Series of annual planting in ha}.
        """
        n = len(self.years)
        price_arr = manley_prices.values

        # Logistic equation → afforestation_rate in '000 ha/yr
        f_arr = self._f_series.values
        profit = (
            self.LEV_logs
            + self.LEV_constant * self.LEV_carbon_per_dollar * price_arr
            - self.LMV_constant * self.LMV
        )
        # afforestation_rate in '000 ha
        rate = f_arr / (1.0 + np.exp(-self.h * (profit - self.g)))

        # Cumulative planting in ha
        cumulative = np.cumsum(rate * 1000)

        # limit_factor: scale down if total would exceed max_aggregate
        final_cumulative = cumulative[-1] if cumulative[-1] > 0 else 1.0
        limit_factor = min(1.0, self.max_aggregate_afforestation / final_cumulative)

        # Annual new planting in ha (capped at LUC_limit)
        annual_diff = np.diff(cumulative, prepend=0)  # same as rate * 1000
        # Apply max_forestry cap (post-logistic hard limit), then LUC_limit
        annual_planting = np.minimum(annual_diff * limit_factor, self._max_forestry_cap)
        annual_planting = np.minimum(annual_planting, self.LUC_limit)

        annual_series = pd.Series(annual_planting, index=self.years)

        # Proportion by forest type using MPI ratios
        planting_by_type = {}
        for forest_type in ['permanent_exotic', 'production_exotic', 'natural_forest']:
            planting_by_type[forest_type] = (
                annual_series * self._mpi_proportions[forest_type]
            )

        return planting_by_type

    # ------------------------------------------------------------------
    # Step 5+6: Vintage convolution
    # ------------------------------------------------------------------

    def _vintage_convolution(
        self, planting_by_type: Dict[str, pd.Series]
    ) -> pd.Series:
        """
        Calculate total new-forest absorption using np.convolve.

        For each forest type:
          absorption[cal_year] = sum over all vintages of
            planting[vintage] × yield_increment[age = cal_year - vintage]

        np.convolve handles this vintage summation automatically.

        Units: planting in ha, increments in tCO2/ha → result in tCO2.
        Divide by 1000 to convert to kt CO2-e (model units).

        Returns: pd.Series of total new-forest absorption indexed by year (kt CO2-e).
        """
        n = len(self.years)
        total = np.zeros(n)

        for forest_type in ['permanent_exotic', 'production_exotic', 'natural_forest']:
            planting = planting_by_type[forest_type].values
            increments = self.yield_increments[forest_type]

            # Full convolution, then trim to model horizon
            absorption = np.convolve(planting, increments)[:n]
            total += absorption

        # Convert tCO2 → kt CO2-e
        return pd.Series(total / 1000.0, index=self.years)

    # ------------------------------------------------------------------
    # Accessors used by the rest of the model
    # ------------------------------------------------------------------

    def get_total_supply(self) -> pd.Series:
        """Total forestry supply across all years."""
        return self.results['total_supply']

    def get_historic_held(self) -> Optional[pd.Series]:
        """Historic held units (endogenous mode only; None in exogenous mode)."""
        if self.mode == 'endogenous':
            return self.historic_held
        return None

    def get_historic_surrender(self) -> Optional[pd.Series]:
        """Historic surrender units (endogenous mode only; None in exogenous mode)."""
        if self.mode == 'endogenous':
            return self.historic_surrender
        return None
