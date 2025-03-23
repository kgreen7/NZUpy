"""
Calculation engine for NZUpy.

This module provides the calculation functions for prices, supply, demand, and
stockpile dynamics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Tuple, Union


class CalculationEngine:
    """
    Calculation engine for the NZ ETS model.
    
    This class handles all calculation functions including price calculations,
    supply and demand calculations, and stockpile dynamics.
    """
    
    def __init__(self, model):
        """
        Initialise the calculation engine.
        
        Args:
            model: The parent NZUpy instance
        """
        self.model = model
    
    def calculate_gap(self, price_change_rate: float) -> float:
        """
        Calculate the gap between supply and demand.
        
        This is the main function used during optimisation to evaluate different
        price change rates and find the optimal rate that minimises the gap.
        
        Args:
            price_change_rate: Annual carbon price change rate to evaluate
            
        Returns:
            Total weighted gap between supply and demand across all years
        """
        if self.model.config.optimiser.debug:
            print(f"\nCalculating gap for price_change_rate={price_change_rate:.6f}")
            print(f"Max iterations for convergence: {self.model.config.optimiser.max_iterations}")
        
        # Store the price change rate
        self.model.price_change_rate = float(price_change_rate)
        
        # Initialise and prepare for iteration
        iteration_data = self._initialise_iteration_tracking()
        
        # IMPORTANT: Reset stockpile component state before each optimisation run
        # This prevents state leakage between optimisation iterations
        self.model.stockpile._initialise_data()
        
        # Perform the iterative calculation until convergence
        self._iterate_until_convergence(iteration_data)
        
        # Compile final supply with stockpile
        self._compile_final_supply()
        
        # Calculate final gap measure
        gap = self._calculate_total_gap(
            penalise_shortfalls=self.model.config.optimiser.penalise_shortfalls
        )
        
        if self.model.config.optimiser.debug:
            self._log_gap_calculation_results(gap)
                
        return gap
    
    def _initialise_iteration_tracking(self) -> Dict[str, Any]:
        """
        Initialise tracking variables for the iteration process.
        
        Returns:
            Dictionary with iteration tracking data
        """
        return {
            'max_iterations': self.model.config.optimiser.max_iterations,
            'converged': False,
            'iteration': 0,
            'stockpile_usage': pd.Series(0.0, index=self.model.years)
        }
    
    def _iterate_until_convergence(self, iteration_data: Dict[str, Any]) -> None:
        """
        Perform the iterative calculation process until convergence is reached.
        
        Args:
            iteration_data: Dictionary with iteration tracking variables
        """
        while not iteration_data['converged'] and iteration_data['iteration'] < iteration_data['max_iterations']:
            iteration_data['iteration'] += 1
            
            # Calculate initial prices based on the price change rate
            self._calculate_initial_prices()
            
            # Calculate demand based on current prices
            self._calculate_demand()
            
            # Calculate base supply (without stockpile)
            self._calculate_base_supply()
            
            # Calculate supply-demand balance and update stockpile
            stockpile_results = self._calculate_stockpile_contribution()
            
            # Get new stockpile usage for convergence check
            new_stockpile_usage = stockpile_results['available_units'].copy()
            
            # Check for price revision conditions and update prices if needed
            prev_prices = self.model.prices.copy()
            self._revise_prices_based_on_stockpile(stockpile_results)
            
            # Check for convergence
            iteration_data['converged'] = self._check_convergence(
                prev_prices, 
                iteration_data['stockpile_usage'],
                new_stockpile_usage
            )
            
            # Update stockpile usage for next iteration
            iteration_data['stockpile_usage'] = new_stockpile_usage
    
    def _calculate_stockpile_contribution(self) -> pd.DataFrame:
        """
        Calculate the stockpile's contribution to supply based on the supply-demand balance.
        
        Returns:
            DataFrame containing stockpile calculation results
        """
        # Calculate initial supply-demand balance
        yearly_balances = {}
        for year in self.model.years:
            yearly_balances[year] = self.model.base_supply.get(year, 0) - self.model.demand.get(year, 0)
        
        supply_demand_balance = pd.Series(yearly_balances)
        
        # Calculate stockpile contribution
        stockpile_results = self.model.stockpile.calculate(
            supply_demand_balance=supply_demand_balance,
            prices=self.model.prices,
            price_change_rates=pd.Series([self.model.price_change_rate] * len(self.model.years), index=self.model.years),
            track_forestry_impact=True
        )
        
        # Ensure results are stored as DataFrame
        if isinstance(stockpile_results, dict):
            stockpile_results = pd.DataFrame(stockpile_results)
        
        # Store results in model for later use
        self.model.stockpile_results = stockpile_results
        
        return stockpile_results
    
    def _check_convergence(self, prev_prices: pd.Series, prev_usage: pd.Series, new_usage: pd.Series) -> bool:
        """
        Check if the iteration has converged.
        
        Args:
            prev_prices: Previous iteration's prices
            prev_usage: Previous iteration's stockpile usage
            new_usage: Current iteration's stockpile usage
            
        Returns:
            True if converged, False otherwise
        """
        # Check changes in both prices and stockpile usage
        price_change = (self.model.prices - prev_prices).abs().sum()
        usage_change = (new_usage - prev_usage).abs().sum()
        
        # Convergence thresholds (small values to ensure stability)
        return price_change < 0.01 and usage_change < 100
    
    def _compile_final_supply(self) -> None:
        """
        Compile the final supply DataFrame with all components.
        """
        self.model.supply = pd.DataFrame(index=self.model.years)
        self.model.supply['auction'] = self.model.auction_results['total_auction']
        self.model.supply['industrial'] = self.model.industrial_results['adjusted_allocation']
        self.model.supply['forestry'] = self.model.forestry_results['total_supply']
        
        # Handle stockpile results
        if hasattr(self.model, 'stockpile_results') and self.model.stockpile_results is not None:
            if 'available_units' in self.model.stockpile_results.columns:
                self.model.supply['stockpile'] = self.model.stockpile_results['available_units']
            else:
                # If available_units not found, try to calculate from surplus and non-surplus
                if 'surplus_used' in self.model.stockpile_results.columns and 'non_surplus_used' in self.model.stockpile_results.columns:
                    self.model.supply['stockpile'] = (
                        self.model.stockpile_results['surplus_used'] + 
                        self.model.stockpile_results['non_surplus_used']
                    )
                else:
                    # If no stockpile data available, set to 0
                    self.model.supply['stockpile'] = 0.0
        else:
            # If no stockpile results available, set to 0
            self.model.supply['stockpile'] = 0.0
        
        # Calculate total supply
        self.model.supply['total'] = (
            self.model.supply['auction'] + 
            self.model.supply['industrial'] + 
            self.model.supply['forestry'] + 
            self.model.supply['stockpile']
        )
        
        # Store stockpile results if they exist
        if hasattr(self.model, 'stockpile_results') and self.model.stockpile_results is not None:
            self.model.stockpile_results = self.model.stockpile_results
    
    def _log_gap_calculation_results(self, gap: float) -> None:
        """
        Log the results of the gap calculation for debugging.
        
        Args:
            gap: Calculated gap value
        """
        print(f"  → Gap calculated: {gap:,.0f}")
        print(f"  → Prices:")
        print(f"     2024: ${self.model.prices[2024]:.2f}")
        print(f"     2050: ${self.model.prices[2050]:.2f}")
        print(f"  → Supply-Demand balance: {(self.model.supply['total'] - self.model.demand).sum():,.0f}")

    def _calculate_initial_prices(self):
        """
        Calculate initial prices for all years, including extended projection years.
        Prices for years beyond end_year are projected at the same rate to ensure
        proper forward price calculations.
        """
        # Load price constraints
        price_constraints = self._get_price_constraints()
        
        # Initialise price series
        self._initialise_price_series()
        
        # Calculate future prices based on price change rate
        self._calculate_future_prices(price_constraints)

    def _get_price_constraints(self) -> Dict[str, Optional[float]]:
        """
        Get minimum and maximum price constraints from model parameters.
        
        Returns:
            Dictionary with long_term_min and long_term_max constraints
        """
        constraints = {'long_term_min': None, 'long_term_max': None}
        
        try:
            # Attempt to get constraints from data handler
            if hasattr(self.model, 'data_handler') and self.model.data_handler is not None:
                model_params = self.model.data_handler.get_model_parameters()
                
                # Process minimum price constraint
                long_term_min = model_params.get('long_term_min', None)
                if long_term_min is not None:
                    try:
                        constraints['long_term_min'] = float(long_term_min)
                    except (ValueError, TypeError):
                        if long_term_min != 'na':
                            print(f"Warning: Could not convert long_term_min value '{long_term_min}' to float. No minimum price will be applied.")
                
                # Process maximum price constraint
                long_term_max = model_params.get('long_term_max', None)
                if long_term_max is not None:
                    try:
                        constraints['long_term_max'] = float(long_term_max)
                    except (ValueError, TypeError):
                        if long_term_max != 'na':
                            print(f"Warning: Could not convert long_term_max value '{long_term_max}' to float. No maximum price will be applied.")
            else:
                # No data handler available, raise an error
                raise ValueError("Cannot load price constraints: data_handler not available")
        except Exception as e:
            raise ValueError(f"Failed to load price constraints from model parameters: {e}")
            
        return constraints

    def _initialise_price_series(self):
        """
        Initialise the price series and set historical prices.
        """
        # Start with all prices set to None
        self.model.prices = pd.Series(index=self.model.calculation_years, dtype=float)
        
        # Set historical prices
        for year in self.model.calculation_years:
            if year in self.model.historical_prices:
                self.model.prices[year] = self.model.historical_prices[year]

    def _calculate_future_prices(self, price_constraints: Dict[str, Optional[float]]):
        """
        Calculate prices for future years using the price change rate and applying constraints.
        
        Args:
            price_constraints: Dictionary with minimum and maximum price constraints
        """
        long_term_min = price_constraints['long_term_min']
        long_term_max = price_constraints['long_term_max']
        
        for year in self.model.calculation_years:
            if year > self.model.last_historical_year:
                # Calculate the price based on whether it's the first projected year or a subsequent year
                if year == self.model.last_historical_year + 1:
                    self._calculate_first_projected_year_price(year)
                else:
                    self._calculate_subsequent_year_price(year)
                
                # Apply price constraints
                self._apply_price_constraints(year, long_term_min, long_term_max)

    def _calculate_first_projected_year_price(self, year: int):
        """
        Calculate price for the first year after historical data.
        
        Args:
            year: Year to calculate price for
        """
        # Get control value from parameter hierarchy
        control_value = self._get_price_control_value(year)

        # Apply control parameter to price change
        if control_value >= 0:
            # Positive control: apply standard growth
            growth_factor = 1 + (self.model.price_change_rate * control_value)
            self.model.prices[year] = self.model.last_historical_price * growth_factor
        else:
            # Negative control: invert the change direction
            reduction_factor = 1 - (self.model.price_change_rate * abs(control_value))
            self.model.prices[year] = self.model.last_historical_price * reduction_factor
    def _calculate_subsequent_year_price(self, year: int):
        """
        Calculate price for years after the first projected year.
        
        Args:
            year: Year to calculate price for
        """
        prev_year = year - 1
        
        # Get control value from parameter hierarchy
        control_value = self._get_price_control_value(year)
        # Apply control parameter to price change
        if control_value >= 0:
            # Positive control: apply standard growth
            growth_factor = 1 + (self.model.price_change_rate * control_value)
            self.model.prices[year] = self.model.prices[prev_year] * growth_factor
        else:
            # Negative control: invert the change direction
            reduction_factor = 1 - (self.model.price_change_rate * abs(control_value))
            self.model.prices[year] = self.model.prices[prev_year] * reduction_factor

    def _get_price_control_value(self, year: int) -> float:
        """
        Get price control value following parameter loading priority.
        
        Args:
            year: Year to get price control for
            
        Returns:
            Price control value (1.0 is neutral)
        """
        # Try getting from direct parameter settings (highest priority)
        if hasattr(self.model, 'price_control_parameter'):
            control_value = self.model.price_control_parameter.get(year)
            if control_value is not None:
                return control_value
        
        # Try getting from currently active config (middle priority)
        # This automatically uses the correct scenario's config based on how the runner works
        if hasattr(self.model, 'data_handler') and hasattr(self.model.data_handler, 'historical_manager'):
            # Getting the active price control config name for the current scenario
            # This is already handled by the model runner when it sets up the scenario
            active_config = self.model.active_price_control_config
            if active_config:
                control_value = self.model.data_handler.historical_manager.get_price_control(
                    year, config=active_config)
                if control_value is not None:
                    return control_value
        
        # Default fallback (lowest priority)
        return 1.0 # TODO: Figure out how to remove fallback here without breaking model

    def _apply_price_constraints(self, year: int, min_price: Optional[float], max_price: Optional[float]):
        """
        Apply minimum and maximum price constraints.
        
        Args:
            year: Year to apply constraints for
            min_price: Minimum price constraint (or None if no minimum)
            max_price: Maximum price constraint (or None if no maximum)
        """
        # Apply minimum price constraint if provided and price is below it
        if min_price is not None and self.model.prices[year] < min_price:
            self.model.prices[year] = min_price
        
        # Apply maximum price constraint if provided and price is above it
        if max_price is not None and self.model.prices[year] > max_price:
            self.model.prices[year] = max_price

    def _revise_prices_based_on_stockpile(self, stockpile_results):
        """
        Revise prices if stockpile is being used and price growth exceeds discount rate.
        This replicates Excel's price revision mechanism in columns Y and Z.
        
        Args:
            stockpile_results: Results from stockpile calculations
        """
        # Identify years needing price revision
        price_revision_flags = self._identify_years_needing_revision(stockpile_results)
        
        # Store original prices before making revisions
        original_prices = self.model.prices.copy()
        self.model.unmodified_prices = original_prices
        
        # Apply price revisions if needed
        if price_revision_flags.any():
            self._apply_price_revisions(price_revision_flags, original_prices)
    
    def _identify_years_needing_revision(self, stockpile_results) -> pd.Series:
        """
        Identify years where price revision is needed.
        
        Args:
            stockpile_results: Results from stockpile calculations
            
        Returns:
            Series of boolean flags indicating years needing revision
        """
        price_revision_flags = pd.Series(False, index=self.model.years)
        
        # Check each year to see if price revision is needed
        for year in self.model.years[1:]:  # Skip first year
            prev_year = year - 1
            
            # Calculate price growth rate
            if self.model.prices[prev_year] > 0:
                price_growth_rate = (self.model.prices[year] / self.model.prices[prev_year]) - 1
            else:
                price_growth_rate = 0
            
            # Check if price is growing faster than discount rate AND stockpile is being used
            if (price_growth_rate > self.model.stockpile.discount_rate and 
                stockpile_results['available_units'].get(year, 0) > 0):
                price_revision_flags[year] = True
        
        return price_revision_flags
    
    def _apply_price_revisions(self, price_revision_flags: pd.Series, original_prices: pd.Series):
        """
        Apply price revisions to years where needed.
        
        Args:
            price_revision_flags: Boolean flags indicating which years need revision
            original_prices: Original prices before revision
        """
        revision_years = [year for year in self.model.years if price_revision_flags.get(year, False)]
        
        # Process each revision segment
        i = 0
        while i < len(revision_years):
            # Identify continuous segment of years needing revision
            segment_info = self._identify_revision_segment(i, revision_years, price_revision_flags)
            start_year = segment_info['start_year']
            end_year = segment_info['end_year']
            last_normal_year = start_year - 1
            last_normal_price = original_prices[last_normal_year]
            
            # Apply revision to each year in the segment
            for rev_year in range(start_year, end_year + 1):
                years_since_normal = rev_year - last_normal_year
                # Grow price at exactly the discount rate from the last normal price
                self.model.prices[rev_year] = last_normal_price * (
                    (1 + self.model.stockpile.discount_rate) ** years_since_normal
                )
            
            # Move to next segment
            i = segment_info['next_index']
    
    def _identify_revision_segment(self, current_index: int, revision_years: List[int], 
                                 price_revision_flags: pd.Series) -> Dict[str, int]:
        """
        Identify a continuous segment of years needing price revision.
        
        Args:
            current_index: Current index in the revision_years list
            revision_years: List of years needing revision
            price_revision_flags: Boolean flags indicating which years need revision
            
        Returns:
            Dictionary with segment information
        """
        start_year = revision_years[current_index]
        
        # Find the end of this continuous segment
        end_idx = current_index
        while (end_idx + 1 < len(revision_years) and 
              revision_years[end_idx + 1] == revision_years[end_idx] + 1):
            end_idx += 1
        
        end_year = revision_years[end_idx]
        
        # Find the first non-revision year after this segment
        next_normal_year = end_year + 1
        while next_normal_year in price_revision_flags and price_revision_flags[next_normal_year]:
            next_normal_year += 1
        
        return {
            'start_year': start_year,
            'end_year': end_year,
            'next_index': end_idx + 1
        }
    
    def _calculate_prices(self):
        """Calculate prices for all years based on historical prices and price change rate."""
        # Initialise price series
        self.model.prices = pd.Series(index=self.model.years, dtype=float)
        
        # Set historical prices
        for year in self.model.years:
            if year in self.model.historical_prices:
                self.model.prices[year] = self.model.historical_prices[year]
        
        # Calculate future prices
        for year in self.model.years:
            if year > self.model.last_historical_year:
                self._calculate_price_for_year(year)
    
    def _calculate_price_for_year(self, year: int):
        """
        Calculate price for a specific year based on previous year and price controls.
        
        Args:
            year: Year to calculate price for
        """
        # Handle first year after historical data differently
        if year == self.model.last_historical_year + 1:
            base_price = self.model.last_historical_price
            control_value = self.model.price_control_parameter[year]
        else:
            # Use previous year's price as base
            prev_year = year - 1
            base_price = self.model.prices[prev_year]
            control_value = self.model.price_control_parameter[year]
        
        # Apply price control parameter
        self.model.prices[year] = self._apply_price_control(base_price, control_value)
        
        # Apply maximum price cap if configured
        if hasattr(self.model.config, 'max_price') and self.model.config.max_price is not None:
            if self.model.prices[year] > self.model.config.max_price:
                self.model.prices[year] = self.model.config.max_price
    
    def _apply_price_control(self, base_price: float, control_value: float) -> float:
        """
        Apply price control to base price.
        
        Args:
            base_price: Base price to apply control to
            control_value: Price control parameter value
            
        Returns:
            New price after applying control
        """
        if control_value >= 0:
            # Positive control: apply standard growth
            growth_factor = 1 + (self.model.price_change_rate * control_value)
            return base_price * growth_factor
        else:
            # Negative control: invert the change direction
            reduction_factor = 1 - (self.model.price_change_rate * abs(control_value))
            return base_price * reduction_factor

    def _calculate_price_growth_rates(self) -> pd.Series:
        """Calculate year-on-year price growth rates."""
        growth_rates = pd.Series(index=self.model.years, dtype=float)
        for i, year in enumerate(self.model.years):
            if i == 0:  # First year has no previous year
                growth_rates[year] = 0.0
            else:
                prev_year = self.model.years[i-1]
                if self.model.prices[prev_year] > 0:  # Avoid division by zero
                    growth_rates[year] = (self.model.prices[year] / self.model.prices[prev_year]) - 1
                else:
                    growth_rates[year] = 0.0
        return growth_rates
    
    def _calculate_base_supply(self, scenario_name=None):
        """
        Calculate base supply components (excluding stockpile).
        
        Args:
            scenario_name: Name of the scenario being calculated
        """
        # Calculate auction supply without scenario_name
        self.model.auction_results = self.model.auction.calculate(
            self.model.prices
        )
        
        # Store the auction component itself for later access to input data
        self.model.auction_supply = self.model.auction
        
        # Calculate industrial allocation without scenario_name
        self.model.industrial_results = self.model.industrial.calculate()
        
        # Calculate forestry supply without scenario_name
        self.model.forestry_results = self.model.forestry.calculate(
            self.model.prices
        )
        
        # Calculate base supply (excluding stockpile)
        self.model.base_supply = {}
        for year in self.model.years:
            # For years before stockpile_usage_start_year, set base_supply equal to demand
            if hasattr(self.model.stockpile, 'stockpile_usage_start_year') and year < self.model.stockpile.stockpile_usage_start_year:
                if year in self.model.demand.index:
                    self.model.base_supply[year] = self.model.demand[year]
                else:
                    # Fallback to sum of components if demand not available
                    self.model.base_supply[year] = (
                        self.model.auction_results['total_auction'].get(year, 0) +
                        self.model.industrial_results['adjusted_allocation'].get(year, 0) +
                        self.model.forestry_results['total_supply'].get(year, 0))
            else:
                # Normal calculation for years from stockpile_usage_start_year onwards
                self.model.base_supply[year] = (
                    self.model.auction_results['total_auction'].get(year, 0) +
                    self.model.industrial_results['adjusted_allocation'].get(year, 0) +
                    self.model.forestry_results['total_supply'].get(year, 0))

    def _calculate_supply(self, scenario_name: Optional[str] = None):
        """
        Calculate all supply components (including stockpile) and total supply.
        
        Args:
            scenario_name: Name of the scenario being calculated
            
        Note: This method should be used after base supply and stockpile
        have been calculated separately.
        """
        # Create supply DataFrame with all components
        self.model.supply = pd.DataFrame(index=self.model.years)
        self.model.supply['auction'] = self.model.auction_results['total_auction']
        self.model.supply['industrial'] = self.model.industrial_results['adjusted_allocation']
        self.model.supply['forestry'] = self.model.forestry_results['total_supply']
        self.model.supply['stockpile'] = self.model.stockpile_results['available_units']
        
        # Calculate total supply
        self.model.supply['total'] = (
            self.model.supply['auction'] + 
            self.model.supply['industrial'] + 
            self.model.supply['forestry'] + 
            self.model.supply['stockpile']
        )
    
    def _calculate_demand(self, scenario_name=None):
        """
        Calculate demand based on emissions and price response.
        
        Args:
            scenario_name: Name of the scenario being calculated
        """
        # Calculate price response - no scenario_name
        price_response_results = self.model.price_response.calculate(
            self.model.prices, 
            self.model.emissions.emissions  # Use the emissions series directly
        )
        
        # Get emissions reductions
        emissions_reduction = self.model.price_response.get_emissions_reduction()
        
        # Calculate emissions with price response - no scenario_name here either
        emissions_results = self.model.emissions.calculate(
            emissions_reduction
        )
        
        # Total demand is the emissions after price response
        self.model.demand = emissions_results['total_demand']
        
        # Store component results for later analysis
        self.model.price_response_results = price_response_results
        self.model.emissions_results = emissions_results
    
    def _calculate_total_gap(self, penalise_shortfalls: bool = True) -> float:
        """
        Calculate the total gap between supply and demand across all years.
        
        Args:
            penalise_shortfalls: If True, applies a 1,000,000x penalty to supply shortfalls (Excel model approach).
                                If False, uses simple absolute differences.
        
        Returns:
            Total gap value across all years.
        """
        # Calculate yearly gaps (supply minus demand)
        yearly_gaps = self._calculate_yearly_gaps()
        
        # Calculate penalised gaps
        penalised_gaps = self._apply_gap_penalties(yearly_gaps, penalise_shortfalls)
        
        # Apply filtering and weighting
        weighted_gaps = self._filter_and_weight_gaps(penalised_gaps)
        
        # Return total gap
        return weighted_gaps.sum()
    
    def _calculate_yearly_gaps(self) -> pd.Series:
        """Calculate the gap between supply and demand for each year."""
        # Get payback units from the current stockpile calculation
        payback_units = pd.Series(0.0, index=self.model.years)
        if hasattr(self.model.stockpile_results, 'payback_units'):
            payback_units = self.model.stockpile_results['payback_units']
        
        # Subtract payback units from supply-demand balance
        return self.model.supply['total'] - self.model.demand - payback_units
    
    def _apply_gap_penalties(self, yearly_gaps: pd.Series, penalise_shortfalls: bool) -> pd.Series:
        """
        Apply penalties to supply shortfalls if requested.
        
        Args:
            yearly_gaps: Series of gaps between supply and demand
            penalise_shortfalls: Whether to apply severe penalties to shortfalls
            
        Returns:
            Series of penalised gaps
        """
        penalised_gaps = pd.Series(index=yearly_gaps.index)
        
        for year in yearly_gaps.index:
            if penalise_shortfalls and yearly_gaps[year] < 0:  # Supply shortfall with penalty
                penalised_gaps[year] = abs(yearly_gaps[year]) * 1000000  # Apply severe penalty matching Excel model approach
            else:  # Supply excess or no penalty requested
                penalised_gaps[year] = abs(yearly_gaps[year])
                
        return penalised_gaps
    
    def _filter_and_weight_gaps(self, penalised_gaps: pd.Series) -> pd.Series:
        """
        Filter gaps to exclude years before stockpile_usage_start_year and apply weights. Currently not used, but set-up for future use.
        
        Args:
            penalised_gaps: Series of penalised gaps
            
        Returns:
            Series of filtered and weighted gaps
        """
        # Filter out years before stockpile_usage_start_year
        if hasattr(self.model, 'stockpile') and hasattr(self.model.stockpile, 'stockpile_usage_start_year'):
            start_year = self.model.stockpile.stockpile_usage_start_year
            # Create mask for years to include in calculation (years >= start_year)
            years_to_include = [year >= start_year for year in penalised_gaps.index]
            filtered_gaps = penalised_gaps[years_to_include]
        else:
            filtered_gaps = penalised_gaps
        
        # Weight recent years more heavily if configured (currently all weights are 1.0)
        weights = np.linspace(1.0, 1.0, len(filtered_gaps))  # adjusted to linear
        weighted_gaps = filtered_gaps * weights
        
        return weighted_gaps