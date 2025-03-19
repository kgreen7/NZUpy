"""
Stockpile supply component for the NZ ETS model.

This module implements the stockpile supply component, which represents
accumulated NZUs that weren't used in previous periods but can be used in future periods.
The stockpile has two parts: surplus stockpile (fully available) and non-surplus stockpile 
(available according to liquidity rate, with payback requirements).

All stockpile and surplus values represent year-end figures, meaning they represent
the balance at the end of the specified year. For example, if initial_stockpile is
159,902 and stockpile_reference_year is 2023, this means the stockpile balance was
159,902 at the end of 2023.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Union, Any


class StockpileSupply:
    """
    Stockpile supply component for NZ ETS model.
    
    This class handles the calculation of available units from the stockpile, and
    updates the stockpile based on the supply-demand balance. The stockpile consists
    of two components:
    1. Surplus units - fully available for immediate use
    2. Non-surplus units - only a fraction (liquidity_factor) available, with payback required
    
    All stockpile and surplus values represent year-end figures, meaning they represent
    the balance at the end of the specified year.
    """
    
    def __init__(
        self,
        years: List[int],
        extended_years: Optional[List[int]] = None,
        stockpile_params: Optional[Dict[str, float]] = None,
        initial_stockpile: Optional[float] = None,
        initial_surplus: Optional[float] = None,
        liquidity_factor: Optional[float] = None,
        payback_period: Optional[int] = None,
        discount_rate: Optional[float] = None,
        stockpile_reference_year: Optional[int] = None,
        stockpile_usage_start_year: Optional[int] = None,
        forestry_variables: Optional[pd.DataFrame] = None,
    ):
        """
        Initialise the stockpile supply component.
        
        Args:
            years: List of years for the model.
            extended_years: List of extended projection years beyond the model period.
            stockpile_params: Dictionary of stockpile parameters.
            initial_stockpile: Initial stockpile volume at end of reference year.
            initial_surplus: Initial surplus component of stockpile at end of reference year.
            liquidity_factor: Proportion of non-surplus stockpile available annually.
            payback_period: Number of years after which borrowed units must be paid back.
            discount_rate: Discount rate used for price signal effects. Default is 0.05.
            stockpile_reference_year: Year-end for which initial values are provided. Default is 2022.
            stockpile_usage_start_year: Year when stockpile becomes available to use. Default is 2024.
            forestry_variables: DataFrame with forestry_held and forestry_surrender data by year.
        """
        self.years = years
        self.extended_years = extended_years or []
        stockpile_params = stockpile_params or {}
        
        # Remove default fallback values and require parameters to be provided
        self.liquidity_factor = liquidity_factor or stockpile_params.get('liquidity_factor')
        if self.liquidity_factor is None:
            raise ValueError("liquidity_factor must be provided either directly or via stockpile_params")
        
        if not 0 <= self.liquidity_factor <= 1:
            raise ValueError(f"liquidity_factor must be between 0 and 1, got {self.liquidity_factor}")
        
        # Set the initial stockpile and surplus (year-end values)
        self.initial_stockpile = initial_stockpile or stockpile_params.get('initial_stockpile', 159902)
        self.initial_surplus = initial_surplus or stockpile_params.get('initial_surplus', 67300)
        self.payback_period = payback_period or stockpile_params.get('payback_period', 15)
        self.discount_rate = discount_rate or stockpile_params.get('discount_rate', 0.05)
        
        # Set reference and usage start years with proper defaults
        self.stockpile_reference_year = stockpile_reference_year or stockpile_params.get('stockpile_reference_year', 2023)
        self.stockpile_usage_start_year = stockpile_usage_start_year or stockpile_params.get('stockpile_usage_start_year', 2024)
        
        # Validate the years make sense
        if self.stockpile_reference_year >= self.stockpile_usage_start_year:
            print(f"Warning: stockpile_reference_year ({self.stockpile_reference_year}) should be before " 
                  f"stockpile_usage_start_year ({self.stockpile_usage_start_year})")
        
        # Process forestry variables
        self.forestry_variables = None
        if forestry_variables is not None:
            # Make a copy to avoid modifying the original
            self.forestry_variables = forestry_variables.copy()
            
            # Ensure index is properly named and of integer type
            if self.forestry_variables.index.name != 'year':
                print(f"  Renaming forestry variables index from '{self.forestry_variables.index.name}' to 'year'")
                self.forestry_variables.index.name = 'year'
            
            # Ensure index is of integer type
            if self.forestry_variables.index.dtype != 'int64':
                try:
                    self.forestry_variables.index = self.forestry_variables.index.astype(int)
                    print("  Converted forestry variables index to integer type")
                except Exception as e:
                    print(f"  Warning: Could not convert forestry variables index to integer type: {e}")
        
        # Validate inputs
        if self.initial_surplus > self.initial_stockpile:
            raise ValueError("Initial surplus cannot exceed initial stockpile")
        
        # Initialise data structures
        self._initialise_data()
    
    def _initialise_data(self):
        """Initialise data structures for stockpile supply."""
        # Stockpile balances by year
        self.stockpile_balance = pd.Series(0.0, index=self.years)
        self.surplus_balance = pd.Series(0.0, index=self.years)
        
        # Set initial values for the first year and up to usage start year
        for year in self.years:
            if year <= self.stockpile_usage_start_year:
                self.stockpile_balance[year] = self.initial_stockpile
                self.surplus_balance[year] = self.initial_surplus
        
        # Track borrowed units that need to be paid back
        all_years = self.years + self.extended_years
        self.borrowed_units = pd.DataFrame(0.0, index=self.years, columns=all_years)
        
        # Results DataFrame
        self.results = pd.DataFrame(
            index=self.years,
            columns=[
                'stockpile_balance', 'surplus_balance', 'non_surplus_balance',
                'liquid_stockpile', 'other_stockpile', 'available_units',
                'surplus_used', 'non_surplus_used', 'borrowed_units', 'payback_units', 'net_change',
                'forestry_held_addition', 'forestry_surrender_addition', 'cumulative_forestry_additions',
                'stockpile_without_forestry'
            ],
            data=0.0
        )
    
    def calculate(
        self, 
        supply_demand_balance=None, 
        prices=None, 
        price_change_rates=None, 
        track_forestry_impact=True
    ):
        """
        Calculate stockpile dynamics and available units.
        
        Args:
            supply_demand_balance: Series of supply minus demand by year.
            prices: Series of NZU prices by year.
            price_change_rates: Series of price change rates by year.
                During optimisation, set to False to exclude forestry from supply-demand calculations.
            track_forestry_impact: Whether to track forestry impact on stockpile separately.
                Always True for final results, can be False during initial optimisation iterations.
                
        Returns:
            DataFrame containing stockpile results by year.
        """
        # Convert supply_demand_balance to Series if provided
        if supply_demand_balance is not None and not isinstance(supply_demand_balance, pd.Series):
            supply_demand_balance = pd.Series(supply_demand_balance)
        
        # Convert prices to Series if provided
        if prices is not None and not isinstance(prices, pd.Series):
            prices = pd.Series(prices)
        
        # Calculate price change rates if prices provided but not rates
        if prices is not None and price_change_rates is None:
            # Calculate actual year-over-year growth rates
            price_change_rates = pd.Series(index=prices.index, data=0.0)
            for i, year in enumerate(prices.index):
                if i > 0:
                    prev_year = prices.index[i-1]
                    if prices[prev_year] > 0:  # Avoid division by zero
                        price_change_rates[year] = (prices[year] / prices[prev_year]) - 1
        
        # Initialise stockpile without forestry to track separately
        stockpile_without_forestry = pd.Series(0.0, index=self.years)
        
        # Process each year
        for i, year in enumerate(self.years):
            # Skip years before stockpile_usage_start_year
            if year < self.stockpile_usage_start_year:
                # Values already set in _initialise_data, just set results
                self.results.loc[year, 'stockpile_balance'] = self.stockpile_balance[year]
                self.results.loc[year, 'surplus_balance'] = self.surplus_balance[year]
                self.results.loc[year, 'non_surplus_balance'] = self.stockpile_balance[year] - self.surplus_balance[year]
                self.results.loc[year, 'liquid_stockpile'] = self.surplus_balance[year]
                self.results.loc[year, 'other_stockpile'] = self.stockpile_balance[year] - self.surplus_balance[year]
                # Important: set available_units to 0 for years before usage start
                self.results.loc[year, 'available_units'] = 0
                self.results.loc[year, 'surplus_used'] = 0
                self.results.loc[year, 'non_surplus_used'] = 0
                self.results.loc[year, 'forestry_held_addition'] = 0
                self.results.loc[year, 'forestry_surrender_addition'] = 0
                stockpile_without_forestry[year] = self.stockpile_balance[year]
                continue
            
            # Get previous year's balances for liquidity calculations
            prev_year = self.years[i-1] if i > 0 else year
            prev_non_surplus = max(0, self.stockpile_balance[prev_year] - self.surplus_balance[prev_year])
            
            # Special case for stockpile_usage_start_year to ensure correct initialisation
            if year == self.stockpile_usage_start_year:
                # Explicitly ensure the values are set correctly
                self.stockpile_balance[year] = self.initial_stockpile
                self.surplus_balance[year] = self.initial_surplus
                stockpile_without_forestry[year] = self.stockpile_balance[year]
                # Update prev_non_surplus for first year of usage
                prev_non_surplus = max(0, self.initial_stockpile - self.initial_surplus)
            # Copy previous year's balance for years after stockpile_usage_start_year
            elif i > 0:
                prev_year = self.years[i-1]
                self.stockpile_balance[year] = self.stockpile_balance[prev_year]
                self.surplus_balance[year] = self.surplus_balance[prev_year]
                stockpile_without_forestry[year] = stockpile_without_forestry[prev_year]
            
            # Process forestry additions
            forestry_held_addition = 0.0
            forestry_surrender_addition = 0.0
            
            if track_forestry_impact and self.forestry_variables is not None and year in self.forestry_variables.index:
                try:
                    forestry_held = self.forestry_variables.loc[year, ('central', 'forestry_held')]
                    if not pd.isna(forestry_held) and forestry_held > 0:
                        forestry_held_addition = forestry_held
                        self.stockpile_balance[year] += forestry_held_addition
                except:
                    pass
                
                try:
                    forestry_surrender = self.forestry_variables.loc[year, ('central', 'forestry_surrender')]
                    if not pd.isna(forestry_surrender):
                        forestry_surrender_addition = forestry_surrender
                        self.stockpile_balance[year] += forestry_surrender_addition
                except:
                    pass
            
            # Process supply-demand balance
            balance = supply_demand_balance.get(year, 0) if supply_demand_balance is not None else 0
            units_used_from_stockpile = 0
            surplus_used = 0
            non_surplus_used = 0
            
            # Check if year is eligible for stockpile usage
            if year >= self.stockpile_usage_start_year and balance < 0:  # Demand exceeds supply
                shortfall = abs(balance)
                
                # First use surplus stockpile - ALWAYS AVAILABLE regardless of price growth
                surplus_used = min(shortfall, self.surplus_balance[year])
                self.surplus_balance[year] -= surplus_used
                self.stockpile_balance[year] -= surplus_used
                stockpile_without_forestry[year] -= surplus_used
                shortfall -= surplus_used
                
                # For non-surplus stockpile, check discount rate condition
                non_surplus_available = False
                if (price_change_rates is not None and 
                    year in price_change_rates.index and 
                    price_change_rates[year] < self.discount_rate):
                    non_surplus_available = True
                
                # Only use non-surplus if price growth < discount rate
                if non_surplus_available and shortfall > 0:
                    # Calculate available non-surplus using previous year's balance
                    liquid_non_surplus = prev_non_surplus * self.liquidity_factor
                    proposed_non_surplus_used = min(shortfall, liquid_non_surplus)
                    
                    # Validate against current non-surplus balance
                    current_non_surplus = max(0, self.stockpile_balance[year] - self.surplus_balance[year])
                    if proposed_non_surplus_used > current_non_surplus:
                        print(f"Warning: Year {year}: Attempted to use {proposed_non_surplus_used} from non-surplus, but only {current_non_surplus} available.")
                        non_surplus_used = min(proposed_non_surplus_used, current_non_surplus)
                    else:
                        non_surplus_used = proposed_non_surplus_used
                    
                    # Update balances
                    self.stockpile_balance[year] -= non_surplus_used
                    stockpile_without_forestry[year] -= non_surplus_used
                    
                    # Record borrowing for future payback
                    payback_year = year + self.payback_period
                    if payback_year in self.borrowed_units.columns:
                        self.borrowed_units.loc[year, payback_year] += non_surplus_used
                    else:
                        last_column = max(self.borrowed_units.columns)
                        self.borrowed_units.loc[year, last_column] += non_surplus_used
                
                units_used_from_stockpile = surplus_used + non_surplus_used
            
            # Handle excess supply
            # Per Excel model, excess units are added to the surplus component of the stockpile
            # as they represent units that weren't needed for compliance in the current year
            if balance > 0:
                # All excess goes to surplus by default - matching Excel model behavior
                surplus_portion = balance
                non_surplus_portion = 0.0
                
                # Update both balances
                self.stockpile_balance[year] += balance
                self.surplus_balance[year] += surplus_portion
                stockpile_without_forestry[year] += balance
            
            # Calculate current year's balances for reporting
            current_non_surplus = max(0, self.stockpile_balance[year] - self.surplus_balance[year])
            current_liquid_non_surplus = current_non_surplus * self.liquidity_factor
            
            # Store results
            self.results.loc[year, 'stockpile_balance'] = self.stockpile_balance[year]
            self.results.loc[year, 'surplus_balance'] = self.surplus_balance[year]
            self.results.loc[year, 'non_surplus_balance'] = current_non_surplus
            self.results.loc[year, 'liquid_stockpile'] = self.surplus_balance[year] + current_liquid_non_surplus
            self.results.loc[year, 'other_stockpile'] = current_non_surplus - current_liquid_non_surplus
            self.results.loc[year, 'available_units'] = units_used_from_stockpile
            self.results.loc[year, 'surplus_used'] = surplus_used
            self.results.loc[year, 'non_surplus_used'] = non_surplus_used
            self.results.loc[year, 'forestry_held_addition'] = forestry_held_addition
            self.results.loc[year, 'forestry_surrender_addition'] = forestry_surrender_addition
            
            # Track borrowing and payback
            year_borrowing = sum(self.borrowed_units.loc[year, :])
            year_payback = 0
            
            # Sum all payback due for this year from previous borrowing
            for borrow_year in self.years:
                if year in self.borrowed_units.columns:
                    year_payback += self.borrowed_units.loc[borrow_year, year]
            
            self.results.loc[year, 'borrowed_units'] = year_borrowing
            self.results.loc[year, 'payback_units'] = year_payback
            self.results.loc[year, 'net_change'] = year_payback - year_borrowing
        
        # Calculate cumulative forestry additions for the stockpile
        if track_forestry_impact:
            cumulative_held = pd.Series(0.0, index=self.years)
            cumulative_surrender = pd.Series(0.0, index=self.years)
            running_held_total = 0.0
            running_surrender_total = 0.0
            
            for year in self.years:
                running_held_total += self.results.loc[year, 'forestry_held_addition']
                running_surrender_total += self.results.loc[year, 'forestry_surrender_addition']
                cumulative_held[year] = running_held_total
                cumulative_surrender[year] = running_surrender_total
            
            # Store cumulative forestry additions for reference
            self.results['cumulative_forestry_additions'] = cumulative_held + cumulative_surrender
            
            # Store stockpile without forestry
            self.results['stockpile_without_forestry'] = stockpile_without_forestry
        
        return self.results
    
    def set_forestry_variables(self, forestry_variables: pd.DataFrame) -> None:
        """
        Set or update the forestry variables data.
        
        Args:
            forestry_variables: DataFrame with forestry_held and forestry_surrender data by year.
        """
        # Validate the input dataframe
        required_columns = ['forestry_held', 'forestry_surrender']
        missing_columns = [col for col in required_columns if col not in forestry_variables.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in forestry_variables: {missing_columns}")
        
        # Ensure index is a proper year index
        if not isinstance(forestry_variables.index, pd.Int64Index):
            # Try to convert index to integer if it's not already
            try:
                forestry_variables = forestry_variables.set_index('year')
            except (KeyError, ValueError):
                raise ValueError("forestry_variables must have a 'year' column or a year-based index")
        
        # Store the forestry variables
        self.forestry_variables = forestry_variables
        
        # Recalculate results with the new forestry variables
        self._initialise_data()
        self.calculate()
    
    def get_available_supply(self) -> pd.Series:
        """
        Get the available supply from the stockpile.
        
        Returns:
            Series of available supply indexed by year.
        """
        return self.results['available_units']
    
    def set_liquidity_factor(self, factor: float) -> None:
        """
        Set the liquidity factor and recalculate results.
        
        Args:
            factor: New liquidity factor (0 to 1).
        """
        if not 0 <= factor <= 1:
            raise ValueError(f"Liquidity factor must be between 0 and 1, got {factor}")
        
        self.liquidity_factor = factor
        
        # Recalculate results with the new liquidity factor
        self.calculate()
    
    def set_payback_period(self, period: int) -> None:
        """
        Set the payback period for borrowed non-surplus units.
        
        Args:
            period: Payback period in years.
        """
        if period <= 0:
            raise ValueError(f"Payback period must be positive, got {period}")
        
        self.payback_period = period
        
        # Reset borrowing and recalculate
        self.borrowed_units = pd.DataFrame(0.0, index=self.years, columns=self.years)
        self.calculate()
    
    def set_discount_rate(self, rate: float) -> None:
        """
        Set the discount rate for price signal effects.
        
        Args:
            rate: Discount rate as a decimal.
        """
        if rate < 0:
            raise ValueError(f"Discount rate cannot be negative, got {rate}")
        
        self.discount_rate = rate
        
        # Recalculate results with the new discount rate
        self.calculate()
    
    def set_stockpile_start_year(self, year: int) -> None:
        """
        Set the year when stockpile becomes available to use.
        
        Args:
            year: Stockpile start year.
        """
        if year not in self.years:
            raise ValueError(f"Start year {year} not in model years")
        
        self.stockpile_usage_start_year = year
        
        # Recalculate with the new start year
        self._initialise_data()
        self.calculate()
    
    @property
    def is_stockpile_used(self) -> pd.Series:
        """Return a Series indicating years where stockpile is being used."""
        return self.results['available_units'] > 0

    @property 
    def stockpile_usage_amounts(self) -> pd.Series:
        """Return a Series with the amount of stockpile used each year."""
        return self.results['available_units']
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the stockpile supply component to a dictionary.
        
        Returns:
            Dictionary representation of the stockpile supply component.
        """
        return {
            'initial_stockpile': self.initial_stockpile,
            'initial_surplus': self.initial_surplus,
            'liquidity_factor': self.liquidity_factor,
            'payback_period': self.payback_period,
            'discount_rate': self.discount_rate,
            'stockpile_reference_year': self.stockpile_reference_year,
            'stockpile_usage_start_year': self.stockpile_usage_start_year,
            'stockpile_balance': self.stockpile_balance.to_dict(),
            'surplus_balance': self.surplus_balance.to_dict(),
            'borrowed_units': self.borrowed_units.to_dict(),
            'results': self.results.to_dict(),
            'is_stockpile_used': self.is_stockpile_used.to_dict(),
            'stockpile_usage_amounts': self.stockpile_usage_amounts.to_dict()
        }