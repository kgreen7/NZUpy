"""
Forestry supply component for the NZ ETS model.

This module implements the forestry supply component of the NZ ETS model,
providing a simplified representation of forestry units supply.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Union, Any


class ForestrySupply:
    """
    Forestry supply component for NZ ETS model.
    
    This class handles the calculation of forestry units supply, which can
    be either static (based on input data) or dynamic (based on the Manley model).
    """
    
    def __init__(
        self,
        years: List[int],
        forestry_data: pd.DataFrame,
        use_manley_equation: bool = False,
        manley_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialise the forestry supply component.
        
        Args:
            years: List of years for the model.
            forestry_data: DataFrame with forestry supply data indexed by year.
                Required column: 'forestry_supply'
            use_manley_equation: Whether to use the Manley equation for dynamic forestry supply.
            manley_params: Parameters for the Manley equation if use_manley_equation is True.
                Required keys: 'alpha', 'beta', 'gamma', 'initial_stock'
        """
        self.years = years
        self.forestry_data = forestry_data
        self.use_manley_equation = use_manley_equation
        self.manley_params = manley_params or {}
        
        # Initialise results
        self.results = pd.DataFrame(
            index=self.years,
            columns=['static_supply', 'manley_supply', 'total_supply'],
            data=0.0
        )
    
    def calculate(self, prices: pd.Series) -> pd.DataFrame:
        """
        Calculate forestry supply based on the selected method.
        
        Args:
            prices: Series of NZU prices indexed by year.
            
        Returns:
            DataFrame containing forestry supply by year.
        """
        # Calculate static supply from the input data
        self.results['static_supply'] = self.forestry_data['forestry_supply']
        
        # Calculate Manley supply if requested
        if self.use_manley_equation:
            self._calculate_manley_supply(prices)
        else:
            self.results['manley_supply'] = 0.0
        
        # Calculate total supply
        if self.use_manley_equation:
            self.results['total_supply'] = self.results['manley_supply']
        else:
            self.results['total_supply'] = self.results['static_supply']
        
        return self.results
    
    def _calculate_manley_supply(self, prices: pd.Series):
        """
        Calculate forestry supply using the Manley equation.
        
        The Manley equation is a behavioral model for forestry based on price and stock levels.
        
        Args:
            prices: Series of NZU prices indexed by year
        """
        # Get Manley parameters
        alpha = self.manley_params['alpha']
        beta = self.manley_params['beta']
        gamma = self.manley_params['gamma']
        stock = self.manley_params['initial_stock']
        
        # Calculate supply for each year
        for year in self.years:
            # Check that price is available for this year
            if year not in prices.index:
                raise ValueError(f"Price not available for year {year}")
            
            # Get price for this year
            price = prices[year]
            
            # Calculate supply (with safety check for stock)
            if stock <= 0:
                supply = 0.0
            else:
                supply = alpha * (price ** beta) * (stock ** gamma)
            
            # Update results
            self.results.loc[year, 'manley_supply'] = supply
            
            # Update stock for next year
            if year != self.years[-1]:
                stock -= supply
    
    def update_forestry_data(self, new_data: pd.DataFrame) -> None:
        """
        Update forestry data with new values.
        
        Args:
            new_data: DataFrame with updated forestry data.
                If 'Variable' column is present, only 'forestry_tradeable' variable will be used.
        """
        # Handle 'Variable' column if present
        if 'Variable' in new_data.columns:
            new_data = new_data[new_data['Variable'] == 'forestry_tradeable']
            # Remove Variable column as it's no longer needed
            if 'Variable' in new_data.columns:
                new_data = new_data.drop(columns=['Variable'])
        
        # Handle column names
        if 'Value' in new_data.columns and 'forestry_supply' not in new_data.columns:
            new_data = new_data.rename(columns={'Value': 'forestry_supply'})
        
        # Handle index
        if 'Year' in new_data.columns:
            new_data = new_data.set_index('Year')
        elif 'year' in new_data.columns:
            new_data = new_data.set_index('year')
        
        # Update existing data with new values
        for col in new_data.columns:
            if col in self.forestry_data.columns:
                self.forestry_data.loc[new_data.index, col] = new_data[col]
    
    def set_manley_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set parameters for the Manley equation.
        
        Args:
            params: Dictionary of parameters for the Manley equation.
                Required keys: 'alpha', 'beta', 'gamma', 'initial_stock'
        """
        # Update parameters
        self.manley_params.update(params)
        
        # Validate parameters
        required_params = ['alpha', 'beta', 'gamma', 'initial_stock']
        missing_params = [param for param in required_params if param not in self.manley_params]
        if missing_params:
            raise ValueError(f"Missing Manley equation parameters: {missing_params}")
    
    def set_use_manley_equation(self, use_manley: bool) -> None:
        """
        Set whether to use the Manley equation for dynamic forestry supply.
        
        Args:
            use_manley: Whether to use the Manley equation.
        """
        self.use_manley_equation = use_manley
        
        # If switching to Manley equation, validate parameters
        if use_manley:
            required_params = ['alpha', 'beta', 'gamma', 'initial_stock']
            missing_params = [param for param in required_params if param not in self.manley_params]
            if missing_params:
                raise ValueError(f"Missing Manley equation parameters: {missing_params}")
    
    def get_total_supply(self) -> pd.Series:
        """
        Get the total forestry supply across all years.
        
        Returns:
            Series of total forestry supply indexed by year.
        """
        return self.results['total_supply']
    
    def extract_forestry_variables(self) -> Optional[pd.DataFrame]:
        """
        Extract forestry_held and forestry_surrender variables if available in the raw data.
        
        Returns:
            DataFrame with forestry_held and forestry_surrender data by year, or None if not available.
        """
        # This method would be implemented in a real scenario to extract these variables
        # from the raw data. In this implementation, it's a placeholder.
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the forestry supply component to a dictionary.
        
        Returns:
            Dictionary representation of the forestry supply component.
        """
        return {
            'forestry_data': self.forestry_data.to_dict(),
            'use_manley_equation': self.use_manley_equation,
            'manley_params': self.manley_params,
            'results': self.results.to_dict()
        }
