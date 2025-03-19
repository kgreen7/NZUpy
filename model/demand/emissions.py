"""
Emissions demand component for the NZ ETS model.

This module implements the emissions demand component, which represents
the baseline emissions that drive demand for NZUs in the model.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Union, Any


class EmissionsDemand:
    """
    Emissions demand component for NZ ETS model.
    
    This class handles the calculation of baseline emissions demand,
    with support for different input configurations.
    """
    
    def __init__(
        self,
        years: List[int],
        emissions_data: pd.DataFrame,
        config_name: str = "central",
    ):
        """
        Initialise the emissions demand component.
        
        Args:
            years: List of years for the model.
            emissions_data: DataFrame containing emissions data with columns:
                - Year: Year of the data
                - Config: Configuration name (central, CCC_DP, CCC_CPR, CCC_mid)
                - Value: Emissions value for that year and config
            config_name: Configuration to use for emissions.
                Options: "central", "CCC_DP", "CCC_CPR", "CCC_mid"
        """
        self.years = years
        self.emissions_data = emissions_data
        self.config_name = config_name
        
        # Initialise results DataFrame
        self.results = pd.DataFrame(
            index=self.years,
            columns=['config_name', 'baseline_emissions', 'price_adjusted_emissions', 'total_demand'],
            data={'config_name': self.config_name, 'baseline_emissions': 0.0, 'price_adjusted_emissions': 0.0, 'total_demand': 0.0}
        )
        
        # Set initial emissions based on config
        self.set_config(config_name)
    
    def calculate(self, price_response: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Calculate emissions demand based on the selected configuration.
        
        Args:
            price_response: Optional series of price-induced emissions reductions indexed by year.
                If None, no price adjustment is applied.
            
        Returns:
            DataFrame containing emissions demand by year.
        """
        # Copy emissions values to results
        self.results['baseline_emissions'] = self.emissions
        
        # Apply price response if provided
        if price_response is not None:
            # Ensure price_response has values for all years
            for year in self.years:
                if year not in price_response.index:
                    raise ValueError(f"Price response missing for year {year}")
                
                # Apply price response as a reduction in emissions
                self.results.loc[year, 'price_adjusted_emissions'] = max(
                    0.0,  # Ensure emissions are non-negative
                    self.results.loc[year, 'baseline_emissions'] - price_response[year]
                )
        else:
            # No price adjustment
            self.results['price_adjusted_emissions'] = self.results['baseline_emissions']
        
        # Calculate total demand (currently equal to price-adjusted emissions)
        self.results['total_demand'] = self.results['price_adjusted_emissions']
        
        return self.results
    
    def update_emissions_data(self, new_data: pd.DataFrame) -> None:
        """
        Update emissions data with new values.
        
        Args:
            new_data: DataFrame with updated emissions data.
        """
        # Update existing data with new values
        for col in new_data.columns:
            if col in self.emissions_data.columns:
                self.emissions_data.loc[new_data.index, col] = new_data[col]
        
        # Re-apply current config to update emissions
        self.set_config(self.config_name)
    
    def set_config(self, config_name: str) -> None:
        """
        Set the configuration to use for emissions.
        
        Args:
            config_name: Configuration to use.
                Options: "central", "CCC_DP", "CCC_CPR", "CCC_mid"
        """
        # Valid config names from emissions_baselines.csv
        valid_configs = ["central", "CCC_DP", "CCC_CPR", "CCC_mid"]
        
        if config_name not in valid_configs:
            raise ValueError(f"Invalid config: {config_name}. Valid options: {valid_configs}")
        
        # Filter data for the selected config
        config_data = self.emissions_data[self.emissions_data['Config'] == config_name]
        
        if config_data.empty:
            raise ValueError(f"No data found for config '{config_name}'")
        
        self.config_name = config_name
        self.results['config_name'] = config_name
        
        # Update the emissions series based on the new config
        self.emissions = config_data.set_index('Year')['Value']
    
    def get_total_demand(self) -> pd.Series:
        """
        Get the total emissions demand across all years.
        
        Returns:
            Series of total emissions demand indexed by year.
        """
        return self.results['total_demand']
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the emissions demand component to a dictionary.
        
        Returns:
            Dictionary representation of the emissions demand component.
        """
        return {
            'emissions_data': self.emissions_data.to_dict(),
            'config_name': self.config_name,
            'results': self.results.to_dict()
        }
