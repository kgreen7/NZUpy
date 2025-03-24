"""
Historical data management for the NZUpy model.

This module provides functionality for loading and accessing historical data
and time-varying configuration parameters.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Union, Any
from model.utils.price_convert import load_cpi_data, convert_nominal_to_real, convert_real_to_nominal

class HistoricalDataManager:
    """Manages historical data and time-varying configuration for the NZUpy model."""
    
    def __init__(self, data_dir: Union[str, Path]):
        """Initialise with path to data directory."""
        self.data_dir = Path(data_dir)
        
        # Set up directories
        self.parameters_dir = self.data_dir / "inputs" / "parameters"
        self.supply_dir = self.data_dir / "inputs" / "supply"
        self.demand_dir = self.data_dir / "inputs" / "demand"
        self.historical_dir = self.data_dir / "inputs" / "historical"
        self.economic_dir = self.data_dir / "inputs" / "economic"
        
        # Load configuration data
        self.price_data = self._load_price_data()
        self.price_control_data = self._load_price_control_data()
        self.stockpile_start_data = self._load_stockpile_start_data()
        
        # Load CPI data
        self.cpi_data = load_cpi_data(self.economic_dir / "CPI.csv")
        
        # Dictionary for other historical data
        self.historical_data = {}
        
        # Load historical data if directory exists
        if self.historical_dir.exists():
            self._discover_and_load_historical_data()
    
    def _load_price_data(self):
        """Load carbon price data from CSV file."""
        try:
            file_path = self.parameters_dir / "price.csv"
            df = pd.read_csv(file_path)
            
            # Pivot data to get configs as columns
            price_data = df.pivot(index='Year', columns='Config', values='Value')
            price_data.index = price_data.index.astype(int)
            return price_data
            
        except Exception as e:
            #print(f"Warning: Could not load price data: {e}")
            return pd.DataFrame(columns=['central'])
    
    def _load_price_control_data(self):
        """Load price control data from CSV file."""
        try:
            file_path = self.parameters_dir / "price_control.csv"
            df = pd.read_csv(file_path)
            
            # Pivot data to get configs as columns
            control_data = df.pivot(index='Year', columns='Config', values='Value')
            control_data.index = control_data.index.astype(int)
            return control_data
            
        except Exception as e:
            print(f"Warning: Could not load price control data: {e}")
            return pd.DataFrame(columns=['central'])
    
    def _load_stockpile_start_data(self):
        """Load stockpile start values from stockpile_balance.csv file."""
        try:
            # Look for stockpile_balance.csv in the stockpile directory
            file_path = self.data_dir / "inputs" / "stockpile" / "stockpile_balance.csv"
            if not file_path.exists():
                print(f"Warning: Stockpile balance data file not found: {file_path}")
                return pd.DataFrame()
                
            df = pd.read_csv(file_path)
            
            # Filter for 'stockpile' and 'surplus' variables 
            df = df[df['Variable'].isin(['stockpile', 'surplus'])]
            
            # Convert year to integer
            df['Year'] = df['Year'].astype(int)
            
            return df
            
        except Exception as e:
            print(f"Warning: Could not load stockpile balance data: {e}")
            return pd.DataFrame()
    
    def _discover_and_load_historical_data(self):
        """Dynamically discover and load historical data files."""
        for file_path in self.historical_dir.glob("*.csv"):
            variable_name = file_path.stem
            try:
                df = pd.read_csv(file_path)
                
                # Special handling for carbon price data
                if variable_name == 'carbon_price':
                    if not all(col in df.columns for col in ['Year', 'Value']):
                        raise ValueError("Carbon price CSV must contain 'Year' and 'Value' columns")
                    self.historical_data[variable_name] = df.set_index('Year')['Value']
                
                # Standard handling for other files
                elif 'Year' in df.columns and 'Value' in df.columns:
                    # Standard Year/Value format
                    self.historical_data[variable_name] = df.set_index('Year')['Value']
                elif 'date' in df.columns and 'price' in df.columns:
                    # NZU price format
                    df['year'] = pd.to_datetime(df['date']).dt.year
                    self.historical_data[variable_name] = df.groupby('year')['price'].mean()
                
                # Ensure index is integer
                if variable_name in self.historical_data:
                    self.historical_data[variable_name].index = self.historical_data[variable_name].index.astype(int)
                    
            except Exception as e:
                if variable_name == 'carbon_price':
                    raise RuntimeError(f"Failed to load historical carbon prices from {file_path}: {e}")
                print(f"Warning: Could not load historical data file {file_path}: {e}")
    
    def get_price_control(self, year: int, config: str = 'central') -> float:
        """Get price control value for a specific year and config."""
        if self.price_control_data.empty or config not in self.price_control_data.columns:
            return 1.0
        
        if year in self.price_control_data.index:
            return self.price_control_data.loc[year, config]
        
        # Default value
        return 1.0
    
    def get_historical_data(self, variable: str, nominal: bool = False) -> Optional[pd.Series]:
        """
        Get historical data for a variable if available.
        
        Args:
            variable: Name of the variable to get historical data for
            nominal: Whether to return nominal prices (only applies to price data)
            
        Returns:
            Series with historical data, or None if not available
        """
        if variable == 'carbon_price':
            # For carbon price, we need to handle both real and nominal prices
            real_prices = self.historical_data.get('carbon_price', None)
            if real_prices is None:
                return None
            
            if nominal:
                # Convert real prices to nominal and return
                return self.convert_to_nominal(real_prices)
            else:
                # Return real prices (already in 2023 NZD)
                return real_prices
        
        # For other variables, return from historical_data
        return self.historical_data.get(variable, None)
    
    def get_combined_series(self, variable: str, model_data: pd.Series, nominal: bool = False) -> pd.Series:
        """
        Combine historical and model data for a variable.
        
        Args:
            variable: Name of the variable to combine data for
            model_data: Model-generated data series
            nominal: If True, return nominal prices; otherwise return real prices (2023 NZD)
            
        Returns:
            Combined series with historical and model data
        """
        hist_data = self.get_historical_data(variable, nominal=nominal)
        if hist_data is None:
            return model_data
            
        # Filter out years that overlap with model_data
        model_years = set(model_data.index)
        hist_data = hist_data[~hist_data.index.isin(model_years)]
        
        # Combine and sort
        return pd.concat([hist_data, model_data]).sort_index()
    
    def get_stockpile_start_values(self, year: int, config: str = 'central') -> Dict[str, float]:
        """
        Get stockpile start values for a specific year and config.
        
        Args:
            year: Reference year for stockpile values
            config: Configuration name (defaults to 'central')
            
        Returns:
            Dictionary with 'stockpile' and 'surplus' values
        """
        if self.stockpile_start_data.empty:
            print(f"Warning: No stockpile start data available, using defaults")
            return {'stockpile': 159902, 'surplus': 67300}  # Default values
        
        # First try exact match on year and config
        year_config_match = self.stockpile_start_data[
            (self.stockpile_start_data['Year'] == year) & 
            (self.stockpile_start_data['Config'].str.lower() == config.lower())
        ]
        
        if not year_config_match.empty:
            # Get stockpile and surplus values
            stockpile = year_config_match[year_config_match['Variable'] == 'stockpile']['Value'].iloc[0]
            surplus = year_config_match[year_config_match['Variable'] == 'surplus']['Value'].iloc[0]
            return {'stockpile': stockpile, 'surplus': surplus}
        
        # If no exact match, try central config for same year
        if config != 'central':
            central_match = self.stockpile_start_data[
                (self.stockpile_start_data['Year'] == year) & 
                (self.stockpile_start_data['Config'] == 'central')
            ]
            if not central_match.empty:
                stockpile = central_match[central_match['Variable'] == 'stockpile']['Value'].iloc[0]
                surplus = central_match[central_match['Variable'] == 'surplus']['Value'].iloc[0]
                return {'stockpile': stockpile, 'surplus': surplus}
        
        # If still no match, use defaults
        print(f"Warning: No stockpile start values found for year {year} and config {config}, using defaults")
        return {'stockpile': 159902, 'surplus': 67300}  # Default values
    
    def convert_to_nominal(self, real_prices: pd.Series) -> pd.Series:
        """Convert real prices (2023 NZD) to nominal prices."""
        return convert_real_to_nominal(real_prices, cpi_data=self.cpi_data)
    
