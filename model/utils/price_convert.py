"""
CPI data processing utilities for converting between nominal and real prices.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union, Any

def load_cpi_data(file_path: Union[str, Path]) -> pd.Series:
    """
    Load CPI data from CSV file.
    
    Args:
        file_path: Path to CPI data file
        
    Returns:
        Series of CPI values indexed by year
    """
    df = pd.read_csv(file_path)
    
    # The Year column contains just years in this case
    # Convert to integer directly
    df['Year'] = df['Year'].astype(int)
    
    # Create a clean Series with year and CPI value
    cpi_series = df.set_index('Year')['Value']
    
    # Ensure index is integer
    cpi_series.index = cpi_series.index.astype(int)
    
    return cpi_series

def get_conversion_factor(from_year: int, to_year: int, cpi_data: pd.Series) -> float:
    """
    Calculate CPI conversion factor from one year to another.
    
    Args:
        from_year: Source year
        to_year: Target year
        cpi_data: CPI data series
        
    Returns:
        Conversion factor to multiply prices by
    """
    # Handle missing years by using nearest available
    def get_nearest_year(year):
        if year in cpi_data.index:
            return year
        # Find nearest available year
        available_years = sorted(cpi_data.index)
        if year < min(available_years):
            return min(available_years)
        elif year > max(available_years):
            return max(available_years)
        else:
            # Find closest available year
            return min(available_years, key=lambda x: abs(x - year))
    
    from_year_adj = get_nearest_year(from_year)
    to_year_adj = get_nearest_year(to_year)
    
    # Calculate conversion factor (CPI_to_year / CPI_from_year)
    return cpi_data[to_year_adj] / cpi_data[from_year_adj]

def convert_nominal_to_real(prices: Union[Dict[int, float], pd.Series], 
                           cpi_data: pd.Series,
                           base_year: int = 2023) -> Dict[int, float]:
    """
    Convert prices from nominal to real terms.
    
    Args:
        prices: Dictionary or Series of prices indexed by year
        cpi_data: CPI data series
        base_year: Base year for real prices (default: 2023)
        
    Returns:
        Dictionary of real prices indexed by year
    """
    # Convert prices to dictionary if it's a Series
    if isinstance(prices, pd.Series):
        prices = prices.to_dict()
    
    real_prices = {}
    for year, price in prices.items():
        conversion_factor = get_conversion_factor(year, base_year, cpi_data)
        real_prices[year] = price * conversion_factor
    
    return real_prices

def convert_real_to_nominal(prices: Union[Dict[int, float], pd.Series], 
                           cpi_data: pd.Series,
                           base_year: int = 2023) -> Dict[int, float]:
    """
    Convert prices from real to nominal terms.
    
    Args:
        prices: Dictionary or Series of prices indexed by year
        cpi_data: CPI data series
        base_year: Base year for real prices (default: 2023)
        
    Returns:
        Dictionary of nominal prices indexed by year
    """
    # Convert prices to dictionary if it's a Series
    if isinstance(prices, pd.Series):
        prices = prices.to_dict()
    
    nominal_prices = {}
    for year, price in prices.items():
        conversion_factor = get_conversion_factor(base_year, year, cpi_data)
        nominal_prices[year] = price * conversion_factor
    
    return nominal_prices 