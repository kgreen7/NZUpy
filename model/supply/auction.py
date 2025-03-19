"""
Auction supply component for the NZ ETS model.

This module implements the government auction supply component of the NZ ETS model,
handling both fixed auction volumes and price-contingent auction volumes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Union, Any


class AuctionSupply:
    """
    Government auction supply component for NZ ETS model.
    
    This class handles the auction supply volumes, including:
    - Base auction volumes
    - Cost containment reserve (CCR) volumes
    - Auction reserve price (ARP) mechanisms
    """
    
    def __init__(
        self,
        years: List[int],
        auction_data: pd.DataFrame,
    ):
        """
        Initialise the auction supply component.
        
        Args:
            years: List of years for the model.
            auction_data: DataFrame with auction data indexed by year.
                Required columns: 'base_volume', 'ccr_volume_1', 'ccr_volume_2',
                'ccr_trigger_price_1', 'ccr_trigger_price_2', 'auction_reserve_price'
        """
        self.years = years
        self.auction_data = auction_data
        
        # Initialise results DataFrame
        self.results = pd.DataFrame(
            index=self.years,
            columns=[
                'base_auction', 'ccr_auction_1', 'ccr_auction_2', 
                'total_auction', 'revenue'
            ],
            data=0.0
        )
    
    def calculate(self, prices: pd.Series) -> pd.DataFrame:
        """
        Calculate auction supply based on current prices.
        
        Args:
            prices: Series of NZU prices indexed by year.
            
        Returns:
            DataFrame containing auction supply volumes by type and year.
        """
        # Check that prices have the expected index (years)
        if not all(year in prices.index for year in self.years):
            missing_years = [year for year in self.years if year not in prices.index]
            raise ValueError(f"Prices missing for years: {missing_years}")
        
        # Initialise results with base auction volumes
        self.results['base_auction'] = self.auction_data['base_volume']
        
        # Calculate CCR volumes based on price triggers
        self.results['ccr_auction_1'] = 0.0
        self.results['ccr_auction_2'] = 0.0
        
        # Initialise revenue column
        self.results['revenue'] = 0.0
        
        for year in self.years:
            # Check if price triggers CCR1
            if prices[year] >= self.auction_data.loc[year, 'ccr_trigger_price_1']:
                self.results.loc[year, 'ccr_auction_1'] = self.auction_data.loc[year, 'ccr_volume_1']
                
                # Check if price also triggers CCR2
                if prices[year] >= self.auction_data.loc[year, 'ccr_trigger_price_2']:
                    self.results.loc[year, 'ccr_auction_2'] = self.auction_data.loc[year, 'ccr_volume_2']
        
        # Calculate total auction volumes
        self.results['total_auction'] = (
            self.results['base_auction'] + 
            self.results['ccr_auction_1'] + 
            self.results['ccr_auction_2']
        )
        
        # Apply auction reserve price mechanism
        # If price is below ARP, no units are auctioned
        for year in self.years:
            if prices[year] < self.auction_data.loc[year, 'auction_reserve_price']:
                self.results.loc[year, ['base_auction', 'ccr_auction_1', 'ccr_auction_2', 'total_auction']] = 0.0
                self.results.loc[year, 'revenue'] = 0.0
            else:
                # Calculate revenue with correct units (price in $/tonne * volume in tonnes)
                self.results.loc[year, 'revenue'] = prices[year] * self.results.loc[year, 'total_auction'] * 1000
        
        return self.results
    
    def update_auction_data(self, new_data: pd.DataFrame) -> None:
        """
        Update auction data with new values.
        
        Args:
            new_data: DataFrame with updated auction data.
        """
        # Update existing data with new values
        for col in new_data.columns:
            if col in self.auction_data.columns:
                self.auction_data.loc[new_data.index, col] = new_data[col]
    
    def get_total_supply(self) -> pd.Series:
        """
        Get the total auction supply across all years.
        
        Returns:
            Series of total auction supply indexed by year.
        """
        return self.results['total_auction']
    
    def get_auction_revenue(self) -> pd.Series:
        """
        Get the auction revenue across all years.
        
        Returns:
            Series of auction revenue indexed by year.
        """
        return self.results['revenue']
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the auction supply component to a dictionary.
        
        Returns:
            Dictionary representation of the auction supply component.
        """
        return {
            'auction_data': self.auction_data.to_dict(),
            'results': self.results.to_dict()
        }