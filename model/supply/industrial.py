"""
Industrial allocation supply component for the NZ ETS model.

This module implements a simplified industrial allocation supply component which
directly passes through data from input files without applying phase-out rates
or activity adjustments.
"""

import pandas as pd
from typing import Dict, Optional, List, Any


class IndustrialAllocation:
    """
    Simplified industrial allocation supply component for NZ ETS model.
    
    This class handles industrial allocation volumes by directly using input data,
    without applying phase-out rates or activity adjustments.
    """
    
    def __init__(
        self,
        years: List[int],
        ia_data: pd.DataFrame,
    ):
        """
        Initialise the industrial allocation component.
        
        Args:
            years: List of years for the model.
            ia_data: DataFrame with industrial allocation data indexed by year.
                Required column: 'baseline_allocation'
        """
        self.years = years
        self.baseline_allocation = ia_data['baseline_allocation'].astype(float)
        
        # Initialise results DataFrame
        self.results = pd.DataFrame(
            index=self.years,
            columns=['baseline_allocation', 'adjusted_allocation'],
            data=0.0
        )
    
    def calculate(self, prices: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Calculate industrial allocation based on input data.
        
        Args:
            prices: Optional series of NZU prices indexed by year.
                   Not used in this implementation but kept for API compatibility.
            
        Returns:
            DataFrame containing industrial allocation volumes by year.
        """
        # Set both baseline and adjusted allocation to be the same
        # (no phase-out or activity adjustment applied)
        self.results['baseline_allocation'] = self.baseline_allocation
        self.results['adjusted_allocation'] = self.baseline_allocation
        
        return self.results
    
    def update_allocation_data(self, new_data: pd.DataFrame) -> None:
        """
        Update industrial allocation data with new values.
        
        Args:
            new_data: DataFrame with updated allocation data.
        """
        # Update baseline allocation if provided
        if 'baseline_allocation' in new_data.columns:
            self.baseline_allocation.loc[new_data.index] = new_data['baseline_allocation']
    
    def get_total_allocation(self) -> pd.Series:
        """
        Get the total industrial allocation across all years.
        
        Returns:
            Series of total industrial allocation indexed by year.
        """
        return self.results['adjusted_allocation']
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the industrial allocation component to a dictionary.
        
        Returns:
            Dictionary representation of the industrial allocation component.
        """
        return {
            'baseline_allocation': self.baseline_allocation.to_dict(),
            'results': self.results.to_dict()
        }