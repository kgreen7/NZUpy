"""
Price response component for the NZ ETS model.

This module implements the price response component, which models how
emissions respond to changes in NZU prices using the demand model approach.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Union, Any


class PriceResponse:
    """
    Price response component for NZ ETS model.
    
    This class models how emissions respond to changes in NZU prices,
    using the demand model approach consistent with the Excel implementation.
    """
    
    def __init__(
        self,
        years: List[int],
        demand_model_params: Dict[str, float],
        base_price: float = 0.0
    ):
        """
        Initialise the price response component.
        
        Args:
            years: List of years for the model.
            demand_model_params: Parameters for the demand model.
                Required keys: 'constant', 'reduction_to_t1', 'price', 'model_number'
                For NPV calculation: 'discount_rate', 'forward_years', 'price_conversion_factor'
            base_price: Base price for reference purposes (default: 0.0)
        """
        self.years = years
        self.base_price = base_price
        self.demand_model_params = demand_model_params
        self._initialise_data()
    
    def _initialise_data(self):
        """Initialise data structures for price response."""
        self.results = pd.DataFrame(
            index=self.years,
            columns=[
                'price', 'forward_price', 'emissions_reduction', 'response_percentage',
                'prev_reduction'
            ],
            data=0.0
        )
        self.price_history = {}
        self.previous_reduction = {}
    
    def calculate(self, prices: pd.Series, base_emissions: pd.Series) -> pd.DataFrame:
        """
        Calculate price response based on the demand model.
        
        Args:
            prices: Series of NZU prices indexed by year.
            base_emissions: Series of baseline emissions indexed by year.
            
        Returns:
            DataFrame containing price response results by year.
        """
        # Update price history
        for year in self.years:
            if year in prices.index:
                self.price_history[year] = prices[year]
        
        # Determine base price if not already set
        if self.base_price is None and len(self.price_history) > 0:
            self.base_price = self.price_history[min(self.price_history.keys())]
        elif self.base_price is None:
            self.base_price = 0  # Default base price if no history
        
        # Get model number from parameters
        model_number = int(self.demand_model_params.get('model_number', 2))
        
        # Use appropriate calculation method based on model number
        if model_number == 1:
            return self._calculate_with_demand_model_1(prices, base_emissions)
        else:  # Default to model 2
            return self._calculate_with_demand_model_2(prices, base_emissions)
    
    def calculate_forward_price_with_npv(self, year: int, prices: pd.Series) -> float:
        """
        Calculate forward price using NPV approach matching Excel model.
        
        Args:
            year: The current year
            prices: Series of prices indexed by year
            
        Returns:
            Forward price calculated using NPV approach
        """
        # Get parameters from demand model params
        discount_rate = float(self.demand_model_params['discount_rate'])
        years_forward = int(self.demand_model_params['forward_years'])
        
        # Get current price
        current_price = prices[year]
        
        # Get price range for future years
        future_prices = []
        for i in range(1, years_forward + 1):
            future_year = year + i
            if future_year in prices.index:
                future_prices.append(prices[future_year])
            else:
                # If future year not available, use the last available price
                future_prices.append(prices[max([y for y in prices.index if y <= max(prices.index)])])
        
        # Calculate NPV of future prices (if years_forward > 0)
        if years_forward > 0 and future_prices:
            npv_sum = 0
            for i, price in enumerate(future_prices):
                npv_sum += price / ((1 + discount_rate) ** (i + 1))
            
            # Calculate PV normalisation factor
            pv_factor = sum(1 / ((1 + discount_rate) ** (i + 1)) for i in range(years_forward))
            
            # Combine current price with NPV of future prices
            forward_price = (current_price + npv_sum) / (1 + pv_factor)
        else:
            forward_price = current_price
        
        return forward_price
    
    def _calculate_with_demand_model_1(self, prices: pd.Series, base_emissions: pd.Series) -> pd.DataFrame:
        """
        Calculate price response using demand model 1 (linear autoregressive).
        
        Args:
            prices: Series of NZU prices indexed by year.
            base_emissions: Series of baseline emissions indexed by year.
            
        Returns:
            DataFrame containing price response results.
        """
        # Get demand model parameters
        constant = self.demand_model_params.get('constant', 0)
        reduction_to_t1 = self.demand_model_params.get('reduction_to_t1', 0)
        price_coef = self.demand_model_params.get('price', 0)
        
        # Get 2019$ conversion factor
        price_conversion_factor = float(self.demand_model_params['price_conversion_factor'])
        
        # Initialise previous reduction for first year
        if min(self.years) not in self.previous_reduction:
            self.previous_reduction[min(self.years)] = 1.0  # Start with 1.0
        
        # Calculate for each year
        for i, year in enumerate(self.years):
            # Skip years without price data
            if year not in prices.index:
                continue
            
            # Record current price
            self.results.loc[year, 'price'] = prices[year]
            
            # Get previous reduction (if any)
            prev_year = year - 1
            prev_reduction = (
                self.results.loc[prev_year, 'emissions_reduction'] 
                if prev_year in self.years and self.results.loc[prev_year, 'emissions_reduction'] > 0
                else 1.0  # Use 1.0 for first year or if previous reduction is zero/negative
            )
            
            # Record previous reduction for reference
            self.results.loc[year, 'prev_reduction'] = prev_reduction
            
            # Calculate forward price using NPV approach
            forward_price = self.calculate_forward_price_with_npv(year, prices)
            
            # Apply 2019$ conversion for demand response calculation
            adjusted_forward_price = forward_price / price_conversion_factor
            
            # Record forward price for reference
            self.results.loc[year, 'forward_price'] = forward_price
            
            # Calculate new reduction using linear formula
            # MAX(0, prev_reduction * coef1 + forward_price * coef2 + constant)
            new_reduction = max(0, 
                prev_reduction * reduction_to_t1 + 
                adjusted_forward_price * price_coef + 
                constant
            )
            
            # Apply year-specific additions (from Excel model)
            if year >= 2040:
                new_reduction += 419 * 2
            elif year >= 2030:
                new_reduction += 419
            
            # Ensure non-negative reduction and not more than base emissions
            if year in base_emissions.index:
                new_reduction = min(max(0, new_reduction), base_emissions[year])
            
            # Record the reduction
            self.results.loc[year, 'emissions_reduction'] = new_reduction
            
            # Record response percentage
            self.results.loc[year, 'response_percentage'] = (
                new_reduction / base_emissions[year] if year in base_emissions.index and base_emissions[year] > 0 
                else 0
            )
        
        return self.results
    
    def _calculate_with_demand_model_2(self, prices: pd.Series, base_emissions: pd.Series) -> pd.DataFrame:
        """
        Calculate price response using demand model 2 (logarithmic formula).
        This matches the Excel implementation's formula:
        exp(ln(prev_reduction) * coef1 + ln(forward_price) * coef2 + constant)
        
        Args:
            prices: Series of NZU prices indexed by year.
            base_emissions: Series of baseline emissions indexed by year.
            
        Returns:
            DataFrame containing price response results.
        """
        # Get demand model parameters
        constant = self.demand_model_params.get('constant', 0)
        reduction_to_t1 = self.demand_model_params.get('reduction_to_t1', 0)
        price_coef = self.demand_model_params.get('price', 0)
        
        # Get 2019$ conversion factor
        price_conversion_factor = float(self.demand_model_params['price_conversion_factor'])
        
        # Initialise previous reduction for first year
        if min(self.years) not in self.previous_reduction:
            self.previous_reduction[min(self.years)] = 1.0  # Start with 1.0 to avoid log(0)
        
        # Calculate for each year
        for i, year in enumerate(self.years):
            # Skip years without price data
            if year not in prices.index:
                continue
            
            # Record current price
            self.results.loc[year, 'price'] = prices[year]
            
            # Get previous reduction (if any)
            prev_year = year - 1
            prev_reduction = (
                self.results.loc[prev_year, 'emissions_reduction'] 
                if prev_year in self.years and self.results.loc[prev_year, 'emissions_reduction'] > 0
                else 1.0  # Use 1.0 for first year or if previous reduction is zero/negative
            )
            
            # Record previous reduction for reference
            self.results.loc[year, 'prev_reduction'] = prev_reduction
            
            # Calculate forward price using NPV approach
            forward_price = self.calculate_forward_price_with_npv(year, prices)
            
            # Apply 2019$ conversion for demand response calculation
            adjusted_forward_price = forward_price / price_conversion_factor
            
            # Record forward price for reference
            self.results.loc[year, 'forward_price'] = forward_price
            
            # Calculate new reduction using logarithmic formula (exactly as in Excel)
            # exp(ln(prev_reduction) * coef1 + ln(forward_price) * coef2 + constant)
            try:
                new_reduction = np.exp(
                    np.log(prev_reduction) * reduction_to_t1 + 
                    np.log(adjusted_forward_price) * price_coef + 
                    constant
                )
            except (ValueError, RuntimeError):
                # Fallback if logarithm issues occur
                new_reduction = prev_reduction * reduction_to_t1 + adjusted_forward_price * price_coef
            
            # Ensure non-negative reduction and not more than base emissions
            if year in base_emissions.index:
                new_reduction = min(max(0, new_reduction), base_emissions[year])
            
            # Record the reduction
            self.results.loc[year, 'emissions_reduction'] = new_reduction
            
            # Record response percentage
            self.results.loc[year, 'response_percentage'] = (
                new_reduction / base_emissions[year] if year in base_emissions.index and base_emissions[year] > 0 
                else 0
            )
        
        return self.results
    
    def update_price_response_data(self, new_data: pd.DataFrame) -> None:
        """
        Update price response data with new values (deprecated, kept for backward compatibility).
        
        Args:
            new_data: DataFrame with updated price response data.
        """
        # This method is kept for backward compatibility but does nothing
        pass

    def get_forward_price(self, year, prices):
        """
        Get forward price for emissions response (looking forward years ahead).
        
        Args:
            year: The current year
            prices: Series of prices indexed by year
        
        Returns:
            The forward_years price for the given year
        """
        # Look 10 years ahead
        forward_year = year + self.demand_model_params['forward_years']
        
        # Use the forward_year price if available, otherwise use the furthest available price
        if forward_year in prices.index:
            return prices[forward_year]
        else:
            # This should not happen with extended projection years, but as fallback
            return prices[max(prices.index)]
    
    def set_base_price(self, price: float) -> None:
        """
        Set the base price for reference purposes.
        
        Args:
            price: Base price.
        """
        self.base_price = price
    
    def get_emissions_reduction(self) -> pd.Series:
        """
        Get the emissions reduction due to price response across all years.
        
        Returns:
            Series of emissions reduction indexed by year.
        """
        return self.results['emissions_reduction']
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the price response component to a dictionary.
        
        Returns:
            Dictionary representation of the price response component.
        """
        return {
            'base_price': self.base_price,
            'demand_model_params': self.demand_model_params,
            'results': self.results.to_dict()
        }
