"""
Output formatter for the NZUpy model.

This module provides functionality for converting raw model results into
structured pandas DataFrames with standardised multi-index columns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path


class OutputFormat:
    """
    Formats raw model results into structured DataFrames.
    
    This class takes the raw model results (nested dictionaries) and converts
    them into standardised pandas DataFrames with multi-index columns for
    scenarios and variables.
    """
    
    def __init__(self, model):
        """
        Initialise the output formatter with a reference to the model.
        
        Args:
            model: Reference to the NZUpy instance
        """
        self.model = model
        self.years = model.years
        self.scenarios = model.scenarios  # These are model scenarios, not configs
        self.results = model.results
        self.data_handler = model.data_handler  # Get data_handler from model
        
        # Variable schema for documentation and helper methods
        self._variable_schema = self._create_variable_schema()
    
    def organise_outputs(self):
        """
        Organise model outputs into structured DataFrames for easier access.
        """
        # Store time array for convenience
        self.model.time = np.array(self.years)
        
        # Create DataFrames for each category
        self.model.prices = self._create_prices_dataframe()
        
        self.model.core = self._create_core_dataframe()
        
        self.model.supply = self._create_supply_dataframe()
        
        self.model.demand = self._create_demand_dataframe()
        
        self.model.auctions = self._create_auctions_dataframe()
        
        self.model.industrial = self._create_industrial_dataframe()
        
        self.model.forestry = self._create_forestry_dataframe()
        
        self.model.stockpile = self._create_stockpile_dataframe()
        
        self.model.inputs = self._create_inputs_dataframe()
    
    def _create_prices_dataframe(self):
        """Create DataFrame for prices with scenarios as columns."""
        data = {}
        
        # Import the CPI processor here to avoid circular imports
        from model.utils.price_convert import convert_real_to_nominal, load_cpi_data
        
        # Load CPI data once for efficiency
        cpi_path = self.data_handler.economic_dir / "CPI.csv"
        try:
            cpi_data = load_cpi_data(cpi_path)
        except Exception as e:
            print(f"Warning: Could not load CPI data from {cpi_path}: {e}")
            cpi_data = load_cpi_data(Path("data/inputs/economic/CPI.csv"))
        
        for scenario in self.scenarios:
            if scenario in self.results:
                result = self.results[scenario]
                
                # Extract real prices from results structure
                real_prices = None
                if 'prices' in result:
                    real_prices = result['prices']
                elif 'model' in result and 'prices' in result['model']:
                    real_prices = result['model']['prices']
                else:
                    raise ValueError(f"Carbon prices not found for scenario '{scenario}'. Results structure: {list(result.keys())}")
                # Store real prices (2023 NZD)
                data[(scenario, 'carbon_price')] = real_prices
                
                # Calculate and store nominal prices
                nominal_prices = pd.Series(
                    convert_real_to_nominal(real_prices, cpi_data=cpi_data), 
                    index=real_prices.index
                )
                data[(scenario, 'carbon_price_nominal')] = nominal_prices
                
                # Extract unmodified prices if available
                if 'unmodified_prices' in result:
                    unmodified_real = result['unmodified_prices']
                    data[(scenario, 'unmodified_carbon_price')] = unmodified_real
                    
                    # Calculate nominal unmodified prices
                    unmodified_nominal = pd.Series(
                        convert_real_to_nominal(unmodified_real, cpi_data=cpi_data),
                        index=unmodified_real.index
                    )
                    data[(scenario, 'unmodified_carbon_price_nominal')] = unmodified_nominal
                
                elif 'model' in result and 'unmodified_prices' in result['model']:
                    unmodified_real = result['model']['unmodified_prices']
                    data[(scenario, 'unmodified_carbon_price')] = unmodified_real
                    
                    # Calculate nominal unmodified prices
                    unmodified_nominal = pd.Series(
                        convert_real_to_nominal(unmodified_real, cpi_data=cpi_data),
                        index=unmodified_real.index
                    )
                    data[(scenario, 'unmodified_carbon_price_nominal')] = unmodified_nominal
        
        # Create DataFrame with multi-index columns
        if data:
            # Convert data to DataFrame
            df = pd.DataFrame(data)
            df.index.name = 'year'

            # Create proper multi-index columns or fix existing ones
            if not isinstance(df.columns, pd.MultiIndex):
                columns = pd.MultiIndex.from_tuples(df.columns, 
                                                names=['scenario', 'variable'])
                df.columns = columns
            else:
                # If already a MultiIndex but names are wrong, fix them
                if df.columns.names != ['scenario', 'variable']:
                    df.columns = pd.MultiIndex.from_tuples(
                        list(df.columns),
                        names=['scenario', 'variable']
                    )
        else:
            # Create empty DataFrame with proper structure including nominal price columns
            columns = pd.MultiIndex.from_product([
                self.scenarios, 
                ['carbon_price', 'carbon_price_nominal', 'unmodified_carbon_price', 'unmodified_carbon_price_nominal']
            ], names=['scenario', 'variable'])
            
            df = pd.DataFrame(columns=columns, index=self.years)
            df.index.name = 'year'
        
        return df

    def _create_core_dataframe(self):
        """Create DataFrame for core model outputs."""
        data = {}
        for scenario in self.scenarios:
            if scenario in self.results:
                result = self.results[scenario]
                
                # Extract supply-demand balance
                supply_demand_balance = None
                if 'supply' in result and 'demand' in result:
                    # For top-level supply and demand
                    if isinstance(result['supply'], pd.DataFrame) and 'total' in result['supply'].columns:
                        total_supply = result['supply']['total']
                    else:
                        total_supply = pd.Series(result['supply'], index=self.years)
                        
                    if isinstance(result['demand'], pd.Series):
                        demand = result['demand']
                    else:
                        demand = pd.Series(result['demand'], index=self.years)
                        
                    supply_demand_balance = total_supply - demand
                    
                elif 'model' in result:
                    # For nested supply and demand
                    model_result = result['model']
                    if 'supply_demand_balance' in model_result:
                        supply_demand_balance = model_result['supply_demand_balance']
                    elif 'supply' in model_result and 'demand' in model_result:
                        if isinstance(model_result['supply'], pd.DataFrame) and 'total' in model_result['supply'].columns:
                            total_supply = model_result['supply']['total']
                        else:
                            total_supply = pd.Series(model_result['supply'], index=self.years)
                            
                        if isinstance(model_result['demand'], pd.Series):
                            demand = model_result['demand']
                        else:
                            demand = pd.Series(model_result['demand'], index=self.years)
                            
                        supply_demand_balance = total_supply - demand
                
                if supply_demand_balance is not None:
                    data[(scenario, 'supply_demand_balance')] = supply_demand_balance
                
                # Extract price change rate
                price_change_rate = None
                if 'price_change_rate' in result:
                    # For single value, create a Series with same value for all years
                    if isinstance(result['price_change_rate'], (int, float)):
                        price_change_rate = pd.Series(result['price_change_rate'], index=self.years)
                    else:
                        price_change_rate = result['price_change_rate']
                elif 'model' in result and 'price_change_rate' in result['model']:
                    if isinstance(result['model']['price_change_rate'], (int, float)):
                        price_change_rate = pd.Series(result['model']['price_change_rate'], index=self.years)
                    else:
                        price_change_rate = result['model']['price_change_rate']
                
                if price_change_rate is not None:
                    data[(scenario, 'price_change_rate')] = price_change_rate
                    
                    # Also add avg_price_change_rate
                    # This is a single value, so create a Series with same value for all years
                    data[(scenario, 'avg_price_change_rate')] = pd.Series(
                        price_change_rate.iloc[0] if isinstance(price_change_rate, pd.Series) else price_change_rate,
                        index=self.years
                    )
        
        # Create DataFrame with multi-index columns
        if data:
            # Convert data to DataFrame
            df = pd.DataFrame(data)
            df.index.name = 'year'
            
            # Create proper multi-index columns or fix existing ones
            if not isinstance(df.columns, pd.MultiIndex):
                columns = pd.MultiIndex.from_tuples(df.columns, 
                                                names=['scenario', 'variable'])
                df.columns = columns
            else:
                # If already a MultiIndex but names are wrong, fix them
                if df.columns.names != ['scenario', 'variable']:
                    df.columns = pd.MultiIndex.from_tuples(
                        list(df.columns),
                        names=['scenario', 'variable']
                    )
        else:
            # Create empty DataFrame with proper structure
            columns = pd.MultiIndex.from_product(
                [self.scenarios, ['supply_demand_balance', 'price_change_rate', 'avg_price_change_rate']], 
                names=['scenario', 'variable']
            )
            df = pd.DataFrame(columns=columns, index=self.years)
            df.index.name = 'year'
        
        return df

    def _create_supply_dataframe(self):
        """Create DataFrame for supply components."""
        data = {}
        for scenario in self.scenarios:
            if scenario in self.results:
                result = self.results[scenario]
                
                # Extract supply components
                supply_df = None
                if 'supply' in result and isinstance(result['supply'], pd.DataFrame):
                    supply_df = result['supply']
                elif 'model' in result and 'supply' in result['model'] and isinstance(result['model']['supply'], pd.DataFrame):
                    supply_df = result['model']['supply']
                        
                if supply_df is not None:
                    for col in supply_df.columns:
                        data[(scenario, col)] = supply_df[col]
                    
                    # Check if surplus supply is available from stockpile component
                    # and add it as an additional supply component
                    stockpile_results = None
                    if 'model' in result and 'stockpile_component' in result['model']:
                        stockpile_comp = result['model']['stockpile_component']
                        if isinstance(stockpile_comp, dict) and 'results' in stockpile_comp:
                            stockpile_results = stockpile_comp['results']
                        elif hasattr(stockpile_comp, 'results'):
                            stockpile_results = stockpile_comp.results
                    
                    if stockpile_results is not None and isinstance(stockpile_results, pd.DataFrame):
                        # Add surplus used as a separate supply component
                        if 'surplus_used' in stockpile_results.columns:
                            data[(scenario, 'surplus')] = stockpile_results['surplus_used']
                        # Add surplus and non-surplus used components
                        if 'surplus_used' in stockpile_results.columns:
                            data[(scenario, 'surplus_used')] = stockpile_results['surplus_used']
                        if 'non_surplus_used' in stockpile_results.columns:
                            data[(scenario, 'non_surplus_used')] = stockpile_results['non_surplus_used']
        
        # Create DataFrame with multi-index columns
        if data:
            # Convert data to DataFrame
            df = pd.DataFrame(data)
            df.index.name = 'year'
            
            # Create proper multi-index columns or fix existing ones
            if not isinstance(df.columns, pd.MultiIndex):
                columns = pd.MultiIndex.from_tuples(df.columns, 
                                                names=['scenario', 'variable'])
                df.columns = columns
            else:
                # If already a MultiIndex but names are wrong, fix them
                if df.columns.names != ['scenario', 'variable']:
                    df.columns = pd.MultiIndex.from_tuples(
                        list(df.columns),
                        names=['scenario', 'variable']
                    )
        else:
            # Create empty DataFrame with proper structure - now including surplus_used and non_surplus_used
            columns = pd.MultiIndex.from_product(
                [self.scenarios, ['auction', 'industrial', 'forestry', 'stockpile', 'surplus', 'surplus_used', 'non_surplus_used', 'total']], 
                names=['scenario', 'variable']
            )
            df = pd.DataFrame(columns=columns, index=self.years)
            df.index.name = 'year'
        
        return df

    def _create_auctions_dataframe(self):
        """Create DataFrame for auction components."""
        data = {}
        
        for scenario in self.scenarios:
            if scenario in self.results:
                result = self.results[scenario]
                
                # Get the auction data directly from the supply DataFrame
                if not hasattr(self.model, 'supply') or not isinstance(self.model.supply, pd.DataFrame):
                    raise ValueError(f"Model does not have a properly formatted supply DataFrame")
                    
                # Check if we have auction data in the supply DataFrame
                if (scenario, 'auction') not in self.model.supply.columns:
                    raise ValueError(f"No auction data found for scenario '{scenario}' in supply DataFrame")
                    
                # Get the auction component from supply
                auction_supply = self.model.supply[(scenario, 'auction')]
                
                # Use this as base_supplied (since they should match)
                data[(scenario, 'base_supplied')] = auction_supply
                data[(scenario, 'total_supplied')] = auction_supply  # Total = base since CCR isn't used
                
                # Set CCR components to zero
                data[(scenario, 'ccr1_supplied')] = pd.Series(0.0, index=auction_supply.index)
                data[(scenario, 'ccr2_supplied')] = pd.Series(0.0, index=auction_supply.index)
                
                # Calculate revenue from prices and auction volume
                if not hasattr(self.model, 'prices') or not isinstance(self.model.prices, pd.DataFrame):
                    raise ValueError(f"Model does not have a properly formatted prices DataFrame")
                    
                if (scenario, 'carbon_price') not in self.model.prices.columns:
                    raise ValueError(f"No carbon price data found for scenario '{scenario}'")
                    
                prices = self.model.prices[(scenario, 'carbon_price')]
                
                # Calculate revenue as price * volume (converting kilotonnes to tonnes)
                revenue = pd.Series(index=auction_supply.index)
                for year in auction_supply.index:
                    if year in prices.index:
                        # Convert kilotonnes to tonnes by multiplying by 1000
                        revenue[year] = prices[year] * auction_supply[year] * 1000
                    else:
                        revenue[year] = 0.0
                        
                data[(scenario, 'revenue')] = revenue
                
                # Get auction data from the model's auction component
                if hasattr(self.model, 'auction_supply') and hasattr(self.model.auction_supply, 'auction_data'):
                    auction_data = self.model.auction_supply.auction_data
                    
                    # Map the auction data to the output columns
                    data[(scenario, 'base_available')] = auction_data['base_volume']
                    data[(scenario, 'ccr1_available')] = auction_data['ccr_volume_1']
                    data[(scenario, 'ccr2_available')] = auction_data['ccr_volume_2']
                    data[(scenario, 'ccr1_price')] = auction_data['ccr_trigger_price_1']
                    data[(scenario, 'ccr2_price')] = auction_data['ccr_trigger_price_2']
                    data[(scenario, 'reserve_price')] = auction_data['auction_reserve_price']
                else:
                    # If auction data not available, use placeholders
                    print("Warning: Auction data not available from model, using placeholders")
                    placeholder_cols = [
                        'base_available', 'ccr1_available', 'ccr2_available', 
                        'ccr1_price', 'ccr2_price', 'reserve_price'
                    ]
                    for col in placeholder_cols:
                        data[(scenario, col)] = pd.Series(0.0, index=self.years)
                
        # Create DataFrame with multi-index columns
        df = pd.DataFrame(data)
        df.index.name = 'year'
        
        # Create proper multi-index columns or fix existing ones
        if not isinstance(df.columns, pd.MultiIndex):
            columns = pd.MultiIndex.from_tuples(df.columns, names=['scenario', 'variable'])
            df.columns = columns
        else:
            # If already a MultiIndex but names are wrong, fix them
            if df.columns.names != ['scenario', 'variable']:
                df.columns = pd.MultiIndex.from_tuples(
                    list(df.columns),
                    names=['scenario', 'variable']
                )
        
        # Ensure no NaN values
        df = df.fillna(0.0) 
        
        return df

    def _create_industrial_dataframe(self):
        """Create DataFrame for industrial allocation."""
        data = {}
        for scenario in self.scenarios:
            if scenario in self.results:
                result = self.results[scenario]
                
                # Extract industrial allocation
                industrial_results = None
                
                if 'model' in result and 'industrial_component' in result['model']:
                    model_result = result['model']
                    if 'industrial_results' in model_result:
                        industrial_results = model_result['industrial_results']
                    elif 'industrial_component' in model_result and 'results' in model_result['industrial_component']:
                        industrial_results = model_result['industrial_component']['results']
                elif 'model' in result and 'industrial_results' in result['model']:
                    industrial_results = result['model']['industrial_results']
                
                if industrial_results is not None:
                    if 'adjusted_allocation' in industrial_results:
                        data[(scenario, 'allocation')] = industrial_results['adjusted_allocation']
                    elif 'baseline_allocation' in industrial_results:
                        data[(scenario, 'allocation')] = industrial_results['baseline_allocation']
        
        # Create DataFrame with multi-index columns
        if data:
            # Convert data to DataFrame
            df = pd.DataFrame(data)
            df.index.name = 'year'
            
            # Create proper multi-index columns or fix existing ones
            if not isinstance(df.columns, pd.MultiIndex):
                columns = pd.MultiIndex.from_tuples(df.columns, 
                                                names=['scenario', 'variable'])
                df.columns = columns
            else:
                # If already a MultiIndex but names are wrong, fix them
                if df.columns.names != ['scenario', 'variable']:
                    df.columns = pd.MultiIndex.from_tuples(
                        list(df.columns),
                        names=['scenario', 'variable']
                    )
        else:
            # Create empty DataFrame with proper structure
            columns = pd.MultiIndex.from_product([self.scenarios, ['allocation']], 
                                            names=['scenario', 'variable'])
            df = pd.DataFrame(columns=columns, index=self.years)
            df.index.name = 'year'
        
        return df
    
    def _create_forestry_dataframe(self):
        """Create DataFrame for forestry components."""
        data = {}
        for scenario in self.scenarios:
            if scenario in self.results:
                result = self.results[scenario]
                
                # Extract forestry results
                forestry_results = None
                
                if 'model' in result and 'forestry_component' in result['model']:
                    model_result = result['model']
                    if 'forestry_results' in model_result:
                        forestry_results = model_result['forestry_results']
                    elif 'forestry_component' in model_result and 'results' in model_result['forestry_component']:
                        forestry_results = model_result['forestry_component']['results']
                elif 'model' in result and 'forestry_results' in result['model']:
                    forestry_results = result['model']['forestry_results']
                
                if forestry_results is not None:
                    if 'total_supply' in forestry_results:
                        data[(scenario, 'removals')] = forestry_results['total_supply']
                    elif 'static_supply' in forestry_results:
                        data[(scenario, 'removals')] = forestry_results['static_supply']
                    elif 'manley_supply' in forestry_results:
                        data[(scenario, 'removals')] = forestry_results['manley_supply']
        
        # Create DataFrame with multi-index columns
        if data:
            # Convert data to DataFrame
            df = pd.DataFrame(data)
            df.index.name = 'year'
            
            # Create proper multi-index columns or fix existing ones
            if not isinstance(df.columns, pd.MultiIndex):
                columns = pd.MultiIndex.from_tuples(df.columns, 
                                                names=['scenario', 'variable'])
                df.columns = columns
            else:
                # If already a MultiIndex but names are wrong, fix them
                if df.columns.names != ['scenario', 'variable']:
                    df.columns = pd.MultiIndex.from_tuples(
                        list(df.columns),
                        names=['scenario', 'variable']
                    )
        else:
            # Create empty DataFrame with proper structure
            columns = pd.MultiIndex.from_product([self.scenarios, ['removals']], 
                                            names=['scenario', 'variable'])
            df = pd.DataFrame(columns=columns, index=self.years)
            df.index.name = 'year'
        
        return df

    def _create_stockpile_dataframe(self):
        """Create DataFrame for stockpile components."""
        data = {}
        for scenario in self.scenarios:
            if scenario in self.results:
                result = self.results[scenario]
                
                # Extract stockpile results
                stockpile_results = None
                
                # Handle both top-level and nested structures
                if 'model' in result and 'stockpile_component' in result['model']:
                    stockpile_comp = result['model']['stockpile_component']
                    
                    # Handle different result structures
                    if isinstance(stockpile_comp, dict):
                        if 'results' in stockpile_comp:
                            stockpile_results = stockpile_comp['results']
                        if 'stockpile_balance' in stockpile_comp:
                            data[(scenario, 'balance')] = stockpile_comp['stockpile_balance']
                        if 'surplus_balance' in stockpile_comp:
                            data[(scenario, 'surplus_balance')] = stockpile_comp['surplus_balance']
                    elif hasattr(stockpile_comp, 'results'):
                        stockpile_results = stockpile_comp.results
                elif 'model' in result and 'stockpile_results' in result['model']:
                    stockpile_results = result['model']['stockpile_results']
                
                if stockpile_results is not None:
                    # Extract all available variables from stockpile_results
                    if isinstance(stockpile_results, pd.DataFrame):
                        if 'stockpile_balance' in stockpile_results.columns:
                            data[(scenario, 'balance')] = stockpile_results['stockpile_balance']
                        if 'surplus_balance' in stockpile_results.columns:
                            data[(scenario, 'surplus_balance')] = stockpile_results['surplus_balance']
                        if 'non_surplus_balance' in stockpile_results.columns:
                            data[(scenario, 'non_surplus_balance')] = stockpile_results['non_surplus_balance']
                        if 'available_units' in stockpile_results.columns:
                            data[(scenario, 'units_used')] = stockpile_results['available_units']
                        if 'surplus_used' in stockpile_results.columns:
                            data[(scenario, 'surplus_used')] = stockpile_results['surplus_used']
                        if 'non_surplus_used' in stockpile_results.columns:
                            data[(scenario, 'non_surplus_used')] = stockpile_results['non_surplus_used']
                        if 'forestry_held_addition' in stockpile_results.columns:
                            data[(scenario, 'forestry_held')] = stockpile_results['forestry_held_addition']
                        if 'forestry_surrender_addition' in stockpile_results.columns:
                            data[(scenario, 'forestry_surrender')] = stockpile_results['forestry_surrender_addition']
                        if 'stockpile_without_forestry' in stockpile_results.columns:
                            data[(scenario, 'without_forestry')] = stockpile_results['stockpile_without_forestry']
                        
                        # Add existing payback tracking
                        if 'borrowed_units' in stockpile_results.columns:
                            data[(scenario, 'borrowed_units')] = stockpile_results['borrowed_units']
                        if 'payback_units' in stockpile_results.columns:
                            data[(scenario, 'payback_units')] = stockpile_results['payback_units']
                        if 'net_change' in stockpile_results.columns:
                            data[(scenario, 'net_borrowing')] = stockpile_results['net_change']
                        
                        # Add new payback metric for cumulative tracking if not already present
                        if 'borrowed_units' in stockpile_results.columns and 'payback_units' in stockpile_results.columns:
                            if 'cumulative_net_borrowing' not in stockpile_results.columns:
                                data[(scenario, 'cumulative_net_borrowing')] = (
                                    stockpile_results['payback_units'] - stockpile_results['borrowed_units']
                                ).cumsum()
                            else:
                                data[(scenario, 'cumulative_net_borrowing')] = stockpile_results['cumulative_net_borrowing']
                        
                        # Add explicit mapping for cumulative_forestry_additions
                        if 'cumulative_forestry_additions' in stockpile_results.columns:
                            data[(scenario, 'cumulative_forestry_additions')] = stockpile_results['cumulative_forestry_additions']
                        
                        # Calculate derived values
                        if 'forestry_held_addition' in stockpile_results.columns and 'forestry_surrender_addition' in stockpile_results.columns:
                            data[(scenario, 'forestry_contribution')] = (
                                stockpile_results['forestry_held_addition'] + 
                                stockpile_results['forestry_surrender_addition']
                            )
                        
                        # Calculate ratio_to_demand if we have demand and stockpile balance
                        if 'stockpile_balance' in stockpile_results.columns:
                            demand = None
                            if 'demand' in result:
                                demand = result['demand']
                            elif 'model' in result and 'demand' in result['model']:
                                demand = result['model']['demand']
                            
                            if demand is not None:
                                # Ensure demand is a Series
                                if isinstance(demand, pd.Series):
                                    demand_series = demand
                                else:
                                    demand_series = pd.Series(demand, index=self.years)
                                
                                # Calculate ratio
                                stockpile_balance = stockpile_results['stockpile_balance']
                                data[(scenario, 'ratio_to_demand')] = stockpile_balance / demand_series
        
        # Create DataFrame with multi-index columns
        if data:
            # Convert data to DataFrame
            df = pd.DataFrame(data)
            df.index.name = 'year'
            
            # Create proper multi-index columns or fix existing ones
            if not isinstance(df.columns, pd.MultiIndex):
                columns = pd.MultiIndex.from_tuples(df.columns, 
                                                names=['scenario', 'variable'])
                df.columns = columns
            else:
                # If already a MultiIndex but names are wrong, fix them
                if df.columns.names != ['scenario', 'variable']:
                    df.columns = pd.MultiIndex.from_tuples(
                        list(df.columns),
                        names=['scenario', 'variable']
                    )
        else:
            # Create empty DataFrame with proper structure - include all variables
            stockpile_vars = [
                'balance', 'surplus_balance', 'non_surplus_balance', 'ratio_to_demand',
                'units_used', 'surplus_used', 'non_surplus_used', 'forestry_held',
                'forestry_surrender', 'forestry_contribution', 'without_forestry',
                'borrowed_units', 'payback_units', 'net_borrowing', 'cumulative_net_borrowing',
                'cumulative_forestry_additions'
            ]
            columns = pd.MultiIndex.from_product([self.scenarios, stockpile_vars], 
                                            names=['scenario', 'variable'])
            df = pd.DataFrame(columns=columns, index=self.years)
            df.index.name = 'year'
        
        return df

    def _create_demand_dataframe(self):
        """Create DataFrame for demand components."""
        data = {}
        for scenario in self.scenarios:
            if scenario in self.results:
                result = self.results[scenario]
                
                # Extract emissions data - first try emissions_results, then emissions_component
                emissions_data = None
                if 'model' in result:
                    if 'emissions_results' in result['model']:
                        emissions_data = result['model']['emissions_results']
                    elif 'emissions_component' in result['model']:
                        emissions_data = result['model']['emissions_component']
                
                if emissions_data is not None:
                    # Handle both DataFrame and EmissionsDemand object cases
                    if isinstance(emissions_data, pd.DataFrame):
                        # Create proper multi-index columns
                        if 'baseline_emissions' in emissions_data:
                            data[(scenario, 'baseline')] = emissions_data['baseline_emissions']
                        if 'total_demand' in emissions_data:
                            data[(scenario, 'emissions')] = emissions_data['total_demand']
                        elif 'price_adjusted_emissions' in emissions_data:
                            data[(scenario, 'emissions')] = emissions_data['price_adjusted_emissions']
                    elif hasattr(emissions_data, 'results'):
                        # If it's an EmissionsDemand object, use its results DataFrame
                        results_df = emissions_data.results
                        if 'baseline_emissions' in results_df:
                            data[(scenario, 'baseline')] = results_df['baseline_emissions']
                        if 'total_demand' in results_df:
                            data[(scenario, 'emissions')] = results_df['total_demand']
                        elif 'price_adjusted_emissions' in results_df:
                            data[(scenario, 'emissions')] = results_df['price_adjusted_emissions']
                    
                    # Calculate gross mitigation if we have both baseline and emissions
                    if (scenario, 'baseline') in data and (scenario, 'emissions') in data:
                        data[(scenario, 'gross_mitigation')] = data[(scenario, 'baseline')] - data[(scenario, 'emissions')]
                
                # Add payback units from stockpile component if available
                if 'model' in result and 'stockpile_component' in result['model']:
                    stockpile_comp = result['model']['stockpile_component']
                    
                    if isinstance(stockpile_comp, dict) and 'results' in stockpile_comp:
                        stockpile_results = stockpile_comp['results']
                        
                        if isinstance(stockpile_results, pd.DataFrame) and 'payback_units' in stockpile_results.columns:
                            # Include payback units as part of demand
                            data[(scenario, 'payback_units')] = stockpile_results['payback_units']
                            
                            # Also create an expanded demand metric that includes both emissions and paybacks
                            if (scenario, 'emissions') in data:
                                data[(scenario, 'total_demand_with_paybacks')] = data[(scenario, 'emissions')] + stockpile_results['payback_units']

        # Create DataFrame with multi-index columns
        if data:
            df = pd.DataFrame(data)
            df.index.name = 'year'
            
            # Create proper multi-index columns or fix existing ones
            if not isinstance(df.columns, pd.MultiIndex):
                df.columns = pd.MultiIndex.from_tuples(df.columns, names=['scenario', 'variable'])
            else:
                # If already a MultiIndex but names are wrong, fix them
                if df.columns.names != ['scenario', 'variable']:
                    df.columns = pd.MultiIndex.from_tuples(
                        list(df.columns),
                        names=['scenario', 'variable']
                    )
            return df
        else:
            # Create empty DataFrame with proper structure
            columns = pd.MultiIndex.from_product(
                [self.scenarios, ['baseline', 'emissions', 'gross_mitigation', 'payback_units', 'total_demand_with_paybacks']], 
                names=['scenario', 'variable']
            )
            return pd.DataFrame(columns=columns, index=self.years)

    def _create_inputs_dataframe(self):
        """Create DataFrame for model inputs."""
        data = {}
        for scenario in self.scenarios:
            # Get the component configuration for this scenario
            scenario_index = self.scenarios.index(scenario)
            component_config = self.model.component_configs[scenario_index]
            
            # Extract key input parameters
            discount_rate = getattr(component_config, 'discount_rate', None)
            data[(scenario, 'discount_rate')] = pd.Series(discount_rate, index=self.years)
            
            data[(scenario, 'scenario_name')] = pd.Series(scenario, index=self.years)
            data[(scenario, 'years')] = pd.Series(self.years, index=self.years)
            
            initial_stockpile = getattr(component_config, 'initial_stockpile', None)
            if initial_stockpile is not None:
                data[(scenario, 'stockpile_start')] = pd.Series(initial_stockpile, index=self.years)
            
            initial_surplus = getattr(component_config, 'initial_surplus', None)
            if initial_surplus is not None:
                data[(scenario, 'surplus_start')] = pd.Series(initial_surplus, index=self.years)
            
            payback_period = getattr(component_config, 'payback_period', None)
            if payback_period is not None:
                data[(scenario, 'payback_period')] = pd.Series(payback_period, index=self.years)
            
            liquidity = getattr(component_config, 'liquidity', None)
            if liquidity is not None:
                data[(scenario, 'liquidity_limit')] = pd.Series(liquidity, index=self.years)
            
            # Get price response configuration
            if hasattr(self.model, 'price_response'):
                price_response = self.model.price_response
                
            # Get demand model number
            demand_model_number = getattr(component_config, 'demand_model_number', None)
            if demand_model_number is not None:
                data[(scenario, 'demand_model')] = pd.Series(demand_model_number, index=self.years)
        
        # Create DataFrame with multi-index columns
        if data:
            # Convert data to DataFrame
            df = pd.DataFrame(data)
            df.index.name = 'year'
            
            # Create proper multi-index columns or fix existing ones
            if not isinstance(df.columns, pd.MultiIndex):
                columns = pd.MultiIndex.from_tuples(df.columns, 
                                                names=['scenario', 'variable'])
                df.columns = columns
            else:
                # If already a MultiIndex but names are wrong, fix them
                if df.columns.names != ['scenario', 'variable']:
                    df.columns = pd.MultiIndex.from_tuples(
                        list(df.columns),
                        names=['scenario', 'variable']
                    )
        else:
            # Create empty DataFrame with proper structure
            input_vars = ['discount_rate', 'scenario_name', 'years', 'stockpile_start', 
                        'surplus_start', 'payback_period', 'liquidity_limit', 
                        'price_response_forward_years', 'demand_model']
            columns = pd.MultiIndex.from_product([self.scenarios, input_vars], 
                                            names=['scenario', 'variable'])
            df = pd.DataFrame(columns=columns, index=self.years)
            df.index.name = 'year'
        
        return df
    
    def list_variables(self):
        """
        List all available variables by category.
        
        This helper method provides a summary of all available variables
        organised by category, with examples of how to access them.
        """
        print("Available variables by category:")
        
        print("\nPrices:")
        print("  self.prices - Carbon prices by scenario")
        print("  Access real prices: model.prices.xs('carbon_price', level='variable', axis=1)")
        print("  Access nominal prices: model.prices.xs('carbon_price_nominal', level='variable', axis=1)")
        print("  Example for specific scenario: model.prices[('central', 'carbon_price')]")
        print("  Example for nominal prices: model.prices[('central', 'carbon_price_nominal')]")
        
        print("\nCore model outputs:")
        for var in sorted(self._get_variables_by_category('core')):
            print(f"  self.core.xs('{var}', level='variable', axis=1) - {self._get_variable_description(var)}")
        
        print("\nSupply components:")
        for var in sorted(self._get_variables_by_category('supply')):
            print(f"  self.supply.xs('{var}', level='variable', axis=1) - {self._get_variable_description(var)}")
        
        print("\nAuction components:")
        for var in sorted(self._get_variables_by_category('auctions')):
            print(f"  self.auctions.xs('{var}', level='variable', axis=1) - {self._get_variable_description(var)}")
        
        print("\nIndustrial allocation:")
        for var in sorted(self._get_variables_by_category('industrial')):
            print(f"  self.industrial.xs('{var}', level='variable', axis=1) - {self._get_variable_description(var)}")
        
        print("\nForestry components:")
        for var in sorted(self._get_variables_by_category('forestry')):
            print(f"  self.forestry.xs('{var}', level='variable', axis=1) - {self._get_variable_description(var)}")
        
        print("\nStockpile components:")
        for var in sorted(self._get_variables_by_category('stockpile')):
            print(f"  self.stockpile.xs('{var}', level='variable', axis=1) - {self._get_variable_description(var)}")
        
        print("\nDemand components:")
        for var in sorted(self._get_variables_by_category('demand')):
            print(f"  self.demand.xs('{var}', level='variable', axis=1) - {self._get_variable_description(var)}")
        
        print("\nInput parameters:")
        for var in sorted(self._get_variables_by_category('inputs')):
            print(f"  self.inputs.xs('{var}', level='variable', axis=1) - {self._get_variable_description(var)}")
        
    
    def _get_variables_by_category(self, category):
        """Helper method to get variables for a specific category."""
        category_map = {
            'core': ['supply_demand_balance', 'price_change_rate', 'avg_price_change_rate'],
            'supply': ['auction', 'industrial', 'forestry', 'stockpile', 'total'],
            'auctions': ['base_available', 'base_supplied', 'ccr1_available', 'ccr1_supplied',
                        'ccr2_available', 'ccr2_supplied', 'total_available', 'total_supplied',
                        'ccr1_price', 'ccr2_price', 'reserve_price', 'revenue'],
            'industrial': ['allocation'],
            'forestry': ['removals'],
            'stockpile': ['balance', 'surplus_balance', 'non_surplus_balance', 'ratio_to_demand',
                        'units_used', 'surplus_used', 'non_surplus_used', 'forestry_held',
                        'forestry_surrender', 'forestry_contribution', 'without_forestry',
                        'borrowed_units', 'payback_units', 'net_borrowing', 'cumulative_net_borrowing',
                        'cumulative_forestry_additions'],
            'demand': ['baseline_emissions', 'emissions', 'gross_mitigation', 'net_mitigation', 
                      'payback_units', 'total_demand_with_paybacks'],
            'inputs': ['discount_rate', 'scenario_name', 'years', 'stockpile_start', 
                      'surplus_start', 'payback_period', 'liquidity_limit', 
                      'price_response_forward_years', 'demand_model']
        }
        
        return category_map.get(category, [])
    
    def variable_info(self, variable_name=None):
        """
        Get information about a specific variable or list all variables.
        
        Args:
            variable_name: Optional name of the variable to get information about.
                         If None, lists all variables with their descriptions.
        
        Returns:
            If variable_name is provided, returns a dictionary with information about the variable.
            Otherwise, prints a list of all variables with their descriptions.
        """
        if variable_name is None:
            # List all variables with their descriptions
            print("Available variables:")
            print("{:<30} {:<50} {:<15} {:<10}".format("Variable", "Description", "Units", "Type"))
            print("-" * 105)
            
            for var_name, info in sorted(self._variable_schema.items()):
                print("{:<30} {:<50} {:<15} {:<10}".format(
                    var_name, 
                    info['description'], 
                    info['units'], 
                    info['type']
                ))
            
            return None
        else:
            # Return information about the specified variable
            if variable_name in self._variable_schema:
                return self._variable_schema[variable_name]
            else:
                print(f"Variable '{variable_name}' not found in schema.")
                print("Use variable_info() without arguments to see all available variables.")
                return None

    def _get_variable_description(self, variable_name):
        """Helper method to get variable description."""
        if variable_name in self._variable_schema:
            return f"{self._variable_schema[variable_name]['description']} ({self._variable_schema[variable_name]['units']})"
        return f"Variable: {variable_name}"
    
    def _create_variable_schema(self):
        """Create variable schema with descriptions, units and types."""
        # Start with the existing schema
        schema = {
            # Prices category
            'carbon_price': {'description': 'Carbon price in real 2023 NZD', 'units': '$/tonne CO₂-e', 'type': 'output'},
            'unmodified_carbon_price': {'description': 'Carbon price before stockpile revision in real 2023 NZD', 'units': '$/tonne CO₂-e', 'type': 'output'},
            
            # Add nominal price entries
            'carbon_price_nominal': {'description': 'Carbon price in nominal NZD', 'units': '$/tonne CO₂-e', 'type': 'output'},
            'unmodified_carbon_price_nominal': {'description': 'Carbon price before stockpile revision in nominal NZD', 'units': '$/tonne CO₂-e', 'type': 'output'},

            # Core category
            'supply_demand_balance': {'description': 'Supply minus demand by year', 'units': 'kt CO₂-e', 'type': 'output'},
            'price_change_rate': {'description': 'Annual price growth rate', 'units': '%', 'type': 'output'},
            'avg_price_change_rate': {'description': 'Average annual price growth', 'units': '%', 'type': 'output'},
            
            # Inputs category
            'discount_rate': {'description': 'Used for stockpile calculations', 'units': '%', 'type': 'input'},
            'scenario_name': {'description': 'Identifier for each scenario', 'units': '-', 'type': 'input'},
            'years': {'description': 'Model years within start/end', 'units': 'year', 'type': 'input'},
            'stockpile_start': {'description': 'Initial stockpile volume', 'units': 'kt CO2-e', 'type': 'input'},
            'surplus_start': {'description': 'Initial surplus volume', 'units': 'kt CO2-e', 'type': 'input'},
            'payback_period': {'description': 'Years to pay back borrowed units', 'units': 'years', 'type': 'input'},
            'price_response_forward_years': {'description': 'Price response look ahead', 'units': 'years', 'type': 'input'},
            'liquidity_limit': {'description': 'Annual non-surplus limit', 'units': '%', 'type': 'input'},
            'demand_model': {'description': 'Price response model number', 'units': 'integer', 'type': 'input'},
            
            # Supply category
            'total': {'description': 'Total unit supply', 'units': 'kt CO₂-e', 'type': 'output'},
            'auction': {'description': 'Units supplied through auction', 'units': 'kt CO₂-e', 'type': 'output'},
            'industrial': {'description': 'Units allocated to industrial activities', 'units': 'kt CO₂-e', 'type': 'output'},
            'forestry': {'description': 'Units from forestry removals', 'units': 'kt CO₂-e', 'type': 'output'},
            'stockpile': {'description': 'Units from stockpile', 'units': 'kt CO₂-e', 'type': 'output'},
            
            # Auctions category
            'base_available': {'description': 'Base auction volume available', 'units': 'kt CO2-e', 'type': 'input'},
            'base_supplied': {'description': 'Base auction volumes actually supplied', 'units': 'kt CO2-e', 'type': 'output'},
            'ccr1_available': {'description': 'CCR1 auction volumes available', 'units': 'kt CO2-e', 'type': 'input'},
            'ccr1_supplied': {'description': 'CCR1 auction volumes supplied', 'units': 'kt CO2-e', 'type': 'output'},
            'ccr2_available': {'description': 'CCR2 auction volumes available', 'units': 'kt CO2-e', 'type': 'input'},
            'ccr2_supplied': {'description': 'CCR2 auction volumes supplied', 'units': 'kt CO2-e', 'type': 'output'},
            'total_supplied': {'description': 'Total auction supplied across tiers', 'units': 'kt CO2-e', 'type': 'output'},
            'total_available': {'description': 'Total auction available across tiers', 'units': 'kt CO2-e', 'type': 'output'},
            'ccr1_price': {'description': 'CCR1 price trigger', 'units': '$/tonne CO2-e', 'type': 'input'},
            'ccr2_price': {'description': 'CCR2 price trigger', 'units': '$/tonne CO2-e', 'type': 'input'},
            'reserve_price': {'description': 'Minimum auction price', 'units': '$/tonne CO2-e', 'type': 'input'},
            'revenue': {'description': 'Annual auction revenue', 'units': '$', 'type': 'output'},
            
            # Industrial category
            'allocation': {'description': 'Free allocation to industry', 'units': 'kt CO2-e', 'type': 'output'},
            
            # Forestry category
            'removals': {'description': 'Units supplied from forestry', 'units': 'kt CO2-e', 'type': 'output'},
            
            # Stockpile category
            'balance': {'description': 'Total stockpile balance', 'units': 'kt CO2-e', 'type': 'output'},
            'surplus_balance': {'description': 'Surplus stockpile balance', 'units': 'kt CO2-e', 'type': 'output'},
            'non_surplus_balance': {'description': 'Non-surplus stockpile balance', 'units': 'kt CO2-e', 'type': 'output'},
            'ratio_to_demand': {'description': 'Stockpile divided by emissions', 'units': 'ratio', 'type': 'output'},
            'units_used': {'description': 'Total units used from stockpile', 'units': 'kt CO2-e', 'type': 'output'},
            'surplus_used': {'description': 'Surplus units used from stockpile', 'units': 'kt CO2-e', 'type': 'output'},
            'non_surplus_used': {'description': 'Non-surplus units used from stockpile', 'units': 'kt CO2-e', 'type': 'output'},
            'forestry_held': {'description': 'Forestry held contributions', 'units': 'kt CO2-e', 'type': 'output'},
            'forestry_surrender': {'description': 'Forestry surrender contributions', 'units': 'kt CO2-e', 'type': 'output'},
            'forestry_contribution': {'description': 'Net forestry contribution', 'units': 'kt CO2-e', 'type': 'output'},
            'without_forestry': {'description': 'Stockpile balance excluding forestry', 'units': 'kt CO2-e', 'type': 'output'},
            'borrowed_units': {'description': 'Annual borrowing from stockpile', 'units': 'kt CO2-e', 'type': 'output'},
            'payback_units': {'description': 'Annual repayment to stockpile', 'units': 'kt CO2-e', 'type': 'output'},
            'net_borrowing': {'description': 'Net borrowing (paybacks minus borrowing)', 'units': 'kt CO2-e', 'type': 'output'},
            'cumulative_net_borrowing': {'description': 'Cumulative net borrowing', 'units': 'kt CO2-e', 'type': 'output'},    
            'cumulative_forestry_additions': {'description': 'Cumulative forestry held and surrender additions', 'units': 'kt CO2-e', 'type': 'output'},

            # Demand category
            'baseline_emissions': {'description': 'Baseline emissions (pre-response)', 'units': 'kt CO2-e', 'type': 'output'},
            'emissions': {'description': 'Emissions after price response', 'units': 'kt CO2-e', 'type': 'output'},
            'gross_mitigation': {'description': 'Difference baseline - emissions', 'units': 'kt CO2-e', 'type': 'output'},
            'net_mitigation': {'description': 'Gross mitigation + forestry', 'units': 'kt CO2-e', 'type': 'output'},
            'payback_units': {'description': 'Annual stockpile repayment obligation', 'units': 'kt CO2-e', 'type': 'output'},
            'total_demand_with_paybacks': {'description': 'Total demand including payback obligations', 'units': 'kt CO2-e', 'type': 'output'},
        }
        
        # Return the updated schema with nominal price variables included
        return schema
    
