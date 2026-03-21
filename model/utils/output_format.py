"""
Output formatter for the NZUpy model.

This module provides functionality for converting raw model results into
structured pandas DataFrames with standardised multi-index columns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any


class OutputFormat:
    """
    Formats raw model results into structured DataFrames.

    This class takes the raw model results (nested dictionaries) and converts
    them into standardised pandas DataFrames with multi-index columns for
    scenarios and variables.

    Result dict structure (from runner.py):
        result['price_change_rate']       — optimal or fixed rate
        result['prices']                  — pd.Series of controlled carbon prices
        result['unmodified_prices']       — pd.Series of pre-control prices
        result['supply']                  — pd.DataFrame of supply components
        result['demand']                  — pd.Series of emissions demand
        result['model']                   — detailed component results:
            ['industrial_results']        — pd.DataFrame with adjusted_allocation
            ['stockpile_results']         — pd.DataFrame with balance columns
            ['stockpile_reference_year']  — int, year before model start
            ['stockpile_initial_values']  — dict with balance/surplus_balance keys
            ['forestry_results']          — dict with total_supply and Manley columns
            ['emissions_results']         — pd.DataFrame with baseline_emissions, total_demand
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

        self.model.supply = self._create_supply_dataframe()
        
        self.model.demand = self._create_demand_dataframe()
        
        self.model.auctions = self._create_auctions_dataframe()
        
        self.model.industrial = self._create_industrial_dataframe()
        
        self.model.forestry = self._create_forestry_dataframe()
        
        self.model.stockpile = self._create_stockpile_dataframe()
    
    def _create_prices_dataframe(self):
        """Create DataFrame for prices with scenarios as columns."""
        data = {}
        
        from model.utils.price_convert import convert_real_to_nominal

        cpi_data = self.data_handler.cpi_data
        
        for scenario in self.scenarios:
            if scenario in self.results:
                result = self.results[scenario]
                
                # Extract real prices (2023 NZD)
                try:
                    real_prices = result['prices']
                except KeyError:
                    raise ValueError(
                        f"Carbon prices not found for scenario '{scenario}'. "
                        f"Results structure: {list(result.keys())}"
                    )
                data[(scenario, 'carbon_price')] = real_prices

                # Calculate nominal prices
                nominal_prices = pd.Series(
                    convert_real_to_nominal(real_prices, cpi_data=cpi_data),
                    index=real_prices.index
                )
                data[(scenario, 'carbon_price_nominal')] = nominal_prices

                # Extract unmodified prices if available
                if 'unmodified_prices' in result:
                    unmodified_real = result['unmodified_prices']
                    data[(scenario, 'unmodified_carbon_price')] = unmodified_real
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

    def _create_supply_dataframe(self):
        """Create DataFrame for supply components."""
        data = {}
        for scenario in self.scenarios:
            if scenario in self.results:
                result = self.results[scenario]
                
                # Extract supply components
                supply_df = result.get('supply')
                if isinstance(supply_df, pd.DataFrame):
                    for col in supply_df.columns:
                        data[(scenario, col)] = supply_df[col]
        
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
                industrial_results = result.get('model', {}).get('industrial_results')
                
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
                forestry_results = result.get('model', {}).get('forestry_results')
                
                if forestry_results is not None:
                    # Primary supply output (always present)
                    if 'total_supply' in forestry_results:
                        data[(scenario, 'removals')] = forestry_results['total_supply']
                    elif 'static_supply' in forestry_results:
                        data[(scenario, 'removals')] = forestry_results['static_supply']
                    elif 'manley_supply' in forestry_results:
                        data[(scenario, 'removals')] = forestry_results['manley_supply']

                    # Manley diagnostic columns (endogenous mode only; absent in exogenous)
                    for col in [
                        'historic_supply', 'manley_supply', 'manley_price',
                        'manley_planting_total', 'manley_planting_permanent',
                        'manley_planting_production', 'manley_planting_natural',
                    ]:
                        series = forestry_results.get(col) if isinstance(forestry_results, dict) \
                            else (forestry_results[col] if col in forestry_results.columns else None)
                        if series is not None and series.any():
                            data[(scenario, col)] = series
        
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
                stockpile_results = result.get('model', {}).get('stockpile_results')
                
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
                            demand = result.get('demand')
                            if demand is not None:
                                if not isinstance(demand, pd.Series):
                                    demand = pd.Series(demand, index=self.years)
                                stockpile_balance = stockpile_results['stockpile_balance']
                                data[(scenario, 'ratio_to_demand')] = stockpile_balance / demand
        
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

            # Prepend reference year row using initial stockpile values
            ref_year = None
            ref_initial = {}
            for scenario in self.scenarios:
                if scenario in self.results:
                    m = self.results[scenario].get('model', {})
                    if ref_year is None and m.get('stockpile_reference_year') is not None:
                        ref_year = m['stockpile_reference_year']
                    if m.get('stockpile_initial_values') is not None:
                        ref_initial[scenario] = m['stockpile_initial_values']

            if ref_year is not None and ref_initial:
                ref_data = {}
                for col in df.columns:
                    scenario, variable = col
                    if scenario in ref_initial and variable in ref_initial[scenario]:
                        ref_data[col] = ref_initial[scenario][variable]
                    else:
                        ref_data[col] = 0.0
                ref_df = pd.DataFrame(ref_data, index=[ref_year])
                ref_df.index.name = 'year'
                ref_df.columns = df.columns  # preserve ['scenario', 'variable'] names
                df = pd.concat([ref_df, df])
        else:
            # Create empty DataFrame with proper structure - include all variables
            stockpile_vars = [
                'balance', 'surplus_balance', 'non_surplus_balance', 'ratio_to_demand',
                'units_used', 'surplus_used', 'non_surplus_used', 'forestry_held',
                'forestry_surrender', 'forestry_contribution',
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
                
                # Extract emissions results
                emissions_data = result.get('model', {}).get('emissions_results')

                if isinstance(emissions_data, pd.DataFrame):
                    if 'baseline_emissions' in emissions_data:
                        data[(scenario, 'baseline')] = emissions_data['baseline_emissions']
                    if 'total_demand' in emissions_data:
                        data[(scenario, 'emissions')] = emissions_data['total_demand']
                    elif 'price_adjusted_emissions' in emissions_data:
                        data[(scenario, 'emissions')] = emissions_data['price_adjusted_emissions']

                    # Calculate gross mitigation if we have both baseline and emissions
                    if (scenario, 'baseline') in data and (scenario, 'emissions') in data:
                        data[(scenario, 'gross_mitigation')] = (
                            data[(scenario, 'baseline')] - data[(scenario, 'emissions')]
                        )

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
        

    def _get_variables_by_category(self, category):
        """Helper method to get variables for a specific category."""
        category_map = {
            'supply': ['auction', 'industrial', 'forestry', 'stockpile', 'total'],
            'auctions': ['base_available', 'base_supplied', 'ccr1_available', 'ccr1_supplied',
                        'ccr2_available', 'ccr2_supplied', 'total_available', 'total_supplied',
                        'ccr1_price', 'ccr2_price', 'reserve_price', 'revenue'],
            'industrial': ['allocation'],
            'forestry': ['removals'],
            'stockpile': ['balance', 'surplus_balance', 'non_surplus_balance', 'ratio_to_demand',
                        'units_used', 'surplus_used', 'non_surplus_used', 'forestry_held',
                        'forestry_surrender', 'forestry_contribution',
                        'borrowed_units', 'payback_units', 'net_borrowing', 'cumulative_net_borrowing',
                        'cumulative_forestry_additions'],
            'demand': ['baseline_emissions', 'emissions', 'gross_mitigation', 'net_mitigation',
                      'payback_units', 'total_demand_with_paybacks'],
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
    
