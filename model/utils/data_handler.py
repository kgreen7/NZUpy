"""
Consolidated data handling utilities for the NZ ETS model.

This module provides functions for loading and validating data from CSV files,
converting to appropriate pandas structures, and providing a clean interface
for model components to access data. It distinguishes between:
- configs: Component input configurations (e.g., 'central', 'high', 'low' input sets)
- scenarios: Different model runs using combinations of configs
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Union, Optional, List, Tuple, Any

from model.utils.historical_data_manager import HistoricalDataManager

class DataHandler:
    """
    Class for handling data loading and processing for the NZ ETS model.
    
    This class handles loading data from CSV files, validating the data,
    and providing access to the data for model components. It supports:
    - Config-based data loading: Loading different input configurations for components
      (e.g., 'central', 'high', 'low' input sets)
    - Standard data loading: Loading default configurations when not using config-based loading
    - Historical data loading: Access to historical data through the HistoricalDataManager
    """
    
    def __init__(
        self, 
        data_dir: Union[str, Path],
        use_config_loading: bool = False
    ):
        """Initialise the DataHandler."""
        # Set up data directory paths
        self.data_dir = Path(data_dir)
        self.use_config_loading = use_config_loading
        
        # Set up directory paths
        self.parameters_dir = self.data_dir / "inputs" / "parameters"
        self.supply_dir = self.data_dir / "inputs" / "supply"
        self.demand_dir = self.data_dir / "inputs" / "demand"
        self.economic_dir = self.data_dir / "inputs" / "economic"  # Add economic directory
        self.stockpile_dir = self.data_dir / "inputs" / "stockpile"  # Add stockpile directory
        
        # Initialise data attributes as None
        self.emissions_baselines_data = None
        self.auctions_data = None
        self.industrial_allocation_data = None
        self.removals_data = None
        self.demand_models_data = None
        self.model_parameters_data = None
        
        # Create historical data manager
        self.historical_manager = HistoricalDataManager(data_dir)
        
        # Always load scenario datasets first
        self._load_config_datasets()
        
        # Then load standard data if not in scenario mode
        if not use_config_loading:
            self._load_all_data()
    
    #
    # Standard data loading methods
    #
    def _load_all_data(self):
        """Load all available data from the data directory."""
        # Load parameters
        self._load_model_parameters()
        self._load_optimisation_parameters()
        
        # Load supply data
        self._load_auction_data()
        self._load_industrial_allocation_data()
        self._load_stockpile_balance_data()
        self._load_forestry_data()
        self._load_forestry_variables()
        
        # Load demand data
        self._load_emissions_data()
    
    def _load_csv(self, file_path: Path, index_col: Optional[Union[str, int]] = None) -> pd.DataFrame:
        """
        Load a CSV file and return as a pandas DataFrame.
        
        Args:
            file_path: Path to the CSV file.
            index_col: Column to use as the index.
            
        Returns:
            DataFrame containing the CSV data.
            
        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the CSV file is empty or cannot be parsed.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path, index_col=index_col)
            return df
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file is empty: {file_path}")
        except pd.errors.ParserError:
            raise ValueError(f"Error parsing CSV file: {file_path}")
    
    def _load_model_parameters(self):
        """Load model parameters from CSV file."""
        try:
            file_path = self.parameters_dir / "model_parameters.csv"
            df = self._load_csv(file_path)
            
            # Filter for central config and convert to dictionary
            central_params = df[df['Config'].str.lower() == 'central']
            self.model_parameters = dict(zip(central_params['Variable'], central_params['Value']))
            
            # Convert numeric parameters
            for param in ['start_year', 'end_year', 'liquidity_factor']:
                param_in_csv = param
                if param not in self.model_parameters and param == 'liquidity_factor' and 'liquidity' in self.model_parameters:
                    param_in_csv = 'liquidity'
                
                if param_in_csv in self.model_parameters:
                    self.model_parameters[param] = float(self.model_parameters[param_in_csv])
                elif param == 'liquidity_factor':
                    raise ValueError("Required parameter 'liquidity_factor' or 'liquidity' not found in model_parameters.csv")
            
            # Convert integer parameters
            for param in ['start_year', 'end_year']:
                if param in self.model_parameters:
                    self.model_parameters[param] = int(self.model_parameters[param])
                    
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Failed to load model parameters: {e}")
    
    def _load_optimisation_parameters(self):
        """Load optimisation parameters from CSV file."""
        file_path = self.parameters_dir / "optimisation_parameters.csv"
        
        try:
            df = self._load_csv(file_path)
            
            # Required parameters and their types
            required_params = {
                'coarse_step': int,
                'fine_step': int,
                'max_rate': int,
                'max_iterations': int
            }
            
            # Validate and convert parameters
            self.optimisation_parameters = {}
            for param, param_type in required_params.items():
                if param not in df['parameter'].values:
                    raise ValueError(f"Missing required parameter: {param}")
                value = df.loc[df['parameter'] == param, 'value'].iloc[0]
                self.optimisation_parameters[param] = param_type(value)
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Optimisation parameters file not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Error loading optimisation parameters: {e}")
    
    def _load_auction_data(self):
        """Load auction data from CSV file."""
        try:
            # Load from auctions.csv
            file_path = self.supply_dir / "auctions.csv"
            df = self._load_csv(file_path)
            
            # Create DataFrame with required columns
            auction_data = pd.DataFrame(index=df['Year'].unique())
            auction_data.index = auction_data.index.astype(int)
            auction_data.index.name = 'year'
            
            # Get data for each required variable
            volume_data = df[df['Variable'] == 'auction_volume']
            price_data = df[df['Variable'] == 'auction_price']
            ccr1_price = df[df['Variable'] == 'CCR_price_1']
            ccr2_price = df[df['Variable'] == 'CCR_price_2']
            ccr1_volume = df[df['Variable'] == 'CCR_volume_1']
            ccr2_volume = df[df['Variable'] == 'CCR_volume_2']
            
            # Add columns with central config data, ensuring numeric conversion
            auction_data['base_volume'] = pd.to_numeric(
                volume_data[volume_data['Config'] == 'central'].set_index('Year')['Value'].str.replace(',', ''),
                errors='coerce'
            )
            auction_data['auction_reserve_price'] = pd.to_numeric(
                price_data[price_data['Config'] == 'central'].set_index('Year')['Value'],
                errors='coerce'
            )
            auction_data['ccr_trigger_price_1'] = pd.to_numeric(
                ccr1_price[ccr1_price['Config'] == 'central'].set_index('Year')['Value'],
                errors='coerce'
            )
            auction_data['ccr_trigger_price_2'] = pd.to_numeric(
                ccr2_price[ccr2_price['Config'] == 'central'].set_index('Year')['Value'],
                errors='coerce'
            )
            auction_data['ccr_volume_1'] = pd.to_numeric(
                ccr1_volume[ccr1_volume['Config'] == 'central'].set_index('Year')['Value'].str.replace(',', ''),
                errors='coerce'
            )
            auction_data['ccr_volume_2'] = pd.to_numeric(
                ccr2_volume[ccr2_volume['Config'] == 'central'].set_index('Year')['Value'].str.replace(',', ''),
                errors='coerce'
            )
            
            self.auction_data = auction_data
            
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Failed to load auction data: {e}")
    
    def _load_industrial_allocation_data(self):
        """Load industrial allocation data from CSV file."""
        try:
            # Load from industrial_allocation.csv
            file_path = self.supply_dir / "industrial_allocation.csv"
            df = self._load_csv(file_path)
            
            # Create DataFrame with baseline_allocation column
            allocation_data = pd.DataFrame(index=df['Year'].unique())
            allocation_data.index = allocation_data.index.astype(int)
            allocation_data.index.name = 'year'
            
            # Add baseline_allocation column from central config
            central_data = df[df['Config'] == 'central']
            allocation_data['baseline_allocation'] = central_data.set_index('Year')['Value']
            
            # Add activity_adjustment column (default to 1.0)
            allocation_data['activity_adjustment'] = 1.0
            
            self.industrial_allocation_data = allocation_data
            
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Failed to load industrial allocation data: {e}")
    
    def _load_stockpile_parameters(self):
        """Load stockpile parameters from CSV file."""
        # No longer needed - we always get from model_parameters.csv
        pass
    
    def _load_stockpile_balance_data(self):
        """Load stockpile balance data from CSV file."""
        try:
            # Load from stockpile_balance.csv in stockpile directory
            file_path = self.stockpile_dir / "stockpile_balance.csv"
            df = self._load_csv(file_path)
            
            # Create DataFrame with required columns
            stockpile_data = pd.DataFrame(index=df['Year'].unique())
            stockpile_data.index = stockpile_data.index.astype(int)
            stockpile_data.index.name = 'year'
            
            # Get data for each required variable
            stockpile_values = df[df['Variable'] == 'stockpile']
            surplus_values = df[df['Variable'] == 'surplus']
            
            # Add columns with central config data
            stockpile_data['stockpile'] = pd.to_numeric(
                stockpile_values[stockpile_values['Config'] == 'central'].set_index('Year')['Value'],
                errors='coerce'
            )
            stockpile_data['surplus'] = pd.to_numeric(
                surplus_values[surplus_values['Config'] == 'central'].set_index('Year')['Value'],
                errors='coerce'
            )
            
            self.stockpile_balance_data = stockpile_data
            
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Failed to load stockpile balance data: {e}")
    
    def _load_forestry_data(self):
        """Load forestry data from CSV file."""
        try:
            # Load from removals.csv
            file_path = self.supply_dir / "removals.csv"
            df = self._load_csv(file_path)
            
            # Filter for forestry_tradeable data
            forestry_df = df[df['Variable'] == 'forestry_tradeable']
            
            # Create DataFrame with forestry_supply column (rename from tradeable)
            forestry_data = forestry_df.pivot(
                index='Year', 
                columns='Config',
                values='Value'
            )
            
            # Add forestry_supply column for compatibility
            forestry_data['forestry_supply'] = forestry_data['central']
            
            # Convert index to int
            forestry_data.index = forestry_data.index.astype(int)
            forestry_data.index.name = 'year'
            
            self.forestry_data = forestry_data
            
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Failed to load forestry data: {e}")
    
    def _load_forestry_variables(self):
        """Load forestry variables (held and surrender) from CSV file."""
        try:
            # Load from removals.csv
            file_path = self.supply_dir / "removals.csv"
            df = self._load_csv(file_path)
            
            # Filter for held and surrender data
            variables_df = df[df['Variable'].isin(['forestry_held', 'forestry_surrender'])]
            
            # Pivot to get variables as columns
            forestry_vars = variables_df.pivot(
                index='Year',
                columns=['Config', 'Variable'],
                values='Value'
            )
            
            # Convert index to int
            forestry_vars.index = forestry_vars.index.astype(int)
            forestry_vars.index.name = 'year'
            
            self.forestry_variables = forestry_vars
            
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Failed to load forestry variables: {e}")
    
    def _load_emissions_data(self):
        """Load emissions data from CSV file."""
        try:
            # Load from emissions_baselines.csv
            file_path = self.demand_dir / "emissions_baselines.csv"
            df = self._load_csv(file_path)
            
            # Create DataFrame with base_emissions column
            emissions_data = pd.DataFrame(index=df['Year'].unique())
            emissions_data.index = emissions_data.index.astype(int)
            emissions_data.index.name = 'year'
            
            # Add base_emissions column from central config
            central_data = df[df['Config'] == 'central']  # Exact column name match
            emissions_data['base_emissions'] = central_data.set_index('Year')['Value']
            
            # Add other available configs
            ccc_cpr = df[df['Config'] == 'CCC_CPR']  # Exact config name match
            ccc_dp = df[df['Config'] == 'CCC_DP']    # Exact config name match
            
            if not ccc_cpr.empty:
                emissions_data['high_scenario'] = ccc_cpr.set_index('Year')['Value']
            if not ccc_dp.empty:
                emissions_data['low_scenario'] = ccc_dp.set_index('Year')['Value']
            
            self.emissions_data = emissions_data
            
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Failed to load emissions data: {e}")
    
    def _load_price_response_data(self):
        """Load price response data from CSV file."""
        try:
            file_path = self.demand_dir / "price_response.csv"
            df = self._load_csv(file_path, index_col='year')
            
            # Convert index to int if it's years
            if df.index.dtype == 'object':
                try:
                    df.index = df.index.astype(int)
                except ValueError:
                    # Not year-based index, keep as is
                    pass
            
            self.price_response_data = df
            
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not load price response data: {e}")
            # Create empty dataframe
            years = range(self.model_parameters.get('start_year', 2022),
                         self.model_parameters.get('end_year', 2050) + 1)
            self.price_response_data = pd.DataFrame(
                index=years,
                columns=['price_elasticity'],
                data=-0.1  # Default elasticity
            )
    
    #
    # Config-based data loading methods
    #
    def _load_config_datasets(self):
        """Load all config-based datasets."""
        try:
            # Load emissions baselines
            emissions_file = self.demand_dir / "emissions_baselines.csv"
            if emissions_file.exists():
                self.emissions_baselines_data = self._load_csv(emissions_file)
            
            # Load auction data
            auctions_file = self.supply_dir / "auctions.csv"
            if auctions_file.exists():
                self.auctions_data = self._load_csv(auctions_file)
            
            # Load industrial allocation data
            industrial_file = self.supply_dir / "industrial_allocation.csv"
            if industrial_file.exists():
                self.industrial_allocation_data = self._load_csv(industrial_file)
            
            # Load stockpile balance data
            stockpile_file = self.stockpile_dir / "stockpile_balance.csv"
            if stockpile_file.exists():
                self.stockpile_balance_data = self._load_csv(stockpile_file)
            
            # Load forestry data
            removals_file = self.supply_dir / "removals.csv"
            if removals_file.exists():
                self.removals_data = self._load_csv(removals_file)
            
            # Load demand models
            demand_models_file = self.demand_dir / "demand_models.csv"
            if demand_models_file.exists():
                self.demand_models_data = self._load_csv(demand_models_file)
            
            # Load model parameters
            model_params_file = self.parameters_dir / "model_parameters.csv"
            if model_params_file.exists():
                self.model_parameters_data = self._load_csv(model_params_file)
            
        except Exception as e:
            print(f"Warning: Error loading config datasets: {e}")
    
    # Getter methods for both standard and config-based loading

    def get_years(self) -> List[int]:
        """
        Get the years covered by the model.
        
        Returns:
            List of years covered by the model.
        """
        if self.use_config_loading:
            # For config loading, use model parameters
            params = self.get_model_parameters("central")
            start_year = int(params.get('start_year', 2022))
            end_year = int(params.get('end_year', 2050))
        else:
            # For standard loading, use model parameters
            start_year = self.model_parameters.get('start_year', 2022)
            end_year = self.model_parameters.get('end_year', 2050)
        
        return list(range(start_year, end_year + 1))
    
    def get_model_parameters(self, config: Optional[str] = None) -> Dict[str, Any]:
        """
        Get model parameters for a specific config.
        
        Args:
            config: Model parameter config name (defaults to "central")
            
        Returns:
            Dictionary of model parameters
            
        Raises:
            KeyError: If config not found or required columns missing
            ValueError: If model_parameters_data is not loaded
        """
        if not self.use_config_loading:
            return self.model_parameters.copy()

        # For config-based loading
        if self.model_parameters_data is None or self.model_parameters_data.empty:
            raise ValueError("Model parameters data not loaded")
        
        config = config.lower() if config else "central"
        
        # Filter for the config
        params = self.model_parameters_data[
            self.model_parameters_data['Scenario'].str.lower() == config
        ]
        
        if params.empty:
            raise KeyError(f"No parameters found for config '{config}'")
        
        return dict(zip(params['Variable'], params['Value']))
    
    def get_scenario_parameters(self, config: Optional[str] = None) -> Dict[str, Any]:
        """
        Get scenario parameters.
        
        Args:
            scenario: Name of the scenario. If None, return all scenarios.
            
        Returns:
            Dictionary of scenario parameters.
        """
        if not self.use_config_loading:
            # For standard loading
            if config is not None:
                if config in self.scenario_parameters:
                    return self.scenario_parameters[config].copy()
                else:
                    raise ValueError(f"Scenario not found: {config}")
            return self.scenario_parameters.copy()
        else:
            # For scenario-based loading, we don't have direct scenario parameters
            # So we'll return a default set
            if config is not None:
                return {"description": f"{config} scenario"}
            return {
                "central": {"description": "central scenario"},
                "1 s.e lower": {"description": "One standard error below central"},
                "1 s.e upper": {"description": "One standard error above central"},
                "95% Lower": {"description": "95% lower bound"},
                "95% Upper": {"description": "95% upper bound"},
            }
    
    def get_optimisation_parameters(self) -> Dict[str, Any]:
        """
        Get optimisation parameters.
        
        Returns:
            Dictionary of optimisation parameters.
        """
        if not self.use_config_loading:
            # For standard loading
            return self.optimisation_parameters.copy()
        else:
            # For scenario-based loading, use default values
            return {
                'coarse_step': 15,
                'fine_step': 1,
                'max_rate': 200,
            }
    
    def get_auction_data(self, config: Optional[str] = None) -> pd.DataFrame:
        """
        Get auction data for a specific configuration.
        
        Args:
            config: The configuration name to load (e.g., 'central', 'high', 'low').
                   If None, returns the default configuration.
        
        Returns:
            DataFrame containing auction data for the specified configuration.
        """
        if not self.use_config_loading or config is None:
            return self.auction_data
        
        if config not in self.auctions_data['Config'].unique():
            raise ValueError(f"Invalid auction config: {config}")
        
        # Filter for the scenario
        scenario_data = self.auctions_data[
            self.auctions_data['Config'].str.lower() == config
        ]
        
        if scenario_data.empty:
            raise KeyError(f"No auction data found for config '{config}'")
        
        # Map from CSV variable names to model column names
        var_map = {
            'auction_volume': 'base_volume',
            'auction_price': 'auction_reserve_price',
            'CCR_price_1': 'ccr_trigger_price_1',
            'CCR_price_2': 'ccr_trigger_price_2',
            'CCR_volume_1': 'ccr_volume_1',
            'CCR_volume_2': 'ccr_volume_2'
        }
        
        # Process each variable into a dictionary
        data_dict = {}
        for csv_var, model_col in var_map.items():
            var_data = scenario_data[scenario_data['Variable'] == csv_var][['Year', 'Value']]
            if not var_data.empty:
                # Convert Year to int
                var_data['Year'] = var_data['Year'].astype(int)
                
                # Handle string values with commas (e.g., "7,000")
                if csv_var in ['CCR_volume_1', 'CCR_volume_2']:
                    var_data['Value'] = var_data['Value'].astype(str).str.replace(',', '').astype(float)
                else:
                    var_data['Value'] = pd.to_numeric(var_data['Value'], errors='coerce')
                
                data_dict[model_col] = var_data.set_index('Year')['Value']
        
        # Create DataFrame with all variables
        auction_df = pd.DataFrame(data_dict)
        
        # Fill missing values with 0 and ensure all columns exist
        required_cols = ['base_volume', 'ccr_volume_1', 'ccr_volume_2',
                        'ccr_trigger_price_1', 'ccr_trigger_price_2', 'auction_reserve_price']
        for col in required_cols:
            if col not in auction_df.columns:
                auction_df[col] = 0.0
        
        auction_df.index.name = 'year'
        return auction_df
    
    def get_industrial_allocation(self, config: Optional[str] = None) -> pd.DataFrame:
        """
        Get industrial allocation data for a specific configuration.
        
        Args:
            config: The configuration name to load (e.g., 'central', 'high', 'low').
                   If None, returns the default configuration.
        
        Returns:
            DataFrame with industrial allocation data for the specified configuration.
        """
        if not self.use_config_loading or config is None:
            return self.industrial_allocation_data
            
        if config not in self.industrial_allocation_data['Config'].unique():
            raise ValueError(f"Invalid allocation config: {config}")
            
        # Filter for the scenario
        filtered_data = self.industrial_allocation_data[
            self.industrial_allocation_data['Config'].str.lower() == config
        ]
        
        if filtered_data.empty:
            raise KeyError(f"No industrial allocation data found for config '{config}'")
        
        # Create DataFrame with baseline_allocation column
        ia_data = pd.DataFrame({
            'baseline_allocation': filtered_data.set_index('Year')['Value']
        })
        
        # Convert year index to proper format
        ia_data.index = ia_data.index.astype(int)
        ia_data.index.name = 'year'
        
        return ia_data
    
    def get_industrial_allocation_data(self, config: Optional[str] = None) -> pd.DataFrame:
        """
        Get industrial allocation data for a specific configuration.
        
        Args:
            config: The configuration name to load (e.g., 'central', 'high', 'low').
                   If None, returns the default configuration.
        
        Returns:
            DataFrame with industrial allocation data for the specified configuration.
        """
        if not self.use_config_loading or config is None:
            return self.industrial_allocation_data
            
        if config not in self.industrial_allocation_data['Config'].unique():
            raise ValueError(f"Invalid allocation config: {config}")
            
        # Filter for the scenario
        filtered_data = self.industrial_allocation_data[
            self.industrial_allocation_data['Config'].str.lower() == config
        ]
        
        if filtered_data.empty:
            raise KeyError(f"No industrial allocation data found for config '{config}'")
        
        # Create DataFrame with baseline_allocation column
        ia_data = pd.DataFrame({
            'baseline_allocation': filtered_data.set_index('Year')['Value']
        })
        
        # Convert year index to proper format
        ia_data.index = ia_data.index.astype(int)
        ia_data.index.name = 'year'
        
        return ia_data
    
    def get_stockpile_parameters(self, config: Optional[str] = None) -> Dict[str, Any]:
        """
        Get stockpile parameters from stockpile balance data and model parameters.
        
        Args:
            config: Configuration name (defaults to "central")
            
        Returns:
            Dictionary of stockpile parameters
            
        Raises:
            ValueError: If required parameters are missing
        """
        # Get model parameters for non-stockpile values
        params = self.get_model_parameters(config)
        
        # Get stockpile balance data for the config
        try:
            stockpile_data = self.get_stockpile_balance(config)
            
            # Get latest year's values from stockpile balance data
            latest_year = max(stockpile_data.index)
            initial_stockpile = stockpile_data.loc[latest_year, 'stockpile']
            initial_surplus = stockpile_data.loc[latest_year, 'surplus']
        except (ValueError, KeyError) as e:
            # If no data for this config, try central config
            if config and config.lower() != 'central':
                print(f"Warning: No stockpile balance data for config '{config}', using 'central'")
                stockpile_data = self.get_stockpile_balance('central')
                latest_year = max(stockpile_data.index)
                initial_stockpile = stockpile_data.loc[latest_year, 'stockpile']
                initial_surplus = stockpile_data.loc[latest_year, 'surplus']
            else:
                raise ValueError(f"Failed to load stockpile balance data: {e}")
        
        # Map CSV names to return names
        param_mapping = {
            'liquidity_factor': 'liquidity_factor',
            'payback_years': 'payback_period',
            'discount_rate': 'discount_rate',
            'start_year': 'stockpile_usage_start_year'
        }
        
        # Check required parameters exist
        missing = [p for p in param_mapping.keys() if p not in params]
        if missing:
            raise ValueError(f"Missing required parameters in model_parameters.csv: {missing}")
        
        # Create return dictionary with mapped names
        result = {}
        for csv_name, return_name in param_mapping.items():
            value = params[csv_name]
            # Convert to appropriate type
            if csv_name in ['payback_years', 'start_year']:
                result[return_name] = int(value)
            else:
                result[return_name] = float(value)
        
        # Add stockpile values
        result['initial_stockpile'] = float(initial_stockpile)
        result['initial_surplus'] = float(initial_surplus)
        
        # Add reference year
        result['stockpile_reference_year'] = latest_year
        
        return result
    
    def get_stockpile_balance(self, config: Optional[str] = None) -> pd.DataFrame:
        """
        Get stockpile balance data for a specific configuration.
        
        Args:
            config: The configuration name to load (e.g., 'central', 'high', 'low').
                   If None, returns the default configuration.
        
        Returns:
            DataFrame containing stockpile balance data for the specified configuration.
        """
        if not self.use_config_loading or config is None:
            if not hasattr(self, 'stockpile_balance_data'):
                self._load_stockpile_balance_data()
            return self.stockpile_balance_data
        
        if not hasattr(self, 'stockpile_balance_data') or self.stockpile_balance_data.empty:
            raise ValueError("Stockpile balance data not loaded")
        
        # Filter for the config and stockpile/surplus variables
        stockpile_data = self.stockpile_balance_data[
            (self.stockpile_balance_data['Config'].str.lower() == config.lower())
        ]
        
        if stockpile_data.empty:
            raise KeyError(f"No stockpile balance data found for config '{config}'")
        
        # Create DataFrame with stockpile and surplus columns
        stockpile_df = pd.DataFrame(index=stockpile_data['Year'].unique())
        stockpile_df.index = stockpile_df.index.astype(int)
        stockpile_df.index.name = 'year'
        
        # Add stockpile and surplus columns
        stockpile_values = stockpile_data[stockpile_data['Variable'] == 'stockpile']
        surplus_values = stockpile_data[stockpile_data['Variable'] == 'surplus']
        
        stockpile_df['stockpile'] = pd.to_numeric(
            stockpile_values.set_index('Year')['Value'],
            errors='coerce'
        )
        stockpile_df['surplus'] = pd.to_numeric(
            surplus_values.set_index('Year')['Value'],
            errors='coerce'
        )
        
        return stockpile_df
    
    def get_forestry_data(self, config: Optional[str] = None) -> pd.DataFrame:
        """
        Get forestry data for a specific configuration.
        
        Args:
            config: The configuration name to load (e.g., 'central', 'high', 'low').
                   If None, returns the default configuration.
        
        Returns:
            DataFrame containing forestry data for the specified configuration.
        """
        if not self.use_config_loading or config is None:
            return self.forestry_data
            
        if config not in self.forestry_data['Config'].unique():
            raise ValueError(f"Invalid forestry config: {config}")
            
        # Filter for the scenario and forestry_tradeable variable
        forestry_data = self.removals_data[
            (self.removals_data['Config'].str.lower() == config) &
            (self.removals_data['Variable'] == 'forestry_tradeable')
        ]
        
        if forestry_data.empty:
            raise KeyError(f"No forestry_tradeable data found for config '{config}'")
        
        # Create DataFrame with forestry_supply column
        forestry_df = pd.DataFrame({
            'forestry_supply': forestry_data.set_index('Year')['Value']
        })
        
        # Ensure year index is integer type
        forestry_df.index = forestry_df.index.astype(int)
        forestry_df.index.name = 'year'
        
        return forestry_df
    
    def get_forestry_removals(self, config: Optional[str] = None) -> pd.DataFrame:
        """
        Get forestry removals data for a specific configuration.
        
        Args:
            config: The configuration name to load (e.g., 'central', 'high', 'low').
                   If None, returns the default configuration.
        
        Returns:
            DataFrame with forestry removals data containing Year, Variable, and Value columns
        """
        if not self.use_config_loading or config is None:
            # For standard loading, convert to expected format
            forest_data = self.get_forestry_data()
            return pd.DataFrame({
                'Year': forest_data.index.tolist(),
                'Value': forest_data['forestry_supply'].tolist()
            })

        if self.removals_data is None or self.removals_data.empty:
            raise ValueError("Removals data not loaded")
        
        # Filter for scenario and return relevant columns
        forest_data = self.removals_data[
            self.removals_data['Config'].str.lower() == config
        ]
        
        if forest_data.empty:
            raise KeyError(f"No forestry data found for config '{config}'")
        
        return forest_data[['Year', 'Variable', 'Value']]
    
    def get_forestry_variables(self, config: Optional[str] = None) -> pd.DataFrame:
        """
        Get forestry variables data.
        
        Args:
            config: The configuration name to load (e.g., 'central', 'high', 'low').
                   If None, returns the default configuration.
        
        Returns:
            DataFrame indexed by year containing columns for each forestry variable:
            - forestry_tradeable
            - forestry_held
            - forestry_surrender
        """
        if not self.use_config_loading or config is None:
            if hasattr(self, 'forestry_variables') and self.forestry_variables is not None:
                return self.forestry_variables.copy()
            return pd.DataFrame()

        if self.removals_data is None or self.removals_data.empty:
            raise ValueError("Removals data not loaded")
        
        # Filter for scenario
        forest_data = self.removals_data[
            self.removals_data['Config'].str.lower() == config
        ]
        
        if forest_data.empty:
            raise KeyError(f"No forestry data found for config '{config}'")
        
        # Pivot the data to get variables as columns
        result_df = pd.DataFrame(index=forest_data['Year'].unique())
        
        # Add each variable as a column
        for variable in ['forestry_tradeable', 'forestry_held', 'forestry_surrender']:
            var_data = forest_data[forest_data['Variable'] == variable]
            if not var_data.empty:
                result_df[variable] = var_data.set_index('Year')['Value']
        
        # Clean up the index
        result_df.index = result_df.index.astype(int)
        result_df.index.name = 'year'
        
        # Fill any missing values with 0.0
        return result_df.fillna(0.0)
    
    def get_emissions_data(self, config: Optional[str] = None) -> pd.DataFrame:
        """
        Get emissions data for a specific configuration.
        
        Args:
            config: The configuration name to load (e.g., 'central', 'CCC_DP', 'CCC_CPR').
                   If None, returns the default configuration.
        
        Returns:
            DataFrame containing emissions data with columns:
                - Year: Year of the data
                - Config: Configuration name
                - Value: Emissions value for that year and config
        """
        if not self.use_config_loading or config is None:
            # For standard loading, return the raw emissions data
            return self.emissions_baselines_data
            
        if self.emissions_baselines_data is None or self.emissions_baselines_data.empty:
            raise ValueError("Emissions baselines data not loaded")
            
        # Filter for the specific config
        emissions_data = self.emissions_baselines_data[
            self.emissions_baselines_data['Config'].str.lower() == config.lower()
        ]
        
        if emissions_data.empty:
            raise KeyError(f"No emissions data found for config '{config}'")
        
        return emissions_data
    
    def get_emissions_baseline(self, config: str = "central") -> pd.DataFrame:
        """
        Get emissions baseline data for a specific configuration.
        
        Args:
            config: The configuration name to load (e.g., 'central', 'high', 'low').
        
        Returns:
            DataFrame containing emissions baseline data for the specified configuration.
        """
        if not self.use_config_loading:
            # For standard loading, extract from emissions data
            emissions_df = self.get_emissions_data()
            return pd.DataFrame({
                'Year': emissions_df.index.tolist(),
                'Value': emissions_df['base_emissions'].tolist()
            })

        if self.emissions_baselines_data is None or self.emissions_baselines_data.empty:
            raise ValueError("Emissions baselines data not loaded")
        
        # Filter for scenario
        baseline_data = self.emissions_baselines_data[
            self.emissions_baselines_data['Config'].str.lower() == config
        ][['Year', 'Value']]
        
        if baseline_data.empty:
            raise KeyError(f"No emissions data found for config '{config}'")
        
        return baseline_data
    
    def _map_demand_model_name(self, config: str) -> str:
        """
        Map demand model config names to CSV file names.
        
        Args:
            config: Display demand model name
            
        Returns:
            Mapped config name for CSV files
        """
        mapping = {
            "95% Lower": "95pc_lower",
            "1 s.e lower": "stde_lower",
            "central": "central",
            "1 s.e upper": "stde_upper",
            "95% Upper": "95pc_upper"
        }
        return mapping.get(config, config)

    def get_demand_model(self, config: str = "central", model_number: int = 2) -> Dict[str, float]:
        """
        Get demand model parameters.
        
        Args:
            config: Configuration name (e.g., "central", "95pc_lower")
            model_number: Demand model number (1 or 2)
            
        Returns:
            Dictionary containing demand model parameters including:
            - constant, reduction_to_t1, price (from demand_models.csv)
            - model_number (passed parameter)
            - discount_rate, forward_years, price_conversion_factor (from model_parameters.csv)
        """
        try:
            # Load from demand_models.csv
            file_path = self.demand_dir / "demand_models.csv"
            df = self._load_csv(file_path)
            
            # Map config name to CSV format
            mapped_config = self._map_demand_model_name(config)
            
            # Filter for config and model using correct column name
            params = df[
                (df['Config'].str.lower() == mapped_config.lower()) &
                (df['Model'] == model_number)
            ]
            
            if params.empty:
                raise ValueError(f"No parameters found for config '{mapped_config}' and model {model_number}")
            
            # Convert demand model parameters to dictionary
            model_params = dict(zip(params['Variable'], params['Value']))
            
            # Add model number to parameters
            model_params['model_number'] = model_number
            
            # Add parameters from model_parameters.csv
            try:
                # Load model parameters for the same config
                model_params_data = self.get_model_parameters(config)
                
                # Add required parameters with their CSV names
                param_mappings = {
                    'discount_rate': 'discount_rate',
                    'forward_years': 'forward_years',
                    'price_conversion_factor': '2019_NZD'
                }
                
                for param_name, csv_name in param_mappings.items():
                    if csv_name in model_params_data:
                        try:
                            model_params[param_name] = float(model_params_data[csv_name])
                        except (ValueError, TypeError):
                            if model_params_data[csv_name] != 'na':
                                print(f"Warning: Could not convert {csv_name} value '{model_params_data[csv_name]}' to float. Using default.")
                            # Set defaults if conversion fails
                            defaults = {
                                'discount_rate': 0.05,
                                'forward_years': 10,
                                'price_conversion_factor': 1.0
                            }
                            model_params[param_name] = defaults[param_name]
                    else:
                        print(f"Warning: {csv_name} not found in model parameters. Using default.")
                        # Set defaults if parameter missing
                        defaults = {
                            'discount_rate': 0.05,
                            'forward_years': 10,
                            'price_conversion_factor': 1.0
                        }
                        model_params[param_name] = defaults[param_name]
                
            except Exception as e:
                print(f"Warning: Could not load additional parameters from model_parameters.csv: {e}")
                # Set defaults if loading fails
                model_params.update({
                    'discount_rate': 0.05,
                    'forward_years': 10,
                    'price_conversion_factor': 1.0
                })
            
            return model_params
            
        except Exception as e:
            raise ValueError(f"Failed to load demand model parameters: {e}")
    
    def list_available_options(self, option_type: str) -> List[str]:
        """List available options for a component type."""
        try:
            # Map option types to their data attributes and column names
            option_mapping = {
                'emissions': (self.emissions_baselines_data, 'Config'),
                'auction': (self.auctions_data, 'Config'),
                'industrial': (self.industrial_allocation_data, 'Config'),
                'forestry': (self.removals_data, 'Config'),
                'demand_model': (self.demand_models_data, 'Config')
            }
            
            if option_type not in option_mapping:
                raise ValueError(f"Unknown option type: {option_type}")
            
            data, column = option_mapping[option_type]
            
            if data is not None and not data.empty and column in data.columns:
                configs = sorted(data[column].unique().tolist())
                return configs
            
            # If we can't find configs, return central only without warning
            return ['central']
            
        except Exception as e:
            # If there's an error, return central only without warning
            return ['central']
            
    def list_available_configs(self) -> Dict[str, List[str]]:
        """
        List all available configurations for each data type.
        
        Returns:
            Dictionary mapping data types to lists of available configs
        """
        if self.use_config_loading:
            # For scenario-based loading
            configs = {}
            
            # Check model parameters for config column
            if hasattr(self, 'model_parameters_data') and not self.model_parameters_data.empty:
                if 'Config' in self.model_parameters_data.columns:
                    configs['model_parameters'] = self.model_parameters_data['Config'].unique().tolist()
                else:
                    # Try to find a similar column
                    config_columns = [col for col in self.model_parameters_data.columns if 'config' in col.lower()]
                    if config_columns:
                        configs['model_parameters'] = self.model_parameters_data[config_columns[0]].unique().tolist()
            
            # Check each dataframe for config columns
            if hasattr(self, 'emissions_baselines_data') and not self.emissions_baselines_data.empty:
                if 'Config' in self.emissions_baselines_data.columns:
                    configs['emissions'] = self.emissions_baselines_data['Config'].unique().tolist()
            
            if hasattr(self, 'auctions_data') and not self.auctions_data.empty:
                # Check if the expected column exists
                if 'Config' in self.auctions_data.columns:
                    configs['auctions'] = self.auctions_data['Config'].unique().tolist()
                else:
                    # Try to find a similar column
                    config_columns = [col for col in self.auctions_data.columns if 'config' in col.lower()]
                    if config_columns:
                        configs['auctions'] = self.auctions_data[config_columns[0]].unique().tolist()
            
            if hasattr(self, 'industrial_allocation_data') and not self.industrial_allocation_data.empty:
                # Check if the expected column exists
                if 'Config' in self.industrial_allocation_data.columns:
                    configs['industrial_allocation'] = self.industrial_allocation_data['Config'].unique().tolist()
                else:
                    # Try to find a similar column
                    config_columns = [col for col in self.industrial_allocation_data.columns if 'config' in col.lower()]
                    if config_columns:
                        configs['industrial_allocation'] = self.industrial_allocation_data[config_columns[0]].unique().tolist()
            
            if hasattr(self, 'removals_data') and not self.removals_data.empty:
                # Check if the expected column exists
                if 'Config' in self.removals_data.columns:
                    configs['forestry'] = self.removals_data['Config'].unique().tolist()
                else:
                    # Try to find a similar column
                    config_columns = [col for col in self.removals_data.columns if 'config' in col.lower()]
                    if config_columns:
                        configs['forestry'] = self.removals_data[config_columns[0]].unique().tolist()
            
            if hasattr(self, 'demand_models_data') and not self.demand_models_data.empty:
                if 'Config' in self.demand_models_data.columns:
                    configs['demand_models'] = self.demand_models_data['Config'].unique().tolist()
            
            return configs
        else:
            # For standard loading, try to extract configs from data
            result = {}
            
            # Get configs from scenario parameters
            if hasattr(self, 'scenario_parameters') and self.scenario_parameters:
                result['configs'] = list(self.scenario_parameters.keys())
            
            # Get configs from emissions data
            if hasattr(self, 'emissions_data') and self.emissions_data is not None:
                # Look for config columns
                config_columns = [col for col in self.emissions_data.columns 
                                  if col not in ['base_emissions', 'low_scenario', 'high_scenario']]
                if config_columns:
                    result['emissions'] = config_columns
            
            return result

    def get_historical_data(self, variable: str) -> Optional[pd.Series]:
        """
        Get historical data for a variable if available.
        
        Args:
            variable: Name of the variable to get historical data for
            
        Returns:
            Series with historical data, or None if not available
        """
        return self.historical_manager.get_historical_data(variable)
    
    def get_carbon_price(self, year: int, config: str = 'central') -> Optional[float]:
        """
        Get carbon price for a specific year and config.
        
        Args:
            year: The year to get the price for
            config: The configuration to use (defaults to 'central')
            
        Returns:
            Carbon price value, or None if not available
        """
        return self.historical_manager.get_carbon_price(year, config)
    
    def get_price_control(self, year: int, config: str = 'central') -> float:
        """
        Get price control value for a specific year and config.
        
        Args:
            year: The year to get the price control value for
            config: The configuration to use (defaults to 'central')
            
        Returns:
            Price control value (defaults to 1.0 if not available)
        """
        return self.historical_manager.get_price_control(year, config)
    
    def get_combined_series(self, variable: str, model_data: pd.Series) -> pd.Series:
        """
        Combine historical and model data for a variable.
        
        Args:
            variable: Name of the variable to combine data for
            model_data: Series containing model data
            
        Returns:
            Combined series with historical and model data
        """
        return self.historical_manager.get_combined_series(variable, model_data)
    
    def get_stockpile_start_values(self, year: int, config: str = 'central') -> Dict[str, float]:
        """
        Get stockpile start values for a specific year and config.
        
        Args:
            year: The year to get stockpile values for
            config: The configuration to use (defaults to 'central')
            
        Returns:
            Dictionary containing 'stockpile' and 'surplus' values
        """
        return self.historical_manager.get_stockpile_start_values(year, config)
