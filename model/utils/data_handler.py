"""
Consolidated data handling utilities for the NZUpy model.

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
    - Scenario-specific data storage: Maintains separate data for each scenario
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
    ):
        """Initialise the DataHandler."""
        # Set up data directory paths
        self.data_dir = Path(data_dir)

        # Set up directory paths
        self.parameters_dir = self.data_dir / "inputs" / "parameters"
        self.supply_dir = self.data_dir / "inputs" / "supply"
        self.demand_dir = self.data_dir / "inputs" / "demand"
        self.economic_dir = self.data_dir / "inputs" / "economic"
        self.stockpile_dir = self.data_dir / "inputs" / "stockpile"
        self.forestry_dir = self.data_dir / "inputs" / "forestry"

        # Initialise data attributes as None
        self.emissions_baselines_data = None
        self.auctions_data = None
        self.industrial_allocation_data = None
        self.removals_data = None
        self.demand_models_data = None
        self.model_parameters_data = None

        # forestry data attributes
        self.historical_removals_data = None
        self.yield_tables_data = None
        self.yield_increments = None # dict of {forest_type: np.array}
        self.afforestation_projections_data = None
        self.manley_parameters_data = None

        # Create historical data manager
        self.historical_manager = HistoricalDataManager(data_dir)

        # Initialize scenario-specific data storage
        self.scenario_data = {}

        # Load all config-based datasets
        self._load_config_datasets()

        # Load forestry datasets
        self._load_forestry_datasets()
    
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
    
    def _load_forestry_datasets(self):
        """Load forestry datasets. Failures are non-fatal — files are only
        required when forestry_mode='endogenous'."""
        for loader in [
            self._load_historical_removals,
            self._load_yield_tables,
            self._load_afforestation_projections,
            self._load_manley_parameters,
        ]:
            try:
                loader()
            except Exception as e:
                print(f"Warning: {loader.__name__} failed: {e}")

    def _load_historical_removals(self):
        file_path = self.supply_dir / "historical_removals.csv"
        if file_path.exists():
            self.historical_removals_data = self._load_csv(file_path)

    def _load_yield_tables(self):
        file_path = self.forestry_dir / "yield_tables.csv"
        if file_path.exists():
            self.yield_tables_data = self._load_csv(file_path)
            # Pre-compute annual yield increments for each forest type.
            # np.diff(cumulative, prepend=0) gives: increment[0] = cumulative[0],
            # increment[i] = cumulative[i] - cumulative[i-1] for i > 0.
            self.yield_increments = {}
            for forest_type in ['permanent_exotic', 'production_exotic', 'natural_forest']:
                cumulative = self.yield_tables_data[forest_type].values.astype(float)
                self.yield_increments[forest_type] = np.diff(cumulative, prepend=0)

    def _load_afforestation_projections(self):
        file_path = self.forestry_dir / "afforestation_projections.csv"
        if file_path.exists():
            self.afforestation_projections_data = self._load_csv(file_path)

    def _load_manley_parameters(self):
        file_path = self.forestry_dir / "manley_parameters.csv"
        if file_path.exists():
            self.manley_parameters_data = self._load_csv(file_path)

    # ------------------------------------------------------------------
    # Endogenous forestry getter methods
    # ------------------------------------------------------------------

    def get_historical_removals(self, config: Optional[str] = None,
                              scenario_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get historic (old-forest) forestry data from historical_removals.csv.

        Returns DataFrame indexed by year with columns:
        historic_forestry_tradeable, historic_forestry_held, historic_forestry_surrender.
        """
        if self.historical_removals_data is None:
            raise ValueError(
                "Historic removals data not loaded. "
                "Ensure historical_removals.csv exists in the supply directory."
            )

        config_lower = config.lower() if config else 'central'
        data = self.historical_removals_data[
            self.historical_removals_data['Config'].str.lower() == config_lower
        ]
        if data.empty:
            raise KeyError(f"No historic removals data found for config '{config}'")

        result = {}
        for var in ['historic_forestry_tradeable', 'historic_forestry_held',
                    'historic_forestry_surrender']:
            var_data = data[data['Variable'] == var][['Year', 'Value']]
            if not var_data.empty:
                result[var] = var_data.set_index('Year')['Value'].astype(float)

        df = pd.DataFrame(result)
        df.index = df.index.astype(int)
        df.index.name = 'year'
        df = df.fillna(0.0)

        # historical_removals.csv stores values in individual NZUs (t CO2-e).
        # The model works in kt CO2-e — divide by 1000. TODO: ensure consistent CSV units
        df = df / 1000.0
        return df

    def get_yield_increments(self) -> Dict[str, Any]:
        """Return pre-computed annual yield increments dict {forest_type: np.array}."""
        if self.yield_increments is None:
            raise ValueError(
                "Yield increments not computed. "
                "Ensure yield_tables.csv exists in the forestry directory."
            )
        return {k: v.copy() for k, v in self.yield_increments.items()}

    def get_afforestation_projections(self, config: Optional[str] = None) -> pd.DataFrame:
        """
        Get MPI afforestation projections pivoted to wide format.

        Returns DataFrame indexed by year with columns:
        permanent_exotic, production_exotic, natural_forest.
        """
        if self.afforestation_projections_data is None:
            raise ValueError(
                "Afforestation projections data not loaded. "
                "Ensure afforestation_projections.csv exists in the forestry directory."
            )

        config_lower = config.lower() if config else 'central'
        data = self.afforestation_projections_data[
            self.afforestation_projections_data['Config'].str.lower() == config_lower
        ]
        if data.empty:
            raise KeyError(f"No afforestation projections found for config '{config}'")

        result = data.pivot_table(index='Year', columns='Forest', values='Area', aggfunc='first')
        result.index = result.index.astype(int)
        result.index.name = 'year'
        result.columns.name = None
        return result.fillna(0.0)

    def get_manley_parameters(self, config: Optional[str] = None) -> Dict[str, Any]:
        """
        Get Manley equation parameters, merging 'default' values with
        config-specific overrides (low/central/high for f and LMV).
        """
        if self.manley_parameters_data is None:
            raise ValueError(
                "Manley parameters data not loaded. "
                "Ensure manley_parameters.csv exists in the forestry directory."
            )

        df = self.manley_parameters_data
        # Start with 'default' params
        default_rows = df[df['Config'].str.lower() == 'default']
        result = dict(zip(default_rows['Variable'],
                          default_rows['Value'].astype(float)))

        # Override with config-specific values (e.g., f, LMV)
        config_lower = config.lower() if config else 'central'
        config_rows = df[df['Config'].str.lower() == config_lower]
        for _, row in config_rows.iterrows():
            result[row['Variable']] = float(row['Value'])

        return result

    def get_years(self) -> List[int]:
        """
        Get the years covered by the model.

        Returns:
            List of years covered by the model.
        """
        params = self.get_model_parameters("central")
        start_year = int(params.get('start_year', 2022))
        end_year = int(params.get('end_year', 2050))
        return list(range(start_year, end_year + 1))
    
    def get_model_parameters(self, config: Optional[str] = None, scenario_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get model parameters for a specific config.
        
        Args:
            config: Model parameter config name (defaults to "central")
            scenario_name: If provided, get scenario-specific model parameters
            
        Returns:
            Dictionary of model parameters
            
        Raises:
            KeyError: If config not found or required columns missing
            ValueError: If model_parameters_data is not loaded
        """
        if self.model_parameters_data is None or self.model_parameters_data.empty:
            raise ValueError("Model parameters data not loaded")
        
        config = config.lower() if config else "central"
        
        # Filter for the config
        params = self.model_parameters_data[
            self.model_parameters_data['Config'].str.lower() == config
        ]
        
        if params.empty:
            raise KeyError(f"No parameters found for config '{config}'")
        
        # Store in scenario-specific data if scenario_name provided
        if scenario_name is not None:
            if scenario_name not in self.scenario_data:
                self.scenario_data[scenario_name] = {}
            self.scenario_data[scenario_name]['model_params'] = params.copy()

        return dict(zip(params['Variable'], params['Value']))
    
    def get_auction_data(self, config: Optional[str] = None, scenario_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get auction data for a specific configuration and scenario.
        This method is used by the scenario manager to get raw auction data.
        
        Parameter Loading Priority:
        1. Scenario-specific data (if scenario_name provided)
        2. Configuration-based data from auction.csv
        3. Standard loading mode data (central config only)
        
        Args:
            config: The configuration name to load (e.g., 'central', 'high', 'low').
                   If None, returns the default configuration.
            scenario_name: The name of the scenario to get data for.
                         If provided, returns scenario-specific data.
        
        Returns:
            DataFrame with auction data for the specified configuration.
            
        Raises:
            ValueError: If auction data is not loaded or if required files are missing.
                      Check that auction.csv exists in the supply directory.
            KeyError: If the specified configuration is not found in the data.
                     Available configurations can be found in auction.csv.
        """
        
        # First check for scenario-specific data
        if scenario_name is not None and scenario_name in self.scenario_data:
            scenario_data = self.scenario_data[scenario_name]
            if 'auction' in scenario_data:
                return scenario_data['auction'].copy()
        
        # If no scenario data found, use config-based loading
        if self.auctions_data is None or self.auctions_data.empty:
            raise ValueError(
                "Auction data not loaded. "
                "Please ensure auction.csv exists in the supply directory "
                f"({self.supply_dir}) and contains the required data."
            )
        
        # Filter for the specified config
        config_lower = config.lower() if config else 'central'
        filtered_data = self.auctions_data[
            self.auctions_data['Config'].str.lower() == config_lower
        ]

        if filtered_data.empty:
            available_configs = self.auctions_data['Config'].unique()
            raise KeyError(
                f"No auction data found for config '{config}'. "
                f"Available configurations are: {', '.join(available_configs)}. "
                "Please check auction.csv for valid configurations."
            )
        
        
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
            var_data = filtered_data[filtered_data['Variable'] == csv_var][['Year', 'Value']]
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
        
        
        # Store scenario-specific data if scenario_name provided
        if scenario_name is not None:
            if scenario_name not in self.scenario_data:
                self.scenario_data[scenario_name] = {}
            self.scenario_data[scenario_name]['auction'] = auction_df.copy()
        
        return auction_df
    
    def get_industrial_allocation_data(self, config: Optional[str] = None, scenario_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get industrial allocation data for a specific configuration and scenario.
        This method is used by the scenario manager to get raw allocation data.
        
        Parameter Loading Priority:
        1. Scenario-specific data (if scenario_name provided)
        2. Configuration-based data from industrial_allocation.csv
        3. Standard loading mode data (central config only)
        
        Args:
            config: The configuration name to load (e.g., 'central', 'high', 'low').
                   If None, returns the default configuration.
            scenario_name: The name of the scenario to get data for.
                         If provided, returns scenario-specific data.
        
        Returns:
            DataFrame with industrial allocation data for the specified configuration.
            
        Raises:
            ValueError: If industrial allocation data is not loaded or if required files are missing.
                      Check that industrial_allocation.csv exists in the supply directory.
            KeyError: If the specified configuration is not found in the data.
                     Available configurations can be found in industrial_allocation.csv.
        """
        
        # First check for scenario-specific data
        if scenario_name is not None and scenario_name in self.scenario_data:
            scenario_data = self.scenario_data[scenario_name]
            if 'industrial' in scenario_data:
                return scenario_data['industrial'].copy()
        
        # If no scenario data found, use config-based loading
        if self.industrial_allocation_data is None or self.industrial_allocation_data.empty:
            raise ValueError(
                "Industrial allocation data not loaded. "
                "Please ensure industrial_allocation.csv exists in the supply directory "
                f"({self.supply_dir}) and contains the required data."
            )
             
        # Filter for the specified config
        config_lower = config.lower() if config else 'central'
        filtered_data = self.industrial_allocation_data[
            self.industrial_allocation_data['Config'].str.lower() == config_lower
        ]

        if filtered_data.empty:
            available_configs = self.industrial_allocation_data['Config'].unique()
            raise KeyError(
                f"No industrial allocation data found for config '{config}'. "
                f"Available configurations are: {', '.join(available_configs)}. "
                "Please check industrial_allocation.csv for valid configurations."
            )
        
        
        # Create DataFrame with baseline_allocation column
        ia_data = pd.DataFrame({
            'baseline_allocation': filtered_data.set_index('Year')['Value']
        })
        
        # Convert year index to proper format
        ia_data.index = ia_data.index.astype(int)
        ia_data.index.name = 'year'
        
        # Store scenario-specific data if scenario_name provided
        if scenario_name is not None:
            if scenario_name not in self.scenario_data:
                self.scenario_data[scenario_name] = {}
            self.scenario_data[scenario_name]['industrial'] = ia_data.copy()
        
        return ia_data
    
    def get_stockpile_parameters(self, config: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None, scenario_name: Optional[str] = None, model_start_year: Optional[int] = None) -> Dict[str, Any]:
        """
        Get stockpile parameters from stockpile balance data and model parameters.

        Args:
            config: Configuration name (defaults to "central")
            overrides: Optional dictionary of parameter overrides
            scenario_name: If provided, get scenario-specific stockpile parameters
            model_start_year: The model's actual start year (from define_time). If provided,
                this is used for stockpile_usage_start_year and stockpile_reference_year
                instead of the value in model_parameters.csv.

        Returns:
            Dictionary of stockpile parameters

        Raises:
            ValueError: If required parameters are missing or cannot be loaded
        """
        # Check if we have scenario-specific data first
        if scenario_name is not None and scenario_name in self.scenario_data:
            if 'stockpile' in self.scenario_data[scenario_name]:
                result = self.scenario_data[scenario_name]['stockpile'].copy()
                # Apply overrides if provided
                if overrides:
                    for param, value in overrides.items():
                        if value is not None:
                            result[param] = value
                return result
        
        # Get model parameters for non-stockpile values (model_parameters.csv only has 'central')
        params = self.get_model_parameters('central', scenario_name)
        
        # Get stockpile balance data for the config
        try:
            if not hasattr(self, 'stockpile_balance_data') or self.stockpile_balance_data is None:
                self._load_stockpile_balance_data()
            
            # Filter for the config
            config_lower = config.lower() if config else 'central'
            config_data = self.stockpile_balance_data[
                self.stockpile_balance_data['Config'].str.lower() == config_lower
            ]
            
            if config_data.empty:
                # If no data for this config, try central config
                if config and config.lower() != 'central':
                    print(f"Warning: No stockpile balance data for config '{config}', using 'central'")
                    config_data = self.stockpile_balance_data[
                        self.stockpile_balance_data['Config'].str.lower() == 'central'
                    ]
                else:
                    raise ValueError("No stockpile balance data found for any configuration")
            
            # Use model_start_year if provided (from define_time), otherwise fall back to CSV value
            start_year = model_start_year if model_start_year is not None else int(params['start_year'])
            target_year = start_year - 1
            
            # Get data for target year
            year_data = config_data[config_data['Year'] == target_year]
            
            if year_data.empty:
                raise ValueError(f"No stockpile balance data found for year {target_year} (year before start year {start_year})")
            
            # Get stockpile value
            stockpile_data = year_data[year_data['Variable'] == 'stockpile']
            if stockpile_data.empty:
                raise ValueError(f"No stockpile data found for year {target_year}")
            initial_stockpile = float(stockpile_data['Value'].iloc[0])
            
            # Get surplus value
            surplus_data = year_data[year_data['Variable'] == 'surplus']
            if surplus_data.empty:
                raise ValueError(f"No surplus data found for year {target_year}")
            initial_surplus = float(surplus_data['Value'].iloc[0])
            
        except (ValueError, KeyError, IndexError) as e:
            raise ValueError(f"Failed to load stockpile balance data: {e}")
        
        # Map CSV names to return names (start_year excluded — driven by model_start_year)
        param_mapping = {
            'liquidity_factor': 'liquidity_factor',
            'payback_years': 'payback_period',
            'discount_rate': 'discount_rate',
        }

        # Check required parameters exist
        missing = [p for p in param_mapping.keys() if p not in params]
        if missing:
            raise ValueError(f"Missing required parameters in model_parameters.csv: {missing}")

        # Create return dictionary with mapped names
        result = {}
        for csv_name, return_name in param_mapping.items():
            value = params[csv_name]
            result[return_name] = int(value) if csv_name == 'payback_years' else float(value)

        # stockpile_usage_start_year comes from define_time() (via model_start_year),
        # not from model_parameters.csv, so it respects the user-defined time window.
        result['stockpile_usage_start_year'] = start_year

        # Add stockpile values
        result['initial_stockpile'] = initial_stockpile
        result['initial_surplus'] = initial_surplus

        # Add reference year (year before the model start year)
        result['stockpile_reference_year'] = target_year
        
        # Apply overrides if provided
        if overrides:
            # Validate overrides before applying
            for param, value in overrides.items():
                if param not in result:
                    raise ValueError(f"Unknown parameter in overrides: {param}")
                
                # Skip None values - they mean "don't override this parameter"
                if value is None:
                    continue
                
                # Validate parameter values
                if param == 'liquidity_factor' and not 0 <= value <= 1:
                    raise ValueError(f"liquidity_factor must be between 0 and 1, got {value}")
                elif param == 'discount_rate' and not 0 <= value <= 1:
                    raise ValueError(f"discount_rate must be between 0 and 1, got {value}")
                elif param == 'initial_stockpile' and value < 0:
                    raise ValueError(f"initial_stockpile cannot be negative, got {value}")
                elif param == 'initial_surplus' and value < 0:
                    raise ValueError(f"initial_surplus cannot be negative, got {value}")
                elif param == 'payback_period' and value <= 0:
                    raise ValueError(f"payback_period must be positive, got {value}")
                elif param == 'stockpile_usage_start_year' and value < start_year:
                    raise ValueError(f"stockpile_usage_start_year ({value}) cannot be before model start year ({start_year})")
                
                # Convert to appropriate type and update result
                if param in ['payback_period', 'stockpile_usage_start_year', 'stockpile_reference_year']:
                    result[param] = int(value)
                else:
                    result[param] = float(value)
            
            # Now that all parameters have been updated, check inter-parameter relationships
            if result['initial_surplus'] > result['initial_stockpile']:
                raise ValueError(f"initial_surplus ({result['initial_surplus']}) cannot exceed initial_stockpile ({result['initial_stockpile']})")
        
        # Store in scenario-specific data if scenario_name provided
        if scenario_name is not None:
            if scenario_name not in self.scenario_data:
                self.scenario_data[scenario_name] = {}
            self.scenario_data[scenario_name]['stockpile'] = result.copy()

        return result

    def get_forestry_data(self, config: Optional[str] = None, scenario_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get forestry data for a specific configuration and scenario.
        
        Args:
            config: The configuration name to load (e.g., 'central', 'high', 'low').
                   If None, returns the default configuration.
            scenario_name: The name of the scenario to get data for.
                         If provided, returns scenario-specific data.
        
        Returns:
            DataFrame containing forestry data for the specified configuration.
        """
        
        # First check for scenario-specific data
        if scenario_name is not None and scenario_name in self.scenario_data:
            scenario_data = self.scenario_data[scenario_name]
            if 'forestry' in scenario_data:
                return scenario_data['forestry'].copy()
        
        # If no scenario data found, use config-based loading
        if self.removals_data is None or self.removals_data.empty:
            raise ValueError("Removals data not loaded")
             
        # Filter for forestry_tradeable data and the specified config
        config_lower = config.lower() if config else 'central'
        forestry_df = self.removals_data[
            (self.removals_data['Variable'] == 'forestry_tradeable') &
            (self.removals_data['Config'].str.lower() == config_lower)
        ]
        
        if forestry_df.empty:
            raise KeyError(f"No forestry data found for config '{config}'")
        
        
        # Create DataFrame with forestry_tradeable column
        result_df = pd.DataFrame({
            'forestry_tradeable': forestry_df.set_index('Year')['Value']
        })
        
        # Ensure year index is integer type
        result_df.index = result_df.index.astype(int)
        result_df.index.name = 'year'
        
        # Store scenario-specific data if scenario_name provided
        if scenario_name is not None:
            if scenario_name not in self.scenario_data:
                self.scenario_data[scenario_name] = {}
            self.scenario_data[scenario_name]['forestry'] = result_df.copy()
        
        return result_df

    def get_forestry_variables(self, config: Optional[str] = None, scenario_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get forestry variables data.
        
        Args:
            config: The configuration name to load (e.g., 'central', 'high', 'low').
                   If None, returns the default configuration.
            scenario_name: If provided, get scenario-specific forestry variables data.
        
        Returns:
            DataFrame indexed by year containing columns for each forestry variable:
            - forestry_tradeable
            - forestry_held
            - forestry_surrender
        """
        # Check if we have scenario-specific data first
        if scenario_name is not None and scenario_name in self.scenario_data:
            if 'forestry_variables' in self.scenario_data[scenario_name]:
                return self.scenario_data[scenario_name]['forestry_variables'].copy()
        
        if self.removals_data is None or self.removals_data.empty:
            raise ValueError("Removals data not loaded")

        config_lower = config.lower() if config else 'central'
        # Filter for scenario
        forest_data = self.removals_data[
            self.removals_data['Config'].str.lower() == config_lower
        ]
        
        if forest_data.empty:
            raise KeyError(f"No forestry data found for config '{config_lower}'")
        
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
        result_df = result_df.fillna(0.0)
        
        # Store in scenario-specific data if scenario_name provided
        if scenario_name is not None:
            if scenario_name not in self.scenario_data:
                self.scenario_data[scenario_name] = {}
            self.scenario_data[scenario_name]['forestry_variables'] = result_df.copy()

        return result_df
    
    def get_emissions_data(self, config: Optional[str] = None, scenario_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get emissions data for a specific configuration and scenario.
        This method is used by the scenario manager to get raw emissions data.
        
        Parameter Loading Priority:
        1. Scenario-specific data (if scenario_name provided)
        2. Configuration-based data from emissions.csv
        3. Standard loading mode data (central config only)
        
        Args:
            config: The configuration name to load (e.g., 'central', 'high', 'low').
                   If None, returns the default configuration.
            scenario_name: The name of the scenario to get data for.
                         If provided, returns scenario-specific data.
        
        Returns:
            DataFrame with emissions data for the specified configuration.
            
        Raises:
            ValueError: If emissions data is not loaded or if required files are missing.
                      Check that emissions.csv exists in the supply directory.
            KeyError: If the specified configuration is not found in the data.
                     Available configurations can be found in emissions.csv.
        """
        
        # First check for scenario-specific data
        if scenario_name is not None and scenario_name in self.scenario_data:
            scenario_data = self.scenario_data[scenario_name]
            if 'emissions' in scenario_data:
                return scenario_data['emissions'].copy()
        
        # If no scenario data found, use config-based loading
        if self.emissions_baselines_data is None or self.emissions_baselines_data.empty:
            raise ValueError(
                "Emissions data not loaded. "
                "Please ensure emissions_baselines.csv exists in the demand directory "
                f"({self.demand_dir}) and contains the required data."
            )
        
        # Filter for the specified config
        config_lower = config.lower() if config else 'central'
        filtered_data = self.emissions_baselines_data[
            self.emissions_baselines_data['Config'].str.lower() == config_lower
        ]

        if filtered_data.empty:
            available_configs = self.emissions_baselines_data['Config'].unique()
            raise KeyError(
                f"No emissions data found for config '{config}'. "
                f"Available configurations are: {', '.join(available_configs)}. "
                "Please check emissions_baselines.csv for valid configurations."
            )

        # Store scenario-specific data if scenario_name provided
        if scenario_name is not None:
            if scenario_name not in self.scenario_data:
                self.scenario_data[scenario_name] = {}
            self.scenario_data[scenario_name]['emissions'] = filtered_data.copy()

        return filtered_data.copy()
    
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

    def get_demand_model(self, config: str = "central", model_number: int = 2, scenario_name: Optional[str] = None) -> Dict[str, float]:
        """
        Get demand model parameters.
        
        Args:
            config: Configuration name (e.g., "central", "95pc_lower")
            model_number: Demand model number (1 or 2)
            scenario_name: If provided, get scenario-specific demand model parameters
                
        Returns:
            Dictionary containing demand model parameters including:
            - constant, reduction_to_t1, price (from demand_models.csv)
            - model_number (passed parameter)
            - discount_rate, forward_years, price_conversion_factor (from model_parameters.csv)
            
        Raises:
            FileNotFoundError: If demand_models.csv cannot be loaded
            ValueError: If no parameters found for given config and model number
            KeyError: If required parameters are missing from model_parameters.csv
        """
        # Check if we have scenario-specific data first
        if scenario_name is not None and scenario_name in self.scenario_data:
            if 'demand_model' in self.scenario_data[scenario_name]:
                return self.scenario_data[scenario_name]['demand_model'].copy()
        
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
        
        # Load and add required parameters from model_parameters.csv (only has 'central')
        model_params_data = self.get_model_parameters('central', scenario_name)
        
        # Required parameters and their CSV names
        param_mappings = {
            'discount_rate': 'discount_rate',
            'forward_years': 'forward_years', 
            'price_conversion_factor': '2019_NZD'
        }
        
        # Add each required parameter
        for param_name, csv_name in param_mappings.items():
            if csv_name not in model_params_data:
                raise KeyError(f"Required parameter '{csv_name}' not found in model parameters")
            
            try:
                model_params[param_name] = float(model_params_data[csv_name])
            except (ValueError, TypeError):
                raise ValueError(f"Could not convert {csv_name} value '{model_params_data[csv_name]}' to float")
        
        # Store in scenario-specific data if scenario_name provided
        if scenario_name is not None:
            if scenario_name not in self.scenario_data:
                self.scenario_data[scenario_name] = {}
            self.scenario_data[scenario_name]['demand_model'] = model_params.copy()

        return model_params
            
    
    def list_available_configs(self, component_type: str) -> List[str]:
        """
        List available predefined input configurations for a component type.
        
        Args:
            component_type: Type of component to list configs for ('emissions', 'auction', 
                        'industrial', 'forestry', 'demand_model', 'stockpile')
        
        Returns:
            List of available configuration names (e.g., ['central', 'high', 'low'])
        """
        # Map component types to their data attributes and column names
        option_mapping = {
            'emissions': (self.emissions_baselines_data, 'Config'),
            'auction': (self.auctions_data, 'Config'),
            'industrial': (self.industrial_allocation_data, 'Config'),
            'forestry': (self.removals_data, 'Config'),
            'demand_model': (self.demand_models_data, 'Config'),
            'stockpile': (self.stockpile_balance_data, 'Config')
        }
        
        if component_type not in option_mapping:
            raise ValueError(f"Invalid component type: {component_type}. Must be one of {list(option_mapping.keys())}")
            
        data, config_col = option_mapping[component_type]
        
        try:
            if data is not None and not data.empty and config_col in data.columns:
                return sorted(data[config_col].unique().tolist())
            else:
                # Try to find similar columns if exact match not found
                similar_cols = [col for col in data.columns if 'config' in col.lower()]
                if similar_cols:
                    return sorted(data[similar_cols[0]].unique().tolist())
                return []
        except Exception as e:
            print(f"Warning: Error getting configs for {component_type}: {e}")
            return []

    def get_historical_data(self, variable: str, nominal: bool = False) -> Optional[pd.Series]:
        """Get historical data for a variable."""
        return self.historical_manager.get_historical_data(variable, nominal)

