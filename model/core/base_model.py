"""
NZUpy main model interface.

Defines the NZUpy class, which coordinates all model components (supply, demand,
stockpile, optimiser) and exposes the setup and run API.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from pathlib import Path

from model.supply.auction import AuctionSupply
from model.supply.industrial import IndustrialAllocation
from model.supply.stockpile import StockpileSupply
from model.supply.forestry import ForestrySupply
from model.demand.emissions import EmissionsDemand
from model.demand.price_response import PriceResponse
from model.utils.data_handler import DataHandler
from model.config import ModelConfig, ComponentConfig
from model.utils.output_format import OutputFormat
from model.core.optimiser import FastSolveOptimiser
from model.core.calculation_engine import CalculationEngine
from model.core.scenario_manager import ScenarioManager
from model.core.runner import ModelRunner

class NZUpy:
    """
    Main model class for the NZ ETS Supply-Demand Model.

    Coordinates all model components and provides the interface for the
    optimisation process. Set up the model in sequence before calling run():

    Example usage:
        nzu = NZUpy()
        nzu.define_time(2024, 2050)
        nzu.define_scenarios(['Baseline', 'High Price'])
        nzu.allocate()
        nzu.fill_defaults()                                   # seed all components with central config
        nzu.fill_component('emissions', config='CCC_CPR', scenario='High Price')
        nzu.run()
        nzu.prices                                            # DataFrame of carbon prices
    """
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        data_handler: Optional[DataHandler] = None,
    ):
        """
        Initialise the ETS model.

        Args:
            data_dir: Path to data directory
            data_handler: Optional pre-configured data handler
        """
        # Set up data handler
        self.data_handler = data_handler or DataHandler(data_dir or "data")
        self.config = ModelConfig()
        
        # Create empty configuration objects - will be populated later
        self.scenarios = []  # Will hold scenario names
        self.component_configs = []  # Will hold component configurations for each scenario
        # Track whether core methods have been called
        self._time_defined = False
        self._scenarios_defined = False
        self._primed = False
        
        # Initialise empty results storage
        self.results = {}

        # Initialise component classes
        self.calculation_engine = CalculationEngine(self)
        self.scenario_manager = ScenarioManager(self)
        self.model_runner = ModelRunner(self)

        # Flag for scenario loading mode
        self.use_scenario_loading = False
    
    def define_time(
        self, 
        optimisation_start_year: int, 
        optimisation_end_year: int,
        historical_start_year: Optional[int] = None,
        projection_horizon: int = 25
    ) -> 'NZUpy':
        """
        Define the time periods for the model.
        
        Args:
            optimisation_start_year: First year of optimisation window
            optimisation_end_year: Last year of optimisation window
            historical_start_year: First year of historical data (defaults to optimisation_start_year)
            projection_horizon: Number of years to extend calculations beyond optimisation_end_year
        
        Returns:
            Self for method chaining
        """
        # Store time parameters
        self.config.start_year = optimisation_start_year
        self.config.end_year = optimisation_end_year
        self.historical_start_year = historical_start_year or optimisation_start_year
        self.forward_horizon = projection_horizon
        
        # Calculate all required year ranges
        self.years = list(range(self.config.start_year, self.config.end_year + 1))
        
        # Historical years (if any)
        if self.historical_start_year < self.config.start_year:
            self.historical_years = list(range(self.historical_start_year, self.config.start_year))
        else:
            self.historical_years = []
        
        # Extended projection years beyond end_year
        self.extended_years = list(range(self.config.end_year + 1, 
                                      self.config.end_year + self.forward_horizon + 1))
        
        # Full calculation years including the extended projection period
        self.calculation_years = self.years + self.extended_years
        
        # All years for data and presentation
        self.all_years = self.historical_years + self.years + self.extended_years
        
        # Set reference year for stockpile calculations
        self.stockpile_reference_year = optimisation_start_year - 1
        
        # Initialise historical prices using data handler
        self._initialise_historical_prices()
        
        # Initialise price control parameter
        self._initialise_price_control()
        
        # Mark time as defined
        self._time_defined = True
        
        # Print informative message
        print(f"Time periods defined:")
        print(f"  Optimisation: {self.config.start_year}-{self.config.end_year}")
        return self
    
    def _initialise_historical_prices(self):
        """Initialise historical prices based on available data."""
        if not hasattr(self.data_handler, 'historical_manager'):
            raise RuntimeError("Data handler does not have historical manager")
            
        # Get historical carbon price data
        historical_prices = self.data_handler.get_historical_data('carbon_price')
        if historical_prices is None or historical_prices.empty:
            raise RuntimeError("No historical carbon price data available")
            
        # Convert to dictionary format
        self.historical_prices = historical_prices.to_dict()
        
        # Determine the last historical year and its price
        self.last_historical_year = max(self.historical_prices.keys())
        self.last_historical_price = self.historical_prices[self.last_historical_year]
    
    def _initialise_price_control(self):
        """Initialise price control parameter from data."""
        # Create empty price control series
        self.price_control_parameter = pd.Series(index=self.years)
        
        # Get configuration name using active_price_control_config property
        config_name = self.active_price_control_config
        # Load values directly from CSV
        try:
            price_control_csv = self.data_handler.parameters_dir / "price_control.csv"
            df = pd.read_csv(price_control_csv)
            
            # Filter for current config and convert to series
            config_values = df[df['Config'] == config_name].set_index('Year')['Value']
            
            # Apply values to our years
            for year in self.years:
                if year in config_values.index:
                    self.price_control_parameter[year] = config_values[year]
                else:
                    # Default to 1.0 if year not found
                    self.price_control_parameter[year] = 1.0
        except Exception as e:
            print(f"Warning: Error loading price control values: {e}")
            # Default all years to 1.0
            self.price_control_parameter.fillna(1.0, inplace=True)
        
        # Apply any year-specific price control values from config
        for year, value in self.config.price_control_values.items():
            if year in self.price_control_parameter.index:
                self.price_control_parameter[year] = value
    
    def define_scenarios(self, scenario_names: List[str]) -> 'NZUpy':
        """
        Define named scenarios for the model.
        
        Args:
            scenario_names: List of scenario names to create
        
        Returns:
            Self for method chaining
        """
        # Validate inputs
        if not scenario_names:
            raise ValueError("At least one scenario name must be provided")
        
        if not self._time_defined:
            raise ValueError("Time must be defined before scenarios. Call define_time() first.")
        
        # Store scenario names
        self.scenarios = scenario_names
        
        # Create a ComponentConfig for each scenario
        self.component_configs = [ComponentConfig() for _ in scenario_names]
        
        # Mark scenarios as defined
        self._scenarios_defined = True
        
        return self
    
    def allocate(self) -> 'NZUpy':
        """
        Allocate the model by initializing core data structures and scenario slots.

        Call this after define_time() and define_scenarios(), before any fill_*() calls.

        Returns:
            Self for method chaining
        """
        # Check if model has already been allocated
        if self._primed:
            print("Model already allocated. Skipping.")
            return self

        # Validate that time and scenarios have been defined
        if not self._time_defined:
            raise ValueError("Time must be defined before allocating. Call define_time() first.")

        if not self._scenarios_defined:
            raise ValueError("Scenarios must be defined before allocating. Call define_scenarios() first.")

        # Initialise internal data structures for calculations
        self.prices = pd.Series(index=self.calculation_years, dtype=float)
        self.supply = pd.DataFrame(index=self.years)
        self.demand = pd.Series(index=self.years, dtype=float)

        # Price change rate (the optimisation variable)
        self.price_change_rate = 0.0

        # Initialise scenario configurations
        self.scenario_manager._initialise_scenarios()

        # Mark as allocated/primed
        self._primed = True

        # Print informative message
        print(f"Model allocated with {len(self.scenarios)} scenarios:")
        for i, name in enumerate(self.scenarios):
            print(f"  [{i}] {name}")

        # Initialise dictionaries for storing results
        self.results = {}

        return self

    # -------------------------------------------------------------------------
    # fill() family
    # -------------------------------------------------------------------------

    # Variable → component routing table (from 02_INPUT_TAXONOMY.md)
    _VARIABLE_COMPONENT_MAP = {
        # Auction (column names as they appear in auction.csv)
        'base_volume': 'auction',
        'auction_reserve_price': 'auction',
        'ccr_trigger_price_1': 'auction',
        'ccr_volume_1': 'auction',
        'ccr_trigger_price_2': 'auction',
        'ccr_volume_2': 'auction',
        # Forestry (exogenous time-series)
        'forestry_tradeable': 'forestry',
        'forestry_held': 'forestry',
        'forestry_surrender': 'forestry',
        # Forestry mode / Manley scalars
        'forestry_mode': 'forestry',
        'manley_sensitivity': 'forestry',
        'forestry_price_assumption': 'forestry',
        'manley_f': 'forestry',
        'manley_LMV': 'forestry',
        'manley_LUC_limit': 'forestry',
        'forestry_discount_rate': 'forestry',
        'forestry_forward_years': 'forestry',
        # Industrial
        'baseline_allocation': 'industrial',
        # Emissions
        'emissions_baseline': 'emissions',
        # Demand model
        'demand_model_number': 'demand_model',
        'constant': 'demand_model',
        'reduction_to_t1': 'demand_model',
        # Stockpile scalars
        'initial_stockpile': 'stockpile',
        'initial_surplus': 'stockpile',
        'liquidity_factor': 'stockpile',
        'payback_period': 'stockpile',
        'stockpile_usage_start_year': 'stockpile',
        'stockpile_reference_year': 'stockpile',
        # Price
        'start_price': 'price',
        'price_control': 'price',
        # Pricing mode
        'pricing_mode': 'pricing',
        'price_path': 'pricing',
        'price_change_rate': 'pricing',
    }

    # Mode switches: name → tuple of valid values
    _MODE_OPTIONS = {
        'forestry_mode':             ('exogenous', 'endogenous'),
        'pricing_mode':              ('optimised', 'fixed_path', 'fixed_rate'),
        'penalise_shortfalls':       (True, False),
        'manley_sensitivity':        ('low', 'central', 'high'),
        'forestry_price_assumption': ('future', 'current'),
    }

    def fill_defaults(self, config: str = 'central') -> 'NZUpy':
        """
        Load the specified config for ALL components across ALL scenarios.

        Call this once after allocate() to seed every scenario with a baseline
        configuration, then use fill_component() / fill() to override specific
        components or values.

        Args:
            config: Config name to use for all components (default: 'central').

        Returns:
            Self for method chaining.

        Example:
            nzu.allocate()
            nzu.fill_defaults()                        # all scenarios → central
            nzu.fill_component('forestry', config='low', scenario='Low Forestry')
        """
        if not self._primed:
            raise ValueError(
                "Model must be allocated before filling. Call allocate() first."
            )
        for i in range(len(self.scenarios)):
            cfg = self.component_configs[i]
            cfg.emissions = config
            cfg.auction = config
            cfg.industrial = config
            cfg.forestry = config
            cfg.demand_sensitivity = config
            cfg.demand_model_number = 2
            cfg.stockpile = config
            cfg.model_params = config

        print(f"Defaults filled: all components set to '{config}' for all scenarios.")
        return self

    def fill_component(
        self,
        component: str,
        config: str,
        scenario: Optional[Union[str, int]] = None,
    ) -> 'NZUpy':
        """
        Load a specific config for one component, optionally scoped to one scenario.

        Args:
            component: Component name — one of 'auction', 'forestry', 'industrial',
                       'emissions', 'demand_model', 'stockpile'.
            config: Config name (e.g., 'central', 'low', 'CCC_2024').
            scenario: Scenario name (str) or index (int). If None, applies to ALL
                      scenarios.

        Returns:
            Self for method chaining.

        Example:
            nzu.fill_component('forestry', config='low', scenario='Alt')
            nzu.fill_component('emissions', config='CCC_CPR')   # all scenarios
        """
        if not self._primed:
            raise ValueError(
                "Model must be allocated before filling. Call allocate() first."
            )

        valid_components = ['auction', 'forestry', 'industrial', 'emissions',
                            'demand_model', 'stockpile']
        if component not in valid_components:
            raise ValueError(
                f"Unknown component: '{component}'. "
                f"Valid options: {', '.join(valid_components)}"
            )

        # Resolve scenario(s)
        if scenario is None:
            indices = list(range(len(self.scenarios)))
        elif isinstance(scenario, int):
            if scenario < 0 or scenario >= len(self.scenarios):
                raise ValueError(
                    f"Scenario index {scenario} out of range "
                    f"(0–{len(self.scenarios) - 1})."
                )
            indices = [scenario]
        else:
            if scenario not in self.scenarios:
                raise ValueError(
                    f"Unknown scenario: '{scenario}'. "
                    f"Available: {', '.join(self.scenarios)}"
                )
            indices = [self.scenarios.index(scenario)]

        for i in indices:
            cfg = self.component_configs[i]
            if component == 'demand_model':
                cfg.demand_sensitivity = config
            else:
                setattr(cfg, component, config)

        scope = f"scenario '{self.scenarios[indices[0]]}'" if len(indices) == 1 else "all scenarios"
        print(f"Filled {component} with config '{config}' for {scope}.")
        return self

    def fill(
        self,
        variable_name: str,
        value: Any,
        scenario: Optional[Union[str, int]] = None,
        component: Optional[str] = None,
    ) -> 'NZUpy':
        """
        Set an individual variable value for one or all scenarios.

        The component is inferred automatically from the variable name.
        For ambiguous variables (e.g., 'discount_rate', 'forward_years') the
        component kwarg is required.

        Args:
            variable_name: Variable to set (must be in the taxonomy routing table,
                           or pass component= for ambiguous variables).
            value: New value. Scalar for parameters; pd.Series for time series.
            scenario: Scenario name or index. If None, applies to ALL scenarios.
            component: Required for ambiguous variables ('discount_rate',
                       'forward_years'). Also accepted for any variable to be
                       explicit.

        Returns:
            Self for method chaining.

        Example:
            nzu.fill('liquidity_factor', 0.15, scenario='Alt')
            nzu.fill('discount_rate', 0.07, scenario='Alt', component='stockpile')
            nzu.fill('forestry_tradeable', my_series, scenario='Alt')
        """
        if not self._primed:
            raise ValueError(
                "Model must be allocated before filling. Call allocate() first."
            )

        # Inform the user when they pass a mode variable through fill()
        if variable_name in self._MODE_OPTIONS:
            print(
                f"Note: '{variable_name}' is a mode switch. "
                f"Consider using set_mode('{variable_name}', ...) instead."
            )

        # Resolve component
        if component is None:
            if variable_name not in self._VARIABLE_COMPONENT_MAP:
                raise ValueError(
                    f"Unknown variable '{variable_name}'. "
                    f"If this is an ambiguous variable (e.g., 'discount_rate'), "
                    f"pass component= to disambiguate."
                )
            component = self._VARIABLE_COMPONENT_MAP[variable_name]

        # Resolve scenario index(es)
        if scenario is None:
            indices = list(range(len(self.scenarios)))
        elif isinstance(scenario, int):
            if scenario < 0 or scenario >= len(self.scenarios):
                raise ValueError(
                    f"Scenario index {scenario} out of range "
                    f"(0–{len(self.scenarios) - 1})."
                )
            indices = [scenario]
        else:
            if scenario not in self.scenarios:
                raise ValueError(
                    f"Unknown scenario: '{scenario}'. "
                    f"Available: {', '.join(self.scenarios)}"
                )
            indices = [self.scenarios.index(scenario)]

        _valid_stockpile_params = [
            "initial_stockpile", "initial_surplus", "liquidity_factor",
            "discount_rate", "stockpile_usage_start_year", "payback_period",
            "stockpile_reference_year",
        ]

        _valid_forestry_scalars = [
            'forestry_mode', 'manley_sensitivity', 'forestry_price_assumption',
            'manley_f', 'manley_LMV', 'manley_LUC_limit',
            'forestry_discount_rate', 'forestry_forward_years',
        ]

        for i in indices:
            if component == 'forestry' and variable_name in _valid_forestry_scalars:
                # Validate forestry_mode and forestry_price_assumption values
                if variable_name == 'forestry_mode' and value not in ('exogenous', 'endogenous'):
                    raise ValueError(
                        f"forestry_mode must be 'exogenous' or 'endogenous', got '{value}'"
                    )
                if variable_name == 'forestry_price_assumption' and value not in ('future', 'current'):
                    raise ValueError(
                        f"forestry_price_assumption must be 'future' or 'current', got '{value}'"
                    )
                if variable_name == 'manley_sensitivity' and value not in ('low', 'central', 'high'):
                    raise ValueError(
                        f"manley_sensitivity must be 'low', 'central', or 'high', got '{value}'"
                    )
                setattr(self.component_configs[i], variable_name, value)
            elif component == 'stockpile':
                if variable_name not in _valid_stockpile_params:
                    raise ValueError(
                        f"Unknown stockpile parameter: '{variable_name}'. "
                        f"Valid options: {', '.join(_valid_stockpile_params)}"
                    )
                if variable_name == 'liquidity_factor' and not 0 <= value <= 1:
                    raise ValueError(f"liquidity_factor must be between 0 and 1, got {value}")
                if variable_name == 'discount_rate' and not 0 <= value <= 1:
                    raise ValueError(f"discount_rate must be between 0 and 1, got {value}")
                setattr(self.component_configs[i], variable_name, value)
                print(f"Set {variable_name}={value} for scenario '{self.scenarios[i]}'")
            elif component == 'demand_model' and variable_name == 'demand_model_number':
                if not isinstance(value, int) or value not in [1, 2]:
                    raise ValueError("demand_model_number must be 1 or 2")
                self.component_configs[i].demand_model_number = value
            elif component == 'pricing':
                if variable_name == 'pricing_mode':
                    if value not in self._MODE_OPTIONS['pricing_mode']:
                        raise ValueError(
                            f"pricing_mode must be one of "
                            f"{list(self._MODE_OPTIONS['pricing_mode'])}, got '{value}'"
                        )
                    self.component_configs[i].pricing_mode = value
                elif variable_name == 'price_path':
                    if not isinstance(value, pd.Series):
                        raise ValueError("price_path must be a pd.Series")
                    self.component_configs[i].price_path = value
                elif variable_name == 'price_change_rate':
                    if not isinstance(value, (int, float)):
                        raise ValueError("price_change_rate must be a number")
                    self.component_configs[i].price_change_rate = float(value)
                else:
                    raise ValueError(
                        f"Unknown pricing variable: '{variable_name}'. "
                        f"Valid options: pricing_mode, price_path, price_change_rate"
                    )
            elif isinstance(value, pd.Series):
                self._store_series(variable_name, value, component=component, scenario_index=i)
            else:
                raise ValueError(
                    f"Cannot set scalar '{variable_name}' for component "
                    f"'{component}' via fill(). "
                    f"For time-series components use a pd.Series value, "
                    f"or use fill_component() to select a named config."
                )

        scope = (f"scenario '{self.scenarios[indices[0]]}'"
                 if len(indices) == 1 else "all scenarios")
        print(f"Filled {component}.{variable_name} for {scope}.")
        return self

    def fill_range_configs(self) -> 'NZUpy':
        """
        Configure demand sensitivity configs to match scenario names for range runs.

        Must be called after define_scenario_type('Range') and allocate().

        Returns:
            Self for method chaining.

        Example:
            nzu.define_scenario_type('Range')
            nzu.allocate()
            nzu.fill_defaults()
            nzu.fill_range_configs()   # maps each scenario → its demand config
            nzu.run()
        """
        return self.scenario_manager.configure_range_scenarios()

    def set_mode(
        self,
        mode_name: str,
        value,
        scenario: Optional[Union[str, int]] = None,
    ) -> 'NZUpy':
        """
        Set a structural mode switch for one or all scenarios.

        Use this for categorical model behaviour switches (e.g. pricing_mode,
        forestry_mode). For data values use fill().

        Args:
            mode_name: Mode to set — one of 'pricing_mode', 'forestry_mode',
                       'penalise_shortfalls', 'manley_sensitivity',
                       'forestry_price_assumption'.
            value: New value. Must be one of the valid options for the mode.
            scenario: Scenario name or index. If None, applies to ALL scenarios.

        Returns:
            Self for method chaining.

        Example:
            nzu.set_mode('pricing_mode', 'fixed_path', scenario='Alt')
            nzu.set_mode('forestry_mode', 'endogenous')   # all scenarios
        """
        if not self._primed:
            raise ValueError(
                "Model must be allocated before setting modes. Call allocate() first."
            )

        if mode_name not in self._MODE_OPTIONS:
            raise ValueError(
                f"Unknown mode '{mode_name}'. "
                f"Valid modes: {list(self._MODE_OPTIONS.keys())}"
            )
        if value not in self._MODE_OPTIONS[mode_name]:
            raise ValueError(
                f"Invalid value '{value}' for mode '{mode_name}'. "
                f"Valid options: {list(self._MODE_OPTIONS[mode_name])}"
            )

        # Resolve scenario index(es)
        if scenario is None:
            indices = list(range(len(self.scenarios)))
        elif isinstance(scenario, int):
            if scenario < 0 or scenario >= len(self.scenarios):
                raise ValueError(
                    f"Scenario index {scenario} out of range "
                    f"(0–{len(self.scenarios) - 1})."
                )
            indices = [scenario]
        else:
            if scenario not in self.scenarios:
                raise ValueError(
                    f"Unknown scenario: '{scenario}'. "
                    f"Available: {', '.join(self.scenarios)}"
                )
            indices = [self.scenarios.index(scenario)]

        for i in indices:
            setattr(self.component_configs[i], mode_name, value)

        scope = (f"scenario '{self.scenarios[indices[0]]}'"
                 if len(indices) == 1 else "all scenarios")
        print(f"Set {mode_name}='{value}' for {scope}.")
        return self

    # -------------------------------------------------------------------------
    # Discoverability methods
    # -------------------------------------------------------------------------

    # Variables belonging to each component (from 02_INPUT_TAXONOMY.md)
    _COMPONENT_VARIABLES = {
        'auction': ['auction_volume', 'auction_reserve_price', 'CCR_price_1',
                    'CCR_volume_1', 'CCR_price_2', 'CCR_volume_2'],
        'forestry': ['forestry_tradeable', 'forestry_held', 'forestry_surrender'],
        'industrial': ['baseline_allocation'],
        'emissions': ['emissions_baseline'],
        'demand_model': ['demand_model_number', 'constant', 'reduction_to_t1',
                         'discount_rate', 'forward_years', 'price_conversion_factor'],
        'stockpile': ['initial_stockpile', 'initial_surplus', 'liquidity_factor',
                      'discount_rate', 'payback_period', 'stockpile_usage_start_year'],
        'price': ['start_price', 'price_control'],
    }

    def list_configs(self, component: str) -> dict:
        """
        List available configs for a component by reading the source CSV.

        Args:
            component: Component name ('auction', 'forestry', 'industrial',
                       'emissions', 'demand_model', 'stockpile').

        Returns:
            Dict mapping variable names to lists of available config names.
            Also prints a summary to stdout.

        Example:
            nzu.list_configs('forestry')
            # {'forestry_tradeable': ['central'], ...}
        """
        available = self.data_handler.list_available_configs(component)
        variables = self._COMPONENT_VARIABLES.get(component, [])

        print(f"\nAvailable configs for '{component}':")
        print(f"  Configs: {available}")
        print(f"  Variables: {variables}")

        return {var: available for var in variables}

    def show_config(self, component: str, config_name: str) -> None:
        """
        Display the values for a specific component config.

        Args:
            component: Component name ('auction', 'forestry', 'industrial',
                       'emissions', 'demand_model', 'stockpile').
            config_name: Config name to display (e.g., 'central', 'low').

        Example:
            nzu.show_config('forestry', 'central')
        """
        print(f"\n=== {component.upper()} — config: '{config_name}' ===")
        try:
            if component == 'auction':
                data = self.data_handler.get_auction_data(config=config_name)
                print(data.to_string())
            elif component == 'forestry':
                data = self.data_handler.get_forestry_data(config=config_name)
                print(data.to_string())
            elif component == 'industrial':
                data = self.data_handler.get_industrial_allocation_data(config=config_name)
                print(data.to_string())
            elif component == 'emissions':
                data = self.data_handler.get_emissions_data(config=config_name)
                print(data.to_string())
            elif component == 'demand_model':
                data = self.data_handler.get_demand_model(config=config_name, model_number=2)
                for k, v in data.items():
                    print(f"  {k}: {v}")
            elif component == 'stockpile':
                model_start_year = self.config.start_year if self._time_defined else None
                data = self.data_handler.get_stockpile_parameters(config=config_name, model_start_year=model_start_year)
                for k, v in data.items():
                    print(f"  {k}: {v}")
            else:
                print(f"show_config() not supported for component '{component}'.")
        except Exception as e:
            print(f"Could not load config '{config_name}' for '{component}': {e}")

    def compare_configs(self, component: str, config_a: str, config_b: str) -> None:
        """
        Display a side-by-side comparison of two configs for a component.

        Args:
            component: Component name.
            config_a: First config name.
            config_b: Second config name.

        Example:
            nzu.compare_configs('emissions', 'central', 'CCC_CPR')
        """
        print(f"\n=== {component.upper()} — '{config_a}' vs '{config_b}' ===")
        try:
            if component in ('auction', 'forestry', 'industrial', 'emissions'):
                getter_map = {
                    'auction': self.data_handler.get_auction_data,
                    'forestry': self.data_handler.get_forestry_data,
                    'industrial': self.data_handler.get_industrial_allocation_data,
                    'emissions': self.data_handler.get_emissions_data,
                }
                getter = getter_map[component]
                df_a = getter(config=config_a)
                df_b = getter(config=config_b)
                # Build comparison DataFrame using first numeric column of each
                comparison = pd.DataFrame({
                    config_a: df_a.select_dtypes('number').iloc[:, 0],
                    config_b: df_b.select_dtypes('number').iloc[:, 0],
                })
                comparison['diff'] = comparison[config_b] - comparison[config_a]
                print(comparison.to_string())
            elif component in ('demand_model', 'stockpile'):
                if component == 'demand_model':
                    kwargs = {'model_number': 2}
                else:
                    kwargs = {'model_start_year': self.config.start_year if self._time_defined else None}
                getter = (self.data_handler.get_demand_model
                          if component == 'demand_model'
                          else self.data_handler.get_stockpile_parameters)
                data_a = getter(config=config_a, **kwargs)
                data_b = getter(config=config_b, **kwargs)
                col_w = max(len(config_a), len(config_b), 12)
                print(f"  {'Variable':<30} {config_a:<{col_w}} {config_b:<{col_w}}")
                print(f"  {'-'*30} {'-'*col_w} {'-'*col_w}")
                for k in sorted(set(list(data_a.keys()) + list(data_b.keys()))):
                    va = data_a.get(k, 'N/A')
                    vb = data_b.get(k, 'N/A')
                    marker = ' *' if va != vb else ''
                    print(f"  {k:<30} {str(va):<{col_w}} {str(vb):<{col_w}}{marker}")
            else:
                print(f"compare_configs() not supported for component '{component}'.")
        except Exception as e:
            print(f"Could not compare configs for '{component}': {e}")

    def show_current(self, scenario: Optional[Union[str, int]] = None) -> dict:
        """
        Display which config is currently loaded for each component.

        Args:
            scenario: Scenario name or index. If None, shows all scenarios.

        Returns:
            Dict of {scenario_name: {component: config_name}}.
            Also prints a formatted table to stdout.

        Example:
            nzu.show_current()
            nzu.show_current(scenario='Alt')
        """
        if not self._primed:
            raise ValueError("Model must be allocated before calling show_current().")

        if scenario is None:
            indices = list(range(len(self.scenarios)))
        elif isinstance(scenario, int):
            if scenario < 0 or scenario >= len(self.scenarios):
                raise ValueError(f"Scenario index {scenario} out of range.")
            indices = [scenario]
        else:
            if scenario not in self.scenarios:
                raise ValueError(
                    f"Unknown scenario: '{scenario}'. "
                    f"Available: {', '.join(self.scenarios)}"
                )
            indices = [self.scenarios.index(scenario)]

        result = {}
        for i in indices:
            name = self.scenarios[i]
            cfg = self.component_configs[i]
            result[name] = {
                'auction':      cfg.auction,
                'industrial':   cfg.industrial,
                'forestry':     cfg.forestry,
                'emissions':    cfg.emissions,
                'demand_model': cfg.demand_sensitivity,
                'stockpile':    cfg.stockpile,
                'model_params': cfg.model_params,
            }

        # Pretty-print table
        scenario_names = [self.scenarios[i] for i in indices]
        col_w = max((len(s) for s in scenario_names), default=12) + 2
        print(f"\nCurrent configuration:")
        header = f"  {'Component':<18}" + "".join(f"  {s:<{col_w}}" for s in scenario_names)
        print(header)
        print("  " + "-" * (18 + (col_w + 2) * len(indices)))
        for comp in ['auction', 'industrial', 'forestry', 'emissions',
                     'demand_model', 'stockpile', 'model_params']:
            row = f"  {comp:<18}"
            for name in scenario_names:
                val = result[name].get(comp, '—')
                row += f"  {str(val):<{col_w}}"
            print(row)

        return result

    def calculate_gap(self, price_change_rate: float) -> float:
        """Calculate the gap between supply and demand."""
        return self.calculation_engine.calculate_gap(price_change_rate)

    def validate(self) -> bool:
        """
        Validate the model configuration before running.
        
        Returns:
            True if the model is valid, otherwise raises ValueError
        """
        # Check that the model has been primed
        if not self._primed:
            raise ValueError("Model must be allocated before validation. Call allocate() first.")

        # Check that we have at least one scenario
        if not self.scenarios:
            raise ValueError("No scenarios defined. Call define_scenarios() first.")

        # Check that we have scenario configurations
        if not hasattr(self, 'component_configs') or not self.component_configs:
            raise ValueError("Scenario configurations not initialised. Call allocate() first.")
        
        # Check scenario configurations
        for i, scenario_config in enumerate(self.component_configs):
            scenario_name = self.scenarios[i]
            
            # Check if required scenario types are set
            if not scenario_config.emissions:
                raise ValueError(f"Emissions scenario not set for '{scenario_name}'. Call fill_defaults() or fill_component() first.")
            
            if not scenario_config.auction:
                raise ValueError(f"Auction scenario not set for '{scenario_name}'. Call fill_defaults() or fill_component() first.")

            if not scenario_config.industrial:
                raise ValueError(f"Industrial allocation scenario not set for '{scenario_name}'. Call fill_defaults() or fill_component() first.")
            
            if not scenario_config.forestry:
                raise ValueError(f"Forestry scenario not set for '{scenario_name}'. Call fill_defaults() or fill_component() first.")
            
            # Validate stockpile parameters if set
            stockpile_params = {}
            try:
                # Try to get stockpile parameters with any custom values
                if scenario_config.stockpile:
                    stockpile_params = self.data_handler.get_stockpile_parameters(
                        scenario_config.stockpile,
                        overrides={
                            'initial_stockpile': scenario_config.initial_stockpile,
                            'initial_surplus': scenario_config.initial_surplus,
                            'liquidity_factor': scenario_config.liquidity_factor,
                            'discount_rate': scenario_config.discount_rate,
                            'payback_period': scenario_config.payback_period,
                            'stockpile_usage_start_year': scenario_config.stockpile_usage_start_year,
                            'stockpile_reference_year': scenario_config.stockpile_reference_year
                        },
                        model_start_year=self.config.start_year
                    )
                else:
                    # If no config specified, use model parameters
                    stockpile_params = self.data_handler.get_stockpile_parameters(
                        model_start_year=self.config.start_year
                    )
            except Exception as e:
                raise ValueError(f"Failed to validate stockpile parameters for scenario '{scenario_name}': {e}")
            
            # Validate relationships between parameters
            if stockpile_params['initial_surplus'] > stockpile_params['initial_stockpile']:
                raise ValueError(
                    f"Invalid stockpile parameters for scenario '{scenario_name}': "
                    f"initial_surplus ({stockpile_params['initial_surplus']}) cannot exceed "
                    f"initial_stockpile ({stockpile_params['initial_stockpile']})"
                )
            
            if stockpile_params['stockpile_usage_start_year'] < self.config.start_year:
                raise ValueError(
                    f"Invalid stockpile parameters for scenario '{scenario_name}': "
                    f"stockpile_usage_start_year ({stockpile_params['stockpile_usage_start_year']}) "
                    f"cannot be before model start year ({self.config.start_year})"
                )
            
            if stockpile_params['stockpile_reference_year'] >= stockpile_params['stockpile_usage_start_year']:
                print(
                    f"Warning: stockpile_reference_year ({stockpile_params['stockpile_reference_year']}) "
                    f"should be before stockpile_usage_start_year ({stockpile_params['stockpile_usage_start_year']}) "
                    f"for scenario '{scenario_name}'"
                )
        
        # All checks passed
        return True

    def define_scenario_type(self, scenario_type: str = 'Single') -> 'NZUpy':
        """
        Define the scenario type for the model.
        
        Args:
            scenario_type: Type of scenario run to perform.
                Options: 'Single', 'Range'
                Default: 'Single'
            
        Returns:
            Self for method chaining
        """
        return self.scenario_manager.define_scenario_type(scenario_type)

    def set_demand_model(self, model_number: int) -> 'NZUpy':
        """
        Set the demand model number to use for all scenarios.
        
        Args:
            model_number: Demand model number (1 or 2)
                1 = MACC model
                2 = ENZ model (default)
            
        Returns:
            Self for method chaining
        """
        return self.scenario_manager.set_demand_model(model_number)

    def run(self) -> Dict[str, Dict[str, Any]]:
        """
        Run the model for all defined scenarios.
        
        Returns:
            Dictionary mapping scenario names to their results
        """
        return self.model_runner.run()
        
    def get_scenario_result(self, scenario_name_or_index: Union[str, int]) -> Dict[str, Any]:
        """
        Retrieve results for a specific scenario by name or index.
        
        Args:
            scenario_name_or_index: Name or index of the scenario
        
        Returns:
            Dictionary containing results for the specified scenario
        
        Raises:
            ValueError: If model has not been run or scenario is not found
        """
        return self.model_runner.get_scenario_result(scenario_name_or_index)

    def run_optimisation(self) -> Dict[str, Any]:
        """
        Run the optimisation process to find the optimal price change rate.
        
        Returns:
            Dict containing optimisation and model results.
        """
        return self.model_runner.run_optimisation()

    def run_model(self, price_change_rate: Optional[float] = None, *, is_final_run: bool = False) -> Dict[str, Any]:
        """
        Run the model with a given price change rate.
        
        Args:
            price_change_rate: The price change rate to use.
                If None, the current price_change_rate will be used.
            is_final_run: Whether this is the final run after optimisation.
                If True, forestry variables will be included.
        
        Returns:
            Dict containing model results.
        """
        return self.model_runner.run_model(price_change_rate, is_final_run=is_final_run)
    
    def set_price_control(self, year_values: Dict[int, float]):
        """
        Set price control parameters for specific years.
        
        Args:
            year_values: Dictionary mapping years to price control values.
                Positive values (e.g., 1.0) apply the normal price change.
                Negative values (e.g., -1.0) invert the price change direction.
                Zero disables price change for that year.
        """
        for year, value in year_values.items():
            if year in self.price_control_parameter.index:
                self.price_control_parameter[year] = value
        
        # Recalculate prices if a price change rate has been set
        if hasattr(self, 'price_change_rate') and self.price_change_rate != 0:
            self.calculation_engine._calculate_prices()

    def set_price_control_range(self, value: float, start_year: int = None, end_year: int = None):
        """
        Set the same price control value for a range of years.
        
        Args:
            value: Price control value to set.
            start_year: First year to apply the control (defaults to first year after historical prices).
            end_year: Last year to apply the control (defaults to last model year).
        """
        if start_year is None:
            start_year = self.last_historical_year + 1
        
        if end_year is None:
            end_year = max(self.years)
        
        year_values = {year: value for year in range(start_year, end_year + 1) 
                    if year in self.price_control_parameter.index}
        
        self.set_price_control(year_values)
    
    @property
    def active_price_control_config(self):
        """Get the active price control configuration for the current run."""
        # The model runner already sets up the scenario's config before calculations
        # so we can just check the current component_configs for the active scenario
        if hasattr(self, '_active_scenario_index'):
            component_config = self.component_configs[self._active_scenario_index]
            return getattr(component_config, 'price_control_config', 'central')
        return 'central'  # Default if not running a specific scenario
    
    def _store_series(
        self,
        series_name: str,
        data: pd.Series,
        component: str,
        scenario_index: int,
    ) -> 'NZUpy':
        """Store a time series override for a specific component and scenario."""
        if not self._primed:
            raise ValueError("Model must be allocated before setting series data. Call allocate() first.")

        # Validate scenario_index
        if scenario_index < 0 or scenario_index >= len(self.scenarios):
            raise ValueError(f"Invalid scenario index: {scenario_index}. Valid range: 0-{len(self.scenarios)-1}")

        scenario_name = self.scenarios[scenario_index]
        component_config = self.component_configs[scenario_index]

        if component == 'auction':
            auction_data = self.data_handler.get_auction_data(
                config=component_config.auction,
                scenario_name=scenario_name
            )
            valid_series = ['base_volume', 'auction_reserve_price', 'ccr_trigger_price_1',
                            'ccr_trigger_price_2', 'ccr_volume_1', 'ccr_volume_2']
            if series_name not in valid_series:
                raise ValueError(f"Invalid auction series: '{series_name}'. Valid options: {', '.join(valid_series)}")
            auction_data[series_name] = data
            if scenario_name not in self.data_handler.scenario_data:
                self.data_handler.scenario_data[scenario_name] = {}
            self.data_handler.scenario_data[scenario_name]['auction'] = auction_data
            print(f"Updated {component}.{series_name} for scenario '{scenario_name}'")

        elif component == 'industrial':
            industrial_data = self.data_handler.get_industrial_allocation_data(
                config=component_config.industrial,
                scenario_name=scenario_name
            )
            valid_series = ['baseline_allocation', 'activity_adjustment']
            if series_name not in valid_series:
                raise ValueError(f"Invalid industrial series: '{series_name}'. Valid options: {', '.join(valid_series)}")
            industrial_data[series_name] = data
            if scenario_name not in self.data_handler.scenario_data:
                self.data_handler.scenario_data[scenario_name] = {}
            self.data_handler.scenario_data[scenario_name]['industrial'] = industrial_data
            print(f"Updated {component}.{series_name} for scenario '{scenario_name}'")

        elif component == 'forestry':
            forestry_data = self.data_handler.get_forestry_data(
                config=component_config.forestry,
                scenario_name=scenario_name
            )
            valid_series = ['forestry_supply']
            if series_name not in valid_series:
                raise ValueError(f"Invalid forestry series: '{series_name}'. Valid options: {', '.join(valid_series)}")
            forestry_data[series_name] = data
            if scenario_name not in self.data_handler.scenario_data:
                self.data_handler.scenario_data[scenario_name] = {}
            self.data_handler.scenario_data[scenario_name]['forestry'] = forestry_data
            print(f"Updated {component}.{series_name} for scenario '{scenario_name}'")

        elif component == 'emissions':
            # Validate series name
            valid_series = ['emissions']
            if series_name not in valid_series:
                raise ValueError(f"Invalid emissions series: '{series_name}'. Valid options: {', '.join(valid_series)}")
            if scenario_name not in self.data_handler.scenario_data:
                self.data_handler.scenario_data[scenario_name] = {}
            self.data_handler.scenario_data[scenario_name]['emissions'] = pd.DataFrame({
                'emissions': data
            })
            print(f"Updated {component}.{series_name} for scenario '{scenario_name}'")

        else:
            raise ValueError(f"Invalid component: '{component}'. Valid options: auction, industrial, forestry, emissions")

        return self

    def use_price_control_config(self, config_name: str, scenario_index: int = None, scenario_name: str = None):
        """
        Load a specific price control configuration.
        
        Args:
            config_name: The configuration name to load (e.g., 'central', 'scarcity_then_surplus')
            scenario_index: Index of the scenario to apply this to (optional)
            scenario_name: Name of the scenario to apply this to (optional)
        
        Returns:
            Self for method chaining
        """
        # Determine which scenario to use
        if scenario_index is not None:
            if scenario_index < 0 or scenario_index >= len(self.scenarios):
                raise ValueError(f"Invalid scenario_index: {scenario_index}")
            scenario = self.scenarios[scenario_index]
        elif scenario_name is not None:
            if scenario_name not in self.scenarios:
                raise ValueError(f"Unknown scenario_name: {scenario_name}")

        # Store configuration name in scenario manager
        self.scenario_manager.set_price_control_config(scenario, config_name)
        
        return self