"""
Base model for NZ ETS Supply-Demand Model.

This module provides the main model interface that coordinates components
and provides the core functionality using a builder pattern for setup.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

# Import from other modules after refactoring
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

# Import refactored components
from model.core.calculation_engine import CalculationEngine
from model.core.scenario_manager import ScenarioManager
from model.core.runner import ModelRunner

class NZUpy:
    """
    Main model class for the NZ ETS Supply-Demand Model.
    
    This class coordinates all model components and provides the interface
    for the optimisation process. It follows a builder pattern for setup:
    
    Example usage:
        nze = NZUpy()
        nze.define_time(2023, 2050)
        nze.define_scenarios(['Low Auction', 'central', 'High Auction'])
        nze.prime()
        
        # Configure scenarios
        nze.use_central_configs(1)  # Set everything to central for scenario 1
        nze.set_parameter("initial_stockpile", 159902, 'stockpile', 1)  # Set stockpile parameter
        nze.use_config(0, 'emissions', 'CCC_CPR')
        
        # Run the model
        results = nze.run()
    """
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        data_handler: Optional[DataHandler] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ):
        """
        Initialise the ETS model.
        
        Args:
            data_dir: Path to data directory
            data_handler: Optional pre-configured data handler
            config_overrides: Optional dictionary of config values to override defaults
        """
        # Set up data handler
        self.data_handler = data_handler or DataHandler(data_dir or "data")
        
        try:
            # Get required parameters from data handler
            stockpile_params = self.data_handler.get_stockpile_parameters()
            
            # Create base config with required parameters
            config_dict = {
                "initial_stockpile": stockpile_params['initial_stockpile'],
                "initial_surplus": stockpile_params['initial_surplus'],
                "liquidity_factor": stockpile_params['liquidity_factor']
            }
            
            # Apply any overrides
            if config_overrides:
                config_dict.update(config_overrides)
            
            # Create config
            self.config = ModelConfig(**config_dict)
            
        except Exception as e:
            raise ValueError(f"Failed to initialise model - could not load required parameters from data: {e}")
        
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
        
        # Get the current scenario's component config
        # If no scenarios defined yet, use default 'central' config
        if not hasattr(self, 'scenarios') or not self.scenarios:
            price_control_config = 'central'
        else:
            # Get current scenario or default to first scenario
            current_scenario = getattr(self, 'current_scenario', None)
            if not current_scenario:
                current_scenario = self.scenarios[0]
            
            scenario_index = self.scenarios.index(current_scenario)
            component_config = self.component_configs[scenario_index]
            price_control_config = getattr(component_config, 'price_control_config', 'central')
        
        # Load values from the CSV via historical data manager
        if hasattr(self.data_handler, 'historical_manager'):
            for year in self.years:
                price_control = self.data_handler.historical_manager.get_price_control(year, config=price_control_config)
                if price_control is not None:
                    self.price_control_parameter[year] = price_control
        
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
    
    def list_available_configs(self, component_type: str) -> List[str]:
        """
        List available predefined configurations for a component type.
        
        Args:
            component_type: Type of configs to list ('emissions', 'auction', 'industrial', 
                        'forestry', 'demand_model')
        
        Returns:
            List of available configuration names
        """
        return self.scenario_manager.list_available_configs(component_type)
    
    def prime(self) -> 'NZUpy':
        """
        Prime the model by initializing core data structures and configurations.
        
        Returns:
            Self for method chaining
        """
        # Check if model has already been primed
        if self._primed:
            print("Model already primed. Skipping.")
            return self
        
        # Validate that time and scenarios have been defined
        if not self._time_defined:
            raise ValueError("Time must be defined before priming. Call define_time() first.")
        
        if not self._scenarios_defined:
            raise ValueError("Scenarios must be defined before priming. Call define_scenarios() first.")
        
        # Initialise internal data structures for calculations
        self.prices = pd.Series(index=self.calculation_years, dtype=float)
        self.supply = pd.DataFrame(index=self.years)  # Supply for model years only
        self.demand = pd.Series(index=self.years, dtype=float)  # Demand for model years only
        
        # Price change rate (the optimisation variable)
        self.price_change_rate = 0.0
        
        # Initialise scenario configurations
        self.scenario_manager._initialise_scenarios()
        
        # Mark as primed
        self._primed = True
        
        # Print informative message
        print(f"Model primed with {len(self.scenarios)} scenarios:")
        for i, name in enumerate(self.scenarios):
            print(f"  [{i}] {name}")
        
        # Initialise dictionaries for storing results
        self.results = {}
        
        return self
    
    def set_parameter(self, parameter_name: str, value: Any, component: str, 
                    scenario_index: int = 0, scenario_name: str = None) -> 'NZUpy':
        """
        Set a parameter value for a specific component and scenario.
        
        Args:
            parameter_name: Name of the parameter to set
            value: New value for the parameter
            component: Component the parameter belongs to ('stockpile', 'auction', etc.)
            scenario_index: Index of the scenario to modify (default: 0)
            scenario_name: Name of the scenario to modify (overrides scenario_index if provided)
            
        Returns:
            Self for method chaining
        """
        if not self._primed:
            raise ValueError("Model must be primed before setting parameters. Call prime() first.")
        
        # Resolve scenario_index if scenario_name provided
        if scenario_name is not None:
            if scenario_name not in self.scenarios:
                raise ValueError(f"Unknown scenario name: '{scenario_name}'. Available scenarios: {', '.join(self.scenarios)}")
            scenario_index = self.scenarios.index(scenario_name)
        
        # Validate scenario_index
        if scenario_index < 0 or scenario_index >= len(self.scenarios):
            raise ValueError(f"Invalid scenario index: {scenario_index}. Valid range: 0-{len(self.scenarios)-1}")
        
        # Get scenario config
        component_config = self.component_configs[scenario_index]
        
        # Get scenario name for display
        scenario_name = self.scenarios[scenario_index]
        
        # Handle different components
        if component == 'stockpile':
            
            # Validate parameter name
            valid_params = ['initial_stockpile', 'initial_surplus', 'liquidity_factor', 
                            'discount_rate', 'payback_period', 'stockpile_usage_start_year', 
                            'stockpile_reference_year']
            
            if parameter_name not in valid_params:
                raise ValueError(f"Invalid stockpile parameter: '{parameter_name}'. Valid options: {', '.join(valid_params)}")
            
            # Validate parameter value based on type
            if parameter_name in ['initial_stockpile', 'initial_surplus']:
                if not isinstance(value, (int, float)) or value < 0:
                    raise ValueError(f"{parameter_name} must be a non-negative number")
            elif parameter_name in ['liquidity_factor', 'discount_rate']:
                if not isinstance(value, (int, float)) or value < 0 or value > 1:
                    raise ValueError(f"{parameter_name} must be a number between 0 and 1")
            elif parameter_name == 'payback_period':
                if not isinstance(value, int) or value <= 0:
                    raise ValueError("payback_period must be a positive integer")
            elif parameter_name == 'stockpile_usage_start_year':
                if not isinstance(value, int) or value < self.config.start_year:
                    raise ValueError(f"stockpile_usage_start_year must be at least {self.config.start_year}")
            
            # Set the parameter
            setattr(component_config, parameter_name, value)

            # Print confirmation
            print(f"Set {component}.{parameter_name} = {value} for scenario '{scenario_name}'")
            
        elif component == 'demand_model':
            # Special case for demand model number
            if parameter_name == 'model_number':
                if not isinstance(value, int) or value not in [1, 2]:
                    raise ValueError("demand_model number must be 1 or 2")
                component_config.demand_model_number = value
                print(f"Set demand_model.model_number = {value} for scenario '{scenario_name}'")
            else:
                print(f"Cannot set parameter '{parameter_name}' directly for component 'demand_model'")
        
        else:
            # For other components, explain that they need to use a different approach
            print(f"Cannot directly set parameter '{parameter_name}' for component '{component}'.")
            print(f"To modify {component} configuration, use:")
            print(f"  - use_config() to select a different predefined configuration")
            print(f"  - create custom datasets and load them with the data handler")
        
        return self
    
    def use_config(self, scenario_index: int, component_type: str, config_name: str, model_number: Optional[int] = None) -> 'NZUpy':
        """
        Use a specific configuration for a component in a scenario.
        
        Args:
            scenario_index: Index of the scenario to modify
            component_type: Type of component to configure ('emissions', 'auction', etc.)
            config_name: Name of the configuration to use
            model_number: Optional model number for demand models
            
        Returns:
            Self for method chaining
        """
        return self.scenario_manager.use_config(scenario_index, component_type, config_name, model_number)
    
    def use_central_configs(self, scenario_index: int) -> 'NZUpy':
        """
        Use central configurations for all components in a specific model scenario.
        
        Args:
            scenario_index: Index of the model scenario to modify
            
        Returns:
            Self for method chaining
        """
        return self.scenario_manager.use_central_configs(scenario_index)
    
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
            raise ValueError("Model must be primed before validation. Call prime() first.")
        
        # Check that we have at least one scenario
        if not self.scenarios:
            raise ValueError("No scenarios defined. Call define_scenarios() first.")
        
        # Check that we have scenario configurations
        if not hasattr(self, 'component_configs') or not self.component_configs:
            raise ValueError("Scenario configurations not initialised. Call prime() first.")
        
        # Check scenario configurations
        for i, scenario_config in enumerate(self.component_configs):
            scenario_name = self.scenarios[i]
            
            # Check if required scenario types are set
            if not scenario_config.emissions:
                raise ValueError(f"Emissions scenario not set for '{scenario_name}'. Call use_config() first.")
            
            if not scenario_config.auctions:
                raise ValueError(f"Auction scenario not set for '{scenario_name}'. Call use_config() first.")
            
            if not scenario_config.industrial_allocation:
                raise ValueError(f"Industrial allocation scenario not set for '{scenario_name}'. Call use_config() first.")
            
            if not scenario_config.forestry:
                raise ValueError(f"Forestry scenario not set for '{scenario_name}'. Call use_config() first.")
            
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
                        }
                    )
                else:
                    # If no config specified, use model parameters
                    stockpile_params = self.data_handler.get_stockpile_parameters()
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

    def configure_range_scenarios(self) -> 'NZUpy':
        """
        Configure all scenarios for a 'Range' scenario type run.
        
        Returns:
            Self for method chaining
        """
        return self.scenario_manager.configure_range_scenarios()

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
    
    def run_scenarios(self, scenarios: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Run the model for multiple scenarios.
        
        Args:
            scenarios: List of scenarios to run.
                Options: "central", "1 s.e lower", "1 s.e upper", "95% Lower", "95% Upper"
        
        Returns:
            Dict mapping scenarios to their results.
        """
        return self.model_runner.run_scenarios(scenarios)
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get the model results.
        
        Returns:
            Dict containing model results.
        """
        return {
            'price_change_rate': self.price_change_rate,
            'prices': self.prices.to_dict() if hasattr(self, 'prices') else {},
            'supply': self.supply.to_dict() if hasattr(self, 'supply') else {},
            'demand': self.demand.to_dict() if hasattr(self, 'demand') else {},
        }
    
    def update_scenario_config(self, scenario_name: str, **kwargs) -> 'NZUpy':
        """
        Update scenario configuration parameters.
        
        Args:
            scenario_name: Name of the scenario to update
            **kwargs: Key-value pairs of configuration parameters to update
            
        Returns:
            Self for method chaining
        """
        # Check if scenario exists
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found. Available scenarios: {', '.join(self.scenarios)}")
        
        # Get the scenario configuration using index
        scenario_index = self.scenarios.index(scenario_name)
        scenario_config = self.component_configs[scenario_index]
        
        # Update the configuration
        for key, value in kwargs.items():
            if hasattr(scenario_config, key):
                setattr(scenario_config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        
        return self

    def set_stockpile_values(self, scenario_index: int, stockpile: float, surplus: float) -> 'NZUpy':
        """
        Set custom stockpile and surplus values for a scenario.
        
        Args:
            scenario_index: Index of the scenario to modify
            stockpile: Total stockpile volume
            surplus: Surplus component volume (must be <= stockpile)
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If scenario_index is invalid or values are invalid
        """
        # Validate scenario index
        if scenario_index < 0 or scenario_index >= len(self.scenarios):
            raise ValueError(f"Invalid scenario index: {scenario_index}. Valid range: 0-{len(self.scenarios)-1}")
        
        # Validate values
        if surplus > stockpile:
            raise ValueError(f"Surplus ({surplus}) cannot exceed total stockpile ({stockpile})")
        
        if stockpile < 0 or surplus < 0:
            raise ValueError(f"Stockpile and surplus values must be non-negative")
        
        # Set values in the component config
        self.component_configs[scenario_index].initial_stockpile = stockpile
        self.component_configs[scenario_index].initial_surplus = surplus
        
        print(f"Set custom stockpile values for scenario {scenario_index}: stockpile={stockpile}, surplus={surplus}")
        return self
    
    def show_inputs(self, component: str, scenario_index: int = 0, scenario_name: str = None):
        """
        Display all inputs for a specific component and scenario.
        
        Args:
            component: Component to inspect ('stockpile', 'auction', 'industrial', 
                    'forestry', 'emissions', 'demand_model')
            scenario_index: Index of the scenario to inspect (default: 0)
            scenario_name: Name of the scenario to inspect (overrides scenario_index if provided)
            
        Returns:
            DataFrame containing all inputs for the component and scenario
        """
        if not self._primed:
            raise ValueError("Model must be primed before showing inputs. Call prime() first.")
        
        # Resolve scenario_index if scenario_name provided
        if scenario_name is not None:
            if scenario_name not in self.scenarios:
                raise ValueError(f"Unknown scenario name: '{scenario_name}'. Available scenarios: {', '.join(self.scenarios)}")
            scenario_index = self.scenarios.index(scenario_name)
        
        # Validate scenario_index
        if scenario_index < 0 or scenario_index >= len(self.scenarios):
            raise ValueError(f"Invalid scenario index: {scenario_index}. Valid range: 0-{len(self.scenarios)-1}")
        
        # Get scenario name for display
        scenario_name = self.scenarios[scenario_index]
        
        # Validate component
        valid_components = ['stockpile', 'auction', 'industrial', 'forestry', 'emissions', 'demand_model']
        if component not in valid_components:
            raise ValueError(f"Invalid component: '{component}'. Valid options: {', '.join(valid_components)}")
        
        # Get component configuration
        component_config = self.component_configs[scenario_index]
        
        # Get config name based on component type
        config_name = None
        if component == 'stockpile':
            config_name = component_config.stockpile
        elif component == 'auction':
            config_name = component_config.auctions
        elif component == 'industrial':
            config_name = component_config.industrial_allocation
        elif component == 'forestry':
            config_name = component_config.forestry
        elif component == 'emissions':
            config_name = component_config.emissions
        elif component == 'demand_model':
            config_name = component_config.demand_sensitivity
        
        if config_name is None:
            print(f"Warning: No configuration found for {component} in scenario {scenario_name}")
            return None
        
        # Get component data using existing data_handler methods
        try:
            # Load data differently based on component type
            if component == 'stockpile':
                data = self.data_handler.get_stockpile_parameters(config_name)
                print(f"\n=== Stockpile Configuration for Scenario '{scenario_name}' ===")
                print(f"Config name: {config_name}")
                
                # Override with scenario-specific values if set
                if component_config.initial_stockpile is not None:
                    data['initial_stockpile'] = component_config.initial_stockpile
                    print(f"* initial_stockpile is custom for this scenario")
                if component_config.initial_surplus is not None:
                    data['initial_surplus'] = component_config.initial_surplus
                    print(f"* initial_surplus is custom for this scenario")
                if component_config.liquidity_factor is not None:
                    data['liquidity_factor'] = component_config.liquidity_factor
                    print(f"* liquidity_factor is custom for this scenario")
                if component_config.discount_rate is not None:
                    data['discount_rate'] = component_config.discount_rate
                    print(f"* discount_rate is custom for this scenario")
                if component_config.payback_period is not None:
                    data['payback_period'] = component_config.payback_period
                    print(f"* payback_period is custom for this scenario")
                if component_config.stockpile_usage_start_year is not None:
                    data['stockpile_usage_start_year'] = component_config.stockpile_usage_start_year
                    print(f"* stockpile_usage_start_year is custom for this scenario")
                
                # Format and print
                print("\nStockpile Parameters:")
                print(f"  Initial Stockpile: {data['initial_stockpile']:,.0f} kt CO₂-e")
                print(f"  Initial Surplus: {data['initial_surplus']:,.0f} kt CO₂-e")
                print(f"  Liquidity Factor: {data['liquidity_factor']:.2%}")
                print(f"  Discount Rate: {data['discount_rate']:.2%}")
                print(f"  Payback Period: {data['payback_period']} years")
                print(f"  Stockpile Usage Start Year: {data['stockpile_usage_start_year']}")
                print(f"  Stockpile Reference Year: {data['stockpile_reference_year']}")
                
                # Return formatted data as dict for potential further use
                return data
                
            elif component == 'auction':
                data = self.data_handler.get_auction_data(config_name)
                print(f"\n=== Auction Configuration for Scenario '{scenario_name}' ===")
                print(f"Config name: {config_name}")
                
                # Print summary of time series
                print("\nAuction Parameters:")
                years_to_show = min(5, len(data.index))
                print(f"  Base Auction Volumes (first {years_to_show} years):")
                for i, year in enumerate(data.index[:years_to_show]):
                    print(f"    {year}: {data.loc[year, 'base_volume']:,.0f} kt CO₂-e")
                if len(data.index) > years_to_show:
                    print(f"    ... {len(data.index) - years_to_show} more years")
                
                # Show CCR parameters
                print("\n  Cost Containment Reserve (CCR) Parameters:")
                ccr_years = min(3, len(data.index))
                for i, year in enumerate(data.index[:ccr_years]):
                    print(f"    {year}:")
                    print(f"      CCR1 Price Trigger: ${data.loc[year, 'ccr_trigger_price_1']:.2f}")
                    print(f"      CCR1 Volume: {data.loc[year, 'ccr_volume_1']:,.0f} kt CO₂-e")
                    print(f"      CCR2 Price Trigger: ${data.loc[year, 'ccr_trigger_price_2']:.2f}")
                    print(f"      CCR2 Volume: {data.loc[year, 'ccr_volume_2']:,.0f} kt CO₂-e")
                if len(data.index) > ccr_years:
                    print(f"    ... {len(data.index) - ccr_years} more years")
                
                # Return data for potential further use
                return data
                
            elif component == 'industrial':
                data = self.data_handler.get_industrial_allocation_data(config_name)
                print(f"\n=== Industrial Allocation Configuration for Scenario '{scenario_name}' ===")
                print(f"Config name: {config_name}")
                
                # Print summary of allocation data
                print("\nIndustrial Allocation Volumes (first 5 years):")
                years_to_show = min(5, len(data.index))
                for i, year in enumerate(data.index[:years_to_show]):
                    print(f"  {year}: {data.loc[year, 'baseline_allocation']:,.0f} kt CO₂-e")
                if len(data.index) > years_to_show:
                    print(f"  ... {len(data.index) - years_to_show} more years")
                    
                # Return data for potential further use
                return data
                
            elif component == 'forestry':
                data = self.data_handler.get_forestry_data(config_name)
                # Also get forestry variables if available
                variables = self.data_handler.get_forestry_variables(config_name)
                
                print(f"\n=== Forestry Configuration for Scenario '{scenario_name}' ===")
                print(f"Config name: {config_name}")
                
                # Print summary of forestry data
                print("\nForestry Supply Volumes (first 5 years):")
                years_to_show = min(5, len(data.index))
                for i, year in enumerate(data.index[:years_to_show]):
                    print(f"  {year}: {data.loc[year, 'forestry_supply']:,.0f} kt CO₂-e")
                if len(data.index) > years_to_show:
                    print(f"  ... {len(data.index) - years_to_show} more years")
                
                if not variables.empty:
                    print("\nForestry Variables Available:")
                    for col in variables.columns:
                        print(f"  - {col}")
                
                # Return data for potential further use
                return {'supply': data, 'variables': variables}
                
            elif component == 'emissions':
                data = self.data_handler.get_emissions_data(config_name)
                
                print(f"\n=== Emissions Configuration for Scenario '{scenario_name}' ===")
                print(f"Config name: {config_name}")
                
                # Process and display data
                if 'Year' in data.columns and 'Value' in data.columns:
                    # Group by year
                    emissions_by_year = data.set_index('Year')['Value']
                    
                    print("\nEmissions Baseline (first 5 years):")
                    years_to_show = min(5, len(emissions_by_year.index))
                    for i, year in enumerate(sorted(emissions_by_year.index)[:years_to_show]):
                        print(f"  {year}: {emissions_by_year[year]:,.0f} kt CO₂-e")
                    if len(emissions_by_year.index) > years_to_show:
                        print(f"  ... {len(emissions_by_year.index) - years_to_show} more years")
                else:
                    print("Warning: Emissions data structure doesn't match expected format.")
                
                # Return data for potential further use
                return data
                
            elif component == 'demand_model':
                model_number = component_config.demand_model_number
                data = self.data_handler.get_demand_model(config_name, model_number)
                
                print(f"\n=== Demand Model Configuration for Scenario '{scenario_name}' ===")
                print(f"Config name: {config_name}")
                print(f"Model number: {model_number}")
                
                # Display scalar parameters
                print("\nDemand Model Parameters:")
                scalar_params = ['constant', 'reduction_to_t1', 'price', 'discount_rate', 'forward_years']
                for param in scalar_params:
                    if param in data:
                        print(f"  {param}: {data[param]}")
                
                # Return data for potential further use
                return data
                
        except Exception as e:
            print(f"Error retrieving {component} data: {str(e)}")
            return None

    def show_parameter(self, parameter_name: str, component: str, scenario_index: int = 0, scenario_name: str = None):
        """
        Display a specific parameter value for a component and scenario.
        
        Args:
            parameter_name: Name of the parameter to display
            component: Component the parameter belongs to
            scenario_index: Index of the scenario to inspect (default: 0)
            scenario_name: Name of the scenario to inspect (overrides scenario_index if provided)
            
        Returns:
            Value of the parameter, or None if not found
        """
        if not self._primed:
            raise ValueError("Model must be primed before showing parameters. Call prime() first.")
        
        # Resolve scenario_index if scenario_name provided
        if scenario_name is not None:
            if scenario_name not in self.scenarios:
                raise ValueError(f"Unknown scenario name: '{scenario_name}'. Available scenarios: {', '.join(self.scenarios)}")
            scenario_index = self.scenarios.index(scenario_name)
        
        # Validate scenario_index
        if scenario_index < 0 or scenario_index >= len(self.scenarios):
            raise ValueError(f"Invalid scenario index: {scenario_index}. Valid range: 0-{len(self.scenarios)-1}")
        
        # Get scenario name for display
        scenario_name = self.scenarios[scenario_index]
        
        # Get all component data
        all_data = self.show_inputs(component, scenario_index)
        
        if all_data is None:
            return None
            
        # Convert DataFrame to dict if needed
        if hasattr(all_data, 'to_dict'):
            all_data = all_data.to_dict()
        
        # Handle nested dictionaries (like forestry with 'supply' and 'variables')
        if isinstance(all_data, dict) and component == 'forestry' and 'supply' in all_data:
            # For forestry, check if parameter is in supply data
            if parameter_name in all_data['supply']:
                print(f"\nParameter '{parameter_name}' for {component} in scenario '{scenario_name}':")
                print(f"  Value: {all_data['supply'][parameter_name]}")
                return all_data['supply'][parameter_name]
            else:
                print(f"Parameter '{parameter_name}' not found in {component} supply data.")
                return None
        
        # Regular parameter lookup
        if parameter_name in all_data:
            value = all_data[parameter_name]
            print(f"\nParameter '{parameter_name}' for {component} in scenario '{scenario_name}':")
            
            # Format display based on parameter type
            if isinstance(value, float):
                if parameter_name in ['liquidity_factor', 'discount_rate']:
                    print(f"  Value: {value:.2%}")
                else:
                    print(f"  Value: {value:,.4f}")
            else:
                print(f"  Value: {value}")
            
            return value
        else:
            print(f"Parameter '{parameter_name}' not found in {component} data.")
            return None

    def show_series(self, series_name: str, component: str, scenario_index: int = 0, scenario_name: str = None, max_rows: int = 10):
        """
        Display a time series from a component for a specific scenario.
        
        Args:
            series_name: Name of the time series to display
            component: Component the series belongs to
            scenario_index: Index of the scenario to inspect (default: 0)
            scenario_name: Name of the scenario to inspect (overrides scenario_index if provided)
            max_rows: Maximum number of rows to display (default: 10)
            
        Returns:
            Series object containing the requested time series, or None if not found
        """
        if not self._primed:
            raise ValueError("Model must be primed before showing series data. Call prime() first.")
        
        # Resolve scenario_index if scenario_name provided
        if scenario_name is not None:
            if scenario_name not in self.scenarios:
                raise ValueError(f"Unknown scenario name: '{scenario_name}'. Available scenarios: {', '.join(self.scenarios)}")
            scenario_index = self.scenarios.index(scenario_name)
        
        # Validate scenario_index
        if scenario_index < 0 or scenario_index >= len(self.scenarios):
            raise ValueError(f"Invalid scenario index: {scenario_index}. Valid range: 0-{len(self.scenarios)-1}")
        
        # Get scenario name for display
        scenario_name = self.scenarios[scenario_index]
        
        # Get component configuration
        component_config = self.component_configs[scenario_index]
        
        # Get config name based on component type
        config_name = None
        if component == 'stockpile':
            config_name = component_config.stockpile
        elif component == 'auction':
            config_name = component_config.auctions
        elif component == 'industrial':
            config_name = component_config.industrial_allocation
        elif component == 'forestry':
            config_name = component_config.forestry
        elif component == 'emissions':
            config_name = component_config.emissions
        elif component == 'demand_model':
            config_name = component_config.demand_sensitivity
        
        if config_name is None:
            print(f"Warning: No configuration found for {component} in scenario {scenario_name}")
            return None
        
        try:
            # Get series data based on component type
            series_data = None
            
            if component == 'auction':
                auction_data = self.data_handler.get_auction_data(config_name)
                if series_name in auction_data.columns:
                    series_data = auction_data[series_name]
                else:
                    available_series = ", ".join(auction_data.columns)
                    print(f"Series '{series_name}' not found. Available series: {available_series}")
                    return None
                    
            elif component == 'industrial':
                industrial_data = self.data_handler.get_industrial_allocation_data(config_name)
                if series_name in industrial_data.columns:
                    series_data = industrial_data[series_name]
                else:
                    available_series = ", ".join(industrial_data.columns)
                    print(f"Series '{series_name}' not found. Available series: {available_series}")
                    return None
                    
            elif component == 'forestry':
                forestry_data = self.data_handler.get_forestry_data(config_name)
                if series_name in forestry_data.columns:
                    series_data = forestry_data[series_name]
                else:
                    # Try forestry variables as well
                    forestry_vars = self.data_handler.get_forestry_variables(config_name)
                    if series_name in forestry_vars.columns:
                        series_data = forestry_vars[series_name]
                    else:
                        supply_series = ", ".join(forestry_data.columns)
                        var_series = ", ".join(forestry_vars.columns) if not forestry_vars.empty else "none"
                        print(f"Series '{series_name}' not found.")
                        print(f"Available supply series: {supply_series}")
                        print(f"Available variable series: {var_series}")
                        return None
                    
            elif component == 'emissions':
                emissions_data = self.data_handler.get_emissions_data(config_name)
                # Typically emissions data comes as Year/Value pairs, so we need to process it
                if 'Year' in emissions_data.columns and 'Value' in emissions_data.columns:
                    series_data = emissions_data.set_index('Year')['Value']
                else:
                    print(f"Emissions data does not have expected Year/Value structure")
                    return None
                    
            else:
                print(f"Series data not available for component '{component}'")
                return None
            
            # Display the series
            if series_data is not None:
                print(f"\n=== {series_name} Series for {component.title()} in Scenario '{scenario_name}' ===")
                
                # Get number of rows to display
                num_rows = min(max_rows, len(series_data))
                
                # Format display based on component and series type
                print(f"\nFirst {num_rows} values (out of {len(series_data)} total):")
                
                for i, (year, value) in enumerate(series_data.items()[:num_rows]):
                    # Format based on series type
                    if series_name in ['base_volume', 'ccr_volume_1', 'ccr_volume_2', 'baseline_allocation', 'forestry_supply']:
                        print(f"  {year}: {value:,.0f} kt CO₂-e")
                    elif series_name in ['ccr_trigger_price_1', 'ccr_trigger_price_2', 'auction_reserve_price']:
                        print(f"  {year}: ${value:.2f}")
                    else:
                        print(f"  {year}: {value}")
                
                if len(series_data) > max_rows:
                    print(f"  ... {len(series_data) - max_rows} more values")
                    
                return series_data
                
        except Exception as e:
            print(f"Error retrieving series '{series_name}' from {component}: {str(e)}")
            return None

    def set_series(self, series_name: str, data: pd.Series, component: str, 
                scenario_index: int = 0, scenario_name: str = None) -> 'NZUpy':
        """
        Set a time series for a specific component and scenario.
        
        Args:
            series_name: Name of the time series to set
            data: New data for the time series (pandas Series)
            component: Component the series belongs to ('auction', 'industrial', 'forestry', 'emissions')
            scenario_index: Index of the scenario to modify (default: 0)
            scenario_name: Name of the scenario to modify (overrides scenario_index if provided)
            
        Returns:
            Self for method chaining
        """
        if not self._primed:
            raise ValueError("Model must be primed before setting series data. Call prime() first.")
        
        # Resolve scenario_index if scenario_name provided
        if scenario_name is not None:
            if scenario_name not in self.scenarios:
                raise ValueError(f"Unknown scenario name: '{scenario_name}'. Available scenarios: {', '.join(self.scenarios)}")
            scenario_index = self.scenarios.index(scenario_name)
        
        # Validate scenario_index
        if scenario_index < 0 or scenario_index >= len(self.scenarios):
            raise ValueError(f"Invalid scenario index: {scenario_index}. Valid range: 0-{len(self.scenarios)-1}")
        
        # Get scenario name for display
        scenario_name = self.scenarios[scenario_index]
        
        # Get component configuration
        component_config = self.component_configs[scenario_index]
        
        # Handle different components
        if component == 'auction':
            # Get auction data
            auction_data = self.data_handler.get_auction_data(component_config.auctions)
            
            # Validate series name
            valid_series = ['base_volume', 'auction_reserve_price', 'ccr_trigger_price_1', 
                           'ccr_trigger_price_2', 'ccr_volume_1', 'ccr_volume_2']
            if series_name not in valid_series:
                raise ValueError(f"Invalid auction series: '{series_name}'. Valid options: {', '.join(valid_series)}")
            
            # Update the series
            auction_data[series_name] = data
            
            # Update the data handler's auction data
            if hasattr(self.data_handler, 'auction_data'):
                self.data_handler.auction_data = auction_data
            
            print(f"Updated {component}.{series_name} for scenario '{scenario_name}'")
            
        elif component == 'industrial':
            # Get industrial allocation data
            industrial_data = self.data_handler.get_industrial_allocation_data(component_config.industrial_allocation)
            
            # Validate series name
            valid_series = ['baseline_allocation', 'activity_adjustment']
            if series_name not in valid_series:
                raise ValueError(f"Invalid industrial series: '{series_name}'. Valid options: {', '.join(valid_series)}")
            
            # Update the series
            industrial_data[series_name] = data
            
            # Update the data handler's industrial allocation data
            if hasattr(self.data_handler, 'industrial_allocation_data'):
                self.data_handler.industrial_allocation_data = industrial_data
            
            print(f"Updated {component}.{series_name} for scenario '{scenario_name}'")
            
        elif component == 'forestry':
            # Get forestry data
            forestry_data = self.data_handler.get_forestry_data(component_config.forestry)
            
            # Validate series name
            valid_series = ['forestry_supply']
            if series_name not in valid_series:
                raise ValueError(f"Invalid forestry series: '{series_name}'. Valid options: {', '.join(valid_series)}")
            
            # Update the series
            forestry_data[series_name] = data
            
            # Update the data handler's forestry data
            if hasattr(self.data_handler, 'forestry_data'):
                self.data_handler.forestry_data = forestry_data
            
            print(f"Updated {component}.{series_name} for scenario '{scenario_name}'")
            
        elif component == 'emissions':
            # Get emissions data
            emissions_data = self.data_handler.get_emissions_data(component_config.emissions)
            
            # Validate series name
            valid_series = ['emissions']
            if series_name not in valid_series:
                raise ValueError(f"Invalid emissions series: '{series_name}'. Valid options: {', '.join(valid_series)}")
            
            # Update the series
            if isinstance(emissions_data, pd.DataFrame) and 'Year' in emissions_data.columns:
                emissions_data.loc[emissions_data['Year'].isin(data.index), 'Value'] = data
            
            print(f"Updated {component}.{series_name} for scenario '{scenario_name}'")
            
        else:
            raise ValueError(f"Invalid component: '{component}'. Valid options: auction, industrial, forestry, emissions")
        
        return self

    def list_parameters(self, component: str, scenario_index: int = 0, scenario_name: str = None):
        """
        List all available parameters for a specific component and scenario.
        
        Args:
            component: Component to list parameters for
            scenario_index: Index of the scenario to inspect (default: 0)
            scenario_name: Name of the scenario to inspect (overrides scenario_index if provided)
            
        Returns:
            List of parameter names
        """
        if not self._primed:
            raise ValueError("Model must be primed before listing parameters. Call prime() first.")
        
        # Resolve scenario_index if scenario_name provided
        if scenario_name is not None:
            if scenario_name not in self.scenarios:
                raise ValueError(f"Unknown scenario name: '{scenario_name}'. Available scenarios: {', '.join(self.scenarios)}")
            scenario_index = self.scenarios.index(scenario_name)
        
        # Validate scenario_index
        if scenario_index < 0 or scenario_index >= len(self.scenarios):
            raise ValueError(f"Invalid scenario index: {scenario_index}. Valid range: 0-{len(self.scenarios)-1}")
        
        # Get scenario name for display
        scenario_name = self.scenarios[scenario_index]
        
        # Validate component
        valid_components = ['stockpile', 'auction', 'industrial', 'forestry', 'emissions', 'demand_model']
        if component not in valid_components:
            raise ValueError(f"Invalid component: '{component}'. Valid options: {', '.join(valid_components)}")
        
        # Define parameters for each component
        parameters = {
            'stockpile': ['initial_stockpile', 'initial_surplus', 'liquidity_factor', 
                        'discount_rate', 'payback_period', 'stockpile_usage_start_year', 
                        'stockpile_reference_year'],
            'demand_model': ['model_number', 'constant', 'reduction_to_t1', 'price', 
                            'discount_rate', 'forward_years', 'price_conversion_factor'],
            'auction': ['auction_reserve_price', 'ccr_trigger_price_1', 'ccr_trigger_price_2'],
            'industrial': [],  # No scalar parameters, only time series
            'forestry': [],    # No scalar parameters, only time series
            'emissions': []    # No scalar parameters, only time series
        }
        
        param_list = parameters.get(component, [])
        
        print(f"\nAvailable parameters for '{component}' in scenario '{scenario_name}':")
        if param_list:
            for param in param_list:
                print(f"  - {param}")
        else:
            print(f"  No scalar parameters available for {component}")
            if component in ['industrial', 'forestry', 'emissions']:
                print(f"  {component.title()} component primarily uses time series data.")
                print(f"  Use show_series() to view available series.")
        
        return param_list

    def list_series(self, component: str, scenario_index: int = 0, scenario_name: str = None):
        """
        List all available time series for a specific component and scenario.
        
        Args:
            component: Component to list time series for
            scenario_index: Index of the scenario to inspect (default: 0)
            scenario_name: Name of the scenario to inspect (overrides scenario_index if provided)
            
        Returns:
            List of series names
        """
        if not self._primed:
            raise ValueError("Model must be primed before listing series. Call prime() first.")
        
        # Resolve scenario_index if scenario_name provided
        if scenario_name is not None:
            if scenario_name not in self.scenarios:
                raise ValueError(f"Unknown scenario name: '{scenario_name}'. Available scenarios: {', '.join(self.scenarios)}")
            scenario_index = self.scenarios.index(scenario_name)
        
        # Validate scenario_index
        if scenario_index < 0 or scenario_index >= len(self.scenarios):
            raise ValueError(f"Invalid scenario index: {scenario_index}. Valid range: 0-{len(self.scenarios)-1}")
        
        # Get scenario name for display
        scenario_name = self.scenarios[scenario_index]
        
        # Validate component
        valid_components = ['auction', 'industrial', 'forestry', 'emissions']
        if component not in valid_components:
            raise ValueError(f"Invalid component: '{component}'. Valid options for time series: {', '.join(valid_components)}")
        
        # Get component configuration
        component_config = self.component_configs[scenario_index]
        
        # Get config name based on component type
        config_name = None
        if component == 'auction':
            config_name = component_config.auctions
        elif component == 'industrial':
            config_name = component_config.industrial_allocation
        elif component == 'forestry':
            config_name = component_config.forestry
        elif component == 'emissions':
            config_name = component_config.emissions
        
        if config_name is None:
            print(f"Warning: No configuration found for {component} in scenario {scenario_name}")
            return []
        
        try:
            # Get series data based on component type
            series_names = []
            
            if component == 'auction':
                auction_data = self.data_handler.get_auction_data(config_name)
                series_names = list(auction_data.columns)
                    
            elif component == 'industrial':
                industrial_data = self.data_handler.get_industrial_allocation_data(config_name)
                series_names = list(industrial_data.columns)
                    
            elif component == 'forestry':
                forestry_data = self.data_handler.get_forestry_data(config_name)
                forestry_vars = self.data_handler.get_forestry_variables(config_name)
                
                series_names = list(forestry_data.columns)
                if not forestry_vars.empty:
                    var_series = list(forestry_vars.columns)
                    series_names.extend(var_series)
                    
            elif component == 'emissions':
                # For emissions, we typically just have the emissions series
                series_names = ['emissions']
            
            print(f"\nAvailable time series for '{component}' in scenario '{scenario_name}':")
            for series in series_names:
                print(f"  - {series}")
                
            return series_names
                
        except Exception as e:
            print(f"Error retrieving series list for {component}: {str(e)}")
            return []

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
        print(f"\nDEBUG: Attempting to set price control config '{config_name}'")
        print(f"DEBUG: Target scenario_index: {scenario_index}, scenario_name: {scenario_name}")
        
        # Determine which scenario to use
        if scenario_index is not None:
            if scenario_index < 0 or scenario_index >= len(self.scenarios):
                raise ValueError(f"Invalid scenario_index: {scenario_index}")
            scenario = self.scenarios[scenario_index]
            print(f"DEBUG: Using scenario '{scenario}' from index {scenario_index}")
        elif scenario_name is not None:
            if scenario_name not in self.scenarios:
                raise ValueError(f"Unknown scenario_name: {scenario_name}")
            scenario = scenario_name
            print(f"DEBUG: Using scenario '{scenario}' from name")
        else:
            scenario = self.current_scenario
            print(f"DEBUG: Using current scenario '{scenario}'")
        
        # Store configuration name in scenario manager
        self.scenario_manager.set_price_control_config(scenario, config_name)
        
        print(f"DEBUG: Price control config set to '{config_name}' for scenario '{scenario}'")
        return self