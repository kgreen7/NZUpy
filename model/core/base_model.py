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
        nze.set_parameter(0, "initial_stockpile", 159902)
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
        # Create default price control series
        self.price_control_parameter = pd.Series(
            index=self.years, 
            data=self.config.price_control_default
        )
        
        # Try to get price control data from historical data manager
        if hasattr(self.data_handler, 'historical_manager'):
            for year in self.years:
                price_control = self.data_handler.get_price_control(year)
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
        
        # Print informative message
        print(f"Defined {len(scenario_names)} scenarios: {', '.join(scenario_names)}")
        
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
    
    def set_parameter(self, scenario_index: int, parameter_name: str, value: Any) -> 'NZUpy':
        """
        Set a model parameter for a specific scenario.
        
        Args:
            scenario_index: Index of the scenario to modify
            parameter_name: Name of the parameter to set. Valid options:
                - initial_stockpile: Initial stockpile value
                - initial_surplus: Initial surplus value
                - liquidity_factor: Liquidity factor (previously 'liquidity')
                - discount_rate: Discount rate for calculations
                - stockpile_usage_start_year: Year to start using stockpile
                - payback_period: Payback period in years
            value: Value to set for the parameter
        
        Returns:
            Self for method chaining
        """
        return self.scenario_manager.set_parameter(scenario_index, parameter_name, value)
    
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
        return self.scenario_manager.use_scenario(scenario_index, component_type, config_name, model_number)
    
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
            # Check if required scenario types are set
            if not scenario_config.emissions:
                raise ValueError(f"Emissions scenario not set for scenario {i}. Call use_config() first.")
            
            if not scenario_config.auctions:
                raise ValueError(f"Auction scenario not set for scenario {i}. Call use_config() first.")
            
            if not scenario_config.industrial_allocation:
                raise ValueError(f"Industrial allocation scenario not set for scenario {i}. Call use_config() first.")
            
            if not scenario_config.forestry:
                raise ValueError(f"Forestry scenario not set for scenario {i}. Call use_config() first.")
        
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

    def get_price_control_parameters(self) -> pd.Series:
        """
        Get the current price control parameters.
        
        Returns:
            Series of price control parameters indexed by year.
        """
        return self.price_control_parameter.copy()
    
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