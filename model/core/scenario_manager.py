"""
Scenario management for NZ ETS Supply-Demand Model.

This module provides functionality for managing model scenarios (different model runs)
and their associated component configurations (input settings).

Key terminology:
- Scenario: A specific model run with its own set of component configurations
- Config: A set of input parameters for a specific component (e.g., 'central', 'high', 'low')
"""

import pandas as pd
from typing import Dict, Any, List, Optional, Union, NewType
from model.config import ComponentConfig

# Type aliases for clarity
ScenarioName = NewType('ScenarioName', str)  # Name of a model run
ConfigName = NewType('ConfigName', str)  # Name of a component input configuration

class ScenarioManager:
    """
    Manages scenario configurations and parameter settings for the NZ ETS model.
    
    This class handles the creation and management of different scenarios,
    including their configurations and parameter settings.
    """
    
    def __init__(self, model: 'model.core.base_model.NZUpy'):
        """
        Initialise the scenario manager.
        
        Args:
            model: Reference to the parent NZUpy model instance
        """
        self.model = model
    
    def _initialise_scenarios(self):
        """Initialise scenario configurations with default component configs."""
        # Create a ComponentConfig for each model scenario if not already created
        if not hasattr(self.model, 'component_configs') or not self.model.component_configs:
            self.model.component_configs = [ComponentConfig() for _ in self.model.scenarios]
    
    def list_available_configs(self, component_type: str) -> List[str]:
        """
        List available predefined input configurations for a component type.
        
        Args:
            component_type: Type of component to list configs for ('emissions', 'auction', 
                        'industrial', 'forestry', 'demand_model', 'stockpile')
        
        Returns:
            List of available configuration names (e.g., ['central', 'high', 'low'])
        """
        return self.model.data_handler.list_available_configs(component_type)
    
    def use_config(self, scenario_index: int, component_type: str, config_name: str, model_number: Optional[int] = None) -> 'model.core.base_model.NZUpy':
        """
        Use a specific configuration for a component in a scenario.
        
        Args:
            scenario_index: Index of the scenario to modify
            component_type: Type of component to configure ('emissions', 'auction', etc.)
            config_name: Name of the configuration to use
            model_number: Optional model number for demand models
            
        Returns:
            NZUpy for method chaining
        """
        # Validate that the model has been primed
        if not self.model._primed:
            raise ValueError("Model must be primed before using configurations. Call prime() first.")
        
        # Validate scenario index
        if scenario_index < 0 or scenario_index >= len(self.model.scenarios):
            raise ValueError(f"Invalid scenario index: {scenario_index}. Valid range: 0-{len(self.model.scenarios)-1}")
        
        # Get scenario name for display
        scenario_name = self.model.scenarios[scenario_index]
        
        # Handle different component types
        if component_type == 'stockpile':
            # Set the stockpile configuration just like other components
            self.model.component_configs[scenario_index].stockpile = config_name
            print(f"Using {config_name} config for stockpile in scenario {scenario_index} ({scenario_name})")
            
            
        elif component_type == 'emissions':
            self.model.component_configs[scenario_index].emissions = config_name
            print(f"Set emissions configuration to '{config_name}' for scenario '{scenario_name}'")
            
        elif component_type == 'auction':
            self.model.component_configs[scenario_index].auctions = config_name
            print(f"Set auction configuration to '{config_name}' for scenario '{scenario_name}'")
            
        elif component_type == 'industrial':
            self.model.component_configs[scenario_index].industrial_allocation = config_name
            print(f"Set industrial allocation configuration to '{config_name}' for scenario '{scenario_name}'")
            
        elif component_type == 'forestry':
            self.model.component_configs[scenario_index].forestry = config_name
            print(f"Set forestry configuration to '{config_name}' for scenario '{scenario_name}'")
            
        elif component_type == 'demand_model':
            if model_number is not None:
                if model_number not in [1, 2]:
                    raise ValueError(f"Invalid demand model number: {model_number}. Valid options: 1, 2")
                self.model.component_configs[scenario_index].demand_model_number = model_number
                print(f"Set demand model number to {model_number} for scenario '{scenario_name}'")
            
            if config_name:
                self.model.component_configs[scenario_index].demand_sensitivity = config_name
                print(f"Set demand sensitivity to '{config_name}' for scenario '{scenario_name}'")
            
        else:
            raise ValueError(f"Unknown component type: '{component_type}'. Valid options: emissions, auction, industrial, forestry, demand_model")
        
        return self.model
    
    def use_central_configs(self, scenario_index: int) -> 'model.core.base_model.NZUpy':
        """
        Use central input configurations for all components in a specific model scenario.
        
        This is a convenience method that sets all components within the specified
        scenario to use their 'central' input configurations.
        
        Args:
            scenario_index: Index of the model scenario to modify
            
        Returns:
            NZUpy for method chaining
        """
        # Validate that the model has been primed
        if not self.model._primed:
            raise ValueError("Model must be primed before using configs. Call prime() first.")
        
        # Validate scenario index
        if scenario_index < 0 or scenario_index >= len(self.model.scenarios):
            raise ValueError(f"Invalid scenario index: {scenario_index}. Valid range: 0-{len(self.model.scenarios)-1}")
        
        # Use central configs for all components using the use_config method
        # for consistent case-insensitive matching
        self.use_config(scenario_index, 'emissions', 'central')
        self.use_config(scenario_index, 'auction', 'central')
        self.use_config(scenario_index, 'industrial', 'central')
        self.use_config(scenario_index, 'forestry', 'central')
        self.use_config(scenario_index, 'demand_model', 'central', model_number=2)
        self.use_config(scenario_index, 'stockpile', 'central')  # Add stockpile config
        
        # Also set model_params to central
        self.model.component_configs[scenario_index].model_params = "central"
        
        print(f"Using central configs for all components in model scenario {scenario_index} ({self.model.scenarios[scenario_index]})")
        
        return self.model

    def set_parameter(self, scenario_index: int, parameter_name: str, value: Any) -> 'model.core.base_model.NZUpy':
        """
        Set a model parameter for a specific scenario run.
        
        Args:
            scenario_index: Index of the model scenario to modify
            parameter_name: Name of the parameter to set. Valid options:
                - initial_stockpile: Initial stockpile value
                - initial_surplus: Initial surplus value
                - liquidity_factor: Liquidity factor (previously 'liquidity')
                - discount_rate: Discount rate for calculations
                - stockpile_usage_start_year: Year to start using stockpile
                - payback_period: Payback period in years
            value: Value to set for the parameter
        
        Returns:
            NZUpy for method chaining
        """
        # Validate that the model has been primed
        if not self.model._primed:
            raise ValueError("Model must be primed before setting parameters. Call prime() first.")
        
        # Validate scenario index
        if scenario_index < 0 or scenario_index >= len(self.model.scenarios):
            raise ValueError(f"Invalid scenario index: {scenario_index}. Valid range: 0-{len(self.model.scenarios)-1}")

        # Check which kind of parameter we're setting
        valid_parameters = [
            "initial_stockpile", 
            "initial_surplus", 
            "liquidity_factor", 
            "discount_rate",
            "stockpile_usage_start_year",
            "payback_period"
        ]
        
        if parameter_name in valid_parameters:
            # Validate parameter values
            if parameter_name == "liquidity_factor" and (value < 0 or value > 1):
                raise ValueError(f"liquidity_factor must be between 0 and 1, got {value}")
            elif parameter_name == "discount_rate" and (value < 0 or value > 1):
                raise ValueError(f"discount_rate must be between 0 and 1, got {value}")
            elif parameter_name == "stockpile_usage_start_year" and value < self.model.config.start_year:
                raise ValueError(f"stockpile_usage_start_year cannot be before model start year {self.model.config.start_year}")
            elif parameter_name in ["initial_stockpile", "initial_surplus"] and value < 0:
                raise ValueError(f"{parameter_name} cannot be negative, got {value}")
            elif parameter_name == "payback_period" and value <= 0:
                raise ValueError(f"payback_period must be positive, got {value}")
            
            # Store in the scenario config object for later use
            setattr(self.model.component_configs[scenario_index], parameter_name, value)
            print(f"Set parameter {parameter_name}={value} for scenario {scenario_index} ({self.model.scenarios[scenario_index]})")
        else:
            raise ValueError(f"Unknown parameter: {parameter_name}. Valid parameters: {', '.join(valid_parameters)}")
        
        return self.model
    
    def define_scenario_type(self, scenario_type: str = 'Single') -> 'model.core.base_model.NZUpy':
        """
        Define the type of scenario analysis to perform.
        
        Args:
            scenario_type: Type of scenario analysis to perform.
                Options:
                - 'Single': Run individual scenarios independently
                - 'Range': Run a range of scenarios for sensitivity analysis
                Default: 'Single'
            
        Returns:
            NZUpy for method chaining
        """
        valid_types = ['Single', 'Range']
        if scenario_type not in valid_types:
            raise ValueError(f"Invalid scenario type: {scenario_type}. Valid options: {valid_types}")
        
        self.model.scenario_type = scenario_type
        
        # If 'Range' is selected, configure the scenarios automatically
        if scenario_type == 'Range':
            # Define standard sensitivity scenarios with correct mapping to CSV names
            sensitivity_scenarios = {
                "95% Lower": "95pc_lower",
                "1 s.e lower": "stde_lower", 
                "central": "central",
                "1 s.e upper": "stde_upper",
                "95% Upper": "95pc_upper"
            }
            
            # If scenarios are already defined, warn the user
            if hasattr(self.model, '_scenarios_defined') and self.model._scenarios_defined:
                print(f"Warning: Redefining scenarios for '{scenario_type}' scenario type.")
            
            # Define the scenarios using display names
            self.model.define_scenarios(list(sensitivity_scenarios.keys()))
            
            # Store the mapping for later use
            self.model.sensitivity_scenario_mapping = sensitivity_scenarios
            
            # Store current demand model
            self.model.demand_model_number = getattr(self.model, 'demand_model_number', 2) # Default to model 2 (ENZ)
        
        print(f"Scenario type set to '{scenario_type}'")
        return self.model

    def set_demand_model(self, model_number: int) -> 'model.core.base_model.NZUpy':
        """
        Set which demand model number to use for all scenarios.
        
        This setting applies across all scenarios and affects which demand model's
        input configurations are used.
        
        Args:
            model_number: Demand model number (1 or 2)
                1 = MACC model
                2 = ENZ model (default)
            
        Returns:
            NZUpy for method chaining
        """
        if model_number not in [1, 2]:
            raise ValueError(f"Invalid demand model number: {model_number}. Valid options: 1, 2")
        
        self.model.demand_model_number = model_number
        
        # Update scenarios if they're already defined
        if hasattr(self.model, 'component_configs') and self.model.component_configs:
            for i, _ in enumerate(self.model.component_configs):
                if hasattr(self.model.component_configs[i], 'demand_model_number'):
                    self.model.component_configs[i].demand_model_number = model_number
        
        print(f"Demand model set to {model_number}")
        return self.model

    def configure_range_scenarios(self) -> 'model.core.base_model.NZUpy':
        """
        Configure scenarios for range/sensitivity analysis.
        
        This method sets up multiple model scenarios with different input configurations
        to analyse sensitivity to input assumptions. Each scenario uses a different
        set of component configurations (e.g., low, central, high).
        
        Returns:
            NZUpy for method chaining
        """
        if not hasattr(self.model, 'scenario_type') or self.model.scenario_type != 'Range':
            raise ValueError("This method should only be called for 'Range' scenario type")
        
        if not hasattr(self.model, '_scenarios_defined') or not self.model._scenarios_defined:
            raise ValueError("Scenarios must be defined first. Call define_scenarios() first.")
        
        # Default to model 2 if not specified
        model_number = getattr(self.model, 'demand_model_number', 2)
        
        # Configure each scenario
        for i, scenario in enumerate(self.model.scenarios):
            # Start with central settings for all components
            self.use_central_configs(i)
            
            # Then override the demand model to match the scenario name
            self.use_config(i, 'demand_model', scenario, model_number=model_number)
        
        print(f"Configured all scenarios for 'Range' run with demand model {model_number}")
        return self.model
        
    def _initialise_scenario_components(self, scenario_config):
        """
        Initialise components for a specific scenario with their input configurations.
        
        Args:
            scenario_config: Configuration object containing which input config
                           each component should use in this scenario
        """
        # Get data for current scenarios
        try:
            auction_data = self.model.data_handler.get_auction_data(config=scenario_config.auctions)
        except Exception as e:
            raise ValueError(f"Failed to load auction data for config '{scenario_config.auctions}': {e}")
            
        try:
            ia_data = self.model.data_handler.get_industrial_allocation_data(config=scenario_config.industrial_allocation)
        except Exception as e:
            raise ValueError(f"Failed to load industrial allocation data for config '{scenario_config.industrial_allocation}': {e}")
            
        try:
            forestry_data = self.model.data_handler.get_forestry_data(config=scenario_config.forestry)
        except Exception as e:
            raise ValueError(f"Failed to load forestry data for config '{scenario_config.forestry}': {e}")
            
        try:
            emissions_data = self.model.data_handler.get_emissions_data(config=scenario_config.emissions)
        except Exception as e:
            raise ValueError(f"Failed to load emissions data for config '{scenario_config.emissions}': {e}")
        
        # Get forestry variables data
        try:
            forestry_variables = self.model.data_handler.get_forestry_variables(config=scenario_config.forestry)
        except Exception as e:
            raise ValueError(f"Failed to load forestry variables for config '{scenario_config.forestry}': {e}")
        
        # Get model parameters
        try:
            model_params = self.model.data_handler.get_model_parameters(config=scenario_config.model_params)
        except Exception as e:
            raise ValueError(f"Failed to load model parameters for config '{scenario_config.model_params}': {e}")
        
        # Get demand model parameters
        try:
            demand_model_params = self.model.data_handler.get_demand_model(
                config=scenario_config.demand_sensitivity,
                model_number=scenario_config.demand_model_number
            )
        except Exception as e:
            raise ValueError(f"Failed to load demand model parameters for config '{scenario_config.demand_sensitivity}' and model number {scenario_config.demand_model_number}: {e}")
        
        # Get stockpile parameters with explicit hierarchy
        stockpile_params = {}

        try:
            # STEP 1: Load base parameters from config file or defaults
            base_params = {}
            
            # First try to get parameters from config file if specified
            if scenario_config.stockpile:
                try:
                    config_params = self.model.data_handler.get_stockpile_parameters(scenario_config.stockpile)
                    base_params = {
                        'initial_stockpile': config_params['initial_stockpile'],
                        'initial_surplus': config_params['initial_surplus'],
                        'liquidity_factor': config_params['liquidity_factor'],
                        'payback_period': config_params['payback_period'],
                        'stockpile_usage_start_year': config_params['stockpile_usage_start_year'],
                        'discount_rate': config_params['discount_rate'],
                        'stockpile_reference_year': config_params.get('stockpile_reference_year', min(self.model.years) - 1)
                    }
                except Exception as e:
                    print(f"Warning: Failed to load stockpile parameters from config '{scenario_config.stockpile}': {e}")
                    print("Falling back to model parameters")
            
            # If no config specified or config loading failed, use model parameters
            if not base_params:
                param_mappings = {
                    'initial_stockpile': 'stockpile_start',
                    'initial_surplus': 'surplus_start',
                    'liquidity_factor': 'liquidity_factor',
                    'payback_period': 'payback_years',
                    'stockpile_usage_start_year': 'start_year',
                    'discount_rate': 'discount_rate'
                }
                
                for scenario_name, model_param_name in param_mappings.items():
                    value = model_params.get(model_param_name)
                    if value is None:
                        raise ValueError(f"Required parameter '{scenario_name}' not found in model parameters")
                    
                    # Convert to appropriate type
                    if scenario_name in ['payback_period', 'stockpile_usage_start_year']:
                        base_params[scenario_name] = int(value)
                    else:
                        base_params[scenario_name] = float(value)
                
                # Set default reference year
                base_params['stockpile_reference_year'] = min(self.model.years) - 1
            
            # STEP 2: Override with any explicitly set parameters
            param_list = ['initial_stockpile', 'initial_surplus', 'liquidity_factor', 
                         'discount_rate', 'payback_period', 'stockpile_usage_start_year',
                         'stockpile_reference_year']
            
            for param in param_list:
                custom_value = getattr(scenario_config, param, None)
                if custom_value is not None:
                    # Validate parameter values before overriding
                    if param == 'liquidity_factor' and not 0 <= custom_value <= 1:
                        raise ValueError(f"liquidity_factor must be between 0 and 1, got {custom_value}")
                    elif param == 'discount_rate' and not 0 <= custom_value <= 1:
                        raise ValueError(f"discount_rate must be between 0 and 1, got {custom_value}")
                    elif param == 'initial_stockpile' and custom_value < 0:
                        raise ValueError(f"initial_stockpile cannot be negative, got {custom_value}")
                    elif param == 'initial_surplus' and custom_value < 0:
                        raise ValueError(f"initial_surplus cannot be negative, got {custom_value}")
                    elif param == 'initial_surplus' and custom_value > base_params['initial_stockpile']:
                        raise ValueError(f"initial_surplus ({custom_value}) cannot exceed initial_stockpile ({base_params['initial_stockpile']})")
                    elif param == 'payback_period' and custom_value <= 0:
                        raise ValueError(f"payback_period must be positive, got {custom_value}")
                    elif param == 'stockpile_usage_start_year' and custom_value < self.model.config.start_year:
                        raise ValueError(f"stockpile_usage_start_year cannot be before model start year {self.model.config.start_year}")
                    
                    # Use the custom parameter instead of the config/default value
                    base_params[param] = custom_value
            
            # STEP 3: Set final parameters
            stockpile_params = base_params
            
        except Exception as e:
            raise ValueError(f"Failed to load stockpile parameters: {e}")
        
        # Initialise components with proper error handling
        try:
            from model.supply.stockpile import StockpileSupply
            
            self.model.stockpile = StockpileSupply(
                years=self.model.years,
                extended_years=self.model.extended_years,
                forestry_variables=forestry_variables,
                **stockpile_params
            )
        except Exception as e:
            raise ValueError(f"Failed to initialise stockpile component: {e}")
        
        try:
            from model.supply.auction import AuctionSupply
            
            self.model.auction = AuctionSupply(
                years=self.model.years,
                auction_data=auction_data
            )
        except Exception as e:
            raise ValueError(f"Failed to initialise auction component: {e}")
        
        try:
            from model.supply.industrial import IndustrialAllocation
            
            self.model.industrial = IndustrialAllocation(
                years=self.model.years,
                ia_data=ia_data
            )
        except Exception as e:
            raise ValueError(f"Failed to initialise industrial allocation component: {e}")
        
        try:
            from model.supply.forestry import ForestrySupply
            
            self.model.forestry = ForestrySupply(
                years=self.model.years,
                forestry_data=forestry_data
            )
        except Exception as e:
            raise ValueError(f"Failed to initialise forestry component: {e}")
        
        # Store component results for later use
        self.model.auction_results = {}
        self.model.industrial_results = {}
        self.model.stockpile_results = {}
        self.model.forestry_results = {}
        self.model.price_response_results = {}
        self.model.emissions_results = {}
        
        # Create place for base supply
        self.model.base_supply = pd.Series(index=self.model.years, dtype=float)
        
        # Map scenario name to what EmissionsDemand expects
        emissions_scenario = scenario_config.emissions
        # The EmissionsDemand class expects specific scenario names
        # If the scenario name is 'central', map it to 'central' for proper recognition
        if emissions_scenario and emissions_scenario.lower() == 'central':
            emissions_scenario = 'central'
        
        # Initialise emissions demand with emissions data
        try:
            from model.demand.emissions import EmissionsDemand
            
            self.model.emissions = EmissionsDemand(
                years=self.model.years,
                emissions_data=self.model.data_handler.get_emissions_data(config=emissions_scenario),
                config_name=emissions_scenario
            )
        except Exception as e:
            raise ValueError(f"Failed to initialise emissions component: {e}")
        
        # Initialise price response with demand model parameters
        try:
            from model.demand.price_response import PriceResponse
            
            self.model.price_response = PriceResponse(
                years=self.model.years,
                demand_model_params=demand_model_params
            )
        except Exception as e:
            raise ValueError(f"Failed to initialise price response component: {e}")
        
        # Important: Update config with relevant parameters for compatibility
        self.model.config.liquidity_factor = float(stockpile_params['liquidity_factor'])
        self.model.config.initial_stockpile = float(stockpile_params['initial_stockpile'])
        self.model.config.initial_surplus = float(stockpile_params['initial_surplus'])

    def set_price_control_config(self, scenario_name, config_name):
        """Set price control configuration for a specific scenario."""
        
        # Get the scenario index
        scenario_index = self.model.scenarios.index(scenario_name)

        # Store the config name in the scenario's component config
        self.model.component_configs[scenario_index].price_control_config = config_name

        # Verify the setting
        stored_config = getattr(self.model.component_configs[scenario_index], 'price_control_config', None)
