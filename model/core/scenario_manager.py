"""
Scenario management for NZUpy Model.

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
        
        # Store existing configurations if they exist
        existing_configs = {}
        if hasattr(self.model, 'component_configs') and self.model.component_configs:
            for i, config in enumerate(self.model.component_configs):
                existing_configs[i] = {
                    'forestry': getattr(config, 'forestry', 'central'),
                    'auction': getattr(config, 'auction', 'central'),
                    'industrial': getattr(config, 'industrial', 'central'),
                    'emissions': getattr(config, 'emissions', 'central'),
                    'stockpile': getattr(config, 'stockpile', 'central'),
                    'demand_sensitivity': getattr(config, 'demand_sensitivity', 'central'),
                    'demand_model_number': getattr(config, 'demand_model_number', 2)
                }
        
        # Configure each scenario
        for i, scenario in enumerate(self.model.scenarios):
            # Set all components to central defaults
            cfg = self.model.component_configs[i]
            cfg.emissions = 'central'
            cfg.auction = 'central'
            cfg.industrial = 'central'
            cfg.forestry = 'central'
            cfg.demand_sensitivity = 'central'
            cfg.demand_model_number = model_number
            cfg.stockpile = 'central'
            cfg.model_params = 'central'

            # Restore any non-central configurations set before this call
            if i in existing_configs:
                for component_type, value in existing_configs[i].items():
                    if value != 'central':
                        if component_type == 'demand_model_number':
                            cfg.demand_model_number = value
                        elif component_type == 'demand_sensitivity':
                            cfg.demand_sensitivity = value
                        else:
                            setattr(cfg, component_type, value)

            # Override the demand sensitivity to match the scenario name
            cfg.demand_sensitivity = scenario
            cfg.demand_model_number = model_number

        # Mark range scenarios as configured
        self.model._range_scenarios_configured = True
        
        return self.model
        
    def _initialise_scenario_components(self, component_config, scenario_name=None):
        """
        Initialise components for a specific scenario with their input configurations.
        
        Args:
            component_config: Configuration object containing which input config
                           each component should use in this scenario
            scenario_name: The name of the scenario being initialised (for scenario-specific data)
        """
        
        # Get data for current scenarios, using scenario-specific data where available
        try:
            # First check if we have scenario-specific auction data
            if (scenario_name in self.model.data_handler.scenario_data and 
                'auction' in self.model.data_handler.scenario_data[scenario_name]):
                auction_data = self.model.data_handler.scenario_data[scenario_name]['auction']
            else:
                # Fall back to config-based data
                auction_data = self.model.data_handler.get_auction_data(
                    config=component_config.auction,
                    scenario_name=scenario_name
                )
        except Exception as e:
            raise ValueError(f"Failed to load auction data for config '{component_config.auction}': {e}")

        try:
            # First check if we have scenario-specific industrial allocation data
            if (scenario_name in self.model.data_handler.scenario_data and
                'industrial' in self.model.data_handler.scenario_data[scenario_name]):
                ia_data = self.model.data_handler.scenario_data[scenario_name]['industrial']
            else:
                # Fall back to config-based data
                ia_data = self.model.data_handler.get_industrial_allocation_data(
                    config=component_config.industrial,
                    scenario_name=scenario_name
                )
        except Exception as e:
            raise ValueError(f"Failed to load industrial allocation data for config '{component_config.industrial}': {e}")
            
        try:
            # First check if we have scenario-specific forestry data
            if (scenario_name in self.model.data_handler.scenario_data and 
                'forestry' in self.model.data_handler.scenario_data[scenario_name]):
                forestry_data = self.model.data_handler.scenario_data[scenario_name]['forestry']
            else:
                # Load forestry data using the explicitly set config
                forestry_data = self.model.data_handler.get_forestry_data(
                    config=component_config.forestry,
                    scenario_name=scenario_name
                )
                
                # Ensure this data persists by explicitly storing it
                if scenario_name not in self.model.data_handler.scenario_data:
                    self.model.data_handler.scenario_data[scenario_name] = {}
                self.model.data_handler.scenario_data[scenario_name]['forestry'] = forestry_data.copy()
        except Exception as e:
            raise ValueError(f"Failed to load forestry data for config '{component_config.forestry}': {e}")
            
        try:
            # First check if we have scenario-specific emissions data
            if (scenario_name in self.model.data_handler.scenario_data and 
                'emissions' in self.model.data_handler.scenario_data[scenario_name]):
                emissions_data = self.model.data_handler.scenario_data[scenario_name]['emissions']
            else:
                # Fall back to config-based data
                emissions_data = self.model.data_handler.get_emissions_data(
                    config=component_config.emissions,
                    scenario_name=scenario_name
                )
        except Exception as e:
            raise ValueError(f"Failed to load emissions data for config '{component_config.emissions}': {e}")
        
        # Load endogenous forestry data if needed (Task 3/8)
        forestry_mode = getattr(component_config, 'forestry_mode', 'exogenous')
        historical_removals = None
        yield_increments = None
        afforestation_projections = None
        manley_params = None

        if forestry_mode == 'endogenous':
            try:
                forestry_config = component_config.forestry or 'central'
                historical_removals = self.model.data_handler.get_historical_removals(
                    config=forestry_config
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to load historical_removals for config '{component_config.forestry}': {e}"
                )
            try:
                yield_increments = self.model.data_handler.get_yield_increments()
            except Exception as e:
                raise ValueError(f"Failed to load yield increments: {e}")
            try:
                # Afforestation projections use the forestry config (low/central/high)
                # if available, otherwise fall back to 'central'
                proj_config = component_config.forestry or 'central'
                afforestation_projections = self.model.data_handler.get_afforestation_projections(
                    config=proj_config
                )
            except Exception:
                try:
                    afforestation_projections = self.model.data_handler.get_afforestation_projections(
                        config='central'
                    )
                except Exception as e:
                    raise ValueError(f"Failed to load afforestation projections: {e}")
            try:
                manley_sensitivity = getattr(component_config, 'manley_sensitivity', 'central')
                manley_params = self.model.data_handler.get_manley_parameters(
                    config=manley_sensitivity
                )
            except Exception as e:
                raise ValueError(f"Failed to load Manley parameters: {e}")

        # Get forestry variables data (held/surrender → fed to StockpileSupply)
        # Task 8: In endogenous mode, use historical_removals held/surrender instead
        #         of the combined values from removals.csv.
        try:
            if forestry_mode == 'endogenous' and historical_removals is not None:
                # Build forestry_variables from historical_removals columns
                forestry_variables = historical_removals.rename(columns={
                    'historic_forestry_tradeable': 'forestry_tradeable',
                    'historic_forestry_held': 'forestry_held',
                    'historic_forestry_surrender': 'forestry_surrender',
                })
            elif (scenario_name in self.model.data_handler.scenario_data and
                  'forestry_variables' in self.model.data_handler.scenario_data[scenario_name]):
                forestry_variables = self.model.data_handler.scenario_data[scenario_name]['forestry_variables']
            else:
                forestry_variables = self.model.data_handler.get_forestry_variables(
                    config=component_config.forestry,
                    scenario_name=scenario_name
                )
        except Exception as e:
            raise ValueError(f"Failed to load forestry variables for config '{component_config.forestry}': {e}")
        
        # Get model parameters with scenario-specific data
        try:
            # First check if we have scenario-specific model parameters
            if (scenario_name in self.model.data_handler.scenario_data and 
                'model_params' in self.model.data_handler.scenario_data[scenario_name]):
                model_params = self.model.data_handler.scenario_data[scenario_name]['model_params']
            else:
                # Fall back to config-based data
                model_params = self.model.data_handler.get_model_parameters(
                    config=component_config.model_params,
                    scenario_name=scenario_name
                )
        except Exception as e:
            raise ValueError(f"Failed to load model parameters for config '{component_config.model_params}': {e}")
        
        # Get demand model parameters with scenario-specific data
        try:
            # First check if we have scenario-specific demand model parameters
            if (scenario_name in self.model.data_handler.scenario_data and 
                'demand_model' in self.model.data_handler.scenario_data[scenario_name]):
                demand_model_params = self.model.data_handler.scenario_data[scenario_name]['demand_model']
            else:
                # Fall back to config-based data
                demand_model_params = self.model.data_handler.get_demand_model(
                    config=component_config.demand_sensitivity,
                    model_number=getattr(component_config, 'demand_model_number', 2),
                    scenario_name=scenario_name
                )
        except Exception as e:
            raise ValueError(f"Failed to load demand model parameters for config '{component_config.demand_sensitivity}' and model number {getattr(component_config, 'demand_model_number', 2)}: {e}")
        
        # Get stockpile parameters, ensuring we at least try the central config
        try:
            stockpile_config = component_config.stockpile or "central"
            # First check if we have scenario-specific stockpile parameters
            if (scenario_name in self.model.data_handler.scenario_data and 
                'stockpile' in self.model.data_handler.scenario_data[scenario_name]):
                stockpile_params = self.model.data_handler.scenario_data[scenario_name]['stockpile']
            else:
                # Fall back to config-based data
                stockpile_params = self.model.data_handler.get_stockpile_parameters(
                    config=stockpile_config,
                    overrides={
                        'initial_stockpile': getattr(component_config, 'initial_stockpile', None),
                        'initial_surplus': getattr(component_config, 'initial_surplus', None),
                        'liquidity_factor': getattr(component_config, 'liquidity_factor', None),
                        'discount_rate': getattr(component_config, 'discount_rate', None),
                        'payback_period': getattr(component_config, 'payback_period', None),
                        'stockpile_usage_start_year': getattr(component_config, 'stockpile_usage_start_year', None),
                        'stockpile_reference_year': getattr(component_config, 'stockpile_reference_year', None)
                    },
                    scenario_name=scenario_name,
                    model_start_year=self.model.config.start_year
                )
        except Exception as e:
            raise ValueError(f"Failed to load stockpile parameters for config '{stockpile_config}': {e}")
        
        # Import required component classes
        from model.supply.stockpile import StockpileSupply
        from model.supply.auction import AuctionSupply
        from model.supply.industrial import IndustrialAllocation
        from model.supply.forestry import ForestrySupply
        from model.demand.emissions import EmissionsDemand
        from model.demand.price_response import PriceResponse
        
        # Initialise components with proper error handling
        try:
            # Initialise StockpileSupply with proper parameters from config
            self.model.stockpile = StockpileSupply(
                years=self.model.years,
                extended_years=getattr(self.model, 'extended_years', None),
                stockpile_params=stockpile_params,
                forestry_variables=forestry_variables
            )
        except Exception as e:
            raise ValueError(f"Failed to initialise stockpile component: {e}")
        
        try:
            self.model.auction = AuctionSupply(
                years=self.model.years,
                auction_data=auction_data
            )
        except Exception as e:
            raise ValueError(f"Failed to initialise auction component: {e}")
        
        try:
            self.model.industrial = IndustrialAllocation(
                years=self.model.years,
                ia_data=ia_data
            )
        except Exception as e:
            raise ValueError(f"Failed to initialise industrial allocation component: {e}")
        
        try:
            self.model.forestry = ForestrySupply(
                years=self.model.years,
                forestry_data=forestry_data,
                mode=forestry_mode,
                manley_config=component_config if forestry_mode == 'endogenous' else None,
                historical_removals=historical_removals,
                yield_increments=yield_increments,
                afforestation_projections=afforestation_projections,
                manley_params=manley_params,
            )
        except Exception as e:
            raise ValueError(f"Failed to initialise forestry component: {e}")
        
        try:
            self.model.emissions = EmissionsDemand(
                years=self.model.years,
                emissions_data=emissions_data,
                config_name=component_config.emissions
            )
        except Exception as e:
            raise ValueError(f"Failed to initialise emissions component: {e}")
        
        try:
            self.model.price_response = PriceResponse(
                years=self.model.years,
                demand_model_params=demand_model_params
            )
        except Exception as e:
            raise ValueError(f"Failed to initialise price response component: {e}")
        
        # Store scenario name for reference
        self.model._active_scenario_name = scenario_name

    def set_price_control_config(self, scenario_name, config_name):
        """Set price control configuration for a specific scenario."""
        
        # Get the scenario index
        scenario_index = self.model.scenarios.index(scenario_name)

        # Store the config name in the scenario's component config
        self.model.component_configs[scenario_index].price_control_config = config_name

        # Verify the setting
        stored_config = getattr(self.model.component_configs[scenario_index], 'price_control_config', None)
