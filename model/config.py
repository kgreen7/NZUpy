"""
Configuration module for the NZ ETS model.

This module provides functionality for loading, validating, and managing
configuration parameters for the NZ ETS model.
"""

import yaml
import json
import os
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import pandas as pd

# Type aliases for clarity
ConfigName = str       # Name of a component input configuration (e.g., "central", "CCC_CPR")
ScenarioName = str     # Name of a model run/scenario (e.g., "Low Carbon Price")
Year = int             # Calendar year

@dataclass
class SupplyConfig:
    """Configuration for supply components."""
    
    # Auction parameters
    auction_volumes: Dict[Year, float] = field(default_factory=dict)

    # Forestry parameters
    forestry_method: str = "static"  # Options: "static", "dynamic"
    forestry_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DemandConfig:
    """Configuration for demand components."""
    
    # Base emissions parameters
    base_emissions: Dict[Year, float] = field(default_factory=dict)
    
    # Price sensitivity parameters
    price_elasticity: float = 1
    price_response_lag: int = 10


@dataclass
class OptimiserConfig:
    """Configuration for the optimisation process."""
    
    coarse_step: int = 10
    fine_step: int = 1
    max_rate: int = 200
    find_min: bool = True
    max_iterations: int = 5
    debug: bool = False
    penalise_shortfalls: bool = True


@dataclass
class ModelConfig:
    """
    Full configuration for the NZ ETS model.
    
    This class represents the complete configuration for the model,
    including all parameters and settings for each component.
    """
    
    # Required parameters (no defaults)
    initial_stockpile: float
    initial_surplus: float  
    liquidity_factor: float
    
    # Parameters with defaults
    start_year: Year = 2024
    end_year: Year = 2050
    
    # Price control parameters
    price_control_values: Dict[Year, float] = field(default_factory=dict)
    price_control_default: float = 1.0     # Default multiplier applied to all future years
    
    # Component configurations
    supply: SupplyConfig = field(default_factory=SupplyConfig)
    demand: DemandConfig = field(default_factory=DemandConfig)
    optimiser: OptimiserConfig = field(default_factory=OptimiserConfig)
    
    # Input configuration selection
    input_config_set: ConfigName = "central"  # Options: "central", "CCC_CPR", "CCC_DP", etc.
    
    # Historical prices
    historical_prices: Dict[Year, float] = field(default_factory=dict)

    # Additional parameters
    additional_params: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class ComponentConfig:
    """
    Configuration for component input selection in the NZ ETS model.
    
    This class provides a centralised way to specify which input configurations to use
    for different components of the model. Each component can use different input
    configurations (e.g., central estimates, high/low variants, or specific policy settings).
    
    Attributes:
        model_params: Model parameters configuration name (default: "central")
        emissions: Emissions configuration name (e.g., "CCC_CPR", "CCC_DP", "central")
        industrial_allocation: Industrial allocation configuration name (default: "central")
        auctions: Auction configuration name (e.g., "central")
        forestry: Forestry configuration name (e.g., "low", "central", "high")
        demand_sensitivity: Demand sensitivity configuration (e.g., "central", "95pc_lower", "stde_upper")
        demand_model_number: Demand model selection (1=MACC model, 2=ENZ model)
    """
    
    # Input configuration selection for each component
    model_params: ConfigName = "central"
    emissions: ConfigName = "central"
    industrial_allocation: ConfigName = "central"
    auctions: ConfigName = "central"
    forestry: ConfigName = "central"
    
    # Demand model selection
    demand_sensitivity: ConfigName = "central"
    demand_model_number: int = 2 #ENZ model
    
    # Optional stockpile parameters (these override datahandler values when set)
    initial_stockpile: Optional[float] = None
    initial_surplus: Optional[float] = None
    liquidity_factor: Optional[float] = None
    discount_rate: Optional[float] = None
    payback_period: Optional[int] = None
    stockpile_usage_start_year: Optional[int] = None


class ConfigManager:
    """
    Manager for loading, validating, and accessing configuration.
    
    This class is responsible for loading configuration from files and
    providing access to configuration parameters.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialise the configuration manager.
        
        Args:
            config: Initial configuration. If None, a default configuration is created.
        """
        self.config = config or ModelConfig(
            initial_stockpile=0.0,  # These will be overridden when loading from file
            initial_surplus=0.0,
            liquidity_factor=0.0,
        )
    
    def load_from_file(self, file_path: str) -> None:
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to the configuration file (YAML or JSON).
        
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported or if validation fails.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.yaml', '.yml']:
            with open(file_path, 'r') as file:
                config_dict = yaml.safe_load(file)
        elif file_ext == '.json':
            with open(file_path, 'r') as file:
                config_dict = json.load(file)
        else:
            raise ValueError(f"Unsupported configuration file format: {file_ext}")
        
        self._update_from_dict(config_dict)
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save the current configuration to a file.
        
        Args:
            file_path: Path to save the configuration file (YAML or JSON).
        
        Raises:
            ValueError: If the file format is not supported.
        """
        config_dict = asdict(self.config)
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.yaml', '.yml']:
            with open(file_path, 'w') as file:
                yaml.dump(config_dict, file, default_flow_style=False)
        elif file_ext == '.json':
            with open(file_path, 'w') as file:
                json.dump(config_dict, file, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {file_ext}")
    
    def _update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Update the configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values.
        """
        # Update top-level parameters
        for key, value in config_dict.items():
            if hasattr(self.config, key) and key not in ['supply', 'demand', 'optimiser']:
                setattr(self.config, key, value)
        
        # Update supply configuration
        if 'supply' in config_dict:
            for key, value in config_dict['supply'].items():
                if hasattr(self.config.supply, key):
                    setattr(self.config.supply, key, value)
        
        # Update demand configuration
        if 'demand' in config_dict:
            for key, value in config_dict['demand'].items():
                if hasattr(self.config.demand, key):
                    setattr(self.config.demand, key, value)
        
        # Update optimiser configuration
        if 'optimiser' in config_dict:
            for key, value in config_dict['optimiser'].items():
                if hasattr(self.config.optimiser, key):
                    setattr(self.config.optimiser, key, value)