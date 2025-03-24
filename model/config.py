"""
Configuration module for the NZUpy model.

This module provides functionality for loading, validating, and managing
configuration parameters for the NZUpy model. NOTE: Mostly phased out.
"""


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




@dataclass
class DemandConfig:
    """Configuration for demand components."""
    
    # Base emissions parameters
    base_emissions: Dict[Year, float] = field(default_factory=dict)




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
        stockpile: Stockpile configuration name (e.g., "central", "high", "low")
    """
    
    # Input configuration selection for each component
    model_params: ConfigName = "central"
    emissions: ConfigName = "central"
    industrial_allocation: ConfigName = "central"
    auctions: ConfigName = "central"
    forestry: ConfigName = "central"
    stockpile: ConfigName = "central"
    
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
    stockpile_reference_year: Optional[int] = None

