"""
Configuration module for the NZUpy model.
"""


from typing import Dict, Optional
from dataclasses import dataclass, field

# Type aliases for clarity
ConfigName = str       # Name of a component input configuration (e.g., "central", "CCC_CPR")
ScenarioName = str     # Name of a model run/scenario (e.g., "Low Carbon Price")
Year = int             # Calendar year


@dataclass
class ModelConfig:
    """Temporal, price-control, and optimiser configuration for the NZ ETS model."""

    start_year: Year = 2024
    end_year: Year = 2050
    price_control_values: Dict[Year, float] = field(default_factory=dict)

    # Optimiser settings
    coarse_step: int = 10
    fine_step: int = 1
    max_rate: int = 200
    max_iterations: int = 5
    debug: bool = False
    penalise_shortfalls: bool = False
    

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
        industrial: Industrial allocation configuration name (default: "central")
        auction: Auction configuration name (e.g., "central")
        forestry: Forestry configuration name (e.g., "low", "central", "high")
        demand_sensitivity: Demand sensitivity configuration (e.g., "central", "95pc_lower", "stde_upper")
        demand_model_number: Demand model selection (1=MACC model, 2=ENZ model)
        stockpile: Stockpile configuration name (e.g., "central", "high", "low")
    """

    # Input configuration selection for each component
    model_params: ConfigName = "central"
    emissions: ConfigName = "central"
    industrial: ConfigName = "central"
    auction: ConfigName = "central"
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

    # Forestry mode settings
    forestry_mode: str = 'exogenous'            # 'exogenous' or 'endogenous'
    manley_sensitivity: str = 'central'          # 'low', 'central', 'high' — selects f and LMV
    forestry_price_assumption: str = 'future'    # 'future' or 'current'

    # Manley parameter overrides (None = use CSV value)
    manley_f: Optional[float] = None
    manley_LMV: Optional[float] = None
    manley_LUC_limit: Optional[float] = None
    forestry_discount_rate: Optional[float] = None
    forestry_forward_years: Optional[int] = None

    # Pricing mode settings
    pricing_mode: str = 'optimised'             # 'optimised' | 'fixed_path' | 'fixed_rate'
    price_path: Optional[object] = None         # pd.Series of prices, used when pricing_mode='fixed_path'
    price_change_rate: Optional[float] = None   # scalar rate, used when pricing_mode='fixed_rate'

    # Per-scenario price control override (set via fill('price_control', pd.Series(...)))
    price_control_override: Optional[object] = None  # pd.Series mapping year → control value

