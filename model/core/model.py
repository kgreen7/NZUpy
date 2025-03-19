"""
Core model for NZ ETS Supply-Demand Model.

This module is maintained for backward compatibility.
It re-exports the NZUpy class from base_model.py.
"""

# Import the NZUpy class from its new location
from model.core.base_model import NZUpy

# Import other required components to maintain backward compatibility
from model.core.calculation_engine import CalculationEngine
from model.core.scenario_manager import ScenarioManager
from model.core.runner import ModelRunner

# Re-export the NZUpy class
__all__ = ['NZUpy']