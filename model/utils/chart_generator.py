"""
Chart generator for NZUpy model results.

This module provides a unified interface for generating standardised charts from
NZUpy model results, handling both single-scenario and range-scenario modes automatically.
"""

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, Optional, List, Union, Tuple, Any

from model.interface.single_charts import (
    carbon_price_chart,
    emissions_pathway_chart,
    supply_components_chart,
    stockpile_balance_chart,
    supply_demand_balance_chart,
    auction_volume_revenue_chart
)

from model.interface.range_charts import (
    carbon_price_chart_with_uncertainty,
    emissions_pathway_chart_with_uncertainty,
    stockpile_chart_with_uncertainty
)


class ChartGenerator:
    """
    Generate standardised charts for NZUpy model results.
    
    This class provides methods to generate standard charts for visualising
    NZUpy model results with consistent styling and historical data integration.
    It automatically detects whether the model was run in single scenario or 
    range scenario mode, and generates the appropriate charts.
    """
    
    def __init__(self, model, data_handler=None):
        """
        Initialise the chart generator.
        
        Args:
            model: NZUpy instance with results
            data_handler: Optional data handler for historical data integration
        """
        self.model = model
        self.data_handler = data_handler
        
        # Determine if this is a range scenario run
        self.is_range_scenario = self._is_range_scenario()
        
        # Verify the model has the required structured data
        self._verify_model_data()
    
    def _is_range_scenario(self) -> bool:
        """
        Determine if the model was run with range scenarios.
        
        Returns:
            bool: True if the model was run in Range mode
        """
        # Check if model has scenario_type attribute set to 'Range'
        if hasattr(self.model, 'scenario_type'):
            return self.model.scenario_type == 'Range'
        
        # Check scenario names for typical range scenario names
        range_scenario_names = [
            "95% Lower", "1 s.e lower", "central", "1 s.e upper", "95% Upper"
        ]
        if hasattr(self.model, 'scenarios'):
            return set(range_scenario_names).issubset(set(self.model.scenarios))
        
        return False
    
    def _verify_model_data(self):
        """
        Verify the model has the required structured data for charting.
        
        Raises:
            ValueError: If required data structures are missing
        """
        required_attrs = ['prices', 'demand', 'supply']
        missing_attrs = [attr for attr in required_attrs if not hasattr(self.model, attr)]
        
        if missing_attrs:
            raise ValueError(f"Model is missing required attributes: {', '.join(missing_attrs)}")
        
        # Verify that the DataFrames have the expected structure
        for attr in ['prices', 'demand', 'supply']:
            df = getattr(self.model, attr)
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Model.{attr} is not a DataFrame")
            
            if df.empty:
                raise ValueError(f"Model.{attr} DataFrame is empty")
            
            if not isinstance(df.columns, pd.MultiIndex):
                raise ValueError(f"Model.{attr} DataFrame columns are not MultiIndex")
            
            if df.columns.names != ['scenario', 'variable']:
                raise ValueError(f"Model.{attr} DataFrame column levels must be named 'scenario' and 'variable'")
    
    def carbon_price_chart(self, scenario=None, start_year=None, end_year=None) -> go.Figure:
        """
        Generate carbon price chart appropriate for model type.
        
        Args:
            scenario: Scenario name (only used for single scenario mode)
            start_year: Optional start year for chart
            end_year: Optional end year for chart
            
        Returns:
            Plotly figure object
        """
        # For range scenario, call the uncertainty version
        if self.is_range_scenario:
            return carbon_price_chart_with_uncertainty(self.model, start_year, end_year)
        
        # For single scenario, use the first scenario if none specified
        if scenario is None and hasattr(self.model, 'scenarios') and self.model.scenarios:
            scenario = self.model.scenarios[0]
            
        return carbon_price_chart(self.model, scenario, start_year, end_year)
    
    def emissions_pathway_chart(self, scenario=None, start_year=None, end_year=None) -> go.Figure:
        """
        Generate emissions pathway chart appropriate for model type.
        
        Args:
            scenario: Scenario name (only used for single scenario mode)
            start_year: Optional start year for chart
            end_year: Optional end year for chart
            
        Returns:
            Plotly figure object
        """
        if self.is_range_scenario:
            return emissions_pathway_chart_with_uncertainty(self.model, start_year, end_year)
        
        if scenario is None and hasattr(self.model, 'scenarios') and self.model.scenarios:
            scenario = self.model.scenarios[0]
            
        return emissions_pathway_chart(self.model, scenario, start_year, end_year)
    
    def stockpile_balance_chart(self, scenario=None, start_year=None, end_year=None) -> go.Figure:
        """
        Generate stockpile balance chart appropriate for model type.
        
        Args:
            scenario: Scenario name (only used for single scenario mode)
            start_year: Optional start year for chart
            end_year: Optional end year for chart
            
        Returns:
            Plotly figure object
        """
        if self.is_range_scenario:
            return stockpile_chart_with_uncertainty(self.model, start_year, end_year)
        
        if scenario is None and hasattr(self.model, 'scenarios') and self.model.scenarios:
            scenario = self.model.scenarios[0]
            
        return stockpile_balance_chart(self.model, scenario, start_year, end_year)
    
    def supply_components_chart(self, scenario=None, start_year=None, end_year=None) -> go.Figure:
        """
        Generate supply components chart (single scenario only).
        
        Args:
            scenario: Scenario name to use
            start_year: Optional start year for chart
            end_year: Optional end year for chart
            
        Returns:
            Plotly figure object
        """
        # Supply components chart is only available for single scenarios
        # For range mode, use the first scenario
        if self.is_range_scenario:
            if hasattr(self.model, 'scenarios') and self.model.scenarios:
                scenario = self.model.scenarios[0]
                print(f"Supply components chart not available in range mode. Using '{scenario}' scenario.")
            else:
                raise ValueError("No scenarios available for supply components chart")
        
        # Use first scenario if none specified
        if scenario is None and hasattr(self.model, 'scenarios') and self.model.scenarios:
            scenario = self.model.scenarios[0]
            
        return supply_components_chart(self.model, scenario, start_year, end_year)
    
    def supply_demand_balance_chart(self, scenario=None, start_year=None, end_year=None) -> go.Figure:
        """
        Generate supply-demand balance chart.
        
        Args:
            scenario: Scenario name (only used for single scenario mode)
            start_year: Optional start year for chart
            end_year: Optional end year for chart
            
        Returns:
            Plotly figure object
        """
        # Use first scenario if none specified
        if scenario is None and hasattr(self.model, 'scenarios') and self.model.scenarios:
            scenario = self.model.scenarios[0]
            
        return supply_demand_balance_chart(self.model, scenario, start_year, end_year)
    
    def auction_volume_revenue_chart(self, scenario=None, start_year=None, end_year=None) -> go.Figure:
        """
        Generate auction volume and revenue chart (single scenario only).
        
        Args:
            scenario: Scenario name to use
            start_year: Optional start year for chart
            end_year: Optional end year for chart
            
        Returns:
            Plotly figure object
        """
        # Auction volume and revenue chart is only available for single scenarios
        # For range mode, use the first scenario
        if self.is_range_scenario:
            if hasattr(self.model, 'scenarios') and self.model.scenarios:
                scenario = self.model.scenarios[0]
                print(f"Auction volume and revenue chart not available in range mode. Using '{scenario}' scenario.")
            else:
                raise ValueError("No scenarios available for auction volume and revenue chart")
        
        # Use first scenario if none specified
        if scenario is None and hasattr(self.model, 'scenarios') and self.model.scenarios:
            scenario = self.model.scenarios[0]
            
        return auction_volume_revenue_chart(self.model, scenario, start_year, end_year)
    
    def generate_standard_charts(self, output_dir=None, format="png") -> Dict[str, go.Figure]:
        """
        Generate all standard charts appropriate for the model type.
        
        Args:
            output_dir: Optional directory to save chart images
            format: Image format for saving (png, pdf, svg)
            
        Returns:
            Dictionary mapping chart names to Plotly figure objects
        """
        charts = {}
        
        try:
            # Generate common charts for both single and range modes
            charts["carbon_price"] = self.carbon_price_chart()
            charts["emissions_pathway"] = self.emissions_pathway_chart()
            charts["stockpile_balance"] = self.stockpile_balance_chart()
            charts["supply_demand_balance"] = self.supply_demand_balance_chart()
            
            # For single scenario only, generate additional charts
            if not self.is_range_scenario:
                charts["supply_components"] = self.supply_components_chart()
                charts["auction_volume_revenue"] = self.auction_volume_revenue_chart()
            
            # Save charts if output directory specified
            if output_dir:
                # Ensure the output directory exists
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                for name, fig in charts.items():
                    try:
                        fig_path = output_path / f"{name}.{format}"
                        fig.write_image(str(fig_path))
                        print(f"Saved {name} chart to {fig_path}")
                    except Exception as e:
                        print(f"Error saving {name} chart: {e}")
            
        except Exception as e:
            print(f"Error generating standard charts: {e}")
        
        return charts
    
    def export_csv_data(self, output_dir=None):
        """
        Export model data to CSV files.
        
        Args:
            output_dir: Directory to save CSV files
        """
        if output_dir is None:
            return
        
        # Ensure the output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export model's structured DataFrames
        dataframes = [
            ('prices', 'prices.csv'),
            ('core', 'core.csv'),
            ('supply', 'supply.csv'),
            ('demand', 'demand.csv'),
            ('stockpile', 'stockpile.csv'),
            ('auctions', 'auctions.csv'),
            ('industrial', 'industrial.csv'),
            ('forestry', 'forestry.csv')
        ]
        
        for attr_name, filename in dataframes:
            if hasattr(self.model, attr_name):
                df = getattr(self.model, attr_name)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    try:
                        csv_path = output_path / filename
                        df.to_csv(str(csv_path))
                        print(f"Exported {attr_name} data to {csv_path}")
                    except Exception as e:
                        print(f"Error exporting {attr_name} data: {e}")
    
    def export_combined_data(self, output_path):
        """
        Export key model data to a single combined CSV file.
        
        This function extracts key metrics from the model's structured DataFrames
        and combines them into a single CSV file for easy analysis.
        
        Args:
            output_path: Path to save the CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create DataFrame to hold combined data
            combined_data = pd.DataFrame(index=self.model.years)
            
            # Process each scenario
            for scenario in self.model.scenarios:
                # Add prices
                try:
                    price_data = self.model.prices.xs('carbon_price', level='variable', axis=1)
                    if scenario in price_data.columns:
                        combined_data[f"{scenario}_price"] = price_data[scenario]
                except KeyError:
                    print(f"Could not extract price data for {scenario}: 'carbon_price' not found")
                except Exception as e:
                    print(f"Error extracting price data for {scenario}: {e}")
                
                # Add demand
                try:
                    demand_data = self.model.demand.xs('emissions', level='variable', axis=1)
                    if scenario in demand_data.columns:
                        combined_data[f"{scenario}_demand"] = demand_data[scenario]
                except KeyError:
                    print(f"Could not extract demand data for {scenario}: 'emissions' not found")
                except Exception as e:
                    print(f"Error extracting demand data for {scenario}: {e}")
                
                # Add supply total
                try:
                    supply_data = self.model.supply.xs('total', level='variable', axis=1)
                    if scenario in supply_data.columns:
                        combined_data[f"{scenario}_supply_total"] = supply_data[scenario]
                except KeyError:
                    print(f"Could not extract total supply data for {scenario}: 'total' not found")
                except Exception as e:
                    print(f"Error extracting total supply data for {scenario}: {e}")
                
                # Add supply components
                for component in ['auction', 'industrial', 'forestry', 'stockpile']:
                    try:
                        component_data = self.model.supply.xs(component, level='variable', axis=1)
                        if scenario in component_data.columns:
                            combined_data[f"{scenario}_supply_{component}"] = component_data[scenario]
                    except KeyError:
                        pass
                    except Exception as e:
                        print(f"Error extracting {component} supply for {scenario}: {e}")
                
                # Add key stockpile metrics
                for metric in ['balance', 'ratio_to_demand', 'units_used', 'surplus_balance', 'non_surplus_balance']:
                    try:
                        stockpile_data = self.model.stockpile.xs(metric, level='variable', axis=1)
                        if scenario in stockpile_data.columns:
                            combined_data[f"{scenario}_stockpile_{metric}"] = stockpile_data[scenario]
                    except KeyError:
                        pass
                    except Exception as e:
                        print(f"Error extracting stockpile {metric} for {scenario}: {e}")
            
            # Ensure the output directory exists
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            combined_data.to_csv(output_file)
            print(f"Exported combined data to {output_file}")
            return True
            
        except Exception as e:
            print(f"Error exporting combined data: {e}")
            return False