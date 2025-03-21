"""
Single chart implementations for NZUPy model results.

This module provides visualisation functions for single scenario runs.

These functions are used when the model is run in 'Single' mode to visualise
the results for a specific scenario.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Optional, List, Union, Tuple, Any

from model.utils.chart_config import (
    NZUPY_CHART_STYLE, 
    apply_nzupy_style, 
    create_uncertainty_color,
    QUALITATIVE_COLORS,
    DIVERGING_COLORS,
    get_band_colors
)

def _get_scenario_data(model, data_type: str, scenario: Optional[str] = None, variable: Optional[str] = None):
    """
    Get data for a specific scenario and variable from model's structured DataFrames.
    
    Args:
        model: NZUpy instance
        data_type: Type of data to extract ('prices', 'demand', etc.)
        scenario: Scenario name (defaults to first scenario)
        variable: Variable name within data type (e.g., 'carbon_price' for prices)
        
    Returns:
        Series with requested data, or None if not found
    """
    # Use first scenario if none specified
    if scenario is None and hasattr(model, 'scenarios') and model.scenarios:
        scenario = model.scenarios[0]
    
    # Map data types to model attributes and variable names - using exact schema names
    data_type_map = {
        # Working ones
        'price': {'attr': 'prices', 'var': 'carbon_price'},
        'demand': {'attr': 'demand', 'var': 'emissions'},
        'baseline_emissions': {'attr': 'demand', 'var': 'baseline'},
        'supply': {'attr': 'supply', 'var': 'total'},
        
        # Supply components chart - using exact schema names
        'auction': {'attr': 'supply', 'var': 'auction'},
        'industrial': {'attr': 'supply', 'var': 'industrial'},
        'forestry': {'attr': 'supply', 'var': 'forestry'},
        'stockpile': {'attr': 'supply', 'var': 'stockpile'},
        
        # Stockpile category - using exact schema names
        'units_used': {'attr': 'stockpile', 'var': 'units_used'},
        'surplus_used': {'attr': 'stockpile', 'var': 'surplus_used'},
        'non_surplus_used': {'attr': 'stockpile', 'var': 'non_surplus_used'},
        'balance': {'attr': 'stockpile', 'var': 'balance'},
        'surplus_balance': {'attr': 'stockpile', 'var': 'surplus_balance'},
        'non_surplus_balance': {'attr': 'stockpile', 'var': 'non_surplus_balance'},
        'without_forestry': {'attr': 'stockpile', 'var': 'without_forestry'},
        
        # Auction category - using exact schema names
        'base_supplied': {'attr': 'auctions', 'var': 'base_supplied'},
        'ccr1_supplied': {'attr': 'auctions', 'var': 'ccr1_supplied'},
        'ccr2_supplied': {'attr': 'auctions', 'var': 'ccr2_supplied'},
        'auction_revenue': {'attr': 'auctions', 'var': 'revenue'}
    }
    
    # Get attribute and variable names
    attr_name = data_type_map[data_type]['attr'] if data_type in data_type_map else data_type
    var_name = variable if variable is not None else data_type_map[data_type]['var'] if data_type in data_type_map else None
    # Get the DataFrame
    df = getattr(model, attr_name, None)
    if df is None:
        print(f"Warning: Could not find attribute '{attr_name}' in model")
        return None
    
    try:
        # Handle different DataFrame structures
        if isinstance(df.columns, pd.MultiIndex):
            # For multi-index columns (scenario, variable)
            if var_name is not None:
                return df.xs((scenario, var_name), level=['scenario', 'variable'], axis=1)
            else:
                return df.xs(scenario, level='scenario', axis=1)
        else:
            # For single-index columns
            return df[scenario] if scenario in df.columns else None
    except Exception as e:
        print(f"Error accessing data: {str(e)}")
        return None

def _get_historical_data(model, data_handler, data_type: str, model_start_year: Optional[int] = None):
    """
    Get historical data for a specific data type if available.
    
    Args:
        model: NZUpy instance
        data_handler: Data handler instance, potentially with historical data
        data_type: Type of data to get historical values for
        model_start_year: Optional start year of model data, to filter historical data
        
    Returns:
        Series with historical data, or None if not available
    """
    
    # Attempt to get historical data using standard method
    if hasattr(data_handler, 'get_historical_data'):
        try:
            hist_data = data_handler.get_historical_data(data_type)
            return hist_data
        except Exception as e:
            raise Exception(f"Failed to get historical data: {e}")
    else:
        raise Exception("data_handler does not have get_historical_data method")

def _filter_by_years(data, start_year: Optional[int] = None, end_year: Optional[int] = None):
    """
    Filter time series data by start and end years.
    
    Args:
        data: Series or DataFrame indexed by year
        start_year: Optional start year (inclusive)
        end_year: Optional end year (inclusive)
        
    Returns:
        Filtered Series or DataFrame
    """
    if data is None:
        return None
        
    mask = pd.Series(True, index=data.index)
    
    if start_year is not None:
        mask &= data.index >= start_year
        
    if end_year is not None:
        mask &= data.index <= end_year
        
    return data[mask]


def carbon_price_chart(model, scenario: Optional[str] = None, start_year: Optional[int] = None, 
                   end_year: Optional[int] = None, show_nominal: bool = True) -> go.Figure:
    """
    Generate carbon price chart for a single scenario.
    
    Args:
        model: NZUpy instance
        scenario: Optional scenario name (defaults to first scenario)
        start_year: Optional start year for chart
        end_year: Optional end year for chart (defaults to model.end_year if not specified)
        show_nominal: Whether to show nominal prices in addition to real prices
        
    Returns:
        Plotly figure object
    """
    # Get colors for uncertainty bands
    base_color = NZUPY_CHART_STYLE["colors"]["central"]
    band_colors = get_band_colors(base_color)
    
    # Use first scenario if none specified
    if scenario is None and hasattr(model, 'scenarios') and model.scenarios:
        scenario = model.scenarios[0]
    
    # If end_year not specified, use the model's end_year attribute
    if end_year is None:
        if hasattr(model, 'end_year'):
            end_year = model.end_year
        elif hasattr(model, 'years') and len(model.years) > 0:
            end_year = max(model.years)
    
    # Create figure
    fig = go.Figure()
    
    try:
        # Print available columns to debug
        if hasattr(model, 'prices') and isinstance(model.prices, pd.DataFrame):
            if isinstance(model.prices.columns, pd.MultiIndex):
                # Extract real price data
                real_price_data = model.prices[(scenario, 'carbon_price')]
                # Filter by years if specified
                if start_year is not None or end_year is not None:
                    mask = pd.Series(True, index=real_price_data.index)
                    if start_year is not None:
                        mask &= real_price_data.index >= start_year
                    if end_year is not None:
                        mask &= real_price_data.index <= end_year
                    real_price_data = real_price_data[mask]
                
                # Add real price trace
                fig.add_trace(go.Scatter(
                    x=real_price_data.index,
                    y=real_price_data.values,
                    name="Real Price (2023 NZD)",
                    line=dict(color=band_colors['central'], width=4),
                    mode='lines',
                    hovertemplate="Year: %{x}<br>Real Price: $%{y:.1f} (2023 NZD)<extra></extra>"
                ))
                
                # Get historical data if available
                data_handler = getattr(model, 'data_handler', None)
                if data_handler is not None:
                    hist_real = data_handler.get_historical_data('carbon_price')
                    if hist_real is not None and not hist_real.empty:
                        # Filter historical data
                        if start_year is not None:
                            hist_real = hist_real[hist_real.index >= start_year]
                        if end_year is not None:
                            hist_real = hist_real[hist_real.index <= end_year]
                        # Only use historical years not in model data
                        hist_real = hist_real[~hist_real.index.isin(real_price_data.index)]
                        
                        fig.add_trace(go.Scatter(
                            x=hist_real.index,
                            y=hist_real.values,
                            name="Historical Real Price (2023 NZD)",
                            line=dict(color=NZUPY_CHART_STYLE["colors"]["reference_primary"], 
                                    width=4, dash='solid'),
                            mode='lines',
                            hovertemplate="Year: %{x}<br>Historical Real Price: $%{y:.1f} (2023 NZD)<extra></extra>"
                        ))
                
                # Add nominal prices if requested
                if show_nominal:
                    try:
                        # Get nominal price data
                        nominal_price_data = model.prices[(scenario, 'carbon_price_nominal')]
                        # Filter by years if specified
                        if start_year is not None or end_year is not None:
                            mask = pd.Series(True, index=nominal_price_data.index)
                            if start_year is not None:
                                mask &= nominal_price_data.index >= start_year
                            if end_year is not None:
                                mask &= nominal_price_data.index <= end_year
                            nominal_price_data = nominal_price_data[mask]
                        
                        # Add nominal price trace
                        fig.add_trace(go.Scatter(
                            x=nominal_price_data.index,
                            y=nominal_price_data.values,
                            name="Nominal Price",
                            line=dict(color=DIVERGING_COLORS[2], width=4, dash='dash'),
                            mode='lines',
                            hovertemplate="Year: %{x}<br>Nominal Price: $%{y:.1f}<extra></extra>"
                        ))
                        
                        # Add historical nominal price data if available
                        if data_handler is not None:
                            hist_nominal = data_handler.get_historical_data('carbon_price', nominal=True)
                            if hist_nominal is not None and not hist_nominal.empty:
                                # Filter historical data
                                if start_year is not None:
                                    hist_nominal = hist_nominal[hist_nominal.index >= start_year]
                                if end_year is not None:
                                    hist_nominal = hist_nominal[hist_nominal.index <= end_year]
                                # Only use historical years not in model data
                                hist_nominal = hist_nominal[~hist_nominal.index.isin(nominal_price_data.index)]
                                
                                fig.add_trace(go.Scatter(
                                    x=hist_nominal.index,
                                    y=hist_nominal.values,
                                    name="Historical Nominal Price",
                                    line=dict(color=DIVERGING_COLORS[3], width=4, dash='dash'),
                                    mode='lines',
                                    hovertemplate="Year: %{x}<br>Historical Nominal Price: $%{y:.1f}<extra></extra>"
                                ))
                    except Exception as e:
                        print(f"Warning: Could not add nominal price data: {e}")
            else:
                raise ValueError("Prices DataFrame does not have MultiIndex columns")
        else:
            raise ValueError("Prices DataFrame not found or not properly initialised")
            
    except Exception as e:
        # Add error annotation
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"Error: Could not access price data - {str(e)}",
            showarrow=False,
            font=dict(size=14)
        )
        
        # Print full error for debugging
        import traceback
        print(f"Error in carbon_price_chart: {e}")
        traceback.print_exc()
    
    # Update layout
    title_suffix = " (Real & Nominal)" if show_nominal else " (Real 2023 NZD)"
    
    fig.update_layout(
        title=f"Carbon Price Projection{title_suffix}",
        xaxis_title="Year",
        yaxis_title="Price ($/tonne CO₂-e)",
        xaxis=dict(
            range=[start_year if start_year is not None else None, 
                   end_year if end_year is not None else None]
        ),
        yaxis=dict(
            rangemode='nonnegative',
            range=[0, None]  # Set minimum to 0, let max auto-scale
        )
    )
    
    # Apply NZUpy styling
    apply_nzupy_style(fig)
    
    return fig

def emissions_pathway_chart(model, scenario=None, start_year=None, end_year=None) -> go.Figure:
    """
    Generate emissions pathway chart for a single scenario.
    
    Args:
        model: NZUpy instance with results
        scenario: Scenario name to plot (defaults to first scenario)
        start_year: Optional start year for chart
        end_year: Optional end year for chart
        
    Returns:
        Plotly figure object
    """
# Get colors for uncertainty bands
    base_color = NZUPY_CHART_STYLE["colors"]["central"]
    band_colors = get_band_colors(base_color)
    
    # Use first scenario if none specified
    if scenario is None and hasattr(model, 'scenarios') and model.scenarios:
        scenario = model.scenarios[0]
    
    # Verify model has the required DataFrame with proper structure
    if not hasattr(model, 'demand') or not isinstance(model.demand, pd.DataFrame):
        raise ValueError("Model does not have a properly structured demand DataFrame")
    
    # Verify multi-index columns structure
    if not isinstance(model.demand.columns, pd.MultiIndex):
        raise ValueError("Demand DataFrame must have multi-index columns")
    
    # Ensure column levels are properly named
    if model.demand.columns.names != ['scenario', 'variable']:
        # Fix the column names
        model.demand.columns = pd.MultiIndex.from_tuples(
            list(model.demand.columns),
            names=['scenario', 'variable']
        )
    
    # Check if scenario is in the DataFrame
    if scenario not in model.demand.columns.get_level_values('scenario'):
        raise ValueError(f"Scenario '{scenario}' not found in demand DataFrame")
    
    # Access emissions and baseline data using the multi-index structure
    try:
        emissions_data = model.demand.xs((scenario, 'emissions'), level=['scenario', 'variable'], axis=1)
        if isinstance(emissions_data, pd.DataFrame) and emissions_data.shape[1] == 1:
            emissions_data = emissions_data.iloc[:, 0]
    except KeyError:
        raise ValueError(f"Emissions data not found for scenario '{scenario}'")
    
    try:
        baseline_data = model.demand.xs((scenario, 'baseline'), level=['scenario', 'variable'], axis=1)
        if isinstance(baseline_data, pd.DataFrame) and baseline_data.shape[1] == 1:
            baseline_data = baseline_data.iloc[:, 0]
    except KeyError:
        raise ValueError(f"Baseline data not found for scenario '{scenario}'")
    
    # Filter by years if specified
    if start_year is not None or end_year is not None:
        mask = pd.Series(True, index=emissions_data.index)
        if start_year is not None:
            mask &= emissions_data.index >= start_year
        if end_year is not None:
            mask &= emissions_data.index <= end_year
        
        emissions_data = emissions_data[mask]
        baseline_data = baseline_data[mask]
    
    # Create figure
    fig = go.Figure()
    
    # Get historical data if available
    data_handler = getattr(model, 'data_handler', None)
    hist_data = None
    if data_handler is not None and hasattr(data_handler, 'get_historical_data'):
        try:
            hist_data = data_handler.get_historical_data('emissions')
        except Exception as e:
            print(f"Warning: Could not get historical data: {e}")
    
    # Add historical data trace if available
    if hist_data is not None and not hist_data.empty:
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data.values,
            name="Historical Emissions",
            line=dict(color=NZUPY_CHART_STYLE["colors"]["reference_primary"], 
                    width=4, dash='solid'),
            mode='lines',
            hovertemplate="Year: %{x}<br>Emissions: %{y:.2f} kt CO₂-e<extra></extra>"
        ))
    
    # Add baseline emissions trace
    if baseline_data is not None:
        fig.add_trace(go.Scatter(
            x=baseline_data.index,
            y=baseline_data.values,
            name="Baseline Emissions (Pre-Response)",
            line=dict(color=DIVERGING_COLORS[1], width=4, dash='solid'),
            mode='lines',
            hovertemplate="Year: %{x}<br>Baseline: %{y:.2f} kt CO₂-e<extra></extra>"
        ))
    
    # Add emissions after price response trace
    if emissions_data is not None:
        fig.add_trace(go.Scatter(
            x=emissions_data.index,
            y=emissions_data.values,
            name="Emissions After Price Response",
            line=dict(color=band_colors['central'], width=4),
            mode='lines',
            hovertemplate="Year: %{x}<br>Emissions: %{y:.2f} kt CO₂-e<extra></extra>"
        ))
    
    # Update layout
    fig.update_layout(
        title="Emissions Pathway",
        xaxis_title="Year",
        yaxis_title="Emissions (kt CO₂-e)",
        yaxis=dict(
            rangemode='nonnegative',
            range=[0, None]  # Set minimum to 0, let max auto-scale
        )
    )
    
    # Apply NZUpy styling
    apply_nzupy_style(fig)
    
    return fig

def supply_components_chart(model, scenario=None, start_year=None, end_year=None):
    """Generate supply components chart with split stockpile."""
    if scenario is None and hasattr(model, 'scenarios') and model.scenarios:
        scenario = model.scenarios[0]
    
    fig = go.Figure()
    
    try:
        if hasattr(model, 'supply') and isinstance(model.supply, pd.DataFrame):
            if isinstance(model.supply.columns, pd.MultiIndex):
                # Extract supply components using direct indexing
                auction = model.supply[(scenario, 'auction')] if (scenario, 'auction') in model.supply.columns else None
                industrial = model.supply[(scenario, 'industrial')] if (scenario, 'industrial') in model.supply.columns else None
                forestry = model.supply[(scenario, 'forestry')] if (scenario, 'forestry') in model.supply.columns else None
                total_supply = model.supply[(scenario, 'total')] if (scenario, 'total') in model.supply.columns else None
                
                # Try to get stockpile components from stockpile DataFrame
                has_split_stockpile = False
                if hasattr(model, 'stockpile') and isinstance(model.stockpile, pd.DataFrame):
                    if (scenario, 'surplus_used') in model.stockpile.columns and (scenario, 'non_surplus_used') in model.stockpile.columns:
                        surplus_used = model.stockpile[(scenario, 'surplus_used')]
                        non_surplus_used = model.stockpile[(scenario, 'non_surplus_used')]
                        has_split_stockpile = True
                
                # Fall back to total stockpile from supply DataFrame if split not available
                if not has_split_stockpile:
                    stockpile = model.supply[(scenario, 'stockpile')] if (scenario, 'stockpile') in model.supply.columns else None
                
                # Get demand for reference line
                if hasattr(model, 'demand') and isinstance(model.demand, pd.DataFrame):
                    demand = model.demand[(scenario, 'emissions')] if (scenario, 'emissions') in model.demand.columns else None
                else:
                    demand = None
                
                # Continue with chart creation only if we have core supply components
                if auction is not None and industrial is not None and forestry is not None:
                    # Filter by years if specified
                    if start_year is not None or end_year is not None:
                        mask = pd.Series(True, index=auction.index)
                        if start_year is not None:
                            mask &= auction.index >= start_year
                        if end_year is not None:
                            mask &= auction.index <= end_year
                            
                        auction = auction[mask]
                        industrial = industrial[mask]
                        forestry = forestry[mask]
                        
                        if has_split_stockpile:
                            surplus_used = surplus_used[mask]
                            non_surplus_used = non_surplus_used[mask]
                        elif stockpile is not None:
                            stockpile = stockpile[mask]
                            
                        if total_supply is not None:
                            total_supply = total_supply[mask]
                        if demand is not None:
                            demand = demand[mask]
                    
                    # Define colors for components using NZUpy chart style
                    component_colors = {
                        'auction_transparent': NZUPY_CHART_STYLE['colors']['auction_transparent'],
                        'industrial_transparent': NZUPY_CHART_STYLE['colors']['industrial_transparent'], 
                        'forestry_transparent': NZUPY_CHART_STYLE['colors']['forestry_transparent'],
                        'non_surplus_transparent': NZUPY_CHART_STYLE['colors']['non_surplus_transparent'],
                        'surplus_transparent': NZUPY_CHART_STYLE['colors']['surplus_transparent'],
                        'non_surplus': NZUPY_CHART_STYLE['colors']['non_surplus_transparent'],
                        'auction': NZUPY_CHART_STYLE['colors']['auction'],
                        'surplus': NZUPY_CHART_STYLE['colors']['surplus_transparent'],
                        'industrial': NZUPY_CHART_STYLE['colors']['industrial'],
                        'forestry': NZUPY_CHART_STYLE['colors']['forestry'],
                        'stockpile': NZUPY_CHART_STYLE['colors']['stockpile']
                    }
                    
                    
                    # Add components as stacked area
                    fig.add_trace(go.Scatter(
                        x=auction.index,
                        y=auction.values,
                        name="Auction",
                        mode='lines',
                        line=dict(width=1, color=component_colors['auction']),
                        stackgroup='one',
                        fillcolor=component_colors['auction_transparent'],
                        hovertemplate="Year: %{x}<br>Auction: %{y:.2f} kt CO₂-e<extra></extra>"
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=industrial.index,
                        y=industrial.values,
                        name="Industrial",
                        mode='lines',
                        line=dict(width=1, color=component_colors['industrial']),
                        stackgroup='one',
                        fillcolor=component_colors['industrial_transparent'],
                        hovertemplate="Year: %{x}<br>Industrial: %{y:.2f} kt CO₂-e<extra></extra>"
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forestry.index,
                        y=forestry.values,
                        name="Forestry",
                        mode='lines',
                        line=dict(width=1, color=component_colors['forestry']),
                        stackgroup='one',
                        fillcolor=component_colors['forestry_transparent'],
                        hovertemplate="Year: %{x}<br>Forestry: %{y:.2f} kt CO₂-e<extra></extra>"
                    ))
                    
                    # Add stockpile components - either split or total
                    if has_split_stockpile:
                        fig.add_trace(go.Scatter(
                            x=surplus_used.index,
                            y=surplus_used.values,
                            name="Surplus",
                            mode='lines',
                            line=dict(width=1, color=component_colors['surplus']),
                            stackgroup='one',
                            fillcolor=component_colors['surplus_transparent'],
                            hovertemplate="Year: %{x}<br>Surplus: %{y:.2f} kt CO₂-e<extra></extra>"
                        ))
                        # Add non-surplus first
                        fig.add_trace(go.Scatter(
                            x=non_surplus_used.index,
                            y=non_surplus_used.values,
                            name="Non-Surplus",
                            mode='lines',
                            line=dict(width=1, color=component_colors['non_surplus']),
                            stackgroup='one',
                            fillcolor=component_colors['non_surplus_transparent'],
                            hovertemplate="Year: %{x}<br>Non-Surplus: %{y:.2f} kt CO₂-e<extra></extra>"
                        ))

                    elif stockpile is not None:
                        # Add total stockpile if split components not available
                        fig.add_trace(go.Scatter(
                            x=stockpile.index,
                            y=stockpile.values,
                            name="Stockpile",
                            mode='lines',
                            line=dict(width=1, color=component_colors['stockpile']),
                            stackgroup='one',
                            fillcolor=component_colors['stockpile_transparent'],
                            hovertemplate="Year: %{x}<br>Stockpile: %{y:.2f} kt CO₂-e<extra></extra>"
                        ))
                    
                    # Add total supply line
                    if total_supply is not None:
                        fig.add_trace(go.Scatter(
                            x=total_supply.index,
                            y=total_supply.values,
                            name="Total Supply",
                            mode='lines',
                            line=dict(color=NZUPY_CHART_STYLE["colors"]["reference_primary"], width=4),
                            hovertemplate="Year: %{x}<br>Total Supply: %{y:.2f} kt CO₂-e<extra></extra>"
                        ))
                    
                    # Add demand line
                    if demand is not None:
                        fig.add_trace(go.Scatter(
                            x=demand.index,
                            y=demand.values,
                            name="Demand",
                            mode='lines',
                            line=dict(color=NZUPY_CHART_STYLE["colors"]["reference_secondary"], width=4, dash='dash'),
                            hovertemplate="Year: %{x}<br>Demand: %{y:.2f} kt CO₂-e<extra></extra>"
                        ))
                    
                    # Set up layout
                    fig.update_layout(
                        title=f"Supply Components - {scenario}",
                        xaxis_title="Year",
                        yaxis_title="NZUs (thousands) /Kt CO₂-e)",
                        yaxis=dict(
                            rangemode='nonnegative',
                            range=[0, None]  # Set minimum to 0, let max auto-scale
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.10,  # Increased from 1.05 to add more space
                            xanchor="center",
                            x=0.5,
                            bgcolor='rgba(255, 255, 255, 0.9)',
                            font=dict(size=12)
                        ),
                        margin=dict(t=50, r=50, l=50, b=50),  # Increased top margin from 120 to 150
                        height=600,
                        width=900
                    )
                else:
                    raise KeyError("Some required supply components are missing")
            else:
                raise ValueError("Supply DataFrame does not have MultiIndex columns")
        else:
            raise ValueError("Supply DataFrame not found or not properly initialised")
            
    except Exception as e:
        # Add error annotation
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"Error: Could not access supply component data - {str(e)}",
            showarrow=False,
            font=dict(size=14)
        )
        
        # Print full error for debugging
        import traceback
        print(f"Error in supply_components_chart: {e}")
        traceback.print_exc()
    
    # Apply NZUpy styling
    apply_nzupy_style(fig)
    
    return fig

def stockpile_balance_chart(model, scenario=None, start_year=None, end_year=None):
    """Generate stockpile balance chart."""
    if scenario is None and hasattr(model, 'scenarios') and model.scenarios:
        scenario = model.scenarios[0]
    
    fig = go.Figure()
    
    try:
        # Print available columns to debug
        if hasattr(model, 'stockpile') and isinstance(model.stockpile, pd.DataFrame):
            
            if isinstance(model.stockpile.columns, pd.MultiIndex):
                # Check the level names
                level_names = model.stockpile.columns.names
                
                # Extract data directly using column tuples
                stockpile_balance = model.stockpile[scenario, 'balance'] if (scenario, 'balance') in model.stockpile.columns else None
                surplus_balance = model.stockpile[scenario, 'surplus_balance'] if (scenario, 'surplus_balance') in model.stockpile.columns else None
                non_surplus_balance = model.stockpile[scenario, 'non_surplus_balance'] if (scenario, 'non_surplus_balance') in model.stockpile.columns else None
                
                # Optional: Get stockpile without forestry if available
                has_without_forestry = False
                if (scenario, 'without_forestry') in model.stockpile.columns:
                    stockpile_without_forestry = model.stockpile[scenario, 'without_forestry']
                    has_without_forestry = True
                
                # Continue with chart creation only if we have required data
                if stockpile_balance is not None and surplus_balance is not None and non_surplus_balance is not None:
                    # Filter by years if specified
                    if start_year is not None or end_year is not None:
                        mask = pd.Series(True, index=stockpile_balance.index)
                        if start_year is not None:
                            mask &= stockpile_balance.index >= start_year
                        if end_year is not None:
                            mask &= stockpile_balance.index <= end_year
                            
                        stockpile_balance = stockpile_balance[mask]
                        surplus_balance = surplus_balance[mask]
                        non_surplus_balance = non_surplus_balance[mask]
                        if has_without_forestry:
                            stockpile_without_forestry = stockpile_without_forestry[mask]
                    
                    # Add surplus area
                    fig.add_trace(go.Scatter(
                        x=surplus_balance.index,
                        y=surplus_balance.values,
                        name="Surplus Balance",
                        mode='lines',
                        line=dict(width=1, color=NZUPY_CHART_STYLE['colors']['diverging_09_bold']),
                        stackgroup='one',
                        fillcolor=NZUPY_CHART_STYLE['colors']['forestry_transparent'],
                        hovertemplate="Year: %{x}<br>Surplus Balance: %{y:.2f} kt CO₂-e<extra></extra>"
                    ))
                    
                    # Add non-surplus area
                    fig.add_trace(go.Scatter(
                        x=non_surplus_balance.index,
                        y=non_surplus_balance.values,
                        name="Non-Surplus Balance",
                        mode='lines',
                        line=dict(width=1, color=NZUPY_CHART_STYLE['colors']['diverging_01_bold']),
                        stackgroup='one',
                        fillcolor=NZUPY_CHART_STYLE['colors']['diverging_01_transparent'],
                        hovertemplate="Year: %{x}<br>Non-Surplus Balance: %{y:.2f} kt CO₂-e<extra></extra>"
                    ))
                    
                    # Add total stockpile line
                    fig.add_trace(go.Scatter(
                        x=stockpile_balance.index,
                        y=stockpile_balance.values,
                        name="Total Stockpile",
                        mode='lines',
                        line=dict(color=NZUPY_CHART_STYLE['colors']['reference_primary'], width=4),
                        hovertemplate="Year: %{x}<br>Total Stockpile: %{y:.2f} kt CO₂-e<extra></extra>"
                    ))
                    
                    # Add stockpile without forestry if available
                    if has_without_forestry:
                        fig.add_trace(go.Scatter(
                            x=stockpile_without_forestry.index,
                            y=stockpile_without_forestry.values,
                            name="Stockpile Without Forestry",
                            mode='lines',
                            line=dict(color=NZUPY_CHART_STYLE['colors']['reference_secondary'], width=4, dash='dash'),
                            hovertemplate="Year: %{x}<br>Stockpile Without Forestry: %{y:.2f} kt CO₂-e<extra></extra>"
                        ))
                    
                    # Set up layout
                    fig.update_layout(
                        title=f"Stockpile Balance - {scenario}",
                        xaxis_title="Year",
                        yaxis_title="NZUs (thousands) /Kt CO₂-e)",
                        yaxis=dict(
                            rangemode='tozero'
                        )
                    )
                else:
                    raise KeyError("Some required stockpile variables are missing")
            else:
                raise ValueError("Stockpile DataFrame does not have MultiIndex columns")
        else:
            raise ValueError("Stockpile DataFrame not found or not properly initialised")
            
    except Exception as e:
        # Add error annotation
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"Error: Could not access stockpile data - {str(e)}",
            showarrow=False,
            font=dict(size=14)
        )
    
    # Apply NZUpy styling
    apply_nzupy_style(fig)
    
    return fig

def supply_demand_balance_chart(model, scenario: Optional[str] = None, start_year: Optional[int] = None, 
                               end_year: Optional[int] = None) -> go.Figure:
    """
    Generate supply-demand balance chart for a single scenario.
    
    Args:
        model: NZUpy instance with results
        scenario: Scenario name to plot (defaults to first scenario)
        start_year: Optional start year for chart
        end_year: Optional end year for chart
        
    Returns:
        Plotly figure object
    """
    # Use first scenario if none specified
    if scenario is None and hasattr(model, 'scenarios') and model.scenarios:
        scenario = model.scenarios[0]

    # Get colors
    base_color = NZUPY_CHART_STYLE["colors"]["central"]
    band_colors = get_band_colors(base_color)

    # Verify model has the required DataFrames with proper structure
    if not hasattr(model, 'supply') or not isinstance(model.supply, pd.DataFrame):
        raise ValueError("Model does not have a properly structured supply DataFrame")
    if not hasattr(model, 'demand') or not isinstance(model.demand, pd.DataFrame):
        raise ValueError("Model does not have a properly structured demand DataFrame")
    # Verify multi-index columns structure for supply
    if not isinstance(model.supply.columns, pd.MultiIndex):
        raise ValueError("Supply DataFrame must have multi-index columns")
    # Verify multi-index columns structure for demand
    if not isinstance(model.demand.columns, pd.MultiIndex):
        raise ValueError("Demand DataFrame must have multi-index columns")
    
    # Ensure column levels are properly named for supply
    if model.supply.columns.names != ['scenario', 'variable']:
        # Fix the column names
        model.supply.columns = pd.MultiIndex.from_tuples(
            list(model.supply.columns),
            names=['scenario', 'variable']
        )
    
    # Ensure column levels are properly named for demand
    if model.demand.columns.names != ['scenario', 'variable']:
        # Fix the column names
        model.demand.columns = pd.MultiIndex.from_tuples(
            list(model.demand.columns),
            names=['scenario', 'variable']
        )
    
    # Check if scenario is in the DataFrames
    if scenario not in model.supply.columns.get_level_values('scenario'):
        raise ValueError(f"Scenario '{scenario}' not found in supply DataFrame")
    
    if scenario not in model.demand.columns.get_level_values('scenario'):
        raise ValueError(f"Scenario '{scenario}' not found in demand DataFrame")
    
    # Access supply and demand data using the multi-index structure
    try:
        supply = model.supply.xs((scenario, 'total'), level=['scenario', 'variable'], axis=1)
        if isinstance(supply, pd.DataFrame) and supply.shape[1] == 1:
            supply = supply.iloc[:, 0]
    except KeyError:
        raise ValueError(f"Total supply data not found for scenario '{scenario}'")
    
    try:
        demand = model.demand.xs((scenario, 'emissions'), level=['scenario', 'variable'], axis=1)
        if isinstance(demand, pd.DataFrame) and demand.shape[1] == 1:
            demand = demand.iloc[:, 0]
    except KeyError:
        raise ValueError(f"Emissions data not found for scenario '{scenario}'")
    
    # Filter by years if specified
    if start_year is not None or end_year is not None:
        mask = pd.Series(True, index=supply.index)
        if start_year is not None:
            mask &= supply.index >= start_year
        if end_year is not None:
            mask &= supply.index <= end_year
        
        supply = supply[mask]
        demand = demand[mask]
    
    # Create figure
    fig = go.Figure()
    
    # Get historical data if available
    data_handler = getattr(model, 'data_handler', None)
    hist_supply = None
    hist_demand = None
    if data_handler is not None and hasattr(data_handler, 'get_historical_data'):
        try:
            hist_supply = data_handler.get_historical_data('supply')
            hist_demand = data_handler.get_historical_data('emissions')
        except Exception as e:
            print(f"Warning: Could not get historical data: {e}")
    
    # Add historical supply trace if available
    if hist_supply is not None:
        # Filter by start year if specified
        if start_year is not None:
            hist_supply = hist_supply[hist_supply.index >= start_year]
        
        # Only use historical years not in model data
        if supply is not None:
            hist_supply = hist_supply[~hist_supply.index.isin(supply.index)]
        
        # Add trace
        if not hist_supply.empty:
            fig.add_trace(go.Scatter(
                x=hist_supply.index,
                y=hist_supply.values,
                name="Historical Supply",
                line=dict(color=DIVERGING_COLORS[1], 
                        width=4),
                mode='lines',
                hovertemplate="Year: %{x}<br>Supply: %{y:.2f} kt CO₂-e<extra></extra>"
            ))
    
    # Add historical demand trace if available
    if hist_demand is not None:
        # Filter by start year if specified
        if start_year is not None:
            hist_demand = hist_demand[hist_demand.index >= start_year]
        
        # Only use historical years not in model data
        if demand is not None:
            hist_demand = hist_demand[~hist_demand.index.isin(demand.index)]
        
        # Add trace
        if not hist_demand.empty:
            fig.add_trace(go.Scatter(
                x=hist_demand.index,
                y=hist_demand.values,
                name="Historical Demand",
                line=dict(color=band_colors['central'], 
                        width=4, dash='dash'),
                mode='lines',
                hovertemplate="Year: %{x}<br>Demand: %{y:.2f} kt CO₂-e<extra></extra>"
            ))
    
    # Add supply trace
    if supply is not None:
        fig.add_trace(go.Scatter(
            x=supply.index,
            y=supply.values,
            name="Supply",
            line=dict(color=DIVERGING_COLORS[1], width=4),
            mode='lines',
            hovertemplate="Year: %{x}<br>Supply: %{y:.2f} kt CO₂-e<extra></extra>"
        ))
    
    # Add demand trace
    if demand is not None:
        fig.add_trace(go.Scatter(
            x=demand.index,
            y=demand.values,
            name="Demand",
            line=dict(color=band_colors['central'], width=4, dash='dash'),
            mode='lines',
            hovertemplate="Year: %{x}<br>Demand: %{y:.2f} kt CO₂-e<extra></extra>"
        ))
    
    # Calculate and add the gap between supply and demand
    if supply is not None and demand is not None:
        common_years = supply.index.intersection(demand.index)
        
        if len(common_years) > 0:
            supply_common = supply.loc[common_years]
            demand_common = demand.loc[common_years]
            
            # Add single fill area between supply and demand
            fig.add_trace(go.Scatter(
                x=common_years.tolist() + common_years.tolist()[::-1],
                y=supply_common.tolist() + demand_common.tolist()[::-1],
                fill='toself',
                fillcolor=NZUPY_CHART_STYLE['colors']['reference_secondary_transparent'],
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=True,
                name="Supply-Demand Gap",
                hovertemplate="Year: %{x}<br>Supply-Demand Gap: %{y:.2f} kt CO₂-e<extra></extra>"
            ))
    
    # Update layout
    fig.update_layout(
        title=f"Supply-Demand Balance - {scenario}",
        xaxis_title="Year",
        yaxis_title="NZUs (thousands) /Kt CO₂-e)",
        yaxis=dict(
            rangemode='nonnegative',
            range=[0, None]  # Set minimum to 0, let max auto-scale
        )
    )
    
    # Apply NZUpy styling
    apply_nzupy_style(fig)
    
    return fig

def auction_volume_revenue_chart(model, scenario=None, start_year=None, end_year=None):
    """Generate auction volume and revenue chart."""
    if scenario is None and hasattr(model, 'scenarios') and model.scenarios:
        scenario = model.scenarios[0]
    
    # Get colors for uncertainty bands
    base_color = NZUPY_CHART_STYLE["colors"]["central"]
    band_colors = get_band_colors(base_color)
    fig = go.Figure()
    
    try:
        # Print available columns to debug
        if hasattr(model, 'auctions') and isinstance(model.auctions, pd.DataFrame):
            print(f"\nAuctions columns: {model.auctions.columns.tolist()}")
            
            # Check column structure
            column_tuples = [(scenario, 'base_supplied'), 
                             (scenario, 'ccr1_supplied'), 
                             (scenario, 'ccr2_supplied'), 
                             (scenario, 'revenue')]
            
            # Extract data directly using column tuples
            base_supplied = model.auctions[column_tuples[0]]
            ccr1_supplied = model.auctions[column_tuples[1]]
            ccr2_supplied = model.auctions[column_tuples[2]]
            revenue = model.auctions[column_tuples[3]]
            
            # Filter by years if specified
            if start_year is not None or end_year is not None:
                mask = pd.Series(True, index=base_supplied.index)
                if start_year is not None:
                    mask &= base_supplied.index >= start_year
                if end_year is not None:
                    mask &= base_supplied.index <= end_year
                    
                base_supplied = base_supplied[mask]
                ccr1_supplied = ccr1_supplied[mask]
                ccr2_supplied = ccr2_supplied[mask]
                revenue = revenue[mask]
            
            # Create the stacked bar chart for auction volumes
            fig.add_trace(go.Bar(
                x=base_supplied.index,
                y=base_supplied.values,
                name="Base Auction Volume",
                marker_color=NZUPY_CHART_STYLE["colors"]["auction_volume_dark"],
                opacity=0.7,
                hovertemplate="Year: %{x}<br>Base Volume: %{y:.2f} kt CO₂-e<extra></extra>"
            ))
            
            fig.add_trace(go.Bar(
                x=ccr1_supplied.index,
                y=ccr1_supplied.values,
                name="CCR1 Volume",
                marker_color=NZUPY_CHART_STYLE["colors"]["auction_volume_medium"],
                opacity=0.7,
                hovertemplate="Year: %{x}<br>CCR1 Volume: %{y:.2f} kt CO₂-e<extra></extra>"
            ))
            
            fig.add_trace(go.Bar(
                x=ccr2_supplied.index,
                y=ccr2_supplied.values,
                name="CCR2 Volume",
                marker_color=NZUPY_CHART_STYLE["colors"]["auction_volume_light"],
                opacity=0.7,
                hovertemplate="Year: %{x}<br>CCR2 Volume: %{y:.2f} kt CO₂-e<extra></extra>"
            ))
            
            # Add revenue line on second y-axis
            fig.add_trace(go.Scatter(
                x=revenue.index,
                y=revenue.values,
                name="Auction Revenue",
                line=dict(color=band_colors['central'], width=4),
                marker=dict(size=8),
                yaxis="y2",
                hovertemplate="Year: %{x}<br>Revenue: $%{y:.0f}<extra></extra>"
            ))
            
            # Set up dual y-axes and layout
            fig.update_layout(
                title=f"Auction Volume and Revenue - {scenario}",
                xaxis_title="Year",
                yaxis_title="Auction Volume (kt CO₂-e)",
                yaxis=dict(
                    rangemode='nonnegative',
                    range=[0, None]  # Set minimum to 0, let max auto-scale
                ),
                yaxis2=dict(
                    title="Auction Revenue ($)",
                    titlefont=dict(color='#0d3941'),
                    tickfont=dict(color='#0d3941'),
                    overlaying="y",
                    side="right",
                    rangemode='nonnegative',
                    range=[0, None]  # Set minimum to 0, let max auto-scale
                ),
                barmode='stack',
                bargap=0.15
            )
        else:
            raise ValueError("Auctions DataFrame not found or not properly initialised")
            
    except Exception as e:
        # Add error annotation
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"Error: Could not access auction data - {str(e)}",
            showarrow=False,
            font=dict(size=14)
        )
        
        # Print full error for debugging
        import traceback
        print(f"Error in auction_volume_revenue_chart: {e}")
        traceback.print_exc()
    
    # Apply NZUpy styling
    apply_nzupy_style(fig)
    
    return fig