"""
Range chart implementations for NZUPy model results.

This module provides visualisation functions for range scenario runs,
focusing on uncertainty visualisation across the five standard scenarios:
central, 1 s.e lower, 1 s.e upper, 95% Lower, and 95% Upper.

These functions are used when the model is run in 'Range' mode to visualise
the uncertainty in the model outputs across different demand scenarios.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Optional, List, Union, Tuple, Any

from model.utils.chart_config import (
    NZUPY_CHART_STYLE, 
    apply_nzupy_style, 
    create_uncertainty_color,
    get_band_colors
)

def carbon_price_chart_with_uncertainty(model, start_year: Optional[int] = None, 
                                        end_year: Optional[int] = None) -> go.Figure:
    """
    Generate carbon price chart with uncertainty bands for range scenario.
    
    Args:
        model: NZUpy instance with range scenario results
        start_year: Optional start year for chart
        end_year: Optional end year for chart (defaults to model.end_year if not specified)
        
    Returns:
        Plotly figure object with uncertainty bands
    """
    fig = go.Figure()
    
    # Define scenario groups for uncertainty bands
    central_scenario = "central"
    std_error_lower = "1 s.e lower"
    std_error_upper = "1 s.e upper"
    confidence_lower = "95% Lower"
    confidence_upper = "95% Upper"
    
    # Get colors for uncertainty bands
    base_color = NZUPY_CHART_STYLE["colors"]["central"]
    band_colors = get_band_colors(base_color)
    
    # Check if model has the expected structure
    if not hasattr(model, 'prices') or not isinstance(model.prices, pd.DataFrame):
        raise ValueError("Model does not have a properly structured prices DataFrame")
    
    # Verify we have a multi-index with the right levels
    if not isinstance(model.prices.columns, pd.MultiIndex) or 'scenario' not in model.prices.columns.names or 'variable' not in model.prices.columns.names:
        raise ValueError("Prices DataFrame must have a multi-index with 'scenario' and 'variable' levels")
    
    # Extract carbon prices for all scenarios
    try:
        prices_df = model.prices.xs('carbon_price', level='variable', axis=1)
    except KeyError:
        raise ValueError("Could not find 'carbon_price' in the model's prices DataFrame")
    
    # Check if we have all the required scenarios
    required_scenarios = [central_scenario, std_error_lower, std_error_upper, confidence_lower, confidence_upper]
    missing_scenarios = [s for s in required_scenarios if s not in prices_df.columns]
    if missing_scenarios:
        raise ValueError(f"Missing required scenarios in prices DataFrame: {missing_scenarios}")
    
    # If end_year not specified, use the model's end_year attribute
    if end_year is None:
        if hasattr(model, 'end_year'):
            end_year = model.end_year
        elif hasattr(model, 'years') and len(model.years) > 0:
            end_year = max(model.years)
    
    # Filter by years if specified
    years_to_include = prices_df.index
    if start_year is not None:
        years_to_include = years_to_include[years_to_include >= start_year]
    if end_year is not None:
        years_to_include = years_to_include[years_to_include <= end_year]
    
    # Filter data to selected years
    filtered_df = prices_df.loc[years_to_include]
    
    # Get historical data if available
    data_handler = getattr(model, 'data_handler', None)
    
    if data_handler is not None:
        try:
            historical_prices = data_handler.get_historical_data('carbon_price')
            
            if historical_prices is not None:
                # Filter by start_year if specified
                if start_year is not None:
                    historical_prices = historical_prices[historical_prices.index >= start_year]
                
                # Filter by end_year if specified
                if end_year is not None:
                    historical_prices = historical_prices[historical_prices.index <= end_year]
                
                # Add historical data trace
                fig.add_trace(go.Scatter(
                    x=historical_prices.index,
                    y=historical_prices.values,
                    line=dict(color=NZUPY_CHART_STYLE["colors"]["reference_primary"], width=3, dash='solid'),
                    name='Historical',
                    mode='lines',
                    hovertemplate="Year: %{x}<br>Price: $%{y:.1f}<extra></extra>"
                ))
        except Exception as e:
            pass
    
    # Add 95% confidence interval band (between 95% Lower and 95% Upper)
    fig.add_trace(go.Scatter(
        x=filtered_df.index.tolist() + filtered_df.index.tolist()[::-1],
        y=filtered_df[confidence_upper].tolist() + filtered_df[confidence_lower].tolist()[::-1],
        fill='toself',
        fillcolor=band_colors['confidence'],
        line=dict(color=band_colors['confidence']),
        name='95% Confidence Interval',
        showlegend=True,
        hoverinfo='skip',
        hovertemplate="Year: %{x}<br>Price: $%{y:.1f}<extra></extra>"
    ))
    
    # Add 1 standard error band (between 1 s.e lower and 1 s.e upper)
    fig.add_trace(go.Scatter(
        x=filtered_df.index.tolist() + filtered_df.index.tolist()[::-1],
        y=filtered_df[std_error_upper].tolist() + filtered_df[std_error_lower].tolist()[::-1],
        fill='toself',
        fillcolor=band_colors['std_error'],
        line=dict(color=band_colors['confidence']),
        name='1 Standard Error',
        showlegend=True,
        hoverinfo='skip',
        hovertemplate="Year: %{x}<br>Price: $%{y:.1f}<extra></extra>"
    ))
    
    # Add central estimate line
    fig.add_trace(go.Scatter(
        x=filtered_df.index,
        y=filtered_df[central_scenario],
        line=dict(color=band_colors['central'], width=4),
        name='Central Estimate',
        mode='lines',
        hovertemplate="Year: %{x}<br>Price: $%{y:.1f}<extra></extra>"
    ))
    
    # Update layout
    fig.update_layout(
        title='Carbon Price with Uncertainty Bands',
        xaxis_title='Year',
        yaxis_title='Price ($/tonne CO₂-e)',
        yaxis=dict(
            rangemode='nonnegative',
            range=[0, None]  # Set minimum to 0, let max auto-scale
        ),
        xaxis=dict(
            range=[start_year if start_year is not None else None, 
                   end_year if end_year is not None else None]
        )
    )
    
    # Apply NZUpy styling
    apply_nzupy_style(fig)
    
    return fig

def emissions_pathway_chart_with_uncertainty(model, start_year: Optional[int] = None, 
                                           end_year: Optional[int] = None) -> go.Figure:
    """
    Generate emissions pathway chart with uncertainty bands for range scenario.
    
    Args:
        model: NZUpy instance with range scenario results
        start_year: Optional start year for chart
        end_year: Optional end year for chart
        
    Returns:
        Plotly figure object with uncertainty bands
    """
    fig = go.Figure()
    
    # Define scenario groups for uncertainty bands
    central_scenario = "central"
    std_error_lower = "1 s.e lower"
    std_error_upper = "1 s.e upper"
    confidence_lower = "95% Lower"
    confidence_upper = "95% Upper"
    
    # Get colors for uncertainty bands
    base_color = NZUPY_CHART_STYLE["colors"]["central"]
    band_colors = get_band_colors(base_color)
    
    # Check if model has the expected structure
    if not hasattr(model, 'demand') or not isinstance(model.demand, pd.DataFrame):
        raise ValueError("Model does not have a properly structured demand DataFrame")
    
    # Verify we have a multi-index with the right levels
    if not isinstance(model.demand.columns, pd.MultiIndex) or 'scenario' not in model.demand.columns.names or 'variable' not in model.demand.columns.names:
        raise ValueError("Demand DataFrame must have a multi-index with 'scenario' and 'variable' levels")
    
    # Extract emissions and baseline data for all scenarios
    try:
        emissions_df = model.demand.xs('emissions', level='variable', axis=1)
        baseline_df = model.demand.xs('baseline', level='variable', axis=1)
    except KeyError as e:
        raise ValueError(f"Could not find required variable in the model's demand DataFrame: {e}")
    
    # Check if we have all the required scenarios
    required_scenarios = [central_scenario, std_error_lower, std_error_upper, confidence_lower, confidence_upper]
    missing_scenarios = [s for s in required_scenarios if s not in emissions_df.columns]
    if missing_scenarios:
        raise ValueError(f"Missing required scenarios in demand DataFrame: {missing_scenarios}")
    
    # Filter by years if specified
    years_to_include = emissions_df.index
    if start_year is not None:
        years_to_include = years_to_include[years_to_include >= start_year]
    if end_year is not None:
        years_to_include = years_to_include[years_to_include <= end_year]
    
    # Filter data to selected years
    filtered_emissions = emissions_df.loc[years_to_include]
    filtered_baseline = baseline_df.loc[years_to_include]
    
    # Add baseline emissions as a reference line (using central scenario's baseline)
    fig.add_trace(go.Scatter(
        x=filtered_baseline.index,
        y=filtered_baseline[central_scenario],
        line=dict(color=NZUPY_CHART_STYLE["colors"]["reference_secondary"], width=4, dash='dash'),
        name='Baseline Emissions (Pre-Response)',
        mode='lines',
        hovertemplate="Year: %{x}<br>Baseline: %{y:.2f} kt CO₂-e<extra></extra>"
    ))
    
    # Add 95% confidence interval band (between 95% Lower and 95% Upper)
    fig.add_trace(go.Scatter(
        x=filtered_emissions.index.tolist() + filtered_emissions.index.tolist()[::-1],
        y=filtered_emissions[confidence_upper].tolist() + filtered_emissions[confidence_lower].tolist()[::-1],
        fill='toself',
        fillcolor=band_colors['confidence'],
        line=dict(color=band_colors['confidence']),
        name='95% Confidence Interval',
        showlegend=True,
        hoverinfo='skip',
        hovertemplate="Year: %{x}<br>Emissions: %{y:.2f} kt CO₂-e<extra></extra>"
    ))
    
    # Add 1 standard error band (between 1 s.e lower and 1 s.e upper)
    fig.add_trace(go.Scatter(
        x=filtered_emissions.index.tolist() + filtered_emissions.index.tolist()[::-1],
        y=filtered_emissions[std_error_upper].tolist() + filtered_emissions[std_error_lower].tolist()[::-1],
        fill='toself',
        fillcolor=band_colors['std_error'],
        line=dict(color=band_colors['confidence']),
        name='1 Standard Error',
        showlegend=True,
        hoverinfo='skip',
        hovertemplate="Year: %{x}<br>Emissions: %{y:.2f} kt CO₂-e<extra></extra>"
    ))
    
    # Add central estimate line
    fig.add_trace(go.Scatter(
        x=filtered_emissions.index,
        y=filtered_emissions[central_scenario],
        line=dict(color=band_colors['central'], width=4),
        name='Emissions After Price Response',
        mode='lines',
        hovertemplate="Year: %{x}<br>Emissions: %{y:.2f} kt CO₂-e<extra></extra>"
    ))
    
    # Get historical data if available
    data_handler = getattr(model, 'data_handler', None)
    if data_handler is not None:
        try:
            # Try to get historical emissions data
            historical_emissions = data_handler.get_historical_data('emissions')
            
            if historical_emissions is not None:
                # Filter to only include years before the model start
                historical_years = historical_emissions.index[historical_emissions.index < min(filtered_emissions.index)]
                
                if len(historical_years) > 0:
                    historical_data = historical_emissions.loc[historical_years]
                    
                    # Add historical data trace
                    fig.add_trace(go.Scatter(
                        x=historical_data.index,
                        y=historical_data.values,
                        line=dict(color=NZUPY_CHART_STYLE["colors"]["reference_primary"], width=4, dash='solid'),
                        name='Historical Emissions',
                        mode='lines',
                        hovertemplate="Year: %{x}<br>Historical: %{y:.2f} kt CO₂-e<extra></extra>"
                    ))
        except Exception as e:
            print(f"Could not add historical data: {e}")
    
    # Update layout
    fig.update_layout(
        title='Emissions Pathway with Uncertainty',
        xaxis_title='Year',
        yaxis_title='Emissions (kt CO₂-e)',
        yaxis=dict(
            rangemode='nonnegative',
            range=[0, None]  # Set minimum to 0, let max auto-scale
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.9)',
            font=dict(size=12)
        ),
        margin=dict(t=150, r=50, l=50, b=50),
        height=600,
        width=900
    )
    
    # Apply NZUpy styling
    apply_nzupy_style(fig)
    
    return fig

def stockpile_chart_with_uncertainty(model, start_year: Optional[int] = None, 
                                    end_year: Optional[int] = None) -> go.Figure:
    """
    Generate stockpile chart with uncertainty bands for range scenario.
    
    Args:
        model: NZUpy instance with range scenario results
        start_year: Optional start year for chart
        end_year: Optional end year for chart
        
    Returns:
        Plotly figure object with uncertainty bands
    """
    fig = go.Figure()
    
    # Define scenario groups for uncertainty bands
    central_scenario = "central"
    std_error_lower = "1 s.e lower"
    std_error_upper = "1 s.e upper"
    confidence_lower = "95% Lower"
    confidence_upper = "95% Upper"
    
    # Get colors for uncertainty bands
    base_color = NZUPY_CHART_STYLE["colors"]["central"]
    band_colors = get_band_colors(base_color)
    
    # Check if model has the expected structure
    if not hasattr(model, 'stockpile') or not isinstance(model.stockpile, pd.DataFrame):
        raise ValueError("Model does not have a properly structured stockpile DataFrame")
    
    # Verify we have a multi-index with the right levels
    if not isinstance(model.stockpile.columns, pd.MultiIndex) or 'scenario' not in model.stockpile.columns.names or 'variable' not in model.stockpile.columns.names:
        raise ValueError("Stockpile DataFrame must have a multi-index with 'scenario' and 'variable' levels")
    
    # Extract stockpile balance data for all scenarios
    try:
        stockpile_df = model.stockpile.xs('balance', level='variable', axis=1)
    except KeyError:
        raise ValueError("Could not find 'balance' in the model's stockpile DataFrame")
    
    # Check if we have all the required scenarios
    required_scenarios = [central_scenario, std_error_lower, std_error_upper, confidence_lower, confidence_upper]
    missing_scenarios = [s for s in required_scenarios if s not in stockpile_df.columns]
    if missing_scenarios:
        raise ValueError(f"Missing required scenarios in stockpile DataFrame: {missing_scenarios}")
    
    # Filter by years if specified
    years_to_include = stockpile_df.index
    if start_year is not None:
        years_to_include = years_to_include[years_to_include >= start_year]
    if end_year is not None:
        years_to_include = years_to_include[years_to_include <= end_year]
    
    # Filter data to selected years
    filtered_df = stockpile_df.loc[years_to_include]
    
    # Add 95% confidence interval band (between 95% Lower and 95% Upper)
    fig.add_trace(go.Scatter(
        x=filtered_df.index.tolist() + filtered_df.index.tolist()[::-1],
        y=filtered_df[confidence_upper].tolist() + filtered_df[confidence_lower].tolist()[::-1],
        fill='toself',
        fillcolor=band_colors['confidence'],
        line=dict(color=band_colors['confidence']),
        name='95% Confidence Interval',
        showlegend=True,
        hoverinfo='skip',
        hovertemplate="Year: %{x}<br>Stockpile: %{y:.2f} kt CO₂-e<extra></extra>"
    ))
    
    # Add 1 standard error band (between 1 s.e lower and 1 s.e upper)
    fig.add_trace(go.Scatter(
        x=filtered_df.index.tolist() + filtered_df.index.tolist()[::-1],
        y=filtered_df[std_error_upper].tolist() + filtered_df[std_error_lower].tolist()[::-1],
        fill='toself',
        fillcolor=band_colors['std_error'],
        line=dict(color=band_colors['confidence']),
        name='1 Standard Error',
        showlegend=True,
        hoverinfo='skip',
        hovertemplate="Year: %{x}<br>Stockpile: %{y:.2f} kt CO₂-e<extra></extra>"
    ))
    
    # Add central estimate line
    fig.add_trace(go.Scatter(
        x=filtered_df.index,
        y=filtered_df[central_scenario],
        line=dict(color=band_colors['central'], width=4),
        name='Central Estimate',
        mode='lines',
        hovertemplate="Year: %{x}<br>Stockpile: %{y:.2f} kt CO₂-e<extra></extra>"
    ))
    
    # Try to add stockpile without forestry for the central scenario
    try:
        without_forestry_df = model.stockpile.xs('without_forestry', level='variable', axis=1)
        if central_scenario in without_forestry_df.columns:
            central_without_forestry = without_forestry_df[central_scenario].loc[years_to_include]
            
            fig.add_trace(go.Scatter(
                x=central_without_forestry.index,
                y=central_without_forestry.values,
                line=dict(color=NZUPY_CHART_STYLE["colors"]["reference_secondary"], width=4, dash='dash'),
                name='Without Forestry (Central)',
                mode='lines',
                hovertemplate="Year: %{x}<br>Without Forestry: %{y:.2f} kt CO₂-e<extra></extra>"
            ))
    except KeyError:
        print("Could not add 'without_forestry' trace - variable not found")
    
    # Update layout
    fig.update_layout(
        title='Stockpile Balance with Uncertainty',
        xaxis_title='Year',
        yaxis_title='Stockpile Balance (kt CO₂-e)',
        yaxis=dict(
            rangemode='tozero'  # Using 'tozero' since stockpile can be negative
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.9)',
            font=dict(size=12)
        ),
        margin=dict(t=150, r=50, l=50, b=50),
        height=600,
        width=900
    )
    
    # Apply NZUpy styling
    apply_nzupy_style(fig)
    
    return fig
