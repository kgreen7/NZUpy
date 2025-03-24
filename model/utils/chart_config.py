"""
NZUpy chart configuration and styling utilities.

This module provides styling configuration and helper functions for
consistent chart generation across the NZUpy model.
"""

import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List

# Colour schemes
QUALITATIVE_COLORS = px.colors.qualitative.Prism
DIVERGING_COLORS = px.colors.diverging.PRGn

# NZUpy chart styling configuration
NZUPY_CHART_STYLE = {
    # Colour palette
    "colors": {
        # Component colours (for supply stacked area charts)
        "auction": QUALITATIVE_COLORS[0].replace("rgb", "rgba").replace(")", ", 0.6)"),       
        "industrial": QUALITATIVE_COLORS[1].replace("rgb", "rgba").replace(")", ", 0.6)"),      
        "forestry": QUALITATIVE_COLORS[2].replace("rgb", "rgba").replace(")", ", 0.6)"),          
        "surplus": QUALITATIVE_COLORS[3].replace("rgb", "rgba").replace(")", ", 0.6)"),
        "non_surplus": QUALITATIVE_COLORS[4].replace("rgb", "rgba").replace(")", ", 0.6)"), 

        "stockpile": QUALITATIVE_COLORS[3].replace("rgb", "rgba").replace(")", ", 0.6)"),

        # Semi-transparent versions (with 0.3 alpha)
        "auction_transparent": QUALITATIVE_COLORS[0].replace("rgb", "rgba").replace(")", ", 0.4)"),
        "industrial_transparent": QUALITATIVE_COLORS[1].replace("rgb", "rgba").replace(")", ", 0.4)"),
        "forestry_transparent": QUALITATIVE_COLORS[2].replace("rgb", "rgba").replace(")", ", 0.4)"),
        "surplus_transparent": QUALITATIVE_COLORS[3].replace("rgb", "rgba").replace(")", ", 0.4)"),
        "non_surplus_transparent": QUALITATIVE_COLORS[4].replace("rgb", "rgba").replace(")", ", 0.4)"),

        "stockpile_transparent": QUALITATIVE_COLORS[3].replace("rgb", "rgba").replace(")", ", 0.4)"),


        "diverging_01_bold": DIVERGING_COLORS[1].replace("rgb", "rgba").replace(")", ", 0.6)"),   
        "diverging_01_transparent": DIVERGING_COLORS[1].replace("rgb", "rgba").replace(")", ", 0.4)"),
        "diverging_02_bold": DIVERGING_COLORS[2].replace("rgb", "rgba").replace(")", ", 0.6)"),
        "diverging_02_transparent": DIVERGING_COLORS[2].replace("rgb", "rgba").replace(")", ", 0.4)"),
        "diverging_03_bold": DIVERGING_COLORS[3].replace("rgb", "rgba").replace(")", ", 0.6)"),
        "diverging_03_transparent": DIVERGING_COLORS[3].replace("rgb", "rgba").replace(")", ", 0.4)"),
        "diverging_04_bold": DIVERGING_COLORS[4].replace("rgb", "rgba").replace(")", ", 0.6)"),
        "diverging_04_transparent": DIVERGING_COLORS[4].replace("rgb", "rgba").replace(")", ", 0.4)"),
        "diverging_05_bold": DIVERGING_COLORS[5].replace("rgb", "rgba").replace(")", ", 0.6)"),
        "diverging_05_transparent": DIVERGING_COLORS[5].replace("rgb", "rgba").replace(")", ", 0.4)"),
        "diverging_06_bold": DIVERGING_COLORS[6].replace("rgb", "rgba").replace(")", ", 0.6)"),
        "diverging_06_transparent": DIVERGING_COLORS[6].replace("rgb", "rgba").replace(")", ", 0.4)"),
        "diverging_07_bold": DIVERGING_COLORS[7].replace("rgb", "rgba").replace(")", ", 0.6)"),
        "diverging_07_transparent": DIVERGING_COLORS[7].replace("rgb", "rgba").replace(")", ", 0.4)"),
        "diverging_08_bold": DIVERGING_COLORS[8].replace("rgb", "rgba").replace(")", ", 0.6)"),
        "diverging_08_transparent": DIVERGING_COLORS[8].replace("rgb", "rgba").replace(")", ", 0.4)"),
        "diverging_09_bold": DIVERGING_COLORS[9].replace("rgb", "rgba").replace(")", ", 0.6)"),
        "diverging_09_transparent": DIVERGING_COLORS[9].replace("rgb", "rgba").replace(")", ", 0.4)"),
        
        
        # Scenario colours
        "central": QUALITATIVE_COLORS[2],       
        "lower": DIVERGING_COLORS[2],           
        "upper": DIVERGING_COLORS[8],       

        #Semi-transparent versions (with 0.3 alpha)
        "central_transparent": QUALITATIVE_COLORS[2].replace("rgb", "rgba").replace(")", ", 0.4)"),
        "lower_transparent": DIVERGING_COLORS[1].replace("rgb", "rgba").replace(")", ", 0.4)"),
        "upper_transparent": DIVERGING_COLORS[8].replace("rgb", "rgba").replace(")", ", 0.4)"),
        
        # Uncertainty bands
        "uncertainty": "rgba(153, 153, 153, 0.2)",
        
        # Reference lines
        "reference_primary": "rgb(47, 47, 47)",      # Dark slate grey
        "reference_secondary": "rgb(108, 108, 108)",  # Lighter slate grey
        "reference_secondary_transparent": "rgba(108, 108, 108, 0.2)",
        
        # Auction volume colours
        "auction_volume_dark": "rgb(17, 17, 17)",     # Darkest grey
        "auction_volume_medium": "rgb(108, 108, 108)", # Medium grey
        "auction_volume_light": "rgb(169, 169, 169)",  # Lightest grey
    },
    
    # Layout defaults
    "layout": {
        "font": {
            "family": "Arial, sans-serif",
            "size": 12,
        },
        "title": {
            "font": {
                "size": 16,
                "color": "rgb(47, 79, 79)",
            }
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1
        },
        "margin": {"l": 60, "r": 30, "t": 60, "b": 60},
        "hovermode": "closest",
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "xaxis": {
            "showgrid": True,
            "gridcolor": "#eeeeee",
            "title": {"standoff": 15}
        },
        "yaxis": {
            "showgrid": True,
            "gridcolor": "#eeeeee",
            "title": {"standoff": 15}
        },
    },
    
    # Chart-specific configurations
    "supply_components_order": ["auction", "industrial", "forestry", "non_surplus", "surplus"],
    
    "uncertainty_opacity": {
        "std_error": 0.2,     # 1 standard error band
        "confidence": 0.1     # 95% confidence interval
    }
}

def apply_nzupy_style(fig: go.Figure) -> go.Figure:
    """
    Apply NZUpy styling to a Plotly figure.
    
    Args:
        fig: Plotly figure to style
        
    Returns:
        Styled Plotly figure
    """
    fig.update_layout(
        font=NZUPY_CHART_STYLE["layout"]["font"],
        title=NZUPY_CHART_STYLE["layout"]["title"],
        legend=NZUPY_CHART_STYLE["layout"]["legend"],
        margin=NZUPY_CHART_STYLE["layout"]["margin"],
        hovermode=NZUPY_CHART_STYLE["layout"]["hovermode"],
        plot_bgcolor=NZUPY_CHART_STYLE["layout"]["plot_bgcolor"],
        paper_bgcolor=NZUPY_CHART_STYLE["layout"]["paper_bgcolor"],
    )
    fig.update_xaxes(**NZUPY_CHART_STYLE["layout"]["xaxis"])
    fig.update_yaxes(**NZUPY_CHART_STYLE["layout"]["yaxis"])
    return fig


def create_uncertainty_color(base_color: str, alpha: float = 0.2) -> str:
    """Create a transparent version of a colour for uncertainty bands"""
    # Handle both rgb and rgba formats
    if base_color.startswith('rgb('):
        # Extract the RGB components
        rgb = base_color.replace('rgb(', '').replace(')', '').split(',')
        return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"
    elif base_color.startswith('rgba('):
        # Replace the existing alpha
        rgba = base_color.replace('rgba(', '').replace(')', '').split(',')
        return f"rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {alpha})"
    else:
        # Return as-is if not in expected format
        return base_color

def get_band_colors(base_color: str) -> Dict[str, str]:
    """Generate colours for uncertainty bands based on a base colour"""
    return {
        'central': base_color,
        'std_error': create_uncertainty_color(base_color, NZUPY_CHART_STYLE["uncertainty_opacity"]["std_error"]),
        'confidence': create_uncertainty_color(base_color, NZUPY_CHART_STYLE["uncertainty_opacity"]["confidence"])
    }