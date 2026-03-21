"""
Chart display functionality for NZUpy model results.

This module provides functions to create HTML pages displaying charts generated
from NZUpy model results, making it easy to showcase visualisations without
manually writing HTML code.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import plotly.graph_objects as go
from datetime import datetime


def create_chart_page(charts: Dict[str, go.Figure], 
                     title: str = "NZUpy Model Results",
                     subtitle: Optional[str] = None,
                     output_dir: Optional[str] = None,
                     filename: str = "charts.html",
                     include_timestamp: bool = True,
                     chart_width: int = 900,
                     chart_height: int = 600,
                     additional_html: Optional[str] = None) -> str:
    """
    Create an HTML page displaying multiple charts.
    
    Args:
        charts: Dictionary mapping chart names to Plotly figure objects
        title: Main title for the HTML page
        subtitle: Optional subtitle to display below the main title
        output_dir: Directory to save the HTML file (if None, HTML is only returned)
        filename: Name of the HTML file to create
        include_timestamp: Whether to include a timestamp in the page
        chart_width: Width in pixels for each chart
        chart_height: Height in pixels for each chart
        additional_html: Optional HTML to include at the end of the page
        
    Returns:
        HTML content as a string
    """
    # Start building the HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                background-color: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                margin: 0;
                color: #2c3e50;
                font-size: 28px;
            }}
            h2 {{
                color: #7f8c8d;
                font-size: 18px;
                font-weight: normal;
                margin-top: 10px;
            }}
            .timestamp {{
                color: #95a5a6;
                font-size: 14px;
                margin-top: 10px;
            }}
            .charts-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 20px;
            }}
            .chart-wrapper {{
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                padding: 15px;
                margin-bottom: 20px;
                width: {chart_width}px;
            }}
            .chart-title {{
                text-align: center;
                font-size: 18px;
                margin-bottom: 10px;
                color: #3498db;
            }}
            .navigation {{
                position: fixed;
                top: 20px;
                right: 20px;
                background-color: #fff;
                padding: 10px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                max-height: 80vh;
                overflow-y: auto;
                z-index: 1000;
            }}
            .navigation h3 {{
                margin-top: 0;
                font-size: 16px;
                color: #2c3e50;
            }}
            .navigation ul {{
                list-style-type: none;
                padding: 0;
                margin: 0;
            }}
            .navigation li {{
                margin-bottom: 8px;
            }}
            .navigation a {{
                color: #3498db;
                text-decoration: none;
                font-size: 14px;
            }}
            .navigation a:hover {{
                text-decoration: underline;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                padding: 20px;
                font-size: 14px;
                color: #7f8c8d;
            }}
            @media (max-width: 768px) {{
                .chart-wrapper {{
                    width: 100%;
                }}
                .navigation {{
                    position: static;
                    margin-bottom: 20px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{title}</h1>
            {f'<h2>{subtitle}</h2>' if subtitle else ''}
            {f'<div class="timestamp">Generated on {datetime.now().strftime("%d %B %Y, %H:%M:%S")}</div>' if include_timestamp else ''}
        </div>
        
        <div class="navigation">
            <h3>Navigation</h3>
            <ul>
    """
    
    # Add navigation links
    for chart_name in charts.keys():
        chart_id = chart_name.lower().replace(' ', '-')
        html += f'        <li><a href="#{chart_id}">{chart_name}</a></li>\n'
    
    html += """
            </ul>
        </div>
        
        <div class="charts-container">
    """
    
    # Add each chart
    for chart_name, fig in charts.items():
        chart_id = chart_name.lower().replace(' ', '-')
        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        html += f"""
        <div class="chart-wrapper" id="{chart_id}">
            <div class="chart-title">{chart_name}</div>
            {chart_html}
        </div>
        """
    
    # Add a footer and close the HTML
    html += f"""
        </div>
        
        {'<div class="additional-content">' + additional_html + '</div>' if additional_html else ''}
        
        <div class="footer">
            Created with NZUpy - New Zealand Emissions Trading Scheme Model
        </div>
    </body>
    </html>
    """
    
    # Save the HTML file if output_dir is provided
    if output_dir:
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"Chart page saved to: {output_path}")
    
    return html


def create_comparison_page(models: Dict[str, Any], 
                          chart_type: str,
                          chart_params: Dict[str, Any] = None,
                          title: str = "NZUpy Model Comparison",
                          output_dir: Optional[str] = None,
                          filename: str = "comparison.html",
                          use_scenario_names: bool = False) -> str:
    """
    Create an HTML page comparing chart results from multiple model runs.
    
    Args:
        models: Dictionary mapping model names to NZUpy model instances or (model, scenario) tuples
        chart_type: Type of chart to create ('carbon_price', 'emissions_pathway', etc.)
        chart_params: Additional parameters to pass to the chart method
        title: Title for the comparison page
        output_dir: Directory to save the HTML file (if None, HTML is only returned)
        filename: Name of the HTML file to create
        use_scenario_names: If True, models are (model, scenario) tuples 
        
    Returns:
        HTML content as a string
    """
    from model.utils.chart_generator import ChartGenerator
    
    # Valid chart types
    valid_chart_types = [
        'carbon_price', 'emissions_pathway', 'supply_components', 
        'stockpile_balance', 'supply_demand_balance', 'auction_volume_revenue'
    ]
    
    if chart_type not in valid_chart_types:
        raise ValueError(f"Unknown chart type: {chart_type}. Valid types: {valid_chart_types}")
    
    chart_params = chart_params or {}
    
    # Generate charts for each model
    charts = {}
    for model_name, model_info in models.items():
        try:
            # Handle tuple format (model, scenario)
            if use_scenario_names and isinstance(model_info, tuple) and len(model_info) == 2:
                model, scenario = model_info
                # Create chart generator for this model
                chart_gen = ChartGenerator(model)
                
                # Get the chart method (e.g., carbon_price_chart)
                chart_method = getattr(chart_gen, f"{chart_type}_chart")
                
                # Generate the chart with specified scenario
                fig = chart_method(scenario=scenario, **chart_params)
            else:
                # Standard behavior for separate model instances
                model = model_info
                # Create chart generator for this model
                chart_gen = ChartGenerator(model)
                
                # Get the chart method (e.g., carbon_price_chart)
                chart_method = getattr(chart_gen, f"{chart_type}_chart")
                
                # Generate the chart
                fig = chart_method(**chart_params)
            
            # Update layout to include model name in title
            current_title = fig.layout.title.text
            fig.update_layout(title=f"{current_title} - {model_name}")
            
            charts[model_name] = fig
        except Exception as e:
            print(f"Error creating chart for model '{model_name}': {e}")
    
    # Create the HTML page
    return create_chart_page(
        charts=charts,
        title=title,
        subtitle=f"Comparison of {chart_type.replace('_', ' ').title()}",
        output_dir=output_dir,
        filename=filename
    )


