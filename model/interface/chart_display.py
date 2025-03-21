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


def create_single_charts_showcase(model, 
                                 output_dir: Optional[str] = None,
                                 filename: str = "single_charts.html",
                                 scenario: Optional[str] = None,
                                 start_year: Optional[int] = None,
                                 end_year: Optional[int] = None) -> str:
    """
    Create an HTML showcase of single scenario charts from an NZUpy model.
    
    Args:
        model: NZUpy model instance with results
        output_dir: Directory to save the HTML file (if None, HTML is only returned)
        filename: Name of the HTML file to create
        scenario: Scenario to use for charts (defaults to first scenario)
        start_year: Start year for charts (optional)
        end_year: End year for charts (optional)
        
    Returns:
        HTML content as a string
    """
    # Use the ChartGenerator class to create charts
    from model.utils.chart_generator import ChartGenerator
    
    # Create a chart generator instance
    chart_gen = ChartGenerator(model, model.data_handler)
    
    # Use first scenario if none specified
    if scenario is None and hasattr(model, 'scenarios') and model.scenarios:
        scenario = model.scenarios[0]
    
    # Set up charts dictionary with all single charts
    charts = {}
    
    try:
        charts["Carbon Price"] = chart_gen.carbon_price_chart(scenario, start_year, end_year)
    except Exception as e:
        print(f"Error creating carbon price chart: {e}")
    
    try:
        charts["Emissions Pathway"] = chart_gen.emissions_pathway_chart(scenario, start_year, end_year)
    except Exception as e:
        print(f"Error creating emissions pathway chart: {e}")
    
    try:
        charts["Supply Components"] = chart_gen.supply_components_chart(scenario, start_year, end_year)
    except Exception as e:
        print(f"Error creating supply components chart: {e}")
    
    try:
        charts["Stockpile Balance"] = chart_gen.stockpile_balance_chart(scenario, start_year, end_year)
    except Exception as e:
        print(f"Error creating stockpile balance chart: {e}")
    
    try:
        charts["Supply-Demand Balance"] = chart_gen.supply_demand_balance_chart(scenario, start_year, end_year)
    except Exception as e:
        print(f"Error creating supply-demand balance chart: {e}")
    
    try:
        charts["Auction Volume & Revenue"] = chart_gen.auction_volume_revenue_chart(scenario, start_year, end_year)
    except Exception as e:
        print(f"Error creating auction volume & revenue chart: {e}")
    
    # Create the HTML page
    subtitle = f"Scenario: {scenario}" if scenario else None
    return create_chart_page(
        charts=charts,
        title="NZUpy Single Scenario Results",
        subtitle=subtitle,
        output_dir=output_dir,
        filename=filename
    )


def create_range_charts_showcase(model, 
                                output_dir: Optional[str] = None,
                                filename: str = "range_charts.html",
                                start_year: Optional[int] = None,
                                end_year: Optional[int] = None) -> str:
    """
    Create an HTML showcase of range scenario charts from an NZUpy model.
    
    Args:
        model: NZUpy model instance with range scenario results
        output_dir: Directory to save the HTML file (if None, HTML is only returned)
        filename: Name of the HTML file to create
        start_year: Start year for charts (optional)
        end_year: End year for charts (optional)
        
    Returns:
        HTML content as a string
    """
    # Use the ChartGenerator class to create charts
    from model.utils.chart_generator import ChartGenerator
    
    # Create a chart generator instance for range scenarios
    chart_gen = ChartGenerator(model, model.data_handler)
    
    # Verify we're using a range scenario model
    if not chart_gen.is_range_scenario:
        print("Warning: Model does not appear to be a range scenario model. Charts may not display uncertainty bands.")
    
    # Set up charts dictionary with all range charts
    charts = {}
    
    try:
        charts["Carbon Price with Uncertainty"] = chart_gen.carbon_price_chart(start_year=start_year, end_year=end_year)
    except Exception as e:
        print(f"Error creating carbon price chart with uncertainty: {e}")
    
    try:
        charts["Emissions Pathway with Uncertainty"] = chart_gen.emissions_pathway_chart(start_year=start_year, end_year=end_year)
    except Exception as e:
        print(f"Error creating emissions pathway chart with uncertainty: {e}")
    
    try:
        charts["Stockpile Balance with Uncertainty"] = chart_gen.stockpile_balance_chart(start_year=start_year, end_year=end_year)
    except Exception as e:
        print(f"Error creating stockpile chart with uncertainty: {e}")
    
    # Create the HTML page
    return create_chart_page(
        charts=charts,
        title="NZUpy Range Scenario Results",
        subtitle="Showing central, standard error, and 95% confidence intervals",
        output_dir=output_dir,
        filename=filename
    )


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


def save_all_charts(model, 
                   output_dir: str,
                   formats: List[str] = ['html', 'png'], 
                   scenario: Optional[str] = None,
                   start_year: Optional[int] = None,
                   end_year: Optional[int] = None) -> Dict[str, List[str]]:
    """
    Save all charts in multiple formats.
    
    Args:
        model: NZUpy model instance
        output_dir: Directory to save charts
        formats: List of formats to save ('html', 'png', 'svg', 'pdf', 'json')
        scenario: Scenario to use (defaults to first scenario)
        start_year: Start year for charts (optional)
        end_year: End year for charts (optional)
        
    Returns:
        Dictionary mapping chart names to lists of saved file paths
    """
    # Use the ChartGenerator class
    from model.utils.chart_generator import ChartGenerator
    
    # Use first scenario if none specified
    if scenario is None and hasattr(model, 'scenarios') and model.scenarios:
        scenario = model.scenarios[0]
    
    # Create chart generator
    chart_gen = ChartGenerator(model, model.data_handler)
    
    # Define chart types
    chart_types = [
        'carbon_price',
        'emissions_pathway',
        'supply_components',
        'stockpile_balance',
        'supply_demand_balance',
        'auction_volume_revenue'
    ]
    
    # Ensure output directory exists
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # Generate and save each chart in requested formats
    for chart_type in chart_types:
        try:
            # Get the chart method
            chart_method = getattr(chart_gen, f"{chart_type}_chart")
            
            # Generate the chart
            fig = chart_method(scenario, start_year, end_year)
            
            saved_files[chart_type] = []
            
            for fmt in formats:
                file_path = out_path / f"{chart_type}.{fmt}"
                
                if fmt == 'html':
                    fig.write_html(file_path)
                elif fmt in ['png', 'svg', 'pdf', 'jpeg']:
                    fig.write_image(file_path)
                elif fmt == 'json':
                    fig.write_json(file_path)
                
                saved_files[chart_type].append(str(file_path))
                print(f"Saved {chart_type} as {file_path}")
                
        except Exception as e:
            print(f"Error saving {chart_type} chart: {e}")
    
    # Also create the combined HTML showcase
    try:
        showcase_path = out_path / "showcase.html"
        html = create_single_charts_showcase(
            model, 
            output_dir=output_dir,
            filename="showcase.html",
            scenario=scenario,
            start_year=start_year,
            end_year=end_year
        )
        saved_files['showcase'] = [str(showcase_path)]
    except Exception as e:
        print(f"Error creating showcase: {e}")
    
    return saved_files
