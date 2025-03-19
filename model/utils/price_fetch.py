"""
NZU Price Data Fetch Utility - awaiting integration into the model.

Downloads historical NZU price data from theecanmole's GitHub repository,
calculates yearly averages, and updates the historical_prices dictionary
in the NZUpy model configuration.

Usage as script:
    python -m model.utils.price_fetch [--output model/config.py] [--url URL]

Usage as module:
    from model.utils.price_fetch import get_nzu_price_data, calculate_yearly_averages
    
    # Get price data
    df = get_nzu_price_data()
    
    # Calculate yearly averages
    yearly_averages = calculate_yearly_averages(df)
"""

import argparse
import csv
import io
import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Union, List, Tuple
from model.utils.price_convert import convert_nominal_to_real, load_cpi_data


def get_nzu_price_data(url: Optional[str] = None, cpi_file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Download NZU price data from theecanmole GitHub repository or load from local file.
    
    Args:
        url: URL to the CSV data. If None, uses the default GitHub URL.
        cpi_file_path: Path to CPI data file. Required for real price conversion.
        
    Returns:
        DataFrame with date, price (nominal), and real_price (2023 NZD) columns
    """
    if cpi_file_path is None:
        raise ValueError("cpi_file_path must be provided for real price conversion")
        
    default_url = "https://raw.githubusercontent.com/theecanmole/nzu/refs/heads/master/nzu-month-price.csv"
    
    if url is None:
        url = default_url
        print(f"Sourcing data from theecanmole GitHub repository, refer license conditions: https://github.com/theecanmole/nzu/blob/master/Licence.txt.")
    
    # Check if the URL is a local file path
    if os.path.exists(url):
        print(f"Loading NZU price data from local file: {url}")
        df = pd.read_csv(url, parse_dates=['date'])
    else:
        # Otherwise, attempt to download from URL
        try:
            print(f"Downloading NZU price data from: {url}")
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception if the request failed
            
            # Parse CSV data
            csv_data = response.text
            df = pd.read_csv(io.StringIO(csv_data), parse_dates=['date'])
            
        except Exception as e:
            print(f"Error downloading NZU price data: {e}")
            print("Falling back to local file if available...")
            
            # Attempt to fall back to a local file with default name
            local_file = "nzu-month-price.csv"
            if os.path.exists(local_file):
                df = pd.read_csv(local_file, parse_dates=['date'])
            else:
                # If all else fails, raise the original exception
                raise
    
    # Add year column for grouping
    df['year'] = df['date'].dt.year
    
    # Load CPI data and convert nominal prices to real (2023 NZD)
    cpi_data = load_cpi_data(cpi_file_path)
    nominal_prices = df.groupby('year')['price'].mean()
    real_prices = convert_nominal_to_real(nominal_prices, cpi_data)
    
    # Add real prices to DataFrame
    df['real_price'] = df['year'].map(real_prices)
    
    return df


def calculate_yearly_averages(df: pd.DataFrame) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Calculate yearly average prices from monthly data.
    
    Args:
        df: DataFrame with 'date', 'price' (nominal), and 'real_price' columns
        
    Returns:
        Tuple of (nominal_yearly_avg, real_yearly_avg) dictionaries
    """
    # Extract year from date
    df['year'] = df['date'].dt.year
    
    # Calculate yearly averages for both nominal and real prices
    nominal_yearly_avg = df.groupby('year')['price'].mean()
    real_yearly_avg = df.groupby('year')['real_price'].mean()
    
    # Convert to dictionaries
    return nominal_yearly_avg.to_dict(), real_yearly_avg.to_dict()


def update_config_file(nominal_yearly_averages: Dict[int, float], 
                     real_yearly_averages: Dict[int, float],
                     output_file: Union[str, Path] = "model/config.py") -> bool:
    """
    Update the historical_prices dictionary in the config file.
    
    Args:
        nominal_yearly_averages: Dictionary mapping years to average nominal prices
        real_yearly_averages: Dictionary mapping years to average real prices (2023 NZD)
        output_file: Path to the config file to update
        
    Returns:
        True if successful, False otherwise
    """
    output_path = Path(output_file)
    
    # First check if file exists
    if not output_path.exists():
        print(f"Config file not found: {output_path}")
        print("Generating new config file with historical prices...")
        
        # Create a new config file with both nominal and real prices
        historical_prices_str = "# Historical NZU prices - generated by model.utils.price_fetch\n"
        historical_prices_str += "# Real prices are in 2023 NZD\n"
        historical_prices_str += "historical_prices = {\n"
        
        for year in sorted(set(nominal_yearly_averages.keys()) | set(real_yearly_averages.keys())):
            nominal = nominal_yearly_averages.get(year, 0.0)
            real = real_yearly_averages.get(year, 0.0)
            historical_prices_str += f"    {year}: {{'nominal': {nominal:.2f}, 'real': {real:.2f}}},\n"
        
        historical_prices_str += "}\n"
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(historical_prices_str)
        
        print(f"Created new config file with historical prices: {output_path}")
        return True
    
    # Read the existing file
    with open(output_path, 'r') as f:
        config_content = f.readlines()
    
    # Find where historical_prices is defined
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(config_content):
        if 'historical_prices = {' in line:
            start_idx = i
        elif start_idx is not None and '}' in line:
            end_idx = i
            break
    
    if start_idx is None or end_idx is None:
        print("Couldn't find historical_prices dictionary in config file")
        return False
    
    # Build the new historical_prices dictionary
    new_dict_lines = ["historical_prices = {\n"]
    
    for year in sorted(set(nominal_yearly_averages.keys()) | set(real_yearly_averages.keys())):
        nominal = nominal_yearly_averages.get(year, 0.0)
        real = real_yearly_averages.get(year, 0.0)
        new_dict_lines.append(f"    {year}: {{'nominal': {nominal:.2f}, 'real': {real:.2f}}},\n")
    
    new_dict_lines.append("}\n")
    
    # Replace the old dictionary with the new one
    new_config_content = config_content[:start_idx] + new_dict_lines + config_content[end_idx+1:]
    
    # Write the updated content back to the file
    with open(output_path, 'w') as f:
        f.writelines(new_config_content)
    
    print(f"Updated historical_prices in {output_path}")
    print(f"Added {len(nominal_yearly_averages)} years of NZU price data")
    return True


def get_latest_price(yearly_averages: Tuple[Dict[int, float], Dict[int, float]]) -> Tuple[int, float, float]:
    """
    Get the latest year and prices from the yearly averages.
    
    Args:
        yearly_averages: Tuple of (nominal_yearly_avg, real_yearly_avg) dictionaries
        
    Returns:
        Tuple of (latest_year, latest_nominal_price, latest_real_price)
    """
    nominal_avg, real_avg = yearly_averages
    
    if not nominal_avg or not real_avg:
        return (0, 0.0, 0.0)
    
    latest_year = max(set(nominal_avg.keys()) | set(real_avg.keys()))
    return (latest_year, nominal_avg.get(latest_year, 0.0), real_avg.get(latest_year, 0.0))


def main():
    """Main function to update NZU price data."""
    parser = argparse.ArgumentParser(description="Update NZU price data in NZUpy model")
    parser.add_argument("--output", type=str, default="model/config.py",
                        help="Output file to update (default: model/config.py)")
    parser.add_argument("--url", type=str, 
                        default="https://raw.githubusercontent.com/theecanmole/nzu/refs/heads/master/nzu-month-price.csv",
                        help="URL to CSV data (default: theecanmole/nzu GitHub repo)")
    parser.add_argument("--local", type=str, default=None,
                        help="Path to local CSV file (overrides URL)")
    parser.add_argument("--print-only", action="store_true",
                        help="Print the historical_prices dictionary without updating any file")
    parser.add_argument("--cpi-file", type=str, default="data/inputs/economic/CPI.csv",
                        help="Path to CPI data file (default: data/inputs/economic/CPI.csv)")
    
    args = parser.parse_args()
    
    # Use local file if provided
    data_source = args.local if args.local else args.url
    
    try:
        # Download or load the price data
        df = get_nzu_price_data(data_source, cpi_file_path=args.cpi_file)
        print(f"Loaded {len(df)} records of NZU price data")
        
        # Calculate yearly averages
        yearly_averages = calculate_yearly_averages(df)
        print(f"Calculated average prices for {len(yearly_averages[0])} years")
        
        # Print the dictionary
        if args.print_only:
            print("\nhistorical_prices = {")
            for year in sorted(set(yearly_averages[0].keys()) | set(yearly_averages[1].keys())):
                nominal = yearly_averages[0].get(year, 0.0)
                real = yearly_averages[1].get(year, 0.0)
                print(f"    {year}: {{'nominal': {nominal:.2f}, 'real': {real:.2f}}},")
            print("}")
            return
        
        # Update the config file
        success = update_config_file(yearly_averages[0], yearly_averages[1], args.output)
        
        if success:
            print("Successfully updated historical prices")
            # Get the latest prices for info
            latest_year, latest_nominal, latest_real = get_latest_price(yearly_averages)
            print(f"Latest prices (Year {latest_year}):")
            print(f"  Nominal: ${latest_nominal:.2f}")
            print(f"  Real (2023 NZD): ${latest_real:.2f}")
        else:
            print("Failed to update historical prices")
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()