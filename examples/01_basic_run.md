# Basic run: How to run NZUpy and customise inputs

## Overview of NZUpy
NZUpy is an optimisation model for simulating the New Zealand Emissions Trading Scheme (NZ ETS). The model simulates interactions between carbon price, emissions, various supply components and the stockpile of units held by private participants.

The model largely repliacates the functionality of the Ministry for the Environment's (MfE's) NZ ETS model that operates in excel (using macros/VBA). However given NZUpy is coded in Python, it allows for greater flexibility and customisation by users who wish to enhance certain components (e.g., incorporating more sophisticated industrial allocation behaviours). 

## Some basics

NZUpy is structured around the `NZUpy` class, which contains all information about the scenario(s), time periods, and components of the model (supply, demand, stockpile dynamics, etc.).

There are four main steps to running NZUpy:

1. **Creating an instance**: Initialise the model
2. **Define the scope**: Specify years, and scenarios
3. **Configure scenarios**: Customise your configuration of inputs and parameters for each scenario
4. **Run the model**: Execute the model and analyse results

The model follows a builder pattern, which allows for flexible configuration.

## 1. Create NZUpy instance

Let's start by importing the NZUpy class and libaries we'll use in the notebook


```python
# Import necessary libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
import os
```

As we're operating in a subfolder off the project root, we'll need to ensure we can find the NZUpy model, and its data directory containing necessary inputs and parameters.


```python
# Add the project root to the path
project_root = Path().absolute().parent
sys.path.insert(0, str(project_root))

# Import the NZUpy class
from model.core.base_model import NZUpy

# Set our input and output directories
data_dir = project_root / "data"
output_dir = project_root / "examples" / "outputs" / "01_basic_run"
os.makedirs(output_dir, exist_ok=True)
```

After that, we can initialise NZUpy, pointing it to our input data directory..


```python
NZU = NZUpy(data_dir=data_dir)
```

## 2. Define the scope

NZUpy requires users to define the time period they wish to model. 

By default, the model is set up for a 2024 start year, and 2050 end year (aligned with MfE's excel-based model), and is recommended for the vast bulk of use cases.


```python
# Define time periods: start year, end year
NZU.define_time(2024, 2050)
```

    Time periods defined:
      Optimisation: 2024-2050
    




    <model.core.base_model.NZUpy at 0x1e7776fc4a0>



For this run, we'll keep things simple with a single scenario. 

Scenario naming has no effect on the model itself, and is used to enable distinct runs with their own configured settings. So we'll call ours 'Basic run'


```python
# Define scenarios - you can have multiple scenarios
NZU.define_scenarios(['Basic run'])
```

    Defined 1 scenarios: Basic run
    




    <model.core.base_model.NZUpy at 0x1e7776fc4a0>



We then prime the model, which initialises dataframes housing our variables


```python
NZU.prime()
```

    Model primed with 1 scenarios:
      [0] Basic run
    




    <model.core.base_model.NZUpy at 0x1e7776fc4a0>



## 3. Configure the scenarios

After priming the model, we need to configure our scenario with the input data and parameters we want to use. 

NZUpy makes it easy to operate the model using central default inputs for all of its supply, demand and stockpile components, loaded in from CSV files in subfolders of `data/inputs/`.

We can use the `use_central_configs()` method to set all components to their central configurations for our scenario to get everything set-up quickly.



```python
# Configure our scenario (index 0) to use central configs for all components
NZU.use_central_configs(0)
```

    Using central config for emissions in scenario 0 (Basic run)
    Using central config for auction in scenario 0 (Basic run)
    Using central config for industrial in scenario 0 (Basic run)
    Using central config for forestry in scenario 0 (Basic run)
    Using central config for demand_model in scenario 0 (Basic run)
    Using central config for stockpile in scenario 0 (Basic run)
    Using central configs for all components in model scenario 0 (Basic run)
    




    <model.core.base_model.NZUpy at 0x1e7776fc4a0>



Most users will want to customise certain inputs however, such as loading in an alternative set of auction settings, or trying out a different govt. forecast for forestry removals. 

Each of the CSV files in `data/inputs/` subfolders contains a range of alternative configurations that draw on different data sources (e.g., MfE & MPI published projections, Climate Change Commission annual estimates of the surplus, etc.)

We can use the `list_available_configs()` method to quicly pull up alternative options for each component.


```python
# Show available configurations for each component
print("Available configurations:")
print(f"Emissions configs:", NZU.list_available_configs('emissions'))
print(f"Auction configs:", NZU.list_available_configs('auction'))
print(f"Industrial configs:", NZU.list_available_configs('industrial'))
print(f"Forestry configs:", NZU.list_available_configs('forestry'))
print(f"Demand model configs:", NZU.list_available_configs('demand_model'))
print(f"Stockpile configs:", NZU.list_available_configs('stockpile'))

```

    Available configurations:
    Emissions configs: ['CCC_CPR', 'CCC_DP', 'CCC_mid', 'central']
    Auction configs: ['CCC_2024', 'central']
    Industrial configs: ['central']
    Forestry configs: ['central', 'high', 'low']
    Demand model configs: ['95pc_lower', '95pc_upper', 'central', 'std_error', 'stde_lower', 'stde_upper']
    Stockpile configs: ['CCC22_central', 'CCC22_high', 'CCC22_low', 'CCC24_central', 'CCC24_high', 'CCC24_low', 'EY24_central', 'EY24_high', 'EY24_low', 'MFE24_central', 'MFE24_high', 'MFE24_low', 'central', 'high', 'low']
    

For this demonstration notebook, we'll look more closely at stockpile and surplus trends. 

We'll do this by swapping out NZUpy's default config (drawn from the Climate Change Commission's 2024 estimates) with a new Ernst & Young central estimate featured in a Sep 2024 report to the Ministry for the Environment. This stockpile config is listed as `EY24_central`.

_nb. Further metadata on where each config is sourced from can be found in respective CSV files in `data/inputs` subfolders._


```python
# Get stockpile data
print("~~~~~~~~| STOCKPILE & SURPLUS BEFORE EDIT |~~~~~~~~")
stockpile_data = NZU.show_inputs('stockpile', scenario_name='Basic run')

# Use the EY24_central config for our scenario (index 0)
NZU.use_config(0, 'stockpile', 'EY24_central')

print("~~~~~~~~| STOCKPILE & SURPLUS AFTER EDIT |~~~~~~~~")
# Get updated stockpile data
stockpile_data = NZU.show_inputs('stockpile', scenario_name='Basic run')

```

    ~~~~~~~~| STOCKPILE & SURPLUS BEFORE EDIT |~~~~~~~~
    
    === Stockpile Configuration for Scenario 'Basic run' ===
    Config name: central
    
    Stockpile Parameters:
      Initial Stockpile: 159,902 kt CO₂-e
      Initial Surplus: 68,100 kt CO₂-e
      Liquidity Factor: 12.00%
      Discount Rate: 5.00%
      Payback Period: 25 years
      Stockpile Usage Start Year: 2024
      Stockpile Reference Year: 2023
    Using EY24_central config for stockpile in scenario 0 (Basic run)
    ~~~~~~~~| STOCKPILE & SURPLUS AFTER EDIT |~~~~~~~~
    
    === Stockpile Configuration for Scenario 'Basic run' ===
    Config name: EY24_central
    
    Stockpile Parameters:
      Initial Stockpile: 159,902 kt CO₂-e
      Initial Surplus: 52,400 kt CO₂-e
      Liquidity Factor: 12.00%
      Discount Rate: 5.00%
      Payback Period: 25 years
      Stockpile Usage Start Year: 2024
      Stockpile Reference Year: 2023
    

_You'll see we now have a lower starting surplus balance for our reference year (2023), down to 52.4 million NZUs from the Climate Commission's central estimate of 68.1 million..._

## 4. Run the model

Now that we've configured our scenario, we can run the model:


```python
# Run the model
results = NZU.run()
```

    
    Running scenario 0: Basic run
    Completed NZUpy run for Basic run
    

## 5. Analyse the results

Now that we've got our results in tow, we can undertake any number of analyses. We'll keep things simple here, and call a list of variables available to us, so we can pick some result variables we're interested in


```python
NZU.list_variables()
```

    Available variables by category:
    
    Prices:
      self.prices - Carbon prices by scenario
      Access real prices: model.prices.xs('carbon_price', level='variable', axis=1)
      Access nominal prices: model.prices.xs('carbon_price_nominal', level='variable', axis=1)
      Example for specific scenario: model.prices[('central', 'carbon_price')]
      Example for nominal prices: model.prices[('central', 'carbon_price_nominal')]
    
    Core model outputs:
      self.core.xs('avg_price_change_rate', level='variable', axis=1) - Average annual price growth (%)
      self.core.xs('price_change_rate', level='variable', axis=1) - Annual price growth rate (%)
      self.core.xs('supply_demand_balance', level='variable', axis=1) - Supply minus demand by year (kt CO₂-e)
    
    Supply components:
      self.supply.xs('auction', level='variable', axis=1) - Units supplied through auction (kt CO₂-e)
      self.supply.xs('forestry', level='variable', axis=1) - Units from forestry removals (kt CO₂-e)
      self.supply.xs('industrial', level='variable', axis=1) - Units allocated to industrial activities (kt CO₂-e)
      self.supply.xs('stockpile', level='variable', axis=1) - Units from stockpile (kt CO₂-e)
      self.supply.xs('total', level='variable', axis=1) - Total unit supply (kt CO₂-e)
    
    Auction components:
      self.auctions.xs('base_available', level='variable', axis=1) - Base auction volume available (kt CO2-e)
      self.auctions.xs('base_supplied', level='variable', axis=1) - Base auction volumes actually supplied (kt CO2-e)
      self.auctions.xs('ccr1_available', level='variable', axis=1) - CCR1 auction volumes available (kt CO2-e)
      self.auctions.xs('ccr1_price', level='variable', axis=1) - CCR1 price trigger ($/tonne CO2-e)
      self.auctions.xs('ccr1_supplied', level='variable', axis=1) - CCR1 auction volumes supplied (kt CO2-e)
      self.auctions.xs('ccr2_available', level='variable', axis=1) - CCR2 auction volumes available (kt CO2-e)
      self.auctions.xs('ccr2_price', level='variable', axis=1) - CCR2 price trigger ($/tonne CO2-e)
      self.auctions.xs('ccr2_supplied', level='variable', axis=1) - CCR2 auction volumes supplied (kt CO2-e)
      self.auctions.xs('reserve_price', level='variable', axis=1) - Minimum auction price ($/tonne CO2-e)
      self.auctions.xs('revenue', level='variable', axis=1) - Annual auction revenue ($)
      self.auctions.xs('total_available', level='variable', axis=1) - Total auction available across tiers (kt CO2-e)
      self.auctions.xs('total_supplied', level='variable', axis=1) - Total auction supplied across tiers (kt CO2-e)
    
    Industrial allocation:
      self.industrial.xs('allocation', level='variable', axis=1) - Free allocation to industry (kt CO2-e)
    
    Forestry components:
      self.forestry.xs('removals', level='variable', axis=1) - Units supplied from forestry (kt CO2-e)
    
    Stockpile components:
      self.stockpile.xs('balance', level='variable', axis=1) - Total stockpile balance (kt CO2-e)
      self.stockpile.xs('borrowed_units', level='variable', axis=1) - Annual borrowing from stockpile (kt CO2-e)
      self.stockpile.xs('cumulative_forestry_additions', level='variable', axis=1) - Cumulative forestry held and surrender additions (kt CO2-e)
      self.stockpile.xs('cumulative_net_borrowing', level='variable', axis=1) - Cumulative net borrowing (kt CO2-e)
      self.stockpile.xs('forestry_contribution', level='variable', axis=1) - Net forestry contribution (kt CO2-e)
      self.stockpile.xs('forestry_held', level='variable', axis=1) - Forestry held contributions (kt CO2-e)
      self.stockpile.xs('forestry_surrender', level='variable', axis=1) - Forestry surrender contributions (kt CO2-e)
      self.stockpile.xs('net_borrowing', level='variable', axis=1) - Net borrowing (paybacks minus borrowing) (kt CO2-e)
      self.stockpile.xs('non_surplus_balance', level='variable', axis=1) - Non-surplus stockpile balance (kt CO2-e)
      self.stockpile.xs('non_surplus_used', level='variable', axis=1) - Non-surplus units used from stockpile (kt CO2-e)
      self.stockpile.xs('payback_units', level='variable', axis=1) - Annual stockpile repayment obligation (kt CO2-e)
      self.stockpile.xs('ratio_to_demand', level='variable', axis=1) - Stockpile divided by emissions (ratio)
      self.stockpile.xs('surplus_balance', level='variable', axis=1) - Surplus stockpile balance (kt CO2-e)
      self.stockpile.xs('surplus_used', level='variable', axis=1) - Surplus units used from stockpile (kt CO2-e)
      self.stockpile.xs('units_used', level='variable', axis=1) - Total units used from stockpile (kt CO2-e)
      self.stockpile.xs('without_forestry', level='variable', axis=1) - Stockpile balance excluding forestry (kt CO2-e)
    
    Demand components:
      self.demand.xs('baseline_emissions', level='variable', axis=1) - Baseline emissions (pre-response) (kt CO2-e)
      self.demand.xs('emissions', level='variable', axis=1) - Emissions after price response (kt CO2-e)
      self.demand.xs('gross_mitigation', level='variable', axis=1) - Difference baseline - emissions (kt CO2-e)
      self.demand.xs('net_mitigation', level='variable', axis=1) - Gross mitigation + forestry (kt CO2-e)
      self.demand.xs('payback_units', level='variable', axis=1) - Annual stockpile repayment obligation (kt CO2-e)
      self.demand.xs('total_demand_with_paybacks', level='variable', axis=1) - Total demand including payback obligations (kt CO2-e)
    
    Input parameters:
      self.inputs.xs('demand_model', level='variable', axis=1) - Price response model number (integer)
      self.inputs.xs('discount_rate', level='variable', axis=1) - Used for stockpile calculations (%)
      self.inputs.xs('liquidity_limit', level='variable', axis=1) - Annual non-surplus limit (%)
      self.inputs.xs('payback_period', level='variable', axis=1) - Years to pay back borrowed units (years)
      self.inputs.xs('price_response_forward_years', level='variable', axis=1) - Price response look ahead (years)
      self.inputs.xs('scenario_name', level='variable', axis=1) - Identifier for each scenario (-)
      self.inputs.xs('stockpile_start', level='variable', axis=1) - Initial stockpile volume (000s) (kt CO2-e)
      self.inputs.xs('surplus_start', level='variable', axis=1) - Initial surplus volume (000s) (kt CO2-e)
      self.inputs.xs('years', level='variable', axis=1) - Model years within start/end (year)
    

As we're dealing with pandas dataframes, we can use typical methods from numpy and pandas to get a quick sense of the results


```python
# Get carbon price data for our scenario
carbon_price = NZU.prices.xs('carbon_price', level='variable', axis=1)
print("Carbon price trajectory ($/tCO₂-e):")
print(carbon_price.head())
```

    Carbon price trajectory ($/tCO₂-e):
    scenario  Basic run
    year               
    2024      62.000000
    2025      65.038000
    2026      68.224862
    2027      71.567880
    2028      75.074706
    

But in this case, given we've swapped in an alternative starting surplus estimate, we're most interested in stockpile and surplus balance, so lets take a look at these.


```python
# Get stockpile balance data
stockpile_balance = NZU.stockpile.xs('balance', level='variable', axis=1)
surplus_balance = NZU.stockpile.xs('surplus_balance', level='variable', axis=1)
non_surplus_balance = NZU.stockpile.xs('non_surplus_balance', level='variable', axis=1)

print("\nStockpile balance in 2050 (kt CO₂-e):")
print(f"Total stockpile: {stockpile_balance.iloc[-1, 0]:,.0f}")
print(f"Surplus: {surplus_balance.iloc[-1, 0]:,.0f}")
print(f"Non-surplus: {non_surplus_balance.iloc[-1, 0]:,.0f}")
```

    
    Stockpile balance in 2050 (kt CO₂-e):
    Total stockpile: 153,492
    Surplus: 36,465
    Non-surplus: 117,027
    

These stockpile balance levels in 2050 are about the same as our balance at the start of time period (end 2023), so it would be interesting to see how these evolve over time.

Helpfully, we can call a pre-configured chart to make our life easier...


```python
# Load chart generator
from model.utils.chart_generator import ChartGenerator

# Initialise chart generator
chart_gen = ChartGenerator(NZU)

# Generate stockpile balance chart - referring to our scenario index 0
stockpile_chart = chart_gen.stockpile_balance_chart(NZU.scenarios[0])

# We can then display the chart in the notebook
display(stockpile_chart) 

# And we can also save the chart directly as PNG
stockpile_chart.write_image(str(output_dir / "stockpile_balance.png"))
```



With that we can see that while EY's reduced starting surplus balance sees the surplus quickly reduced to 0, the surplus begins to increase again in the late 2030s and 2040s... 

While there is plenty more analysis of this trend we could look at, we'll wrap-up this basic run notebook by saving these results to a CSV file.


```python
# Create a DataFrame with the key results we want to export
export_data = pd.DataFrame({
    'year': carbon_price.index,
    'carbon_price': carbon_price['Basic run'],
    'stockpile_balance': stockpile_balance['Basic run'],
    'surplus_balance': surplus_balance['Basic run'],
    'non_surplus_balance': non_surplus_balance['Basic run']
})

# Set the index to year for better readability
export_data.set_index('year', inplace=True)

# Export to CSV
export_path = output_dir / 'basic_run_results.csv'
export_data.to_csv(export_path)
print(f"Results exported to: {export_path}")
```

    Results exported to: c:\Users\nzkri\Python projects\NZUpy\examples\outputs\01_basic_run\basic_run_results.csv
    
