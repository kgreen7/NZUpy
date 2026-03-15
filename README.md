# NZUpy: an NZ ETS supply-demand optimisation model

NZUpy is a Python model that simulates the interaction between unit supply and demand in the New Zealand Emissions Trading Scheme (NZ ETS), and uses calculation methods aligned to the Government's excel-based NZ ETS model.

The model calculates the price path that sees the closest balancing of supply and demand through to mid-century, incorporating stockpile dynamics, forestry response, and price-responsive demand by gross emitters.

## Overview

This model enables users to understand how different regulatory settings and scenarios affect NZU prices, emissions reductions, and market behaviour. NZUpy offers improved flexibility and extensibility compared to the spreadsheet-based implementations.

Key features include:

- **Carbon price projections** - Calculate price paths that balance supply and demand
- **Component-based architecture** - Modular design for supply (auctions, forestry, industrial allocation, stockpile) and demand (emissions, price response)
- **Endogenous forestry** - Price-responsive afforestation using the Manley logistic model, matching the Excel model's methodology
- **Scenario analysis** - Run and compare multiple scenarios with different input configurations
- **Uncertainty analysis** - Generate uncertainty bands around central projections
- **Visualisation tools** - Create standardised charts for model outputs
- **Historical data integration** - Combine model projections with historical data

## Installation

### Requirements

- Python 3.8 or higher

### Setup

1. Clone the repository:

```bash
git clone https://github.com/kgreen7/NZUpy.git
cd NZUpy
```

2. Create and activate a virtual environment (recommended):

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

(Optional) If you plan to develop or modify the code, you can install the package in development mode:

```bash
pip install -e .
```

## Usage

### Basic Example

```python
from model.core.base_model import NZUpy
from model.utils.chart_generator import ChartGenerator

# Initialise model
nzu = NZUpy()
nzu.define_time(2024, 2050)
nzu.define_scenarios(['Baseline'])
nzu.allocate()

# Load central configurations for all components
nzu.fill_defaults()

# Override individual components or parameters as needed
nzu.fill_component('stockpile', config='EY24_central')
nzu.fill_component('emissions', config='CCC_CPR', scenario='Baseline')

# Run and access results
nzu.run()
print(nzu.prices)   # carbon price path

# Visualise results
charts = ChartGenerator(nzu)
carbon_price_chart = charts.carbon_price_chart()
carbon_price_chart.show()
```

### Endogenous Forestry (Manley Model)

```python
nzu = NZUpy()
nzu.define_time(2024, 2050)
nzu.define_scenarios(['Endogenous'])
nzu.allocate()
nzu.fill_defaults()
nzu.fill_component('stockpile', config='EY24_central')

# Enable price-responsive afforestation
nzu.fill('forestry_mode', 'endogenous')
nzu.fill('manley_sensitivity', 'central')  # 'low', 'central', or 'high'

nzu.run()
print(nzu.forestry)  # includes Manley planting and total removals
```

### Running Uncertainty Analysis

```python
# Range mode automatically sets up 5 demand sensitivity scenarios
nzu = NZUpy()
nzu.define_time(2024, 2050)
nzu.define_scenario_type('Range')
nzu.allocate()
nzu.fill_range_configs()  # configures each sensitivity level from CSV
nzu.run()

# Generate uncertainty charts
charts = ChartGenerator(nzu)
price_chart = charts.carbon_price_chart()  # renders with uncertainty bands
price_chart.show()
```

## Project Structure

The project is organised into the following directories:

- **model/** - Main model code
  - **core/** - Core model functionality
  - **supply/** - Supply components (auction, forestry, industrial, stockpile)
  - **demand/** - Demand components (emissions, price response)
  - **utils/** - Utility functions and data handling
  - **interface/** - Charting and user interface functionality
- **data/** - Input data and reference material
  - **inputs/** - Model input data
  - **reference/** - Provides reference material for model and excel model that NZUpy is based on.
- **examples/** - Example notebooks demonstrating operation of model
  - **outputs/** - Results data for example notebooks

## Model Components

### Supply Components

- **Auction** - Government auction volumes and cost containment reserve
- **Industrial Allocation** - Free allocation to emissions-intensive, trade-exposed industries
- **Forestry** - Supply from forestry removals
- **Stockpile** - Accumulated units in private accounts and how these levels change over time

### Demand Components

- **Emissions** - Gross emissions in the NZ ETS
- **Price Response** - How gross emissions respond to price changes

## Data Structure

The model uses a standardised data structure for results with multi-index columns:

- First level: Scenario name
- Second level: Variable name

Example access patterns:

```python
# Access carbon prices for the 'central' scenario
central_prices = model.prices[('central', 'carbon_price')]

# Access emissions for all scenarios
emissions = model.demand.xs('emissions', level='variable', axis=1)
```

## Forthcoming features

- Improved representation of historical years data in output results
- Ability to run model with fixed (exogenous) NZU price paths
- Expanded documentation and user guide

## Frequently asked questions

   ***Will you add additional features that build upon the Government's excel model? For example, SciPy optimisation, enhanced representation of industrial allocation...***

- There are currently no plans for this. This repository aims to serve as a base model with methods aligned with the Government's excel-based NZ ETS model. Those who wish to enhance functionality are encouraged to fork the repository for their own development.

***Can I use NZUpy for commercial purposes? Are there any fees for its use?***

- NZUpy is licensed under a permissive MIT license, so is free to use and adapt, including for commercial purposes (refer LICENSE file for specifics).

***Will you update the repository to reflect new NZ ETS settings as they are announced?***

- Yes, the model will be updated as new auction settings and policy parameters are published to allow users to easily incorporate these.

## Contributions & issues

Contributions to NZUpy are welcome!

For issues: If you find a bug or have a suggestion, please open an issue on GitHub with a clear description.
