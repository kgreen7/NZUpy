# NZUpy: an NZ ETS supply-demand optimisation model

NZUpy is a Python model that simulates the interaction between unit supply and demand in the New Zealand Emissions Trading Scheme (NZ ETS), and uses calculation methods aligned to the Government's excel-based NZ ETS model.

The model calculates the price path that sees the closest balancing of supply and demand through to mid-century, incorporating stockpile dynamics, forestry response, and price-responsive demand by gross emitters.

## Overview

This model enables users to understand how different regulatory settings and scenarios affect NZU prices, emissions reductions, and market behaviour. NZUpy offers improved flexibility and extensibility compared to the spreadsheet-based implementations.

Key features include:

- **Carbon price projections** - Calculate price paths that balance supply and demand
- **Component-based architecture** - Modular design for supply (auctions, forestry, industrial allocation, stockpile) and demand (emissions, price response)
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

4. (Optional) If you plan to develop or modify the code, you can install the package in development mode:

```bash
pip install -e .
```

## Usage

### Basic Example

```python
from model.core.base_model import NZUpy
from model.utils.chart_generator import ChartGenerator

# Create a new model instance
nzu = NZUpy()

# Define time periods
nzu.define_time(2024, 2050)

# Define scenarios
nzu.define_scenarios(['Low Auction', 'Central', 'High Auction'])

# Prime the model
nzu.prime()

# Configure scenarios
nzu.use_central_configs(0)  # Set everything to central for the first scenario
nzu.set_parameter(0, "initial_stockpile", 159902)  # Set specific parameter

# Run the model
results = nzu.run()

# Visualise results
charts = ChartGenerator(nzu)
carbon_price_chart = charts.carbon_price_chart()
carbon_price_chart.show()
```

### Running Uncertainty Analysis

```python
# Define model with range scenarios
nzu = NZUpy()
nzu.define_time(2024, 2050)
nzu.define_scenario_type('Range')  # Use Range mode
nzu.define_scenarios(["95% Lower", "1 s.e lower", "central", "1 s.e upper", "95% Upper"])
nzu.prime()
nzu.configure_range_scenarios()  # Automatically configure the range scenarios
results = nzu.run()

# Generate uncertainty charts
charts = ChartGenerator(nzu)
price_chart = charts.carbon_price_chart()  # Will automatically use uncertainty bands
price_chart.show()
```

## Project Structure

The project is organised into the following directories:

- **model/** - Main model code
  - **core/** - Core model functionality and interfaces
  - **supply/** - Supply components (auction, forestry, industrial, stockpile)
  - **demand/** - Demand components (emissions, price response)
  - **utils/** - Utility functions and data handling
  - **interface/** - Charting and user interface functionality
- **data/** - Input data and model outputs
  - **inputs/** - Model input data
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

A small selection of features from the Government's model have yet to be implemented and are planned for inclusion in the coming weeks/months:

- Enable use of Manley equation afforestation response
- Improved representation of historical years data in output results
- Ability to run model with fxed (exogenous) NZU price paths
- Documentation

The model will also be updated following the release of the Climate Change Commission's forthcoming advice on 2026-2030 NZ ETS auction settings to incorporate their recommendations.

## Frequently asked questions

   ***Will you add additional features that build upon the Government's excel model? For example, SciPy optimisation, enhanced representation of industrial allocation...***

- There are currently no plans for this. This repository aims to serve as a base model with methods aligned with the Government's excel-based NZ ETS model. Those who wish to enhance functionality are encouraged to fork the repository for their own development.

***Can I use NZUpy for commercial purposes? Are there any fees for its use?***

- NZUpy is licensed under a permissive MIT license, so is free to use and adapt, including for commercial purposes (refer LICENSE file for specifics).

***What do you have planned for the repository? Will you update the repository following the Climate Commission's next report on NZ ETS auction settings for 2026-2030?***

- Yes, I plan to release an updated version of the model following annoucement of the Commission's recommendations for auction settings to allow users to easily incorporate these options.

## Contributions & issues

Contributions to NZUpy are welcome!

For issues: If you find a bug or have a suggestion, please open an issue on GitHub with a clear description.
