# NZUpy — API & Module Reference

> **Purpose**: Living source-of-truth for what each public method does, what each internal module is responsible for, what mode switches exist, and what charts the model produces.  Updated whenever the API surface changes.

---

## 1  Public API — methods users call in notebooks

Methods are grouped by workflow stage.  The intended call order is:

```
define_time → define_scenarios / define_scenario_type → allocate → fill_defaults
  → fill_component / fill / set_mode → run → access results
```

All `define_*`, `allocate`, `fill_*`, `set_mode`, and `run` return `self` for method chaining.

### 1.1  Setup stage

| Method | What it does | Key arguments |
|--------|-------------|---------------|
| `define_time(start, end)` | Set the model period (calendar years). | `start_year: int`, `end_year: int` |
| `define_scenarios(names)` | Create named scenarios. | `scenario_names: List[str]` |
| `define_scenario_type('Range')` | Runs the model using 5 standard demand-sensitivity scenarios (`95% Lower` … `95% Upper`). | `scenario_type: str` |
| `allocate()` | Initialise internal data structures. Must follow `define_time` + `define_scenarios`. | — |

### 1.2  Configuration stage — data selection

These methods control **which data** each component uses.

| Method | What it does | Key arguments | Example |
|--------|-------------|---------------|---------|
| `fill_defaults(config)` | Set every component to their default `config` for all scenarios. | `config: str` (default `'central'`) | `nzu.fill_defaults()` |
| `fill_component(component, config, scenario)` | Select a named config for one component; optionally scoped to one scenario.  Valid components: `'auction'`, `'forestry'`, `'industrial'`, `'emissions'`, `'demand_model'`, `'stockpile'`, `'price'` (price control config). | `component: str`, `config: str`, `scenario: str \| int \| None` | `nzu.fill_component('emissions', config='CCC_CPR', scenario='Alt')` · `nzu.fill_component('price', config='scarcity_then_surplus')` |
| `fill(variable, value, scenario, component)` | Set an individual variable value (scalar or `pd.Series`). Component is auto-inferred from the variable name in most cases. | `variable_name: str`, `value: Any`, `scenario: str \| int \| None` | `nzu.fill('liquidity_factor', 0.15)` |
| `fill_range_configs()` | Map each Range scenario to its demand-sensitivity config. Call after `define_scenario_type('Range')`. | — | `nzu.fill_range_configs()` |

### 1.3  Configuration stage — mode switches

> **`set_mode()`** — switches structural model behaviours.  Distinct from `fill()` (data values) and `fill_component()` (named data configurations).

| Method | What it does | Key arguments | Example |
|--------|-------------|---------------|---------|
| `set_mode(mode_name, value, scenario)` | Set a structural mode switch for one or all scenarios. | `mode_name: str`, `value: Any`, `scenario: str \| int \| None` | `nzu.set_mode('forestry_mode', 'endogenous')` |

See **Section 2** for the full table of mode switches.

### 1.4  Execution

| Method | What it does | Notes |
|--------|-------------|-------|
| `run()` | Run all scenarios (optimise or forward-calc depending on `pricing_mode`). Returns results dict. | Calls `validate()` internally. |

### 1.5  Results access — DataFrame properties

After `run()`, structured results are available as MultiIndex DataFrames with columns `(scenario, variable)`:

| Property | Variables it contains |
|----------|---------------------|
| `.prices` | `carbon_price`, `carbon_price_nominal`, `unmodified_carbon_price`, `unmodified_carbon_price_nominal` |
| `.supply` | `auction`, `industrial`, `forestry`, `stockpile`, `total` |
| `.demand` | `baseline`, `emissions`, `gross_mitigation`, `net_mitigation` |
| `.stockpile` | `balance`, `surplus_balance`, `non_surplus_balance`, `units_used`, `surplus_used`, `non_surplus_used`, `forestry_held`, `forestry_surrender`, `payback_units`, `borrowed_units`, and others |
| `.forestry` | `removals` (exogenous), plus `historic_supply`, `manley_supply`, `manley_price`, `manley_planting_total`, `manley_planting_permanent`, `manley_planting_production`, `manley_planting_natural` (endogenous) |
| `.auctions` | `base_available`, `base_supplied`, `ccr1_available`, `ccr1_supplied`, `ccr2_available`, `ccr2_supplied`, `total_available`, `total_supplied`, `reserve_price`, `ccr1_price`, `ccr2_price`, `revenue` |

### 1.6  Results access — helper methods

| Method | What it does |
|--------|-------------|
| `list_variables()` | Print all available output variables with descriptions and units. |
| `variable_info(name)` | Return metadata dict for a specific variable. |
| `get_scenario_result(name_or_index)` | Return raw results dict for one scenario. |

### 1.7  Inspection / discovery

These help users explore available input data before running.

| Method | What it does | Example |
|--------|-------------|---------|
| `list_configs(component)` | Show available named configs for a component (reads from CSV). | `nzu.list_configs('emissions')` |
| `show_config(component, config)` | Display data values for a specific config. | `nzu.show_config('forestry', 'central')` |
| `compare_configs(component, a, b)` | Side-by-side comparison of two configs. | `nzu.compare_configs('emissions', 'central', 'CCC_CPR')` |
| `show_current(scenario)` | Show which config is loaded for each component (for one or all scenarios). | `nzu.show_current()` |

### 1.8  Charting

| Method | What it does |
|--------|-------------|
| `ChartGenerator(model)` | Create chart generator. Pass the NZUpy instance after `run()`. |
| `.carbon_price_chart()` | Carbon price chart (single) or with uncertainty bands (range). |
| `.emissions_pathway_chart()` | Emissions pathway chart. |
| `.supply_components_chart()` | Stacked supply components (single-scenario only). |
| `.stockpile_balance_chart()` | Stockpile balance chart. |
| `.supply_demand_balance_chart()` | Supply minus demand balance (single-scenario only). |
| `.auction_volume_revenue_chart()` | Auction volumes and revenue (single-scenario only). |
| `.generate_standard_charts()` | Generate all applicable charts at once; optionally save to files. |

---

## 2  Mode switches — what `set_mode()` controls

| Mode variable | Valid values | Default | What it changes |
|---|---|---|---|
| `forestry_mode` | `'exogenous'`, `'endogenous'` | `'exogenous'` | Exogenous: fixed forestry series from `removals.csv`.  Endogenous: Manley logistic equation drives new planting; historic forest supply from `historical_removals.csv`. |
| `pricing_mode` | `'optimised'`, `'fixed_path'`, `'fixed_rate'` | `'optimised'` | Optimised: FastSolve finds optimal `price_change_rate`.  Fixed path: user supplies year-by-year `pd.Series` via `fill('price_path', series)`.  Fixed rate: user supplies scalar via `fill('price_change_rate', value)`.  In both fixed modes the optimiser is skipped. |
| `penalise_shortfalls` | `True`, `False` | `False` | Whether supply shortfalls get a 1,000,000× penalty in the gap objective function.  When `True`, the optimiser avoids price paths that produce supply deficits.  Matches the Excel model's asymmetric penalty. |
| `manley_sensitivity` | `'low'`, `'central'`, `'high'` | `'central'` | Selects the `f` (available land) and `LMV` (land market value) parameters for the Manley equation.  Only relevant when `forestry_mode='endogenous'`. |
| `forestry_price_assumption` | `'future'`, `'current'` | `'future'` | Future: Manley uses NPV-weighted forward carbon price.  Current: uses spot price.  Only relevant when `forestry_mode='endogenous'`. |

---

## 3  Variable routing — `fill()` targets

The `_VARIABLE_COMPONENT_MAP` dict in `base_model.py` routes `fill()` calls.  Key mappings:

| Variable | Component | Type | Notes |
|---|---|---|---|
| `base_volume` | auction | time-series | Annual auction volume |
| `auction_reserve_price` | auction | time-series | Reserve / floor price |
| `ccr_trigger_price_1` / `ccr_volume_1` | auction | time-series | Cost containment reserve tier 1 |
| `ccr_trigger_price_2` / `ccr_volume_2` | auction | time-series | Cost containment reserve tier 2 |
| `forestry_tradeable` | forestry | time-series | Exogenous forestry supply |
| `forestry_held` / `forestry_surrender` | forestry | time-series | Held/surrender affecting stockpile |
| `manley_f` / `manley_LMV` / `manley_LUC_limit` | forestry | scalar | Manley equation overrides |
| `forestry_discount_rate` / `forestry_forward_years` | forestry | scalar | Forestry NPV overrides |
| `baseline_allocation` | industrial | time-series | Industrial allocation volume |
| `emissions_baseline` | emissions | time-series | Baseline emissions pathway |
| `demand_model_number` | demand_model | scalar (1 or 2) | 1 = MACC, 2 = ENZ log-linear |
| `initial_stockpile` / `initial_surplus` | stockpile | scalar | Opening balances |
| `liquidity_factor` | stockpile | scalar (0–1) | Fraction of non-surplus available |
| `discount_rate` | stockpile | scalar (0–1) | Requires `component='stockpile'` to disambiguate |
| `payback_period` | stockpile | scalar (int) | Years until borrowed units repaid |
| `stockpile_usage_start_year` / `stockpile_reference_year` | stockpile | scalar (int) | When stockpile kicks in / baseline year |
| `start_price` | price | scalar | Starting carbon price (overrides last historical) |
| `price_control` | price | time-series | Per-year price control parameter |
| `price_path` | price | time-series | Fixed price path (used with `pricing_mode='fixed_path'`) |
| `price_change_rate` | price | scalar | Fixed rate (used with `pricing_mode='fixed_rate'`) |

---

## 4  Internal module map

### 4.1  `model/core/` — orchestration and calculation

**`base_model.py`** — `NZUpy` class.  Central coordinator.  Holds `component_configs` list (one `ComponentConfig` per scenario).  Delegates setup to `ScenarioManager`, execution to `ModelRunner`, calculations to `CalculationEngine`.  Owns all public API methods.  The `_VARIABLE_COMPONENT_MAP` class attribute routes `fill()` calls to the correct component.

**`runner.py`** — `ModelRunner`.  Orchestrates the run loop: iterates over scenarios → calls `ScenarioManager._initialise_scenario_components()` to load data and create component objects → calls the optimiser or forward calculator depending on `pricing_mode` → calls `OutputFormat` to structure results.  Key entry point: `run()`.  Key internal methods: `_run_scenario()` → `_run_scenario_optimisation()` (optimised) / `_run_fixed_path()` (fixed_path) / `_run_fixed_rate()` (fixed_rate).

**`calculation_engine.py`** — `CalculationEngine`.  The computational workhorse.  `calculate_gap(rate)` is the objective function the optimiser minimises.  Internally: (1) set price_change_rate, (2) calculate prices from rate + price control + constraints, (3) calculate base supply (auction + industrial + forestry), (4) calculate demand (baseline emissions − price response), (5) calculate stockpile (fills supply–demand shortfalls), (6) iterate to convergence, (7) return total gap.  Also handles price revision based on stockpile state (Hotelling-rule mechanism).

**`optimiser.py`** — `FastSolveOptimiser`.  Two-phase grid search (coarse then fine) matching the Excel VBA `FastSolve` macro.  Calls `calculate_gap()` repeatedly with different rate values.  Returns optimal rate + metadata.  Algorithm constants: `coarse_step`, `fine_step`, `max_rate`.

**`scenario_manager.py`** — `ScenarioManager`.  Reads a scenario's `ComponentConfig`, loads data from `DataHandler`, instantiates the component objects (`AuctionSupply`, `ForestrySupply`, etc.) onto the model.  Key method: `_initialise_scenario_components()`.  Also handles Range scenario configuration via `configure_range_scenarios()`.

### 4.2  `model/supply/` — supply components

**`auction.py`** — `AuctionSupply`.  Calculates auction volumes given prices: base volume always available; CCR tier 1 released if price ≥ trigger 1; CCR tier 2 if price ≥ trigger 2 (and CCR1 active).  No auction below reserve price.  Also calculates revenue.

**`forestry.py`** — `ForestrySupply`.  Two modes:
- *Exogenous*: passes through `forestry_tradeable` from `removals.csv`.
- *Endogenous*: historic forests contribute fixed supply from `historical_removals.csv`.  New forests are planted via the Manley logistic equation (driven by forward carbon price vs land cost), then convolved with yield curves to produce removals.  New planting is proportioned across permanent/production/natural using MPI afforestation projection ratios.

**`industrial.py`** — `IndustrialAllocation`.  Simple pass-through of `baseline_allocation` from CSV.  No phase-out or activity adjustment applied (matches Excel model).

**`stockpile.py`** — `StockpileSupply`.  Most complex supply component.  Tracks two pools: surplus (fully available) and non-surplus (available at `liquidity_factor` rate, with payback).  Surplus is used first; non-surplus only when price growth < discount rate.  Excess supply (demand < supply) replenishes surplus.  Borrowed non-surplus units are scheduled for payback after `payback_period` years.  Forestry held/surrender flows affect the stockpile balance.

### 4.3  `model/demand/` — demand components

**`emissions.py`** — `EmissionsDemand`.  Loads baseline emissions from config.  Applies emissions reductions from price response to produce net demand.

**`price_response.py`** — `PriceResponse`.  Calculates emissions reduction as a function of carbon price.  Two models: Model 1 (MACC, linear) and Model 2 (ENZ, log-linear).  Uses NPV-weighted forward price over configurable horizon.  Reduction is bounded between 0 and baseline emissions.

### 4.4  `model/utils/` — data loading and output formatting

**`data_handler.py`** — `DataHandler`.  Loads all CSV input files on init.  Provides getter methods filtered by config name: `get_auction_data(config)`, `get_forestry_data(config)`, `get_emissions_data(config)`, `get_stockpile_parameters(config)`, `get_demand_model(config, model_number)`, etc.  Also loads endogenous forestry data: `get_historical_removals()`, `get_yield_increments()`, `get_afforestation_projections()`, `get_manley_parameters()`.  Stores scenario-specific overrides in `scenario_data[scenario_name][component]` dict.

**`output_format.py`** — `OutputFormat`.  Takes raw results dicts from `ModelRunner` and organises into the MultiIndex DataFrames users access via `nzu.prices`, `nzu.supply`, etc.  Provides `list_variables()` and `variable_info()`.  Contains the `_variable_schema` metadata dict describing every output variable.

**`historical_data_manager.py`** — `HistoricalDataManager`.  Loads historical carbon price, CPI data, and price control configs from CSVs in `data/inputs/`.  Provides lookups used by `CalculationEngine` for historical price anchoring and price control values.

**`price_convert.py`** — Real ↔ nominal price conversion using CPI index.  Used by `OutputFormat` to produce nominal price columns.

**`chart_config.py`** — `NZUPY_CHART_STYLE` dict and helper functions (`get_band_colors`, etc.) providing consistent chart colours, fonts, and layout constants for all charts.

**`chart_generator.py`** — `ChartGenerator` class.  User-facing charting interface.  Auto-detects single vs Range scenario mode and delegates to the appropriate chart functions in `interface/`.  Also provides `generate_standard_charts()` and `export_csv_data()`.

### 4.5  `model/interface/` — chart rendering

**`single_charts.py`** — Individual Plotly chart functions for single-scenario mode: `carbon_price_chart()`, `emissions_pathway_chart()`, `supply_components_chart()`, `stockpile_balance_chart()`, `supply_demand_balance_chart()`, `auction_volume_revenue_chart()`.

**`range_charts.py`** — Chart functions for Range/uncertainty mode: `carbon_price_chart_with_uncertainty()`, `emissions_pathway_chart_with_uncertainty()`, `stockpile_chart_with_uncertainty()`.  These render the central scenario with shaded bands for the outer scenarios.

**`chart_display.py`** — HTML page generation.  `create_chart_page()` renders a set of Plotly figures into a single HTML file.  `create_comparison_page()` generates side-by-side chart comparisons across multiple model instances.

### 4.6  `model/config.py` — configuration dataclasses

**`ModelConfig`** — Temporal boundaries (`start_year`, `end_year`), price control values dict, and optimiser settings (`coarse_step`, `fine_step`, `max_rate`, `max_iterations`, `debug`, `penalise_shortfalls`).

**`ComponentConfig`** — Per-scenario configuration.  Holds the config name for each component (auction, forestry, industrial, emissions, demand_sensitivity, stockpile, model_params).  Also holds scalar parameter overrides (initial_stockpile, liquidity_factor, etc.), forestry mode settings (forestry_mode, manley_sensitivity, forestry_price_assumption), and Manley parameter overrides.

---

## 5  Standard charts catalogue

| Chart | Generator method | Underlying function | Notebook(s) | Single/Range | Formats |
|-------|-----------------|---------------------|-------------|-------------|---------|
| Carbon price | `carbon_price_chart()` | `single_charts.carbon_price_chart()` or `range_charts.carbon_price_chart_with_uncertainty()` | 01, 03, 04 | Both | Display, PNG, HTML |
| Emissions pathway | `emissions_pathway_chart()` | `single_charts.emissions_pathway_chart()` or `range_charts.emissions_pathway_chart_with_uncertainty()` | 01, 03 | Both | Display, PNG, HTML |
| Supply components | `supply_components_chart()` | `single_charts.supply_components_chart()` | 01 | Single only | Display, PNG, HTML |
| Stockpile balance | `stockpile_balance_chart()` | `single_charts.stockpile_balance_chart()` or `range_charts.stockpile_chart_with_uncertainty()` | 01, 03 | Both | Display, PNG, HTML |
| Supply–demand balance | `supply_demand_balance_chart()` | `single_charts.supply_demand_balance_chart()` | 01 | Single only | Display, PNG, HTML |
| Auction volume & revenue | `auction_volume_revenue_chart()` | `single_charts.auction_volume_revenue_chart()` | 01 | Single only | Display, PNG, HTML |

**Output modes**:
- *Display*: `fig.show()` renders in Jupyter notebook
- *PNG/PDF/SVG*: `fig.write_image(path)` via `generate_standard_charts(output_dir, format)`
- *HTML*: `create_chart_page(charts, output_dir)` via `chart_display.py`

---

## 6  Data flow — what happens when `run()` is called

1. `run()` calls `validate()` to check all components have configs set.
2. For each scenario in `self.scenarios`:
   a. `ScenarioManager._initialise_scenario_components()` reads the scenario's `ComponentConfig`, calls `DataHandler` getters to load data, and creates fresh component objects (`AuctionSupply`, `ForestrySupply`, etc.) on the model instance.
   b. `_initialise_price_control()` loads the price control parameter series for this scenario.
   c. **If `pricing_mode='optimised'`** (default): `FastSolveOptimiser` is created with `calculate_gap()` as its objective.  It searches coarse then fine over `price_change_rate` values.  For each candidate rate, `calculate_gap()`:
      - Sets the rate, calculates the full price path (historical → projected via rate × price control).
      - Calculates base supply: auction volumes (price-dependent), industrial allocation, forestry supply.
      - Calculates demand: baseline emissions − price response (price-dependent).
      - Calculates stockpile dynamics: fills supply–demand gap using surplus then non-surplus, schedules payback.
      - Iterates steps above until convergence (supply/demand/stockpile stabilise).
      - Returns total gap (sum of absolute |supply − demand| per year, with optional shortfall penalty).
      The optimiser returns the rate with the smallest gap.
   d. **If `pricing_mode='fixed_path'`**: the user's price series is injected directly.  `calculate_gap()` runs once (no optimisation) to produce supply, demand, and stockpile results.
   e. **If `pricing_mode='fixed_rate'`**: the user's scalar rate is applied.  `calculate_gap()` runs once.
   f. A final model run with the chosen rate produces detailed component results.
   g. Results are stored in `self.results[scenario_name]`.
3. `OutputFormat` reads all scenario results and organises them into the MultiIndex DataFrames (`.prices`, `.supply`, `.demand`, `.stockpile`, `.forestry`, `.auctions`).
4. `list_variables()` and `variable_info()` are bound to the model instance.

---

## 7  Input data files

All in `data/inputs/`:

| Directory | File | What it contains |
|-----------|------|-----------------|
| `parameters/` | `model_parameters.csv` | Global parameters: discount_rate, forward_years, price_conversion_factor, long_term_min/max, start_price |
| `parameters/` | `optimisation_parameters.csv` | Optimiser settings: coarse_step, fine_step, max_rate, max_iterations |
| `parameters/` | `price.csv` | Historical carbon prices by year (config-keyed) |
| `parameters/` | `price_control.csv` | Price control parameter series by year (config-keyed: central, scarcity_then_surplus, weakening) |
| `supply/` | `auctions.csv` | Auction volumes, reserve prices, CCR triggers (config-keyed) |
| `supply/` | `removals.csv` | Exogenous forestry: tradeable, held, surrender (config-keyed) |
| `supply/` | `industrial_allocation.csv` | Industrial allocation volumes (config-keyed) |
| `supply/` | `stockpile_balance.csv` | Initial stockpile and surplus values (config-keyed) |
| `supply/` | `historical_removals.csv` | Historic forest removals (tradeable, held, surrender) from existing forests; used only in endogenous forestry mode |
| `forestry/` | `afforestation_projections.csv` | MPI annual afforestation projections by year, forest type (permanent_exotic, production_exotic, natural_forest), and sensitivity config (low/central/high); used by endogenous forestry to proportion new planting |
| `forestry/` | `manley_parameters.csv` | Manley logistic equation parameters by sensitivity (low/central/high), plus shared defaults; drives new-planting response to carbon price in endogenous mode |
| `forestry/` | `yield_tables.csv` | Cumulative carbon yield (tCO₂e/ha) by forest age for each forest type; pre-computed into annual increments by `DataHandler` for use in endogenous forestry convolution |
| `demand/` | `emissions_baselines.csv` | Baseline emissions pathways (config-keyed) |
| `demand/` | `demand_models.csv` | Price-response model parameters (config-keyed, by model number) |
| `economic/` | `CPI.csv` | Consumer Price Index for real↔nominal conversion |

---

Last updated: 20 March 2026 — dead code removal, price control rationalisation (`fill('price_control', ...)` / `fill_component('price', ...)`)
