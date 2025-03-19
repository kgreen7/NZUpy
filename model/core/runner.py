"""
Model runner for NZ ETS Supply-Demand Model.

This module provides functionality for running the model, including
optimisation, scenario runs, and result formatting.
"""

import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple

from model.utils.output_format import OutputFormat
from model.core.optimiser import FastSolveOptimiser


class ModelRunner:
    """
    Model runner for the NZ ETS model.
    
    This class handles the execution of the model, including optimisation,
    scenario runs, and formatting of results.
    """
    
    def __init__(self, model):
        """
        Initialise the model runner.
        
        Args:
            model: The parent NZUpy instance
        """
        self.model = model
    
    def run(self) -> Dict[str, Dict[str, Any]]:
        """
        Run the model for all defined scenarios.
        
        Returns:
            Dictionary mapping scenario names to their results
        """
        # For Range scenario type, ensure scenarios are properly configured
        if hasattr(self.model, 'scenario_type') and self.model.scenario_type == 'Range':
            self.model.configure_range_scenarios()
        
        # Use existing run logic
        if not self.model.validate():
            raise ValueError("Model validation failed. Please check the configuration.")
        
        # Dictionary to store results for each scenario
        scenario_results = {}
        
        # Run each scenario
        for i, scenario_name in enumerate(self.model.scenarios):
            print(f"\nRunning scenario {i}: {scenario_name}")
            
            # Get the component configuration for this scenario
            component_config = self.model.component_configs[i]
            
            # Initialise components for this scenario
            self.model.scenario_manager._initialise_scenario_components(component_config)
            
            # Run optimisation for this scenario
            result = self._run_scenario_optimisation()
            
            # Store results
            scenario_results[scenario_name] = result
            
            # Print summary
            print(f"Completed scenario {scenario_name}")
            print(f"  Final price: ${result['final_price']:.2f}")
            print(f"  Total gap: {result['total_gap']:,.0f}")
        
        # Store all results
        self.model.results = scenario_results
        
        # Create an OutputFormat and organise results into structured DataFrames
        formatter = OutputFormat(self.model)
        formatter.organise_outputs()
        
        # Add helper methods to model as bound methods
        self.model.list_variables = formatter.list_variables
        self.model.variable_info = formatter.variable_info
        
        return scenario_results
        
    def get_scenario_result(self, scenario_name_or_index: Union[str, int]) -> Dict[str, Any]:
        """
        Retrieve results for a specific scenario by name or index.
        
        Args:
            scenario_name_or_index: Name or index of the scenario
        
        Returns:
            Dictionary containing results for the specified scenario
        
        Raises:
            ValueError: If model has not been run or scenario is not found
        """
        # Check if we have results
        if not self.model.results:
            raise ValueError("Model has not been run yet. Call run() first.")
        
        # Handle scenario by index
        if isinstance(scenario_name_or_index, int):
            if scenario_name_or_index < 0 or scenario_name_or_index >= len(self.model.scenarios):
                raise ValueError(f"Invalid scenario index: {scenario_name_or_index}. Valid range: 0-{len(self.model.scenarios)-1}")
            scenario_name = self.model.scenarios[scenario_name_or_index]
        else:
            # Handle scenario by name
            scenario_name = scenario_name_or_index
            if scenario_name not in self.model.scenarios:
                raise ValueError(f"Unknown scenario name: '{scenario_name}'. Available scenarios: {', '.join(self.model.scenarios)}")
        
        # Check if we have results for this scenario
        if scenario_name not in self.model.results:
            raise ValueError(f"No results available for scenario '{scenario_name}'. Run the model first.")
        
        # Return the results for the specified scenario
        return self.model.results[scenario_name]

    def run_optimisation(self) -> Dict[str, Any]:
        """
        Run the optimisation process to find the optimal price change rate.
        
        Returns:
            Dict containing optimisation and model results.
        """
        # Create optimiser
        optimiser = FastSolveOptimiser(self.model.calculation_engine.calculate_gap, debug=False)
        
        # Run optimisation
        optimisation_results = optimiser.optimise()
        
        # Get optimal price change rate
        optimal_price_change_rate = optimisation_results['optimal_rate']
        
        # Calculate final results with optimal rate
        gap = self.model.calculation_engine.calculate_gap(optimal_price_change_rate)
        
        # Run model with optimal rate and get the detailed results
        model_results = self.run_model(optimal_price_change_rate)
        
        # Return results in format compatible with validation script
        return {
            'optimisation': optimisation_results,
            'model': model_results,
            'price_change_rate': optimal_price_change_rate,
            'total_gap': gap,
            'prices': self.model.prices.copy(),
            'supply': self.model.supply.copy(),
            'demand': self.model.demand.copy(),
            'final_price': self.model.prices[self.model.years[-1]]
        }

    def run_model(self, price_change_rate: Optional[float] = None, *, is_final_run: bool = False) -> Dict[str, Any]:
        """
        Run the model with a given price change rate.
        
        Args:
            price_change_rate: The price change rate to use.
                If None, the current price_change_rate will be used.
            is_final_run: Whether this is the final run after optimisation.
                If True, forestry variables will be included.
        
        Returns:
            Dict containing model results.
        """
        if price_change_rate is not None:
            self.model.price_change_rate = price_change_rate
        
        # Set up the model run
        run_data = self._initialise_model_run()
        
        # Perform the iterative calculation until convergence
        self._iterate_until_convergence(run_data)
        
        # Apply post-convergence adjustments for visualisation
        self._apply_post_convergence_adjustments()
        
        # Calculate final gap
        gap = self.model.calculation_engine._calculate_total_gap()
        
        # Compile and return results
        model_results = self._compile_model_results(gap, run_data['iteration'])
        
        # Debug output if enabled
        if self.model.config.optimiser.debug:
            self._print_model_results_summary(gap)
            
        return model_results
    
    def _initialise_model_run(self) -> Dict[str, Any]:
        """
        Initialise data for a model run.
        
        Returns:
            Dictionary with run initialisation data
        """
        # Initialise iteration variables
        run_data = {
            'max_iterations': self.model.config.optimiser.max_iterations,
            'converged': False,
            'iteration': 0,
            'stockpile_usage': pd.Series(0.0, index=self.model.years)
        }
        
        # Reset stockpile component state to prevent state leakage between runs
        self.model.stockpile._initialise_data()
        
        return run_data
    
    def _iterate_until_convergence(self, run_data: Dict[str, Any]) -> None:
        """
        Perform the iterative calculation process until convergence.
        
        Args:
            run_data: Dictionary with model run data
        """
        while not run_data['converged'] and run_data['iteration'] < run_data['max_iterations']:
            run_data['iteration'] += 1
            
            # Perform a single iteration
            stockpile_results, prev_prices, new_stockpile_usage = self._perform_single_iteration()
            
            # Check for convergence
            run_data['converged'] = self._check_convergence(
                prev_prices, 
                run_data['stockpile_usage'],
                new_stockpile_usage
            )
            
            # Update stockpile usage for next iteration
            run_data['stockpile_usage'] = new_stockpile_usage
    
    def _perform_single_iteration(self) -> Tuple[Dict[str, pd.Series], pd.Series, pd.Series]:
        """
        Perform a single iteration of the model calculation.
        
        Returns:
            Tuple of (stockpile_results, previous_prices, new_stockpile_usage)
        """
        # Calculate initial prices based on the price change rate
        self.model.calculation_engine._calculate_initial_prices()
        
        # Calculate demand based on current prices
        self.model.calculation_engine._calculate_demand()
        
        # Calculate base supply (without stockpile)
        self.model.calculation_engine._calculate_base_supply()
        
        # Calculate supply-demand balance
        stockpile_results = self._calculate_stockpile_contribution()
        
        # Get new stockpile usage and previous prices for convergence check
        new_stockpile_usage = stockpile_results['available_units'].copy()
        prev_prices = self.model.prices.copy()
        
        # Revise prices based on stockpile usage
        self.model.calculation_engine._revise_prices_based_on_stockpile(stockpile_results)
        
        # Calculate final supply (including stockpile)
        self.model.calculation_engine._calculate_supply()
        
        return stockpile_results, prev_prices, new_stockpile_usage
    
    def _calculate_stockpile_contribution(self) -> Dict[str, pd.Series]:
        """
        Calculate the stockpile contribution based on supply-demand balance.
        
        Returns:
            Dictionary of stockpile calculation results
        """
        # Calculate initial supply-demand balance
        yearly_balances = {}
        for year in self.model.years:
            yearly_balances[year] = self.model.base_supply.get(year, 0) - self.model.demand.get(year, 0)
        
        supply_demand_balance = pd.Series(yearly_balances)
        
        # Calculate stockpile contribution
        return self.model.stockpile.calculate(
            supply_demand_balance=supply_demand_balance,
            prices=self.model.prices,
            price_change_rates=pd.Series([self.model.price_change_rate] * len(self.model.years), index=self.model.years),
            track_forestry_impact=True   # Always track forestry impact
        )
    
    def _check_convergence(self, prev_prices: pd.Series, prev_usage: pd.Series, new_usage: pd.Series) -> bool:
        """
        Check if the model has converged.
        
        Args:
            prev_prices: Previous iteration's prices
            prev_usage: Previous iteration's stockpile usage
            new_usage: Current iteration's stockpile usage
            
        Returns:
            True if converged, False otherwise
        """
        # Check changes in both prices and stockpile usage
        price_change = (self.model.prices - prev_prices).abs().sum()
        usage_change = (new_usage - prev_usage).abs().sum()
        
        # Small thresholds for convergence
        return price_change < 0.01 and usage_change < 100
    
    def _apply_post_convergence_adjustments(self) -> None:
        """
        Apply post-convergence adjustments for visualisation.
        
        This ensures there are no artificial dips in supply visualisation.
        """
        # For years before stockpile_usage_start_year, override supply to match demand
        if hasattr(self.model.stockpile, 'stockpile_usage_start_year'):
            stockpile_usage_start_year = self.model.stockpile.stockpile_usage_start_year
            for year in self.model.years:
                if year < stockpile_usage_start_year and year in self.model.demand.index:
                    # Ensure 'total' in supply equals demand for pre-stockpile years
                    self.model.supply.loc[year, 'total'] = self.model.demand[year]
                    # Ensure stockpile contribution is 0 for pre-stockpile years
                    self.model.supply.loc[year, 'stockpile'] = 0
    
    def _compile_model_results(self, gap: float, iterations: int) -> Dict[str, Any]:
        """
        Compile model results into a dictionary.
        
        Args:
            gap: Calculated gap value
            iterations: Number of iterations required for convergence
            
        Returns:
            Dictionary with model results
        """
        # Save stockpile results for reference
        stockpile_results = self.model.stockpile_results
        self.model.stockpile_results = stockpile_results
        
        # Ensure payback information is properly captured
        if hasattr(self.model.stockpile, 'borrowed_units'):
            # Calculate total payback due for each year
            payback_units = pd.Series(0.0, index=self.model.years)
            for year in self.model.years:
                for borrow_year in self.model.years:
                    if year in self.model.stockpile.borrowed_units.columns:
                        payback_units[year] += self.model.stockpile.borrowed_units.loc[borrow_year, year]
            
            # Add to stockpile results
            if isinstance(stockpile_results, pd.DataFrame):
                stockpile_results['payback_units'] = payback_units
                stockpile_results['net_borrowing'] = payback_units - stockpile_results['borrowed_units']
        
        # Compile results dictionary
        return {
            'price_change_rate': self.model.price_change_rate,
            'gap': gap,
            'prices': self.model.prices.copy(),
            'unmodified_prices': getattr(self.model, 'unmodified_prices', self.model.prices.copy()),
            'supply': self.model.supply.copy(),
            'demand': self.model.demand.copy(),
            'base_supply': self.model.base_supply.copy(),
            'supply_demand_balance': self.model.supply['total'] - self.model.demand,
            'auction_results': self.model.auction_results.copy(),
            'industrial_results': self.model.industrial_results.copy(),
            'stockpile_results': stockpile_results.copy(),
            'forestry_results': self.model.forestry_results.copy(),
            'price_response_results': self.model.price_response_results.copy(),
            'emissions_results': self.model.emissions_results.copy(),
            'price_control_parameters': self.model.price_control_parameter.copy(),
            'iterations_needed': iterations
        }
    
    def _print_model_results_summary(self, gap: float) -> None:
        """
        Print a summary of model results for debugging.
        
        Args:
            gap: Calculated gap value
        """
        print("\nFinal model results:")
        print(f"  Price change rate: {self.model.price_change_rate:.6f}")
        print(f"  Prices:")
        print(f"     2024: ${self.model.prices[2024]:.2f}")
        print(f"     2050: ${self.model.prices[2050]:.2f}")
        print(f"  Total gap: {gap:,.0f}")
        print(f"  Supply-Demand balance: {(self.model.supply['total'] - self.model.demand).sum():,.0f}\n")
    
    def run_scenarios(self, scenarios: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Run the model for multiple scenarios.
        
        Args:
            scenarios: List of scenarios to run.
                Options: "central", "1 s.e lower", "1 s.e upper", "95% Lower", "95% Upper"
        
        Returns:
            Dict mapping scenarios to their results.
        """
        scenario_results = {}
        
        # Save original scenario
        original_scenario = self.model.config.scenario
        
        # Run each scenario
        for scenario in scenarios:
            # Set scenario
            self.model.config.scenario = scenario
            self.model.emissions.set_scenario(scenario)
            
            # Run model
            results = self.run_model()
            
            # Store results
            scenario_results[scenario] = results
        
        # Restore original scenario
        self.model.config.scenario = original_scenario
        self.model.emissions.set_scenario(original_scenario)
        
        return scenario_results
    
    def _run_scenario_optimisation(self) -> Dict[str, Any]:
        """
        Run optimisation for the current scenario configuration.
        
        This method coordinates the optimisation process to find the optimal
        price change rate, calculates the resulting gap, and compiles the results.
        
        Returns:
            Dictionary with comprehensive results from optimisation
        """
        # Set up and run the optimiser
        optimiser = self._setup_optimiser()
        optimisation_results = optimiser.optimise()
        
        # Get optimal price change rate and calculate resulting gap
        optimal_rate = optimisation_results['optimal_rate']
        gap = self.model.calculation_engine.calculate_gap(optimal_rate)
        
        # Run model with optimal rate for final, detailed results
        model_results = self.run_model(optimal_rate, is_final_run=True)
        
        # Compile and return results
        return self._compile_optimisation_results(optimisation_results, gap, model_results)
    
    def _setup_optimiser(self) -> FastSolveOptimiser:
        """
        Set up the optimiser with appropriate configuration.
        
        Returns:
            Configured FastSolveOptimiser instance
        """
        return FastSolveOptimiser(
            gap_function=self.model.calculation_engine.calculate_gap,
            coarse_step=self.model.config.optimiser.coarse_step,
            fine_step=self.model.config.optimiser.fine_step,
            max_rate=self.model.config.optimiser.max_rate,
            debug=self.model.config.optimiser.debug
        )
    
    def _compile_optimisation_results(self, 
                                     optimisation_results: Dict[str, Any],
                                     gap: float,
                                     model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compile comprehensive results from optimisation and model run.
        
        Args:
            optimisation_results: Results from the optimisation process
            gap: Calculated gap for the optimal rate
            model_results: Results from running the model with optimal rate
            
        Returns:
            Dictionary with comprehensive results
        """
        # Extract optimal rate for convenience
        optimal_rate = optimisation_results['optimal_rate']
        
        # Create the results dictionary with core information
        results = {
            'price_change_rate': optimal_rate,
            'total_gap': gap,
            'prices': self.model.prices.copy(),
            'supply': self.model.supply.copy(),
            'demand': self.model.demand.copy(),
            'final_price': self.model.prices[self.model.config.end_year],
            'convergence_success': not optimisation_results.get('at_boundary', False),
            'optimisation_message': ''
        }
        
        # Add detailed component results for advanced analysis
        results['model'] = self._compile_component_results()
        
        return results
    
    def _compile_component_results(self) -> Dict[str, Any]:
        """
        Compile results from individual model components.
        
        Returns:
            Dictionary with component-specific results
        """
        # Compile stockpile component results
        stockpile_component = {
            'results': self.model.stockpile.results.copy() if hasattr(self.model.stockpile, 'results') else None,
            'stockpile_balance': self.model.stockpile.stockpile_balance.copy() if hasattr(self.model.stockpile, 'stockpile_balance') else None,
            'surplus_balance': self.model.stockpile.surplus_balance.copy() if hasattr(self.model.stockpile, 'surplus_balance') else None
        }
        
        # Add convenience properties if available
        if hasattr(self.model.stockpile, 'is_stockpile_used'):
            stockpile_component['is_stockpile_used'] = self.model.stockpile.is_stockpile_used
            
        if hasattr(self.model.stockpile, 'stockpile_usage_amounts'):
            stockpile_component['stockpile_usage_amounts'] = self.model.stockpile.stockpile_usage_amounts
        
        # Compile all component results
        return {
            'stockpile_component': stockpile_component,
            'emissions_component': self.model.emissions_results,
            'price_response_component': self.model.price_response_results,
            'forestry_component': self.model.forestry_results,
            'auction_component': self.model.auction_results,
            'industrial_component': self.model.industrial_results
        }