from typing import Dict, Any

class ModelRunner:
    def _run_scenario(self, scenario_index: int) -> Dict[str, Any]:
        """
        Run a single scenario.
        
        Args:
            scenario_index: Index of the scenario to run
            
        Returns:
            Dict containing the scenario results
        """
        # Store active scenario index for parameter lookups
        self.model._active_scenario_index = scenario_index
        
        # Get the scenario name
        scenario_name = self.model.scenarios[scenario_index]
        
        # Set price control
        self.model._initialise_price_control()
        
        # Re-initialize components with the scenario name to ensure scenario-specific data is used
        component_config = self.model.component_configs[scenario_index]
        self.model.scenario_manager._initialise_scenario_components(component_config, scenario_name)
        
        # Run optimisation for this scenario
        result = self._run_scenario_optimisation()
        self.model._active_scenario_index = None
        
        return result 

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
            # Get the component configuration for this scenario
            component_config = self.model.component_configs[i]
            
            # Initialise components for this scenario - passing the scenario name
            self.model.scenario_manager._initialise_scenario_components(component_config, scenario_name)
            
            # Run optimisation for this scenario
            result = self._run_scenario(i)  # Only pass the scenario index
            
            # Store results
            scenario_results[scenario_name] = result
            
        # Store all results
        self.model.results = scenario_results
        
        # Create an OutputFormat and organise results into structured DataFrames
        from model.utils.output_format import OutputFormat
        formatter = OutputFormat(self.model)
        formatter.organise_outputs()
        
        # Add helper methods to model as bound methods
        self.model.list_variables = formatter.list_variables
        self.model.variable_info = formatter.variable_info
        
        return scenario_results 