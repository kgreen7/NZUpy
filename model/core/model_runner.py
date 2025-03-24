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
        
        # Initialize components for this scenario
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
        
        # Store existing configurations before any potential reconfiguration
        existing_configs = {}
        if hasattr(self.model, 'component_configs') and self.model.component_configs:
            for i, config in enumerate(self.model.component_configs):
                existing_configs[i] = {
                    'forestry': getattr(config, 'forestry', 'central'),
                    'auctions': getattr(config, 'auctions', 'central'),
                    'industrial_allocation': getattr(config, 'industrial_allocation', 'central'),
                    'emissions': getattr(config, 'emissions', 'central'),
                    'stockpile': getattr(config, 'stockpile', 'central')
                }
        
        # For Range scenario type, ensure scenarios are properly configured
        if hasattr(self.model, 'scenario_type') and self.model.scenario_type == 'Range':
            # Configure range scenarios if needed
            if not hasattr(self.model, '_range_scenarios_configured'):
                self.model.configure_range_scenarios()
                self.model._range_scenarios_configured = True
            
            # Restore any custom configurations that were set after initial range configuration
            if existing_configs:
                for i in existing_configs:
                    for component_type, value in existing_configs[i].items():
                        if value != 'central':  # Only restore non-central configurations
                            self.model.scenario_manager.use_config(i, component_type, value)
        
        # Use existing run logic
        if not self.model.validate():
            raise ValueError("Model validation failed. Please check the configuration.")
        
        # Dictionary to store results for each scenario
        scenario_results = {}
        
        # Run each scenario
        for i, scenario_name in enumerate(self.model.scenarios):
            # Get the component configuration for this scenario
            component_config = self.model.component_configs[i]
            
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