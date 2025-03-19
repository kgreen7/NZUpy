"""
Optimiser for NZ ETS Model

This module implements the FastSolve optimisation algorithm from the original Excel model.
It conducts a two-phase (coarse-fine) search to find the optimal price change rate
that minimises the gap between supply and demand.
"""

import numpy as np
from typing import Callable, Optional, Tuple, Dict, Any


class FastSolveOptimiser:
    """
    Implementation of the FastSolve optimisation algorithm from the NZ ETS Excel model.
    
    This optimiser finds the price change rate that minimises the gap between supply
    and demand using a two-phase search approach: a coarse search followed by a fine-grained
    search around the best coarse solution.
    """
    
    def __init__(
        self,
        gap_function: Callable[[float], float],
        debug: bool = False,
        coarse_step: Optional[int] = 0,    # Now accepts coarse_step
        fine_step: Optional[int] = 0,       # Now accepts fine_step
        max_rate: Optional[int] = 0,      # Now accepts max_rate
        find_min: bool = True,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ):
        """
        Initialise the FastSolve optimiser.
        
        Args:
            gap_function: Function that calculates the gap for a given price change rate.
                           Should take a single float parameter and return a float.
            debug: If True, print debug information during optimisation (default: True).
            coarse_step: Step size for the coarse search phase (default: 15).
            fine_step: Step size for the fine search phase (default: 1).
            max_rate: Maximum price change rate in thousandths (default: 200).
            find_min: If True, find the minimum gap. If False, find the maximum (default: True).
            progress_callback: Optional callback function to report progress.
                              Takes step (0 or 1 for coarse/fine) and description.
        """
        self.gap_function = gap_function
        self.debug = debug
        self.coarse_step = coarse_step
        self.fine_step = fine_step
        self.max_rate = max_rate
        self.find_min = find_min
        self.progress_callback = progress_callback
        
        # Initialise results
        self.optimal_rate = None
        self.min_gap = None
        self.optimisation_path = []
        self.coarse_result = None
        self.fine_result = None
        
        # Direction multiplier (negative for minimisation, positive for maximisation)
        # This matches the 'findMax' parameter in the VBA code
        self.direction = -1 if find_min else 1
    
    def _rate_to_model(self, y: int) -> float:
        """
        Convert an integer index to a price change rate.
        Replicates the calculation from the Excel model:
        currentRate = findMax * (y - maxRate) / 1000
        
        Args:
            y: Integer index in the range [0, max_rate*2]
            
        Returns:
            Float price change rate
        """
        return self.direction * (y - self.max_rate) / 1000
    
    def _evaluate_gap(self, y: int) -> float:
        """
        Evaluate the gap for a given y value.
        
        Args:
            y: Integer value to convert to model rate
            
        Returns:
            Calculated gap value
        """
        rate = self._rate_to_model(y)
        gap = self.gap_function(rate)
        
        if self.debug:
            print(f"  Gap evaluation: y={y:3d}, rate={rate:.6f}, gap={gap:.2e}")
            
        return gap if self.find_min else -gap
    
    def _coarse_search(self) -> Tuple[int, float, float]:
        """Perform the coarse search phase."""
        if self.debug:
            print("\n=== COARSE SEARCH START ===")
            print(f"Step size: {self.coarse_step}")
            print(f"Max rate: {self.max_rate}")
            iterations = (self.max_rate * 2 + 1) // self.coarse_step
            print(f"Will test {iterations} rates")
        
        min_gap = float('inf')
        best_y = None
        
        for y in range(0, self.max_rate * 2 + 1, self.coarse_step):
            gap = self._evaluate_gap(y)
            if self.debug:
                print(f"Testing y={y:3d} (rate={self._rate_to_model(y):.6f})")
                print(f"  Current gap: {gap:,.0f}")
            
            if gap < min_gap:
                min_gap = gap
                best_y = y
                if self.debug:
                    print(f"  ★ NEW BEST! Gap improved by {(min_gap - gap):,.0f}")
        
        if self.debug:
            print(f"\n=== COARSE SEARCH COMPLETE ===")
            print(f"Tested {iterations} different rates")
            print(f"Best rate found: {self._rate_to_model(best_y):.6f}")
            print(f"Best gap: {min_gap:,.0f}\n")
        
        return best_y, self._rate_to_model(best_y), min_gap
    
    def _fine_search(self, coarse_best_y: int) -> Tuple[int, float, float]:
        """Perform the fine search phase."""
        if self.debug:
            print("\n=== FINE SEARCH START ===")
            print(f"Step size: {self.fine_step}")
            print(f"Searching around y={coarse_best_y}")
            # Now using full coarse_step for range
            search_range = self.coarse_step
            iterations = (2 * search_range + 1) // self.fine_step
            print(f"Will test {iterations} rates")
        
        min_gap = float('inf')
        best_y = coarse_best_y
        
        # Modified to use full coarse_step instead of coarse_step // 2
        for y in range(max(0, coarse_best_y - self.coarse_step), 
                      min(self.max_rate * 2, coarse_best_y + self.coarse_step + 1),
                      self.fine_step):
            gap = self._evaluate_gap(y)
            if self.debug:
                print(f"Testing y={y:3d} (rate={self._rate_to_model(y):.6f})")
                print(f"  Current gap: {gap:,.0f}")
            
            if gap < min_gap:
                min_gap = gap
                best_y = y
                if self.debug:
                    print(f"  ★ NEW BEST! Gap improved by {(min_gap - gap):,.0f}")
        
        if self.debug:
            print(f"\n=== FINE SEARCH COMPLETE ===")
            print(f"Tested {iterations} different rates")
            print(f"Best rate found: {self._rate_to_model(best_y):.6f}")
            print(f"Best gap: {min_gap:,.0f}\n")
        
        return best_y, self._rate_to_model(best_y), min_gap
    
    def optimise(self) -> Dict[str, Any]:
        """
        Run the full optimisation process, combining coarse and fine search phases.
        
        Returns:
            Dict containing optimisation results
        """
        
        if self.debug:
            # ADD DEBUG PRINTS
            print("\n=== OPTIMISER DEBUG STATUS ===")
            print(f"Debug mode: {self.debug}")
            print(f"Find minimum: {self.find_min}")
            print(f"Coarse step: {self.coarse_step}")
            print(f"Fine step: {self.fine_step}")
            print(f"Max rate: {self.max_rate}")
            print(f"Debug flag location: {hex(id(self.debug))}")
            print("\n=== Starting Full Optimisation Process ===")
        
        # Reset optimisation path
        self.optimisation_path = []
        
        # Run coarse search
        coarse_best_y, coarse_best_rate, coarse_min_gap = self._coarse_search()
        if self.debug:
            print(f"\nCoarse search results:")
            print(f"  Best y: {coarse_best_y}")
            print(f"  Best rate: {coarse_best_rate:.6f}")
            print(f"  Min gap: {coarse_min_gap:,.2f}")
        
        self.coarse_result = {
            'y': coarse_best_y,
            'rate': coarse_best_rate,
            'gap': coarse_min_gap
        }
        
        # Run fine search
        fine_best_y, fine_best_rate, fine_min_gap = self._fine_search(coarse_best_y)
        if self.debug:
            print(f"\nFine search results:")
            print(f"  Best y: {fine_best_y}")
            print(f"  Best rate: {fine_best_rate:.6f}")
            print(f"  Min gap: {fine_min_gap:,.2f}")
        
        self.fine_result = {
            'y': fine_best_y,
            'rate': fine_best_rate,
            'gap': fine_min_gap
        }
        
        # Store optimal results
        self.optimal_rate = fine_best_rate
        self.min_gap = fine_min_gap
        
        # Check if we hit the boundary
        at_boundary = abs(self.optimal_rate) >= self.max_rate / 1000
        
        if self.debug:
            print(f"\nFinal optimisation results:")
            print(f"  Optimal rate: {self.optimal_rate:.6f}")
            print(f"  Min gap: {self.min_gap:,.2f}")
            print(f"  At boundary: {at_boundary}")
            print("=== Optimisation Complete ===\n")
        
        return {
            'optimal_rate': self.optimal_rate,
            'min_gap': self.min_gap,
            'at_boundary': at_boundary,
            'coarse_result': self.coarse_result,
            'fine_result': self.fine_result,
            'optimisation_path': self.optimisation_path
        }