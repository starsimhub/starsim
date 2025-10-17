"""
MetapopCompatible interface for diseases that support metapopulation transmission
"""

import numpy as np

class MetapopCompatible:
    """
    Mixin for diseases that support metapopulation transmission
    
    This interface provides the contract for diseases to work with
    MetapopulationRoute classes using sequential binomial thinning.
    
    Required states: lat, lon, pop_size (for spatial routing)
    """
    is_metapop = True
    
    def get_shedding(self) -> np.ndarray:
        """
        Return node export field; length = n_nodes
        
        Returns:
            np.ndarray: Shedding intensity per node (infectiousness per capita)
        """
        if not hasattr(self, 'shed'):
            return np.zeros(len(self.sim.people))
        return self.shed
    
    def apply_between(self, lambda_dt_vec: np.ndarray, route_name: str) -> int:
        """
        Immediate ΔE for one route (binomial thinning on remaining S)
        
        Args:
            lambda_dt_vec: Transmission rate vector per node (length n_nodes) multiplied by dt (still needs exponentiation)
            route_name: Name of the route for potential attribution/logging

        Returns:
            int: Total number of new infections applied
        """
        if len(lambda_dt_vec) == 0 or not np.any(lambda_dt_vec > 0):
            return
            
        # Node-level between hazard, destination susceptibility included
        p_between = 1.0 - np.exp(-self.rel_sus * lambda_dt_vec)

        # Apply to remaining susceptibles (age-agnostic between-community transmission)
        return self._apply_binomial_by_age(p_between, route_name)
    
    def _apply_binomial_by_age(self, p_between: np.ndarray, route_name: str) -> int:
        """
        Apply probability to all age groups using Starsim RNG
        
        This method should be implemented by the concrete disease class
        to handle age-specific state transitions.
        
        Args:
            p_between: Infection probability per node (length n_nodes)  
            route_name: Route name for potential logging/attribution

        Returns: 
            int: Total number of new infections applied
        """
        raise NotImplementedError("Concrete disease class must implement _apply_binomial_by_age")