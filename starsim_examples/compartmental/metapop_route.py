"""
GravityNetwork class for gravity-based spatial transmission

Implements pure linear mixer with immediate callback pattern.
Uses zero-row safe matrix normalization and vectorized operations.
"""

import numpy as np
import starsim as ss
from metapop_interface import MetapopCompatible


class GravityNetwork(ss.MetapopulationRoute):
    """
    Gravity-based metapopulation transmission route with immediate callback
    
    Features:
    - Gravity model: (pop_i^α * pop_j^β) / dist^γ  
    - Row-stochastic weight matrix (zero-row safe)
    - Matrix multiplication for transmission: lam_vec = beta_net * (W @ shed)
    - Immediate callback to disease.apply_between()
    - No set_prognoses() - uses sequential binomial thinning
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        self.define_pars(
            x_beta=1.0,                  # Multiplier on disease beta (dimensionless)
            pop_exponent_1=1.0,          # α: origin population exponent
            pop_exponent_2=1.0,          # β: destination population exponent  
            distance_exponent=2.0,       # γ: distance decay exponent
            min_distance=1.0             # Minimum distance (km)
        )
        self.update_pars(**kwargs)
        
        self.W = None                    # Row-stochastic gravity matrix
        self.isolated_nodes = None       # Diagnostic: nodes with no outbound connections
    
    
    def init_pre(self, sim):
        super().init_pre(sim)
        # Don't build matrix here - wait until states are available
        
    def init_post(self):
        super().init_post()
        # Build matrix using spatial data from sim.people
        self.build_gravity_matrix()
    
    def build_gravity_matrix(self):
        """Build row-stochastic (nC, nC) gravity weight matrix - zero-row safe"""
        # Get spatial data from sim.people
        if not hasattr(self.sim, 'people'):
            raise RuntimeError("MetapopulationRoute requires sim.people to be initialized")
        
        people = self.sim.people
        if not all(hasattr(people, attr) for attr in ['lat', 'lon', 'pop_size']):
            raise RuntimeError("sim.people missing required spatial attributes: lat, lon, pop_size")
        
        nC = len(people)
        G = np.zeros((nC, nC))
        
        # Build raw gravity matrix
        for i in range(nC):
            for j in range(nC):
                if i != j:
                    pop_i = float(people.pop_size[i])
                    pop_j = float(people.pop_size[j])
                    
                    # Calculate distance
                    dist = max(
                        self.haversine_distance(
                            people.lat[i], people.lon[i],
                            people.lat[j], people.lon[j]
                        ),
                        self.pars.min_distance
                    )
                    
                    # Gravity model: (pop_i^α * pop_j^β) / dist^γ
                    G[i,j] = ((pop_i ** self.pars.pop_exponent_1) * 
                             (pop_j ** self.pars.pop_exponent_2)) / \
                             (dist ** self.pars.distance_exponent)
        
        # Zero-row safe normalization (V3 fix)
        row_sums = G.sum(axis=1, keepdims=True)
        W = np.zeros_like(G)
        mask = row_sums[:,0] > 0
        W[mask] = G[mask] / row_sums[mask]
        self.W = W
        
        # Store diagnostic information
        self.isolated_nodes = ~mask.flatten()
        
        # Optional: log isolated nodes for debugging
        n_isolated = np.sum(self.isolated_nodes)
        if n_isolated > 0:
            print(f"Warning: {n_isolated} isolated nodes (no outbound connections) in gravity network")
    
    def compute_transmission(self, rel_sus, rel_trans, disease_beta, disease=None):
        """
        Compute transmission and immediately apply via callback
        
        Args:
            rel_sus: Ignored (signature compatibility with Starsim Route interface)
            rel_trans: Ignored (signature compatibility)  
            disease_beta: Ignored (signature compatibility)
            disease: Disease object with MetapopCompatible interface
            
        Returns: 
            Empty UIDs (route doesn't trigger set_prognoses)
        """
        # Only act for metapop-compatible diseases
        if not isinstance(disease, MetapopCompatible):
            return ss.uids([])  # Harmless no-op for regular diseases
        
        # Matrix should already be built in init_post
        if self.W is None:
            raise RuntimeError("Gravity matrix not initialized - check init_post()")
        
        # Get current shedding from disease
        shed = disease.get_shedding()  # (n_nodes,)
        
        if not np.any(shed > 0):
            return ss.uids([])  # No infectious nodes
        
        # Matrix multiplication for transmission (ultra-fast)
        # Use disease_beta multiplied by x_beta (similar to mixing pools)
        beta_dt = self.pars.x_beta * disease_beta.to_events(self.sim.t.dt)
        lambda_dt_vec = beta_dt * (self.W @ shed)  # (n_nodes,)

        # Immediate callback - disease applies binomial thinning to remaining S
        disease.apply_between(lambda_dt_vec, self.name)

        # Route doesn't trigger set_prognoses itself
        return ss.uids([])
    
    def step(self):
        """Route doesn't need step method - all work done in compute_transmission"""
        pass
    
    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculate Haversine distance in km
        
        Args:
            lat1, lon1: Origin coordinates (degrees)
            lat2, lon2: Destination coordinates (degrees)
            
        Returns:
            float: Distance in kilometers
        """
        R = 6371  # Earth's radius in km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * 
             np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
        return 2 * R * np.arcsin(np.sqrt(a))
    
    def distance_matrix(self):
        """
        Calculate full distance matrix between all communities
        
        Returns:
            np.ndarray: (n_communities, n_communities) distance matrix in km
        """
        if not hasattr(self.sim, 'people'):
            raise RuntimeError("Cannot compute distance matrix: sim.people not available")
        
        people = self.sim.people
        if not all(hasattr(people, attr) for attr in ['lat', 'lon']):
            raise RuntimeError("sim.people missing required spatial attributes: lat, lon")
        
        n = len(people)
        D = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = self.haversine_distance(
                    people.lat[i], people.lon[i],
                    people.lat[j], people.lon[j]
                )
                D[i, j] = dist
                D[j, i] = dist  # Symmetric
                
        return D
    
    def __repr__(self):
        """Enhanced representation showing network statistics"""
        base = super().__repr__()
        if self.W is not None:
            n_nodes = self.W.shape[0]
            n_edges = np.sum(self.W > 0)
            n_isolated = np.sum(self.isolated_nodes) if self.isolated_nodes is not None else 0
            stats = f"n_nodes={n_nodes}, n_edges={n_edges}, isolated={n_isolated}"
            # Insert stats before parameters
            pos = base.find('pars=')
            if pos > 0:
                base = base[:pos] + stats + '; ' + base[pos:]
        return base