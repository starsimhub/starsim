"""
MetaPopulation class extending Starsim People for metapopulation modeling

This class adds disease-agnostic spatial metadata (lat, lon, pop_size) to the standard 
Starsim People class. This provides uniform access to population spatial data at the 
sim.people level for all modules (diseases, networks, etc.) while keeping disease-specific 
parameters separate within individual disease modules.

Key principle: MetaPopulation handles spatial/demographic metadata, diseases handle 
epidemiological parameters.
"""

import pandas as pd
import starsim as ss


class MetaPopulation(ss.People):
    """
    Extended People class for metapopulation modeling
    
    Adds spatial states (lat, lon, pop_size) to the standard Starsim People class.
    Each "agent" represents an entire community/population with spatial metadata.
    
    Args:
        communities (pd.DataFrame): DataFrame with community data including lat, lon, pop_size
        **kwargs: Additional arguments passed to ss.People
    """
    
    def __init__(self, communities, **kwargs):
        # Store communities data for initialization
        self.communities = communities
        
        # Derive number of communities from communities DataFrame
        n_communities = len(communities)
        
        # Initialize base People class with age_data=None to get default uniform distribution
        # Note: ss.People still uses n_agents internally, but these represent communities
        super().__init__(n_communities, age_data=None, **kwargs)
        
        # Add spatial states
        spatial_states = [
            ss.FloatArr('lat', default=0.0, label='Latitude'),
            ss.FloatArr('lon', default=0.0, label='Longitude'), 
            ss.FloatArr('pop_size', default=0.0, label='Total population size')
        ]
        
        # Add states to People
        for state in spatial_states:
            self.states.append(state, overwrite=False)
            setattr(self, state.name, state)
            state.link_people(self)
    
    def init_vals(self):
        """Initialize values from communities DataFrame"""
        # Call parent initialization
        super().init_vals()
        
        # Initialize spatial data from communities if provided
        if self.communities is not None:
            self._initialize_from_communities()
    
    def _initialize_from_communities(self):
        """Initialize spatial states from communities DataFrame"""
        if not isinstance(self.communities, pd.DataFrame):
            raise ValueError("Communities must be a pandas DataFrame")
        
        if len(self.communities) != len(self):
            raise ValueError(f"Communities length {len(self.communities)} != n_communities {len(self)}")
        
        # Initialize only spatial/demographic attributes (disease-agnostic)
        spatial_attrs = ['lat', 'lon', 'pop_size']
        for attr in spatial_attrs:
            if attr in self.communities.columns:
                if hasattr(self, attr):
                    getattr(self, attr)[:] = self.communities[attr].values
                else:
                    print(f"Warning: MetaPopulation has no attribute '{attr}'")
            else:
                print(f"Warning: Communities DataFrame missing column '{attr}'")
        
        print(f"Initialized {len(self)} communities with spatial metadata from DataFrame")
    
    
    def __repr__(self):
        """Enhanced representation showing spatial info"""
        base = super().__repr__()
        if hasattr(self, 'lat') and len(self.lat) > 0:
            lat_range = f"{self.lat.min():.1f}-{self.lat.max():.1f}"
            lon_range = f"{self.lon.min():.1f}-{self.lon.max():.1f}"
            pop_range = f"{self.pop_size.min():.0f}-{self.pop_size.max():.0f}"
            spatial_info = f"; lat=[{lat_range}], lon=[{lon_range}], pop=[{pop_range}]"
            # Insert before the closing parenthesis
            base = base[:-1] + spatial_info + ")"
        return base