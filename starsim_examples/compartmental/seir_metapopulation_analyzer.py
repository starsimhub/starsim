"""
SEIRMetapopulationAnalyzer for tracking community-level SEIR dynamics

This analyzer collects SEIR states for each community at every timestep,
providing detailed node-level epidemic tracking with age stratification
and compartment aggregation (E1+E2+E3 → E_total, I1+I2+I3 → I_total).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import starsim as ss
import sciris as sc
import networkx as nx


class SEIRMetapopulationAnalyzer(ss.Analyzer):
    """
    Analyzer for metapopulation SEIR dynamics
    
    Collects and visualizes community-level SEIR timeseries with:
    - Individual community tracking
    - Age group stratification (0-4, 5-19, 20+)
    - E and I stage aggregation
    - Multiple visualization options
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        
        # Data storage
        self.timeseries = {}  # {time: {community_id: {age_group: {S, E, I, R}}}}
        self.times = []       # List of timesteps
        self.community_ids = None  # Will be set from simulation
        self.age_groups = ['0_4', '5_19', '20p']
        
        # Configuration
        self.collect_spatial = True  # Whether to collect spatial coordinates
        self.spatial_data = {}       # Store lat, lon for communities
        
    def init_post(self):
        """Initialize after simulation setup"""
        super().init_post()
        
        # Get community information from simulation
        if hasattr(self.sim, 'people') and hasattr(self.sim.people, 'lat'):
            self.n_communities = len(self.sim.people)
            self.community_ids = list(range(self.n_communities))
            
            # Collect spatial data if available
            if self.collect_spatial:
                self.spatial_data = {
                    'lat': np.array(self.sim.people.lat),
                    'lon': np.array(self.sim.people.lon),
                    'pop_size': np.array(self.sim.people.pop_size)
                }
        else:
            raise RuntimeError("SEIRMetapopulationAnalyzer requires MetaPopulation with spatial data")
        
        print(f"SEIRMetapopulationAnalyzer initialized for {self.n_communities} communities")
    
    def step(self):
        """Collect SEIR data for all communities at current timestep"""
        
        # Find SEIR disease (assumes single SEIR disease in simulation)
        seir_disease = None
        for disease in self.sim.diseases.values():
            if hasattr(disease, 'age_groups') and hasattr(disease, 'get_total_E_by_age'):
                seir_disease = disease
                break
        
        if seir_disease is None:
            return  # Skip if no compatible SEIR disease found
        
        # Current simulation time
        current_time = float(self.sim.ti)
        self.times.append(current_time)
        
        # Collect data for all communities
        community_data = {}
        
        for comm_id in self.community_ids:
            age_data = {}
            
            for age in self.age_groups:
                # Get aggregated compartment totals for this community and age group
                S_total = float(getattr(seir_disease, f'S_{age}')[comm_id])
                E_total = seir_disease.get_total_E_by_age(age)[comm_id]  
                I_total = seir_disease.get_total_I_by_age(age)[comm_id]
                R_total = float(getattr(seir_disease, f'R_{age}')[comm_id])
                
                age_data[age] = {
                    'S': S_total,
                    'E': E_total, 
                    'I': I_total,
                    'R': R_total
                }
            
            community_data[comm_id] = age_data
        
        # Store timestep data
        self.timeseries[current_time] = community_data
    
    def finalize_results(self):
        """Process collected data into convenient formats"""
        super().finalize_results()
        
        # Convert to pandas DataFrame for easier analysis
        self.df = self._create_dataframe()
        print(f"SEIRMetapopulationAnalyzer collected {len(self.times)} timesteps")
    
    def _create_dataframe(self):
        """Convert timeseries data to pandas DataFrame"""
        records = []
        
        for time, communities in self.timeseries.items():
            for comm_id, ages in communities.items():
                for age_group, compartments in ages.items():
                    record = {
                        'time': time,
                        'community_id': comm_id,
                        'age_group': age_group,
                        'S': compartments['S'],
                        'E': compartments['E'],
                        'I': compartments['I'],
                        'R': compartments['R']
                    }
                    
                    # Add spatial data if available
                    if self.collect_spatial and comm_id < len(self.spatial_data['lat']):
                        record['lat'] = self.spatial_data['lat'][comm_id]
                        record['lon'] = self.spatial_data['lon'][comm_id]
                        record['pop_size'] = self.spatial_data['pop_size'][comm_id]
                    
                    records.append(record)
        
        return pd.DataFrame(records)
    
    def plot_community_dynamics(self, community_ids=None, compartments=['S', 'E', 'I', 'R'], mode='by_compartment', normalize=True, **kwargs):
        """
        Plot SEIR dynamics by community with two display modes
        
        Args:
            community_ids: List of community IDs to plot (default: all)
            compartments: List of compartments to plot (default: all)
            mode: 'by_compartment' (1 panel per SEIR, colored by community) or 
                  'by_community' (1 panel per community, traditional view)
            normalize: Whether to normalize by population size (show proportions)
            **kwargs: Additional plotting arguments
        """
        
        if not hasattr(self, 'df'):
            raise RuntimeError("Must run simulation and finalize_results() before plotting")
        
        # Default selections - show all communities or reasonable subset
        if community_ids is None:
            if mode == 'by_compartment':
                community_ids = self.community_ids  # Show all communities
            else:
                community_ids = self.community_ids[:min(16, len(self.community_ids))]  # Limit for grid layout
        
        if mode == 'by_compartment':
            return self._plot_by_compartment(community_ids, compartments, normalize, **kwargs)
        else:
            return self._plot_by_community(community_ids, compartments, normalize, **kwargs)
    
    def _plot_by_compartment(self, community_ids, compartments, normalize=True, **kwargs):
        """Plot with one panel per compartment, colored traces per community"""
        
        # Create subplots - one per compartment
        n_compartments = len(compartments)
        fig, axes = plt.subplots(1, n_compartments, figsize=(5*n_compartments, 6))
        if n_compartments == 1:
            axes = [axes]  # Ensure axes is always a list
        
        # Generate colors for communities using a spectrum
        import matplotlib.cm as cm
        n_communities = len(community_ids)
        colors = cm.viridis(np.linspace(0, 1, n_communities))  # Use viridis colormap for all communities
        
        for i, compartment in enumerate(compartments):
            ax = axes[i]
            
            # Set y-axis label based on normalization option
            y_label = f'{compartment} Proportion' if normalize else f'{compartment} Population'
            
            for j, comm_id in enumerate(community_ids):
                # Get data for this community, aggregate across age groups
                comm_data = self.df[self.df['community_id'] == comm_id].groupby('time')[compartment].sum().reset_index()
                
                # Normalize by population size if requested
                if normalize and comm_id < len(self.spatial_data['pop_size']):
                    pop_size = self.spatial_data['pop_size'][comm_id]
                    y_values = comm_data[compartment] / pop_size
                else:
                    y_values = comm_data[compartment]
                
                # Determine if we should label this community
                # Label every 10th community for large datasets, or all for small datasets
                should_label = False
                if n_communities <= 20:
                    should_label = True
                    label_text = f'Community {comm_id}'
                elif comm_id % max(1, n_communities // 10) == 0:  # Label ~10 communities evenly spaced
                    should_label = True
                    label_text = f'Community {comm_id}'
                else:
                    label_text = None
                
                # Plot with community-specific color
                ax.plot(comm_data['time'], y_values, 
                       color=colors[j], alpha=0.7, linewidth=1.5,
                       label=label_text)
            
            ax.set_title(f'{compartment} Dynamics Across Communities', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)
            
            # Add legend with selective labeling
            if n_communities <= 20:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            elif any(line.get_label() and not line.get_label().startswith('_') for line in ax.get_lines()):
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def _plot_by_community(self, community_ids, compartments, normalize=True, **kwargs):
        """Traditional plot with one panel per community"""
        
        # Set up plotting - arrange in optimal grid layout  
        n_communities = len(community_ids)
        if n_communities <= 3:
            nrows, ncols = 1, n_communities
        elif n_communities <= 6:
            nrows, ncols = 2, 3
        elif n_communities <= 9:
            nrows, ncols = 3, 3
        elif n_communities <= 12:
            nrows, ncols = 3, 4
        elif n_communities <= 16:
            nrows, ncols = 4, 4
        else:
            # For more than 16, use 4 columns
            nrows = int(np.ceil(n_communities / 4))
            ncols = 4
        
        fig, axes = plt.subplots(nrows, ncols, 
                                figsize=(5*ncols, 4*nrows),
                                squeeze=False)
        axes = axes.flatten()  # Flatten for easy indexing
        
        colors = {'S': 'blue', 'E': 'orange', 'I': 'red', 'R': 'green'}
        
        for i, comm_id in enumerate(community_ids):
            ax = axes[i]
            
            # Aggregate across all age groups for this community
            comm_data = self.df[self.df['community_id'] == comm_id].groupby('time')[compartments].sum().reset_index()
            
            # Plot compartments
            for comp in compartments:
                ax.plot(comm_data['time'], comm_data[comp], 
                       label=comp, color=colors.get(comp, 'black'),
                       linewidth=2)
            
            ax.set_title(f'Community {comm_id}')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Population')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_communities, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Leave space for suptitle
        return fig
    
    def plot_aggregate_dynamics(self, by_age=False, **kwargs):
        """
        Plot population-level aggregate SEIR dynamics
        
        Args:
            by_age: Whether to stratify by age group
            **kwargs: Additional plotting arguments
        """
        
        if not hasattr(self, 'df'):
            raise RuntimeError("Must run simulation and finalize_results() before plotting")
        
        if by_age:
            # Aggregate by time and age group
            agg_data = self.df.groupby(['time', 'age_group'])[['S', 'E', 'I', 'R']].sum().reset_index()
            
            fig, axes = plt.subplots(1, len(self.age_groups), figsize=(12, 4))
            colors = {'S': 'blue', 'E': 'orange', 'I': 'red', 'R': 'green'}
            
            for i, age_group in enumerate(self.age_groups):
                ax = axes[i] if len(self.age_groups) > 1 else axes
                age_data = agg_data[agg_data['age_group'] == age_group]
                
                for comp in ['S', 'E', 'I', 'R']:
                    ax.plot(age_data['time'], age_data[comp], 
                           label=comp, color=colors[comp], linewidth=2)
                
                ax.set_title(f'Age Group {age_group}')
                ax.set_xlabel('Time (days)')
                ax.set_ylabel('Total Population')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)  # Leave more space for suptitle
        
        else:
            # Aggregate across all communities and ages
            agg_data = self.df.groupby('time')[['S', 'E', 'I', 'R']].sum().reset_index()
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            colors = {'S': 'blue', 'E': 'orange', 'I': 'red', 'R': 'green'}
            
            for comp in ['S', 'E', 'I', 'R']:
                ax.plot(agg_data['time'], agg_data[comp], 
                       label=comp, color=colors[comp], linewidth=3)
            
            ax.set_title('Total Population SEIR Dynamics')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Total Population')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Leave more space for suptitle
        return fig
    
    def plot_infection_heatmap(self, show_all_ages=True, **kwargs):
        """
        Plot heatmap of infections across communities over time
        
        Args:
            show_all_ages: If True, show all 3 age groups in subplots (default: True)
            **kwargs: Additional plotting arguments
        """
        
        if not hasattr(self, 'df'):
            raise RuntimeError("Must run simulation and finalize_results() before plotting")
        
        if show_all_ages:
            # Create subplot for each age group
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            for i, age_group in enumerate(self.age_groups):
                ax = axes[i]
                
                # Filter for specific age group
                age_data = self.df[self.df['age_group'] == age_group]
                
                # Pivot to create community x time matrix
                pivot_data = age_data.pivot(index='community_id', columns='time', values='I')
                
                # Create heatmap
                im = ax.imshow(pivot_data.values, aspect='auto', cmap='Reds', origin='lower')
                
                # Set labels
                ax.set_xlabel('Time (days)')
                if i == 0:  # Only label y-axis on first subplot
                    ax.set_ylabel('Community ID')
                ax.set_title(f'Age Group {age_group}')
                
                # Add colorbar to each subplot
                plt.colorbar(im, ax=ax, label='Infectious' if i == 1 else '')
                
                # Set ticks for better readability
                time_ticks = np.linspace(0, len(pivot_data.columns)-1, 5, dtype=int)
                ax.set_xticks(time_ticks)
                ax.set_xticklabels([f"{pivot_data.columns[i]:.0f}" for i in time_ticks])
            
            fig.suptitle('Infectious Population Heatmaps by Age Group', fontsize=16, y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(top=0.90)  # Leave space for suptitle
        
        else:
            # Show aggregate across all age groups
            agg_data = self.df.groupby(['time', 'community_id'])['I'].sum().reset_index()
            pivot_data = agg_data.pivot(index='community_id', columns='time', values='I')
            
            # Create heatmap
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            im = ax.imshow(pivot_data.values, aspect='auto', cmap='Reds', origin='lower')
            
            # Set labels
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Community ID')
            ax.set_title('Infectious Population Heatmap - All Ages')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Infectious Population')
            
            # Set ticks for better readability
            time_ticks = np.linspace(0, len(pivot_data.columns)-1, 5, dtype=int)
            ax.set_xticks(time_ticks)
            ax.set_xticklabels([f"{pivot_data.columns[i]:.0f}" for i in time_ticks])
            
            plt.tight_layout()
        
        return fig
    
    def get_peak_infections(self, age_group=None):
        """
        Get peak infection times and values for each community
        
        Args:
            age_group: Specific age group (default: aggregate all ages)
            
        Returns:
            DataFrame with community_id, peak_time, peak_infections
        """
        
        if not hasattr(self, 'df'):
            raise RuntimeError("Must run simulation and finalize_results() before analysis")
        
        if age_group is not None:
            data = self.df[self.df['age_group'] == age_group]
        else:
            # Aggregate across age groups
            data = self.df.groupby(['time', 'community_id'])['I'].sum().reset_index()
        
        # Find peak for each community
        peaks = []
        for comm_id in self.community_ids:
            comm_data = data[data['community_id'] == comm_id]
            if len(comm_data) > 0:
                peak_idx = comm_data['I'].idxmax()
                peak_info = comm_data.loc[peak_idx]
                peaks.append({
                    'community_id': comm_id,
                    'peak_time': peak_info['time'],
                    'peak_infections': peak_info['I']
                })
        
        return pd.DataFrame(peaks)
    
    def plot_network_graph(self, min_edge_weight=0.001, node_size_scale=500, **kwargs):
        """
        Plot NetworkX graph of metapopulation network with geographic positioning
        
        Args:
            min_edge_weight: Minimum edge weight to display (filter weak connections)
            node_size_scale: Scale factor for node sizes based on population
            **kwargs: Additional plotting arguments
        """
        
        if not hasattr(self, 'df') or not hasattr(self, 'spatial_data'):
            raise RuntimeError("Must run simulation and have spatial data for network graph")
        
        # Get gravity network from simulation to extract edge weights
        gravity_network = None
        for route in self.sim.networks.values():
            if hasattr(route, 'W'):  # GravityNetwork has weight matrix W
                gravity_network = route
                break
        
        if gravity_network is None:
            raise RuntimeError("No GravityNetwork found in simulation for edge weights")
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes with geographic positions and population sizes
        for i, comm_id in enumerate(self.community_ids):
            lat = self.spatial_data['lat'][i]
            lon = self.spatial_data['lon'][i] 
            pop = self.spatial_data['pop_size'][i]
            
            G.add_node(comm_id, 
                      lat=lat, 
                      lon=lon, 
                      pop_size=pop,
                      pos=(lon, lat))  # NetworkX uses (x, y) for positions
        
        # Add edges from gravity network weight matrix
        W = gravity_network.W
        for i in range(len(self.community_ids)):
            for j in range(i+1, len(self.community_ids)):  # Upper triangle only (undirected)
                weight = W[i, j]
                if weight > min_edge_weight:  # Filter weak edges
                    G.add_edge(self.community_ids[i], self.community_ids[j], weight=weight)
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Get positions (lon, lat coordinates)
        pos = nx.get_node_attributes(G, 'pos')
        
        # Node sizes based on population
        pop_sizes = [G.nodes[node]['pop_size'] for node in G.nodes()]
        max_pop = max(pop_sizes)
        node_sizes = [node_size_scale * (pop / max_pop) for pop in pop_sizes]
        
        # Edge widths based on weights - normalize for better visibility
        if G.edges():
            edge_weights = [G.edges[edge]['weight'] for edge in G.edges()]
            max_weight = max(edge_weights)
            min_weight = min(edge_weights)
            # Normalize to range [1, 6] for good visibility
            weight_range = max_weight - min_weight
            if weight_range > 0:
                edge_widths = [1 + 5 * (weight - min_weight) / weight_range for weight in edge_weights]
            else:
                edge_widths = [3.0] * len(edge_weights)  # All edges same weight
        else:
            edge_widths = []
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, 
                              node_size=node_sizes,
                              node_color='lightblue',
                              edgecolors='black',
                              alpha=0.7,
                              ax=ax)
        
        if G.edges():
            nx.draw_networkx_edges(G, pos,
                                  width=edge_widths,
                                  alpha=0.6,
                                  edge_color='gray',
                                  ax=ax)
        
        # Add node labels
        nx.draw_networkx_labels(G, pos, 
                               font_size=8,
                               font_weight='bold',
                               ax=ax)
        
        # Set title and labels
        ax.set_title('Metapopulation Network Graph\n(Node size ∝ Population, Edge width ∝ Transmission strength)', 
                    fontsize=14, pad=20)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # Add network statistics as text
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        density = nx.density(G) if n_nodes > 1 else 0.0
        
        stats_text = f"Nodes: {n_nodes}\nEdges: {n_edges}\nDensity: {density:.3f}\nMin edge weight: {min_edge_weight}"
        ax.text(0.02, 0.98, stats_text, 
               transform=ax.transAxes, 
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=10)
        
        # Set aspect ratio and grid
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def __repr__(self):
        """String representation"""
        if hasattr(self, 'df'):
            n_timepoints = len(self.times)
            n_communities = len(self.community_ids)
            return f"SEIRMetapopulationAnalyzer(communities={n_communities}, timepoints={n_timepoints})"
        else:
            return "SEIRMetapopulationAnalyzer(not yet run)"