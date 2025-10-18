"""
Age-stratified metapopulation SEIR model implementation

This module brings together all components for the metapopulation model:
- SEIRPopulation: Age-structured communities with Erlang staging
- GravityNetwork: Gravity model for spatial transmission  
- Helper functions: Community setup and visualization

Usage example:
    from stochastic_metapopulation import make_sim
    sim = make_sim()
    sim.run()
    plot_results(sim)
"""

import numpy as np
import starsim as ss
from seir_population import SEIRPopulation
from metapop_route import GravityNetwork

# Import MetaPopulation from starsim framework
MetaPopulation = ss.MetaPopulation
from seir_metapopulation_analyzer import SEIRMetapopulationAnalyzer
import pandas as pd
import matplotlib.pyplot as plt


def setup_communities(n_communities=50):
    """
    Create communities with key properties for metapopulation modeling
    
    Args:
        n_communities: Number of communities to create
        
    Returns:
        pd.DataFrame: Community data with spatial and demographic information
    """
    communities = []
    
    for _ in range(n_communities):
        lat = 50.0 + 8.0 * np.random.random() 
        lon = -6.0 + 8.0 * np.random.random()
        pop_size = int(np.random.lognormal(10, 1))
        
        # Realistic age distribution
        n_0_4 = int(0.06 * pop_size)    # 6% young children
        n_5_19 = int(0.20 * pop_size)   # 20% school age  
        n_20p = pop_size - n_0_4 - n_5_19  # 74% adults
        
        # Community heterogeneity
        rel_trans = np.random.uniform(0.8, 1.2)  # Transmission heterogeneity
        rel_sus = 1.0  # Default susceptibility

        # Initial prevalence
        init_prev = 0.0  # Start fully susceptible
        
        communities.append({
            'lat': lat, 'lon': lon, 'pop_size': pop_size,
            'N_0_4': n_0_4, 'N_5_19': n_5_19, 'N_20p': n_20p,
            'S_0_4': n_0_4, 'S_5_19': n_5_19, 'S_20p': n_20p,  # Initially all susceptible
            'rel_sus': rel_sus, 'rel_trans': rel_trans,
            'init_prev': init_prev
        })
    
    return pd.DataFrame(communities)


def make_sim(communities, R0=15.0, x_beta=0.80, 
                             duration_years=2.0, seed=42):
    """
    Create a complete metapopulation simulation
    
    Args:
        communities: DataFrame with community data  
        R0: Basic reproduction number (within-community)
        x_beta: Between-community transmission rate
        duration_years: Simulation duration in years
        seed: Random seed
        
    Returns:
        ss.Sim: Configured simulation ready to run
    """
    
    n_communities = len(communities)

    # Pick one community and set init_prev to 1%
    central_idx = n_communities // 2
    communities.at[central_idx, 'init_prev'] = 0.10  # Set initial prevalence to 10%

    # Create MetaPopulation with spatial metadata
    people = MetaPopulation(communities=communities) # Communities for lon, lat, and pop_size

    # Define contact matrix for age-structured transmission
    # Entries are effective contacts per person per day
    contact_matrix = np.array([
        [2, 1, 2],   # 0-4 contacts with [0-4, 5-19, 20+]
        [1, 5, 2],   # 5-19 contacts (high school mixing)
        [2, 2, 3]    # 20+ contacts
    ], dtype=float)
    
    # Create SEIR disease with disease-specific community parameters
    seir = SEIRPopulation(
        R0=R0,
        kE=3, kI=3,                      # Erlang stages
        dur_latent=ss.days(9),           # 9-day latent period
        dur_infectious=ss.days(8),       # 8-day infectious period
        contact_matrix=contact_matrix,   # Required: age-structured contact patterns
        communities=communities          # Disease-specific parameters (init_prev, rel_sus, rel_trans)
    )
    
    # Create gravity network route (simplified - gets data from sim.people)
    network = GravityNetwork(
        x_beta=x_beta,             # Multiplier on disease beta
        pop_exponent_1=1.0,                  # Linear in origin population
        pop_exponent_2=2.0,                  # Linear in destination population
        distance_exponent=2.0                # Inverse square distance decay
    )
    
    # Create analyzer to track community-level SEIR dynamics
    analyzer = SEIRMetapopulationAnalyzer()
    
    # Create simulation with MetaPopulation
    sim = ss.Sim(
        people=people,                  # Use MetaPopulation instead of n_agents
        networks=network,               # Route, not Network
        diseases=seir,
        analyzers=analyzer,             # Add SEIR analyzer
        start='2026-01-01', 
        dur=int(duration_years * 365),  # Number of time steps
        dt=ss.days(1),                   # Daily time steps
        rand_seed=seed,
    )
    
    return sim



def run_metapop_demo(n_communities=20, R0=15.0, x_beta=1.0, duration_years=1.0, seed=42):
    """
    Run a complete metapopulation demonstration
    
    Args:
        n_communities: Number of communities
        R0: Basic reproduction number  
        duration_years: Simulation duration
        seed: Random seed
        
    Returns:
        ss.Sim: Completed simulation
    """
    print("Creating metapopulation simulation...")

    # Create communities
    communities = setup_communities(n_communities)

    sim = make_sim(
        communities=communities, 
        R0=R0, 
        duration_years=duration_years,
        x_beta=x_beta,
        seed=seed
    )
    
    print("Running simulation...")
    sim.run()
    
    print("Simulation complete!")
    return sim


def plot_metapop_results(sim):
    """
    Generate comprehensive plots for metapopulation simulation results
    
    Args:
        sim: Completed simulation with SEIRMetapopulationAnalyzer
    """
    
    # Get the analyzer
    analyzer = sim.analyzers[0] if sim.analyzers else None
    
    if analyzer is None or not hasattr(analyzer, 'df'):
        print("No SEIRMetapopulationAnalyzer found or results not available")
        return
    
    print("Plotting results from analyzer")
    
    # Plot 1: Community dynamics (one panel per compartment, colored by community)
    print("Generating community dynamics plots...")
    n_communities = len(analyzer.community_ids)
    fig1 = analyzer.plot_community_dynamics(mode='by_compartment')  # Show all communities with new layout
    fig1.suptitle(f"SEIR Dynamics Across All Communities (n={n_communities})", fontsize=14, y=0.98)
    plt.subplots_adjust(top=0.88)  # Leave more space for suptitle
    
    # Plot 2: Aggregate dynamics by age group
    print("Generating aggregate dynamics plots...")
    fig2 = analyzer.plot_aggregate_dynamics(by_age=True)
    fig2.suptitle("Population-Level SEIR Dynamics by Age Group", fontsize=14, y=0.95)
    
    # Plot 3: Total population dynamics
    print("Generating total population dynamics...")
    fig3 = analyzer.plot_aggregate_dynamics(by_age=False)
    fig3.suptitle("Total Population SEIR Dynamics", fontsize=14, y=0.95)
    
    # Plot 4: Infection heatmap for all age groups
    print("Generating infection heatmap...")
    fig4 = analyzer.plot_infection_heatmap(show_all_ages=True)
    
    # Plot 5: Network graph visualization
    print("Generating network graph...")
    fig5 = analyzer.plot_network_graph()
    fig5.suptitle("Metapopulation Network Structure", fontsize=16, y=0.95)
    plt.subplots_adjust(top=0.85)  # Leave space for suptitle
    
    # Plot 6: Peak infection analysis
    print("Analyzing peak infections...")
    peak_data = analyzer.get_peak_infections()
    print(f"\nPeak infection statistics:")
    print(f"Mean peak time: {peak_data['peak_time'].mean():.1f} days")
    print(f"Mean peak infections: {peak_data['peak_infections'].mean():.1f}")
    print(f"Peak range: {peak_data['peak_infections'].min():.1f} - {peak_data['peak_infections'].max():.1f}")
    
    return fig1, fig2, fig3, fig4, fig5


# Main execution for testing
if __name__ == "__main__":
    # Run a demonstration with more communities for better visualization
    print("Running metapopulation SEIR simulation...")
    sim = run_metapop_demo(n_communities=256, R0=10.0, x_beta=2e-2, duration_years=0.5)
    
    # Basic results summary
    print(f"\nSimulation completed! Final time: {sim.t} days")
    
    # Generate comprehensive plots
    print("\nGenerating analysis plots...")
    plot_metapop_results(sim)
    
    # Show basic Starsim plot as well
    print("\nShowing default Starsim plot...")
    sim.plot()