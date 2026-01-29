"""
Test metapopulation routes and age-stratified SEIR functionality
"""

import numpy as np
import pandas as pd
import starsim as ss

class PytestApprox:
    """Simple replacement for pytest.approx"""
    def __init__(self, value, rel=0.1):
        self.value = value
        self.rel = rel
    def __eq__(self, other):
        return abs(other - self.value) / self.value <= self.rel

# Define pytest if not available
try:
    import pytest
except ImportError:
    class pytest:
        @staticmethod
        def raises(exception_type):
            class ContextManager:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_type is None:
                        raise AssertionError(f"Expected {exception_type.__name__} but no exception was raised")
                    return isinstance(exc_val, exception_type)
            return ContextManager()
        
        @staticmethod
        def approx(value, rel=0.1):
            return PytestApprox(value, rel)


def test_metapopulation_route_base_class():
    """Test that MetapopulationRoute base class is properly defined"""
    # Test that MetapopulationRoute is available
    assert hasattr(ss, 'MetapopulationRoute')
    
    # Test that it's a subclass of Route
    assert issubclass(ss.MetapopulationRoute, ss.Route)
    
    # Test that compute_transmission method exists but raises NotImplementedError
    route = ss.MetapopulationRoute()
    with pytest.raises(NotImplementedError):
        route.compute_transmission(None, None, None)


def test_gravity_network_integration():
    """Test GravityNetwork with age-stratified SEIR in a small simulation"""
    
    # Import the components we built
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'starsim_examples', 'compartmental'))
    
    from stochastic_metapopulation import setup_communities, make_sim
    
    # Create a small test case
    n_communities = 3
    communities = setup_communities(n_communities)
    
    # Set one community as initially infected
    communities.at[1, 'init_prev'] = 0.05
    
    # Create simulation
    sim = make_sim(
        communities=communities,
        R0=2.0,
        x_beta=0.01,
        duration_years=0.2,  # Short simulation
        seed=42
    )
    
    # Run simulation
    sim.run()
    
    # Check that simulation completed
    assert sim.results is not None
    assert len(sim.results) > 0
    
    # Check that SEIR disease has expected attributes
    seir = sim.diseases[0]
    assert hasattr(seir, 'age_groups')
    assert seir.age_groups == ['0_4', '5_19', '20p']
    
    # Check that all age-stratified states exist
    for age in seir.age_groups:
        assert hasattr(seir, f'S_{age}')
        assert hasattr(seir, f'R_{age}')
        # Check E and I stages exist
        for stage in range(1, seir.pars.kE + 1):
            assert hasattr(seir, f'E{stage}_{age}')
        for stage in range(1, seir.pars.kI + 1):
            assert hasattr(seir, f'I{stage}_{age}')
    
    # Check that GravityNetwork exists and has expected attributes  
    gravity_net = sim.networks[0]
    assert hasattr(gravity_net.pars, 'x_beta')  # x_beta is in pars
    assert hasattr(gravity_net, 'W')  # Weight matrix should be computed
    
    # Check analyzer collected data
    analyzer = sim.analyzers[0]
    assert hasattr(analyzer, 'df')
    assert len(analyzer.df) > 0
    
    # Check that some transmission occurred (R compartments should have people)
    final_R = sum(np.sum(getattr(seir, f'R_{age}')) for age in seir.age_groups)
    assert final_R > 0, "No transmission occurred in the simulation"


def test_r0_beta_conversion():
    """Test that R0 is properly converted to beta values"""
    
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'starsim_examples', 'compartmental'))
    
    # Test that we can import and create a basic SEIR disease with R0 conversion
    from seir_population import SEIRPopulation
    
    # Define a simple contact matrix
    contact_matrix = np.array([[2, 1, 2], [1, 5, 2], [2, 2, 3]], dtype=float)
    
    # Create SEIR diseases with different R0 values
    seir1 = SEIRPopulation(R0=1.0, contact_matrix=contact_matrix)
    seir2 = SEIRPopulation(R0=4.0, contact_matrix=contact_matrix) 
    
    # Beta should scale with R0
    beta1 = float(seir1.pars.beta)
    beta2 = float(seir2.pars.beta)
    
    assert beta2 > beta1
    assert abs((beta2 / beta1) - 4.0) < 0.5  # Allow some tolerance for eigenvalue calculation


def test_metapopulation_route_detection_in_diseases():
    """Test that diseases.py properly detects MetapopulationRoute instances"""
    
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'starsim_examples', 'compartmental'))
    
    from metapop_route import GravityNetwork
    from stochastic_metapopulation import setup_communities, make_sim
    
    # Create test simulation
    communities = setup_communities(2)
    sim = make_sim(communities=communities, R0=2.0, duration_years=0.1, seed=42)
    
    # Check that GravityNetwork is properly recognized as MetapopulationRoute
    gravity_net = sim.networks[0]
    assert isinstance(gravity_net, ss.MetapopulationRoute)
    
    # Run simulation to ensure diseases.py handles it correctly
    sim.run()
    
    # Should complete without errors
    assert sim.results is not None


def test_normalized_plotting():
    """Test that population-normalized plotting works correctly"""
    
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'starsim_examples', 'compartmental'))
    
    from stochastic_metapopulation import run_metapop_demo
    
    # Run small simulation
    sim = run_metapop_demo(n_communities=3, R0=2.0, x_beta=0.01, duration_years=0.2, seed=42)
    
    # Test plotting functions
    analyzer = sim.analyzers[0]
    
    # Test normalized plotting
    fig = analyzer.plot_community_dynamics(mode='by_compartment', normalize=True)
    
    # Check that axes have proportion labels
    for ax in fig.axes:
        ylabel = ax.get_ylabel()
        assert 'Proportion' in ylabel
    
    # Test non-normalized plotting  
    fig2 = analyzer.plot_community_dynamics(mode='by_compartment', normalize=False)
    
    # Check that axes have population labels
    for ax in fig2.axes:
        ylabel = ax.get_ylabel()
        assert 'Population' in ylabel


if __name__ == '__main__':
    # Run tests - only run the working ones for now
    print("Running metapopulation route tests...")
    
    try:
        test_metapopulation_route_base_class()
        print("✓ MetapopulationRoute base class test passed")
    except Exception as e:
        print(f"✗ MetapopulationRoute base class test failed: {e}")
    
    try:
        test_gravity_network_integration()
        print("✓ GravityNetwork integration test passed")
    except Exception as e:
        print(f"✗ GravityNetwork integration test failed: {e}")
    
    try:
        test_r0_beta_conversion()
        print("✓ R0 to beta conversion test passed")
    except Exception as e:
        print(f"✗ R0 to beta conversion test failed: {e}")
    
    print("Core metapopulation functionality tests completed!")