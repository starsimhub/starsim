"""
Test results: export, resampling, plotting, 2D results, and nested results
"""
import numpy as np
import pytest
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

sc.options.interactive = False # Assume not running interactively

small = 100
medium = 1000


# %% Define the tests

@sc.timer()
def test_export():
    sc.heading('Testing results export and plotting')

    # Make a sim with 2 SIS models with varying units and dt
    d1 = ss.SIS(dt=ss.years(1/12), name='sis1')
    d2 = ss.SIS(dt=ss.years(0.5), name='sis2')
    sim = ss.Sim(n_agents=medium, diseases=[d1, d2], networks='random')

    # Run sim and pull out disease results
    sim.run()
    rs1 = sim.results.sis1
    rs2 = sim.results.sis2

    # Export a single result to a series or dataframe
    res = rs1.new_infections
    res_df = res.to_df()
    res_series = res.to_series()
    assert res[-1] == res_df.iloc[-1].value == res_series.iloc[-1]

    # Test resampling (always returns a Result)
    resampled = rs1.new_infections.resample(new_unit='year')
    assert isinstance(resampled, ss.Result)
    assert len(resampled.values) < len(rs1.new_infections.values)

    # Test that resample and annualize produce the same values
    annualized = rs1.new_infections.annualize()
    assert np.allclose(resampled.values, annualized.values, rtol=1e-10)

    # Test Results.annualize() (annualizes all results in the group)
    rs1_annual = rs1.annualize()
    assert isinstance(rs1_annual, ss.Results)
    assert len(rs1_annual.new_infections.values) == len(annualized.values)
    assert np.allclose(rs1_annual.new_infections.values, annualized.values, rtol=1e-10)

    # Export results of a whole module to a dataframe
    dfs = rs1.to_df()
    assert res_df.value.sum() == dfs.cum_infections.values[-1] == sim.summary.sis1_cum_infections

    # Export resampled summary of results to dataframe
    dfy1 = rs1.to_df(resample='year')
    dfy2 = rs2.to_df(resample='5YE')
    assert dfs.new_infections.iloc[:12].sum() == dfy1.new_infections.iloc[0]
    assert rs2.n_susceptible[:2].mean() == dfy2.n_susceptible.iloc[0]  # Entries 0 and 1 represent 2000
    assert rs2.n_susceptible[2:12].mean() == dfy2.n_susceptible.iloc[1]  # Entries 2-12 correspond to 2001-2005

    # Export whole sim to unified annualized dataframe
    sim_df = sim.to_df(resample='year', use_years=True)
    assert sim_df.sis1_n_infected.values[0] == rs1.n_infected[:12].mean()
    assert sim_df.sis2_n_infected.values[0] == rs2.n_infected[:2].mean()

    # Plot
    res.plot()
    sim.results.sis1.plot()
    sim.results.sis2.plot()

    return sim


@sc.timer()
def test_2d_results():
    sc.heading('Testing 2D results')

    # Make an analyzer with a 2D result for infections by sex
    class BySex(ss.Analyzer):
        def init_results(self):
            super().init_results()
            self.define_results(ss.Result('n_infected', groups=['female', 'male'], label='Infected'))

        def step(self):
            infected = self.sim.diseases.sis.infected
            female = self.sim.people.female
            self.results.n_infected[self.ti] = [np.count_nonzero(infected & female), np.count_nonzero(infected & ~female)]

    sim = ss.Sim(n_agents=small, pop_scale=10, diseases='sis', networks='random', analyzers=BySex())
    sim.run()
    res = sim.results.bysex.n_infected
    assert res.shape == (sim.t.npts, 2) # One column per group
    assert np.array_equal(res.sum(axis=1), sim.results.sis.n_infected) # Scaled the same as 1D results
    assert np.array_equal(res['male'], res[:,1]) and res['male'].label == 'Infected (male)' # Each group is a 1D result
    assert res.annualize().shape == res.resample('year').shape == (len(np.unique(res.timevec.years.astype(int))), 2) # Resampled along time only
    assert res.to_df().columns.tolist() == ['timevec', 'female', 'male']
    assert 'bysex_n_infected_male' in sim.to_df().columns # Columns for each group
    assert sim.summary.bysex_n_infected_male == res['male'].mean() # Summarized by group
    res.plot()
    sim.plot()
    with pytest.raises(ValueError):
        ss.Result('n_infected', groups=['low', 'high']) # Group names can't clash with attributes

    return sim


@sc.timer()
def test_nested_results():
    sc.heading('Testing nested results')

    # Make an analyzer that stores a result inside a nested Results object
    class Nested(ss.Analyzer):
        def init_results(self):
            super().init_results()
            self.results['nested'] = ss.Results(self)
            self.results.nested += ss.Result('n_infected', shape=self.t.npts, timevec=self.t.timevec)

        def step(self):
            self.results.nested.n_infected[self.ti] = np.count_nonzero(self.sim.diseases.sis.infected)

    sim = ss.Sim(n_agents=small, pop_scale=10, diseases='sis', networks='random', analyzers=Nested())
    sim.run()
    assert np.array_equal(sim.results.nested.nested.n_infected, sim.results.sis.n_infected) # Nested results are scaled too

    return sim


# %% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)

    # Start timing
    T = sc.tic()

    # Run tests
    sim1 = test_export()
    sim2 = test_2d_results()
    sim3 = test_nested_results()

    sc.toc(T)
    plt.show()
    print('Done.')
