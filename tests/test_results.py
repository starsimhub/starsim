"""
Test results: export, resampling, plotting, 2D results, and nested results
"""
import numpy as np
import pandas as pd
import pytest
import sciris as sc
import starsim as ss
import starsim.library as ssl
import matplotlib.pyplot as plt

sc.options.interactive = False # Assume not running interactively

small = 100
medium = 1000


# %% Define the tests

@sc.timer()
def test_export():
    sc.heading('Testing results export and plotting')

    # Make a sim with 2 SIS models with varying units and dt
    d1 = ssl.AgentSIS(dt=ss.years(1/12), name='sis1') # Written as code, since a declared disease cannot yet have a timestep of its own
    d2 = ssl.AgentSIS(dt=ss.years(0.5), name='sis2')
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

    # Resampling (always returns a Result) sums a flow over each year, and takes a stock at the end of it
    resampled = rs1.new_infections.resample(new_unit='year')
    assert isinstance(resampled, ss.Result)
    assert len(resampled.values) < len(rs1.new_infections.values)
    assert resampled[0] == rs1.new_infections[:12].sum() and resampled.values.sum() == rs1.new_infections.values.sum()
    assert rs1.n_infected.resample('year')[0] == rs1.n_infected[11] # December 2000

    # Results.resample() resamples all the results in the group
    rs1_annual = rs1.resample('year')
    assert isinstance(rs1_annual, ss.Results)
    assert np.allclose(rs1_annual.new_infections.values, resampled.values, rtol=1e-10)

    # Export results of a whole module to a dataframe
    dfs = rs1.to_df()
    assert res_df.value.sum() == dfs.cum_infections.values[-1] == sim.summary.sis1_cum_infections

    # Export resampled summary of results to dataframe
    dfy1 = rs1.to_df(resample='year')
    dfy2 = rs2.to_df(resample=ss.years(5))
    assert dfs.new_infections.iloc[:12].sum() == dfy1.new_infections.iloc[0]
    assert rs2.n_susceptible[10] == dfy2.n_susceptible.iloc[0] # Periods from the start, each including its end: entry 10 is 2005.0
    assert rs2.new_infections[:11].sum() == dfy2.new_infections.iloc[0]

    # Export whole sim to unified annualized dataframe
    sim_df = sim.to_df(resample='year', use_years=True)
    assert sim_df.sis1_n_infected.values[0] == rs1.n_infected[11]
    assert sim_df.sis2_n_infected.values[0] == rs2.n_infected[1]

    # Plot
    res.plot()
    sim.results.sis1.plot()
    sim.results.sis2.plot()

    # Results can be merged like a dict
    merged = sc.mergedicts(sim.results, {})
    assert merged.keys() == sim.results.keys()

    return sim


@sc.timer()
def test_2d_results():
    sc.heading('Testing 2D results')

    # Make an analyzer with a 2D result for infections by sex
    class BySex(ss.Analyzer):
        def init_results(self):
            super().init_results()
            self.define_results(ss.Result('n_infected', columns=['female', 'male'], label='Infected'))

        def step(self):
            infected = self.sim.diseases.sis.infected
            female = self.sim.people.female
            self.results.n_infected[self.ti] = [np.count_nonzero(infected & female), np.count_nonzero(infected & ~female)]

    sim = ss.Sim(n_agents=small, pop_scale=10, diseases='sis', networks='random', analyzers=BySex(), start=2000) # Calendar years, resampled by year below
    sim.run()
    res = sim.results.bysex.n_infected
    assert res.shape == (sim.t.npts, 2) # One column per sex
    assert np.array_equal(res.sum(axis=1), sim.results.sis.n_I) # Scaled the same as 1D results
    assert np.array_equal(res.male, res[:,1]) and res['male'].label == 'Infected (male)' # Each column is a 1D result
    assert res.resample('year').shape == (len(np.unique(res.timevec.years.astype(int))) - 1, 2) # Resampled along time only: each point after the start closes the year before it
    assert res.to_df().columns.tolist() == ['timevec', 'female', 'male']
    assert 'bysex_n_infected_male' in sim.to_df().columns # Dataframe columns for each result column
    assert sim.summary.bysex_n_infected_male == res['male'][-1] # Summarized by column, a stock by its last value
    res.plot()
    sim.plot()
    with pytest.raises(ValueError):
        ss.Result('n_infected', columns=['low', 'high']) # Column names can't clash with attributes

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
    assert np.array_equal(sim.results.nested.nested.n_infected, sim.results.sis.n_I) # Nested results are scaled too

    return sim


@sc.timer()
def test_declared_scale():
    sc.heading('Testing that results declared as sums of states are scaled, as agents')
    S, I, R = ss.states(3)
    sir = ss.Disease(S >> I @ 0.3 * I, I >> R @ ss.days(10), ever=I + R, share=I / (S + I + R), init=dict(I=0.01), name='sir')
    sim = ss.Sim(sir, unit='day', stop=60, total_pop=10_000, n_agents=small*10, solver='abm', verbose=0).run()
    res = sim.results.sir
    assert np.allclose(res.ever, res.n_I + res.n_R) # Numbers of people, not of agents
    assert np.allclose(res.share, res.n_I/(res.n_S + res.n_I + res.n_R)) # A ratio is the same either way
    return sim


def test_flow_ratios():
    sc.heading('Testing flows in results: the per-capita incidence, infection / S')
    S, I, R = ss.states(3)
    infection = S >> I @ 0.3 * I
    sir = ss.Disease(infection, I >> R @ ss.days(10), incidence=infection / S, init=dict(I=0.01), name='sir')
    for solver in ['ode', 'binomial', 'abm']:
        for strat in [False, True]:
            sim = ss.Sim(sir, unit='day', stop=60, total_pop=10_000, n_agents=small*10, solver=solver, verbose=0)
            if strat:
                sim.stratify(age=[0, 50])
            res = sim.run().results.sir
            assert np.allclose(res.incidence[1:], res.infection[1:]/res.n_S[1:])
            if strat: # By group: the flow and the stock of the group
                young = res.incidence.select(age='0-49')[1:]
                assert np.allclose(young, res.infection.select(age='0-49')[1:]/res.n_S.select(age='0-49')[1:])
                weekly = res.infection.resample('week') # Resampled from the start, by group too
                assert np.allclose(weekly.select(age='0-49') + weekly.select(age='50+'), weekly) and np.isclose(weekly[1], res.infection[8:15].sum())
                assert res.n_S.resample('week')[1] == res.n_S[14] # A stock at the end of each week
                assert np.isclose(res.incidence.resample('week')[1], res.infection[8:15].sum()/res.n_S[8:15].sum()) # A ratio of the sums of its parts

    # A ratio with nobody in its denominator is 0 (nothing divided by anything), not an error
    for solver in ['ode', 'abm']:
        S, I, R = ss.states(3)
        sir = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), share=R/R, init=dict(I=0.01), name='sir')
        res = ss.Sim(sir, unit='day', stop=10, total_pop=10_000, n_agents=small*10, solver=solver, verbose=0).run().results.sir
        assert res.share[0] == 0 and res.share[-1] == 1

    return sim


def test_groups_kept(monkeypatch):
    sc.heading('Testing results kept by group: only those named, and every few steps if every step would take too much memory')
    S, I, R = ss.states(3)
    sir = ss.Disease(infection=S >> I @ 0.3 * I, recovery=I >> R @ ss.days(10), share=I / (S + I + R), init=dict(I=0.01), name='sir')
    make = lambda solver, **kw: ss.Sim(sir, unit='day', stop=91, total_pop=10_000, n_agents=small*10, solver=solver, stratify=dict(age=[0, 50]), verbose=0, **kw)
    res = make('ode', results=['n_I', 'infection']).run().results.sir
    assert res.n_I.strata is not None and getattr(res.n_S, 'strata', None) is None and res.infection.cum.strata is not None # Only the named results by group (and the cumulative total of a flow kept by group)
    with pytest.raises(sc.KeyNotFoundError):
        make('ode', results=['n_J']).init()
    for solver in ['ode', 'binomial', 'abm']: # A whole run in parts, and a stepped run
        full = make(solver).run().results.sir
        monkeypatch.setattr(ss.solvers.Engine, 'group_bytes', 1000) # Too little to keep them by group at every step
        sim = make(solver).run()
        monkeypatch.undo()
        res = sim.results.sir
        k = res.n_I.group_ti[1] # The steps in each row
        assert k > 1 and any('kept by group every' in note for note in sim.notes)
        assert np.allclose(res.n_I, full.n_I) # The totals at every step (as agents, with the same seed)
        assert np.isclose(res.infection.select(age='0-49')[2], full.infection.select(age='0-49')[k+1:2*k+1].sum()) # A flow by group: its total over the steps of a row
        assert np.isclose(res.n_I.select(age='50+')[2], full.n_I.select(age='50+')[2*k]) # A state by group: at the end of a row
        assert np.isclose(res.share.select(age='0-49')[-1], full.share.select(age='0-49')[-1]) # A ratio by group: of its parts at the end
        assert np.allclose(res.infection.resample(ss.days(2*k)).select(age='0-49'), full.infection.resample(ss.days(2*k)).select(age='0-49'))

    # Many runs: a ratio by group is the ratio of its parts in each run, as in one sim (before: the numerator, or an error)
    for solver in ['binomial', 'abm']:
        msim = make(solver).run(n_runs=3)
        share = msim.results.sir.share.select(age='0-49')
        assert np.isclose(share.values[-1, -1], msim.sims[-1].results.sir.share.select(age='0-49')[-1]) and share.values.max() <= 1 # The last run's sim

    # Many runs: by group every few steps too, if all of them would take too much memory
    monkeypatch.setattr(ss.solvers.Compartmental, 'max_bytes', 1000)
    n_I = make('binomial').run(n_runs=3).results.sir.n_I
    young, old = n_I.select(age='0-49').values, n_I.select(age='50+').values
    assert young.shape == (3, len(n_I.group_ti)) and np.allclose(young + old, n_I.values[:, n_I.group_ti])
    return sim


def test_result_exprs():
    sc.heading('Testing results declared in a class body')
    class SIR(ss.Disease):
        S = ss.State()
        I = ss.State()
        R = ss.State()
        infection = S >> I @ 0.3*I
        recovery = I >> R @ ss.days(10)
        prev = I / (S + I + R)
        n_ever = I + R
    S, I, R = ss.states(3)
    plain = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), n_ir=I + R, init=dict(I=0.01), name='sir')
    for solver in ['ode', 'abm']:
        for sir in [SIR(init=dict(I=0.01), name='sir'), plain]:
            res = ss.Sim(sir, unit='day', stop=30, total_pop=10_000, n_agents=medium, solver=solver, verbose=0).run().results.sir
            if 'prev' in res:
                assert np.allclose(res.prev, res.n_I/(res.n_S + res.n_I + res.n_R)) and np.allclose(res.n_ever, res.n_I + res.n_R)
            else:
                assert np.allclose(res.n_ir, res.n_I + res.n_R)
    return res


# %% Run as a script
def test_time_points():
    sc.heading('Testing results by time point: the start, then the end of each step')
    # Deaths requested during a step of a declared model are counted in it
    def kill(sim):
        sim.people.request_death(sim.people.auids[:2])
    S, I = ss.states(2)
    dis = ss.Disease(infection=ss.transmission('S -> I', beta=0.0, source='I'), init=dict(I=1), kill=kill, name='dis')
    res = ss.Sim(dis, people=ss.People(1000), unit='day', stop=4, verbose=0).run().results
    assert np.array_equal(res.n_alive, [1000, 998, 996, 994, 992])
    assert np.array_equal(res.new_deaths, [0, 2, 2, 2, 2])

    # Weekly data is matched to the total of a flow over each week, whatever the rounding of the sim's times, and whatever the flow is called
    S, I, R = ss.states(3)
    infection = S >> I @ ss.perday(0.4)*I
    cases = ss.outcome(0.6*infection)
    sim = ss.Sim(ss.Disease(infection, I >> R @ ss.perday(0.15), cases, init=dict(I=10), name='sir'), total_pop=10_000, unit='day', stop=ss.weeks(12), verbose=0).run()
    res = sim.results.sir
    weeks = np.arange(7, 85, 7)
    weekly = lambda flow: np.add.reduceat(flow.values[1:], np.arange(0, 84, 7)) # Days 1 to 7, 8 to 14, ...
    out = ss.calibration.compare(sim, pd.DataFrame(dict(cases=weekly(res.cases), new_infections=weekly(res.infection)), index=weeks))
    assert len(out.cases[1]) == 12 and np.allclose(*out.cases)
    assert 'new_infections' not in res and np.allclose(*out.new_infections) # Starsim 3's name for the one transmission's flow
    assert np.allclose(res.cases.resample('week'), weekly(res.cases)) # The same weekly totals
    return sim


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)

    # Start timing
    T = sc.tic()

    # Run tests
    sim1 = test_export()
    sim2 = test_2d_results()
    sim3 = test_nested_results()
    sim4 = test_declared_scale()
    sim4b = test_flow_ratios()
    sim5 = test_time_points()
    res6 = test_result_exprs()

    sc.toc(T)
    plt.show()
    print('Done.')
