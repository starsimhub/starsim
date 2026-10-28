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
    assert resampled[0] == rs1.new_infections[:13].sum() # The first year's period includes its end (2001.01.01), as resample() says
    assert resampled.values.sum() == rs1.new_infections.values.sum()
    assert rs1.infected.resample('year')[0] == rs1.infected[12] # The end of the first year, 2001.01.01

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
    assert dfs.new_infections.iloc[:13].sum() == dfy1.new_infections.iloc[0] # As above
    assert rs2.susceptible[10] == dfy2.susceptible.iloc[0] # Periods from the start, each including its end: entry 10 is 2005.0
    assert rs2.new_infections[:11].sum() == dfy2.new_infections.iloc[0]

    # Export whole sim to unified annualized dataframe
    sim_df = sim.to_df(resample='year', use_years=True)
    assert sim_df.sis1_infected.values[0] == rs1.infected[12]
    assert sim_df.sis2_infected.values[0] == rs2.infected[2] # The end of the first year, as for sis1

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
    sc.heading('Testing 2D results and nested results')

    # Make an analyzer with a 2D result for infections by sex
    class BySex(ss.Analyzer):
        def init_results(self):
            super().init_results()
            self.define_results(ss.Result('infected', columns=['female', 'male'], label='Infected', kind='stock'))

        def step(self):
            infected = self.sim.diseases.sis.I
            female = self.sim.people.female
            self.results.infected[self.ti] = [np.count_nonzero(infected & female), np.count_nonzero(infected & ~female)]

    # And one that stores a result inside a nested Results object
    class Nested(ss.Analyzer):
        def init_results(self):
            super().init_results()
            self.results['nested'] = ss.Results(self)
            self.results.nested += ss.Result('infected', shape=self.t.npts, timevec=self.t.timevec, kind='stock')

        def step(self):
            self.results.nested.infected[self.ti] = np.count_nonzero(self.sim.diseases.sis.I)

    sim = ss.Sim(n_agents=small, pop_scale=10, diseases='sis', networks='random', analyzers=[BySex(), Nested()], start=2000) # Calendar years, resampled by year below
    sim.run()
    res = sim.results.bysex.infected
    assert res.shape == (sim.t.npts, 2) # One column per sex
    assert np.array_equal(res.sum(axis=1), sim.results.sis.I) # Scaled the same as 1D results
    assert np.array_equal(res.male, res[:,1]) and res['male'].label == 'Infected (male)' # Each column is a 1D result
    assert res.resample('year').shape == (len(np.unique(res.timevec.years.astype(int))) - 1, 2) # Resampled along time only: each point after the start closes the year before it
    assert res.to_df().columns.tolist() == ['time', 'female', 'male']
    assert 'by female, male' in repr(res) # Each column labelled (it was an unlabelled array)
    assert 'bysex_infected_male' in sim.to_df().columns # Dataframe columns for each result column
    assert sim.summary.bysex_infected_male == res['male'][-1] # Summarized by column, a stock by its last value
    res.plot()
    sim.plot()
    with pytest.raises(ValueError):
        ss.Result('infected', columns=['low', 'high']) # Column names can't clash with attributes
    assert np.array_equal(sim.results.nested.nested.infected, sim.results.sis.I) # Nested results are scaled too
    return sim


@sc.timer()
def test_declared_scale():
    sc.heading('Testing that results declared as sums of states are scaled, as agents')
    S, I, R = ss.states(3)
    sir = ss.Disease(S >> I @ 0.3 * I, I >> R @ ss.days(10), ever=I + R, share=I / (S + I + R), init=dict(I=0.01), name='sir')
    sim = ss.Sim(sir, unit='day', stop=60, total_pop=10_000, n_agents=medium, solver='abm', verbose=0).run()
    res = sim.results.sir
    assert np.allclose(res.ever, res.I + res.R) # Numbers of people, not of agents
    assert np.allclose(res.share, res.I/(res.S + res.I + res.R)) # A ratio is the same either way
    return sim


def test_flow_ratios():
    sc.heading('Testing flows in results: the per-capita incidence, infection / S')
    S, I, R = ss.states(3)
    infection = S >> I @ 0.3 * I
    sir = ss.Disease(infection, I >> R @ ss.days(10), incidence=infection / S, init=dict(I=0.01), name='sir')
    for solver in ['ode', 'binomial', 'abm']:
        for strat in [False, True]:
            sim = ss.Sim(sir, unit='day', stop=60, total_pop=10_000, n_agents=medium, solver=solver, verbose=0)
            if strat:
                sim.stratify(age=[0, 50])
            res = sim.run().results.sir
            assert np.allclose(res.incidence[1:], res.infection[1:]/res.S[1:])
            if strat: # By group: the flow and the stock of the group
                young = res.incidence.select(age='0-49')[1:]
                assert np.allclose(young, res.infection.select(age='0-49')[1:]/res.S.select(age='0-49')[1:])
                weekly = res.infection.resample('week') # Resampled from the start, by group too
                assert np.allclose(weekly.select(age='0-49') + weekly.select(age='50+'), weekly) and np.isclose(weekly[1], res.infection[8:15].sum())
                assert res.S.resample('week')[1] == res.S[14] # A stock at the end of each week
                assert np.isclose(res.incidence.resample('week')[1], res.infection[8:15].sum()/res.S[8:15].sum()) # A ratio of the sums of its parts

    # A ratio with nobody in its denominator is 0 (nothing divided by anything), not an error
    for solver in ['ode', 'abm']:
        S, I, R = ss.states(3)
        sir = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), share=R/R, init=dict(I=0.01), name='sir')
        res = ss.Sim(sir, unit='day', stop=10, total_pop=10_000, n_agents=medium, solver=solver, verbose=0).run().results.sir
        assert res.share[0] == 0 and res.share[-1] == 1

    return sim


@sc.timer()
def test_summarize_by():
    sc.heading('Testing kind and summarize_by: guessed from the name, given, None, and added')
    res = ss.Result('new_infections', values=np.arange(5.0))
    assert res.kind == 'flow' and res.summarize_by == 'sum' and res.summarize() == 10 and ss.Result('n_infected').kind == 'stock' and ss.Result('prevalence').summarize_by == 'mean'
    assert ss.Result('x', values=np.arange(5.0), kind='stock').summarize() == 4
    mean = ss.Result('x', values=np.arange(5.0), kind='flow', summarize_by='mean') # A flow summarized another way is still a flow
    assert mean.summarize() == 2 and mean.cum[-1] == 10
    assert ss.Result('x', summarize_by='last').kind == 'stock' # The kind follows the summary if only that is given
    with pytest.raises(ValueError):
        ss.Result('x', kind='count')
    none = ss.Result('timer', values=np.arange(5.0), summarize_by=None)
    with pytest.raises(ValueError):
        none.summarize()
    with pytest.raises(ValueError):
        ss.Result('x', summarize_by='median')
    ss.Result.summarizers['max'] = lambda arr, starts, ends: np.maximum.reduceat(arr, starts, axis=0)
    try:
        assert ss.Result('x', values=np.array([1.0, 5, 2]), summarize_by='max').summarize() == 5
    finally:
        del ss.Result.summarizers['max']

    # A result not summarized over time is left out when all the results are resampled, as in summarize()
    class Timer(ss.Analyzer):
        def init_results(self):
            super().init_results()
            self.define_results(ss.Result('timer', kind=None))
        def step(self):
            self.results.timer[self.ti] = self.ti
    sim = ss.Sim(ss.SIS(), solver='ode', analyzers=Timer(name='a'), start=2000, stop=2002, dt=ss.months(1), verbose=0).run()
    annual = sim.results.resample('year')
    assert 'timer' not in annual.a and len(annual.sis.I) == 2
    assert 'a_timer' not in sim.to_df(resample='year') and 'a_timer' in sim.to_df()
    return res


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
        for sir in [SIR(n_kw=SIR.I + SIR.R, init=dict(I=0.01), name='sir'), plain]: # A result passed to an instance can use the class's states
            res = ss.Sim(sir, unit='day', stop=30, total_pop=10_000, n_agents=medium, solver=solver, verbose=0).run().results.sir
            if 'prev' in res:
                assert np.allclose(res.prev, res.I/(res.S + res.I + res.R)) and np.allclose(res.n_ever, res.I + res.R)
                assert np.allclose(res.n_kw, res.n_ever) # Scaled, as the instance's states are
            else:
                assert np.allclose(res.n_ir, res.I + res.R)
    return res


def test_standard_prevalence():
    sc.heading('Testing the standard prevalence by group, as a ratio')
    for solver in ['ode', 'binomial', 'abm']:
        sis = ss.SIS(beta=ss.peryear(0.3), init_prev=0.05)
        sis.stratify(risk=['low', 'high'], split=[0.8, 0.2], adjust=dict(infection=[0.5, 3]))
        res = ss.Sim(sis, solver=solver, total_pop=10_000, n_agents=10_000, start=2000, stop=2010, verbose=0).run().results.sis
        high, low = res.prevalence.select(risk='high'), res.prevalence.select(risk='low')
        assert high[-1] > 2*low[-1] > 0, f'{solver}: {high[-1]:.3f} against {low[-1]:.3f}'
        assert np.allclose(res.I.select(risk='high')/(res.S.select(risk='high') + res.I.select(risk='high')), high)
        assert np.allclose(0.8*low + 0.2*high, res.prevalence, rtol=0.05) # The groups keep their shares of people, about
    return res


@sc.timer()
def test_results_by_group():
    sc.heading('Testing results by group in a module written as code: counts by sex and age among the living, weighted')
    edges = [0, 15, 25, 50]
    class ByGroup(ss.Analyzer):
        def init_results(self):
            super().init_results()
            self.define_results(ss.Result('n_infected', groups=True), ss.Result('prevalence', scale=False, groups=True))

        def step(self):
            groups, dis = self.sim.groups, self.sim.diseases[0]
            infected = groups.count(dis.I if dis.states else dis.infected) # Declared, or written as code
            self.results.n_infected.set_groups(self.ti, infected)
            self.results.prevalence.set_groups(self.ti, infected, groups.count())

    # Alongside a declared disease, whose results are kept by group already; agents of different weights, and people born and dying
    kw = dict(networks='random', demographics=[ss.Births(birth_rate=20), ss.Deaths(death_rate=15)], analyzers=ByGroup(name='bygroup'), n_agents=medium, start=2000, stop=2010, verbose=0)
    sim = ss.Sim(ss.SIS(beta=ss.peryear(0.3), init_prev=0.1), stratify=dict(sex=['female', 'male'], age=edges), **kw).init()
    sim.people.scale[sim.people.auids[::3]] = 3
    sim.run()
    sis, res = sim.results.sis, sim.results.bygroup
    assert np.array_equal(res.n_infected.strata[1:], sis.I.strata[1:]) # Each group, at each step after the start (whose weights were set after it)
    assert np.allclose(res.prevalence.select(sex='female', age='15-24')[1:], sis.prevalence.select(sex='female', age='15-24')[1:]) # Summed over the groups, then divided
    assert np.allclose(res.prevalence.by('age')[1:], sis.prevalence.by('age')[1:])
    ppl = sim.people
    age = ppl.age.raw - sim.t.dt_year # As counted, before the people aged at the end of the last step
    women = ppl.alive.raw & ppl.female.raw & (age >= 15) & (age < 25)
    assert np.isclose(res.n_infected.select(sex='female', age='15-24')[-1], ppl.scale.raw[women & sim.diseases.sis.I.raw].sum()) # A direct count, by weight

    # With only modules written as code: the stratification is used by the results kept by group
    sim = ss.Sim(ssl.AgentSIS(beta=ss.peryear(0.3), init_prev=0.1), stratify=dict(sex=['female', 'male']), **kw).run()
    res = sim.results.bygroup
    ppl, dis = sim.people, sim.diseases.agentsis
    assert res.n_infected[-1] == (ppl.alive.raw & dis.infected.raw).sum() and res.n_infected.by('sex').shape == (sim.t.npts, 2)
    with pytest.raises(ValueError, match='groups=True'):
        ss.Sim(ssl.AgentSIS(), stratify=dict(sex=['female', 'male']), n_agents=small, verbose=0).init() # Nothing uses it
    return sim


def test_time_points():
    sc.heading('Testing results by time point: the start, then the end of each step')
    # Deaths requested during a step of a declared model are counted in it
    def kill(sim):
        sim.people.request_death(sim.people.auids[:2])
    S, I = ss.states('S I')
    dis = ss.Disease(infection=ss.Transition(S, I, rate=0.0 * I), init_n=dict(I=1), kill=kill, name='dis')
    res = ss.Sim(dis, people=ss.People(1000), unit='day', stop=4, verbose=0).run().results
    assert np.array_equal(res.alive, [1000, 998, 996, 994, 992])
    assert np.array_equal(res.new_deaths, [0, 2, 2, 2, 2])

    # Weekly data is matched to the total of a flow over each week, whatever the rounding of the sim's times, and whatever the flow is called
    S, I, R = ss.states(3)
    infection = S >> I @ ss.perday(0.4)*I
    cases = ss.outcome(0.6*infection)
    sim = ss.Sim(ss.Disease(infection, I >> R @ ss.perday(0.15), cases, init_n=dict(I=10), name='sir'), total_pop=10_000, unit='day', stop=ss.weeks(12), verbose=0).run()
    res = sim.results.sir
    weeks = np.arange(7, 85, 7)
    weekly = lambda flow: np.add.reduceat(flow.values[1:], np.arange(0, 84, 7)) # Days 1 to 7, 8 to 14, ...
    out = ss.results.compare(sim, pd.DataFrame(dict(cases=weekly(res.cases), infection=weekly(res.infection)), index=weeks))
    assert len(out.cases[1]) == 12 and np.allclose(*out.cases) and np.allclose(*out.infection)
    assert np.allclose(res.cases.resample('week'), weekly(res.cases)) # The same weekly totals
    return sim


def cohort(solver, p=0.3, data=None):
    """ 10,000 people who leave P after 5 days on average: a share p to C, and the rest to R """
    P, C, R = ss.states('P C R')
    course = ss.Transition(P, [C, R], dur=ss.days(5), p=[p, None])
    dis = ss.Disease(P, C, R, course=course, cases=ss.outcome(0.5*course.C), first='P', name='dis')
    dis.stratify(age=ss.years([0, 40]))
    return ss.Sim(dis, n_agents=10_000, solver=solver, unit='day', stop=60, data=data, verbose=0)


@sc.timer()
def test_destinations():
    sc.heading('Testing the flow to each destination of a transition with several')
    for solver in ['ode', 'binomial', 'gillespie', 'abm']:
        res = cohort(solver).run().results.dis
        C, R, total = res.course.C, res.course.R, res.course
        print(f'{solver}: {C.sum():n} to C, {R.sum():n} to R')
        assert np.allclose(C + R, total) and np.allclose((C + R).select(age=0), total.select(age=0)) # The total is the sum of its destinations, by group too
        if solver == 'ode':
            assert np.isclose(C.sum(), 3000, rtol=0.01)
        else:
            assert abs(C.sum() - 3000) < 3*np.sqrt(10_000*0.3*0.7) # Within 3 standard errors
        assert C is res['course.C'] and np.array_equal(total.select(to='C'), C) and np.allclose(total.select(to=['C', 'R']), total) # select(to=) is a second spelling
        assert abs(res.cases.sum() - 0.5*C.sum()) <= 3*np.sqrt(0.25*C.sum()) # A destination is a source of an outcome (drawn, except under the ODE)
    with pytest.raises(ValueError):
        res.P.select(to='C') # Not a transition with several destinations

    # A data column named for a destination is matched to it, and fitted
    truth = cohort('ode').run().results.dis.course.C
    data = pd.DataFrame({'course.C': truth.values[1:]}, index=truth.timevec[1:]) # Daily, as the flow is
    fit = cohort('ode', p=ss.par(0.5, bounds=[0.05, 0.95], name='p'), data=data).fit(gof='rmse')
    assert np.isclose(fit.table.best.iloc[0], 0.3, rtol=1e-2)
    return res


@sc.timer()
def test_compare_data():
    sc.heading('Testing lining up results with data')
    sim = ss.Sim(diseases='sis', networks='random', n_agents=small, start=2000, stop=2010, verbose=0).run()
    res = sim.results.sis
    dates = [ss.date(f'{y}-07-01') for y in range(2001, 2010)] # Between steps, so a stock is interpolated
    df = pd.DataFrame({'sis.prevalence': 0.1, 'sis.infection': 5.0}, index=pd.Index(dates, dtype=object))
    compared = ss.results.compare(sim, df)
    lined = ss.results.lineup(sim, df)
    years = ss.results.data_times(sim, df)[1]
    assert np.allclose(compared['sis.prevalence'][1], np.interp(years, sim.t.yearvec, res.prevalence.values)) # A stock, at the data's times
    assert np.allclose(compared['sis.infection'][1][1:], [res.infection.values[(sim.t.yearvec > y-1) & (sim.t.yearvec <= y)].sum() for y in years[1:]]) # A flow, summed over the year to each
    for col,(W,expected,rows) in lined.items(): # The same matrix as calibration uses
        assert np.allclose(W @ ss.results.match_column(sim, col).values, compared[col][1])
    return compared


@sc.timer()
def test_plot():
    sc.heading('Testing what a module plots')
    S, I, R = ss.states(3)
    sim = ss.Sim(S >> I @ 0.3*I, I >> R @ 0.1, verbose=0).run()
    lines = lambda fig: [[line.get_label() for line in ax.get_lines()] for ax in fig.axes]
    assert lines(sim.sir.plot()) == [['S', 'I', 'R']] # The states, together
    assert [len(ax.get_lines()) for ax in sim.sir.plot('flows').axes] == [1, 1] # Each flow, apart
    assert len(sim.sir.plot('all').axes) == 5
    assert lines(sim.sir.plot(['S', 'I'])) == [['S', 'I']]
    assert len(sim.sir.plot('S', 'I').axes) == 2
    assert len(sim.plot().axes) == 5 # The number alive and the deaths of a closed population are left out
    plt.close('all')
    return sim


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)

    # Start timing
    T = sc.tic()

    # Run tests
    sim1 = test_export()
    sim2 = test_2d_results()
    sim4 = test_declared_scale()
    sim4b = test_flow_ratios()
    sim5 = test_time_points()
    res6 = test_destinations()
    res6b = test_summarize_by()
    res7 = test_result_exprs()
    res8 = test_standard_prevalence()
    sim8b = test_results_by_group()
    res9 = test_compare_data()
    sim10 = test_plot()

    sc.toc(T)
    plt.show()
    print('Done.')
