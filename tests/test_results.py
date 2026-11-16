"""
Test results and analyzers: export, resampling, kinds and summaries, results by group, lining up with data, plotting, and the infection log
"""
import warnings
import numpy as np
import pandas as pd
import pytest
import sciris as sc
import starsim as ss
import starsim.library as ssl
import matplotlib.pyplot as plt
import corpus

sc.options.interactive = False # Assume not running interactively

small = 100
medium = 1000


def test_export():
    """ Results exported and resampled: a flow summed over each period, a stock at its start; across modules with steps of their own """
    d1 = ssl.AgentSIS(dt=ss.years(1/12), name='sis1') # Written as code, since a declared disease cannot yet have a timestep of its own
    d2 = ssl.AgentSIS(dt=ss.years(0.5), name='sis2')
    sim = ss.Sim(n_agents=medium, diseases=[d1, d2], networks='random', verbose=0).run()
    rs1, rs2 = sim.results.sis1, sim.results.sis2
    res = rs1.infection
    assert res[-1] == res.to_df().iloc[-1].value == res.to_series().iloc[-1]
    resampled = res.resample(new_unit='year')
    assert isinstance(resampled, ss.Result) and resampled[0] == res[:12].sum() # The first year's period is the steps that start in it
    assert resampled.values.sum() == res.values[:-1].sum() # Whole years only, so not the step from 2050
    assert rs1.infected.resample('year')[0] == rs1.infected[0] # The start of the first year
    assert np.allclose(rs1.resample('year').infection.values, resampled.values, rtol=1e-10) # All the results of a group
    assert res.to_df().value.sum() == res.cum.values[-1] == sim.summarize().value.sis1.infection
    dfy2 = rs2.to_df(resample=ss.years(5))
    assert rs2.susceptible[0] == dfy2.susceptible.iloc[0] and rs2.infection[:10].sum() == dfy2.infection.iloc[0] # Periods from the start, each including its start
    sim_df = sim.to_df(resample='year', use_years=True)
    assert sim_df.sis1_infected.values[0] == rs1.infected[0] and sim_df.sis2_infected.values[0] == rs2.infected[0]

    weekly = ss.Sim(ss.SIR(), solver='ode', start='2000-01-01', stop='2004-06-01', dt=ss.weeks(1), verbose=0).run().results.sir.infection # Calendar periods are labeled by their start
    assert [str(d) for d in weekly.resample('year').timevec] == ['2000-01-01', '2001-01-01', '2002-01-01', '2003-01-01'] # Not 2004, of which the run has five months
    assert str(weekly.resample('month').timevec[1]) == '2000-02-01'
    return sim


def test_2d_results():
    """ 2D results by column, and nested results, scaled as 1D results are """
    sim = ss.Sim(n_agents=small, total_pop=10*small, diseases='sis', networks='random', analyzers=corpus.BySex(), start=2000, verbose=0).run()
    res = sim.results.bysex.infected
    assert res.shape == (sim.t.npts, 2) and np.array_equal(res.sum(axis=1), sim.results.sis.I) # Scaled the same as 1D results
    assert np.array_equal(res.male, res[:,1]) # Each column is a 1D result
    assert res.resample('year').shape == (len(np.unique(res.timevec.years.astype(int))), 2) # Resampled along time only
    assert res.to_df().columns.tolist() == ['time', 'female', 'male']
    assert 'bysex_infected_male' in sim.to_df().columns and sim.summarize().value.bysex.infected.male == res['male'][-1]
    with pytest.raises(ValueError):
        ss.Result('infected', kind='stock', columns=['low', 'high']) # Column names can't clash with attributes
    return sim


def test_declared_results():
    """ Results declared as sums of states, ratios, and flows over states, in the class body or the instance, scaled as people under every solver """
    class SIR(ss.Disease):
        S = ss.State()
        I = ss.State()
        R = ss.State()
        infection = S >> I @ 0.3*I
        recovery = I >> R @ ss.days(10)
        prev = I / (S + I + R)
        n_ever = I + R
        incidence = infection / S
    for solver in ['ode', 'binomial', 'abm']:
        for strat in [False, True]:
            sim = ss.Sim(SIR(n_kw=SIR.I + SIR.R, init=dict(I=0.01), name='sir'), unit='day', stop=60, total_pop=10_000, n_agents=medium, solver=solver, verbose=0) # A result passed to an instance can use the class's states
            if strat:
                sim.stratify(age=[0, 50])
            res = sim.run().results.sir
            assert np.allclose(res.prev, res.I/(res.S + res.I + res.R)) and np.allclose(res.n_ever, res.I + res.R) and np.allclose(res.n_kw, res.n_ever) # Numbers of people, not of agents
            assert np.allclose(res.incidence[1:], res.infection[1:]/res.S[1:])
            if strat: # By group: the flow and the stock of the group
                young = res.incidence.select(age='0-49')[1:]
                assert np.allclose(young, res.infection.select(age='0-49')[1:]/res.S.select(age='0-49')[1:])
                weekly = res.infection.resample('week') # Resampled from the start, by group too
                assert np.allclose(weekly.select(age='0-49') + weekly.select(age='50+'), weekly) and np.isclose(weekly[1], res.infection[7:14].sum())
                assert res.S.resample('week')[1] == res.S[7] # A stock at the start of each week
                assert np.isclose(res.incidence.resample('week')[1], res.incidence[7:14].mean()) # A ratio: its mean over the period, as summarize() takes it

    for solver in ['ode', 'abm']: # A ratio with nobody in its denominator is 0, not an error
        S, I, R = ss.states(3)
        sir = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), frac=R/R, init=dict(I=0.01), name='sir')
        res = ss.Sim(sir, unit='day', stop=10, total_pop=10_000, n_agents=medium, solver=solver, verbose=0).run().results.sir
        assert res.frac[0] == 0 and res.frac[-1] == 1

    for solver in ['ode', 'binomial', 'abm']: # The standard prevalence by group, as a ratio
        sis = ss.SIS(beta=ss.peryear(0.3), dur_inf=ss.years(10), init_prev=0.05)
        sis.stratify(risk={'low': 0.8, 'high': 0.2})
        res = ss.Sim(sis, connectors=ss.modify(sis.infection, multiply=ss.by(risk=[0.5, 3])), solver=solver, total_pop=10_000, n_agents=10_000, start=2000, stop=2010, verbose=0).run().results.sis
        high, low = res.prevalence.select(risk='high'), res.prevalence.select(risk='low')
        assert high[-1] > 2*low[-1] > 0, f'{solver}: {high[-1]:.3f} against {low[-1]:.3f}'
        assert np.allclose(res.I.select(risk='high')/(res.S.select(risk='high') + res.I.select(risk='high')), high)
        assert np.allclose(0.8*low + 0.2*high, res.prevalence, rtol=0.05) # The groups keep their shares of people, about
    return res


def test_summarize_by():
    """ A result's kind and summarize_by: from the name, given, guessed with a warning otherwise, None, and added; what has no one number is left out """
    res = ss.Result('new_infections', values=np.arange(5.0))
    assert res.kind == 'flow' and res.summarize_by == 'sum' and res.summarize() == 10 and ss.Result('n_infected').kind == 'stock' and ss.Result('prevalence').summarize_by == 'mean'
    assert ss.Result('x', values=np.arange(5.0), kind='stock').summarize() == 4
    for kw,kind in [(dict(), 'ratio'), (dict(dtype=int), 'flow'), (dict(scale=True), 'ratio')]: # With no name to go on, guessed from the dtype, with a warning
        with pytest.warns(RuntimeWarning, match='kind='):
            assert ss.Result('doses', **kw).kind == kind
    assert ss.Result('doses', summarize_by='last').kind == 'stock' # From how it is summarized, with no warning
    assert ss.Result('new_x').scale and ss.Result('n_x').scale and not ss.Result('prevalence').scale and ss.Result('prevalence', scale=True).scale # Scaled by kind, unless told
    mean = ss.Result('x', values=np.arange(5.0), kind='flow', summarize_by='mean')
    assert mean.summarize() == 2 and mean.cum[-1] == 10 # A flow summarized another way is still a flow

    class Odd(ss.Analyzer):
        def init_results(self):
            super().init_results()
            self.define_results(ss.Result('timer', kind=None), ss.Result('by_age', dtype=int, kind='flow', shape=(self.t.npts, 3)), ss.Result('times', dtype=list, scale=False))
            self.results['log'] = dict(net=[[0, None]])
        def step(self):
            self.results.timer[self.ti] = self.ti
    sim = ss.Sim(ss.SIS(), analyzers=Odd(name='odd'), n_agents=small, start=2000, stop=2002, dt=ss.months(1), verbose=0).run()
    annual = sim.results.resample('year') # Not summarized over time: left out when resampled, as in summarize()
    assert 'timer' not in annual.odd and len(annual.sis.I) == 2 # 2000 and 2001, not the step from 2002-01-01
    assert 'sis.I' in sim.summarize().index and not any(k.startswith('odd') for k in sim.summarize().index) # A 2D result with no columns=, a list, and other data
    assert len(sim.summarize('odd.by_age')) == 3 and ss.Result('times', dtype=list).kind is None # Asked for by key, one number per column
    return res


def test_results_by_group():
    """ Results by group in a module written as code: counts by sex and age among the living, weighted; and the stratification used only by them """
    class ByGroup(ss.Analyzer):
        def init_results(self):
            super().init_results()
            self.define_results(ss.Result('n_infected', groups=True), ss.Result('prevalence', scale=False, groups=True))
        def record_states(self):
            groups, dis = self.sim.groups, self.sim.diseases[0]
            infected = groups.count(dis.I if dis.states else dis.infected) # Declared, or written as code
            self.results.n_infected.set_groups(self.ti, infected)
            self.results.prevalence.set_groups(self.ti, infected, groups.count())

    kw = dict(networks='random', demographics=[ss.Births(birth_rate=0.02), ss.Deaths(death_rate=0.015)], analyzers=ByGroup(name='bygroup'), n_agents=medium, start=2000, stop=2010, verbose=0)
    sim = ss.Sim(ss.SIS(beta=ss.peryear(0.3), init_prev=0.1), stratify=dict(sex=['female', 'male'], age=[0, 15, 25, 50]), **kw).init()
    sim.people.scale[sim.people.auids[::3]] = 3 # Agents of different weights
    sim.run()
    sis, res = sim.results.sis, sim.results.bygroup
    assert np.array_equal(res.n_infected.strata[1:], sis.I.strata[1:]) # Each group, at each step after the start (whose weights were set after it)
    assert np.allclose(res.prevalence.select(sex='female', age='15-24')[1:], sis.prevalence.select(sex='female', age='15-24')[1:]) # Summed over the groups, then divided
    assert np.allclose(res.prevalence.by('age')[1:], sis.prevalence.by('age')[1:])

    sim = ss.Sim(ssl.AgentSIS(beta=ss.peryear(0.3), init_prev=0.1), stratify=dict(sex=['female', 'male']), **kw).run() # With only modules written as code
    res = sim.results.bygroup
    assert np.array_equal(res.n_infected, sim.diseases.agentsis.results.infected) and res.n_infected.by('sex').shape == (sim.t.npts, 2)
    with pytest.raises(ValueError, match='groups=True'):
        ss.Sim(ssl.AgentSIS(), networks='random', stratify=dict(sex=['female', 'male']), n_agents=small, verbose=0).init() # Nothing uses it
    return sim


def cohort(solver, p=0.3, data=None):
    """ 10,000 people who leave P after 5 days on average: a share p to C, and the rest to R """
    P, C, R = ss.states('P C R')
    course = ss.Transition(P, {C: p, R: None}, dur=ss.days(5))
    dis = ss.Disease(P, C, R, course=course, cases=ss.outcome(0.5*course.C), default_state='P', name='dis')
    dis.stratify(age=ss.years([0, 40]))
    return ss.Sim(dis, n_agents=10_000, total_pop=10_000, solver=solver, unit='day', stop=60, data=data, verbose=0)


def test_destinations():
    """ The flow to each destination of a transition with several, and an outcome of one, by group; a data column named for a destination is fitted """
    for solver in ['ode', 'binomial', 'gillespie', 'abm']:
        res = cohort(solver).run().results.dis
        C, R, total = res.course.C, res.course.R, res.course
        assert np.allclose(C + R, total) and np.allclose((C + R).select(age=0), total.select(age=0)) # The total is the sum of its destinations, by group too
        assert abs(C.sum() - 3000) < (30 if solver == 'ode' else 3*np.sqrt(10_000*0.3*0.7)) # Within 3 standard errors
        assert C is res['course.C'] and np.array_equal(total.select(to='C'), C) and np.allclose(total.select(to=['C', 'R']), total) # select(to=) is a second spelling
        assert abs(res.cases.sum() - 0.5*C.sum()) <= 3*np.sqrt(0.25*C.sum()) # A destination is a source of an outcome
    with pytest.raises(KeyError):
        res.P.select(to='C') # Not a transition with several destinations

    truth = cohort('ode').run().results.dis.course.C
    data = pd.DataFrame({'course.C': truth.values[1:]}, index=truth.timevec[1:])
    fit = cohort('ode', p=ss.par(0.5, bounds=[0.05, 0.95], name='p'), data=data).fit(loss='rmse')
    assert np.isclose(fit.table.best.iloc[0], 0.3, rtol=1e-2)
    return res


def test_compare_data():
    """ Results lined up with data: a stock at the data's times, a flow summed over each period, whatever the rounding of the sim's times; a name close to a result; weeks labeled by their end """
    sis = lambda: ss.SIS(beta=ss.peryear(0.05), dur_inf=ss.years(10)) # In years, for yearly steps
    sim = ss.Sim(diseases=sis(), networks='random', n_agents=small, start=2000, stop=2010, verbose=0).run()
    res = sim.results.sis
    dates = [ss.date(f'{y}-07-01') for y in range(2001, 2010)] # Between steps, so a stock is interpolated
    df = pd.DataFrame({'sis.prevalence': 0.1, 'sis.infection': 5.0}, index=pd.Index(dates, dtype=object))
    compared = ss.results.compare(sim, df)
    years = ss.results.data_times(sim, df)[1]
    assert np.allclose(compared['sis.prevalence'][1], np.interp(years, sim.t.yearvec, res.prevalence.values))
    assert np.allclose(compared['sis.infection'][1][1:], [res.infection.values[(sim.t.yearvec > y-1) & (sim.t.yearvec <= y)].sum() for y in years[1:]])
    for col,(W,expected,rows) in ss.results.lineup(sim, df).items(): # The same matrix as calibration uses
        assert np.allclose(W @ ss.results.match_column(sim, col).values, compared[col][1])
    typo = ss.Sim(diseases=sis(), networks='random', n_agents=small, start=2000, stop=2010, data=df.rename(columns={'sis.infection': 'infectoin'}), verbose=0).init()
    assert any('aliases=' in note for note in typo.notes) # The code to rename it, with the closest result

    def kill(sim): # Deaths requested during a step of a declared model are counted in it
        sim.people.request_death(sim.people.auids[:2])
    S, I = ss.states('S I')
    dis = ss.Disease(infection=ss.Transition(S, I, rate=0.0 * I), init_n=dict(I=1), kill=kill, name='dis')
    res = ss.Sim(dis, people=ss.People(1000), unit='day', stop=4, verbose=0).run().results
    assert np.array_equal(res.alive, [1000, 998, 996, 994, 992]) and np.array_equal(res.death, [2, 2, 2, 2, 2]) # A step from each time point, the last included

    S, I, R = ss.states(3)
    infection = S >> I @ ss.perday(0.4)*I
    sim = ss.Sim(ss.Disease(infection, I >> R @ ss.perday(0.15), ss.outcome(0.6*infection, name='cases'), init_n=dict(I=10), name='sir'), total_pop=10_000, unit='day', dur=ss.weeks(12), verbose=0).run()
    res = sim.results.sir
    weekly = lambda flow: np.add.reduceat(flow.values[:84], np.arange(0, 84, 7)) # Days 0 to 6, 7 to 13, ...
    out = ss.results.compare(sim, pd.DataFrame(dict(cases=weekly(res.cases), infection=weekly(res.infection)), index=np.arange(0, 84, 7)))
    assert len(out.cases[1]) == 12 and np.allclose(*out.cases) and np.allclose(*out.infection)
    assert np.allclose(res.cases.resample('week')[:12], weekly(res.cases)) # The same weekly totals
    ss.results.warned_data.clear()
    with pytest.warns(RuntimeWarning, match='subtract one period'): # Weekly data labeled by the end of each week: the last week runs past the run
        ss.results.check_data(sim, pd.DataFrame({'cases': 1.0}, index=pd.Index(np.arange(7, 85, 7), name='time')))
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        ss.results.check_data(sim, pd.DataFrame({'cases': 1.0}, index=pd.Index(np.arange(0, 78, 7), name='time')))
    return compared


def test_plot():
    """ What a module plots; plots in Jupyter, and overlaid; synthetic data with noise, and its plot """
    S, I, R = ss.states(3)
    sim = ss.Sim(S >> I @ 0.3*I, I >> R @ 0.1, verbose=0).run()
    lines = lambda fig: [[line.get_label() for line in ax.get_lines()] for ax in fig.axes]
    assert lines(sim.sir.plot()) == [['S', 'I', 'R']] # The states, together
    assert [len(ax.get_lines()) for ax in sim.sir.plot('flows').axes] == [1, 1] # Each flow, apart
    assert len(sim.plot().axes) == 7 # The number alive and the deaths of a closed population are left out
    res = sim.results.sir
    assert res.I.plot(is_jupyter=True, do_show=True) is None # In Jupyter, the figure is returned when not shown, and a figure given is not shown
    fig = res.I.plot(is_jupyter=True, do_show=False, label='I')
    assert res.S.plot(fig=fig, label='S', log=True, is_jupyter=True) is fig and len(fig.axes) == 1 # Overlaid

    make = lambda data=None: ss.Sim(ss.Disease(infection=S >> I @ 0.3*I, recovery=I >> R @ ss.days(10), init_n=dict(I=5), name='sir'), data=data, total_pop=10_000, start='2025-01-01', stop='2025-04-01', verbose=0)
    truth = make().run()
    exact = truth.to_synthetic_data('sir.infection', resample='week', noise=None)['sir.infection'].to_numpy()
    over = np.array([truth.to_synthetic_data('sir.infection', resample='week', noise=ss.nbinom(n=2), rand_seed=s)['sir.infection'].to_numpy() for s in range(200)]) # Negative-binomial noise, as the likelihood of a fit takes it
    assert np.allclose(over.mean(axis=0), exact, rtol=0.25) and np.allclose(over.var(axis=0)[3:9], (exact + exact**2/2)[3:9], rtol=0.4)
    with pytest.raises(ValueError, match='as a probability'): # A count is not a probability (it drew n every time)
        truth.to_synthetic_data('sir.infection', noise=ss.binom(n=100))
    data = truth.to_synthetic_data('sir.infection', resample='week').iloc[:6] # Weekly data over part of a daily run: its weekly totals over the whole run
    ax = make(data).run().plot('sir.infection').axes[0]
    assert len(ax.lines[0].get_xdata()) == 13 # The 13 whole weeks of the run's 91 steps (6 with data)
    plt.close('all')
    return sim


def test_find():
    """ A result is found however it is spelled, and only if there is one, also its total by name """
    sim = ss.Sim(diseases='seir', n_agents=100, networks='random', stop=2021, start=2020, verbose=0).run()
    res = sim.results
    for name in ['I', 'seir.I', 'seir_I', 'SEIR.I']:
        assert res[name] is res.seir.I and res.find(name) is res.seir.I and sim.summarize(name) == sim.summarize('seir_I')
    assert res.find('prev', die=False) is None # Part of a name is only a suggestion
    total = res.seir.infection.cum
    assert all(np.array_equal(res[key], total) for key in ['seir.infection.cum', 'seir_infection_cum', 'infection.cum'])
    assert sim.summarize('seir.infection.cum') == total[-1] and ss.results.match_column(sim, 'seir.infection.cum') is not None
    assert res.find('seir.I.cum', die=False) is None # Not a flow
    both = ss.Sim(diseases=['sis', 'sir'], n_agents=100, networks='random', stop=2021, start=2020, verbose=0).run().results
    with pytest.raises(KeyError):
        both['I']
    assert both['sir.I'] is both.sir.I
    return


def test_axes():
    """ select, by, resample, take, quantiles, and to_df work the same way across runs, columns, destinations, and values swept """
    S, I = ss.states(2)
    sis = ss.Disease(S, I, infection=S >> I @ ss.perday(0.1)*I, recovery=I >> S @ ss.perday(0.05), init_prev=0.1, name='sis')
    sis.stratify(sex=['female', 'male'])
    msim = ss.parallel(ss.Sim(sis, unit='day', stop=40, total_pop=1000, solver='binomial', verbose=0), n_runs=3)
    res = msim.results.sis.I
    by = res.by('sex')
    assert by.shape == (3, 41, 2) and np.array_equal(by.values.sum(axis=-1), res.values) and np.array_equal(by.male.values, res.select(sex='male').values) # Across runs, a column for each level
    assert by.resample('week').shape[0] == 3 and np.array_equal(by.resample('week').values[..., 0], res.select(sex='female').resample('week').values)
    assert set(by.to_df().column) == {'female', 'male'} and by.select(run=0).shape == (41, 2)
    assert 'sir' not in res.axes() and list(res.axes()['sex'][0]) == ['female', 'male'] and 'scenario' in res.axes()
    with pytest.raises(ValueError):
        res.by('scenario') # Runs are selected, not split
    assert msim.to_df().columns.tolist() == msim.results.to_df().columns.tolist() and 'quantile' in msim.to_df(quantiles=[0.5])

    S, I, X, R = ss.states('S I X R') # The destinations of a result across runs
    dis = ss.Disease(infection=S >> I @ ss.perday(0.1)*I, progress=ss.Transition(I, {X: 0.3, R: 0.7}, dur=ss.days(5)), init_prev=0.1, name='dis')
    base = ss.Sim(dis, unit='day', stop=20, total_pop=1000, solver='binomial', verbose=0, stratify=dict(age=[0, 15]))
    msim = ss.parallel({'a': base, 'b': base.copy()}, n_runs=2, workers=1, verbose=0)
    res = msim.results.dis.progress
    assert res.median().select(to='X').shape == (2, 21) and res.select(scenario='a').select(to='X').shape == (2, 21)
    assert np.array_equal(res.take(0).select(to='X').values, res.X.values[0]) and res.quantiles().select(to='X').shape == (6, 21)
    assert res.take(0).resample('week').select(to='X').shape == res.take(0).resample('week').shape
    assert np.array_equal(res.cum.select(to='X').values, res.X.cum.values)
    sis = ss.Sim(diseases=ss.SIS(beta=ss.peryear(2), dur_inf=ss.years(1), init_prev=0.1), total_pop=500, start=2000, stop=2004, verbose=0, solver='binomial', stratify=dict(age=[0, 15]))
    sweep = ss.parallel(sis, pars={'sis.beta': [ss.peryear(1), ss.peryear(2)]}, workers=1, verbose=0).results.sis.I # A rate swept can be selected
    assert sweep.select(age='0-14').shape == (2, 5) and sweep.by('age').shape == (2, 5, 2) and sweep.select(beta=ss.peryear(1)).shape == (1, 5)
    return


def test_resample_ratios():
    """ resample() takes a ratio as summarize() does: its mean over each period, with or without groups; a ratio of two flows, the total of one over the total of the other """
    S, I = ss.states(2)
    sis = ss.Disease(S >> I @ ss.perday(0.2)*I, I >> S @ ss.days(10), init_prev=0.05, name='sis')
    vx = ss.vaccinate(ss.Vaccine(efficacy=0.5), when=ss.schedule(every=ss.days(10)), coverage=0.3, name='vx')
    for stratify in [None, dict(age=[0, 15])]:
        sim = ss.Sim(sis, interventions=vx, unit='day', stop=28, total_pop=2000, solver='binomial', stratify=stratify, verbose=0).run()
        prev = sim.results.sis.prevalence
        weekly = prev.resample('week')
        assert np.allclose(weekly, [prev[7*w:7*w+7].mean() for w in range(4)]) and np.isclose(prev[:28].mean(), weekly.mean())
        cov = sim.results.vx.coverage
        num, den = [part.resample('week') for part in cov.parts]
        assert np.allclose(cov.resample('week'), np.divide(num, den, out=np.zeros(len(num)), where=den.values!=0)) # A ratio of flows: total over total
        if stratify:
            young = prev.select(age='0-14') # Groups are added up, then divided, at each time point; then averaged over each period
            assert np.allclose(young.resample('week'), [young[7*w:7*w+7].mean() for w in range(4)])
            with pytest.raises(ValueError, match='select the groups first'):
                weekly.select(age='0-14')
    return sim


def test_infection_log():
    """ The infection log: who infected whom, by which network; several logs share the disease's; each step's entries as the sim runs; the strain """
    sim = ss.Sim(n_agents=1000, dt=0.2, dur=ss.years(15), diseases='sir', networks='random', analyzers='infection_log', verbose=0).run()
    df = sim.analyzers[0].to_df()
    assert len(df) > 900, 'Expect almost everyone to be infected'
    assert set(df.network.dropna()) == {'randomnet'} and df.network.isna().sum() == df.source.lt(0).sum() # Seed infections have no network
    assert df.t.dtype == float # Times as numbers, which can be subtracted (they were durations)
    mixed = ss.Sim(ss.SIR(beta=0.3), n_agents=300, unit='day', stop=10, analyzers=ss.infection_log(), verbose=0).run().analyzers[0].to_df()
    assert 'network' not in mixed.columns # With no networks no route is known, so no column of blanks

    class steps(ss.Analyzer):
        def init_pre(self, sim):
            super().init_pre(sim)
            self.n = []
        def step(self):
            self.n.append(len(self.sim.diseases.d.infection_log.to_df(ti=self.ti)))
    S, I, R = ss.states('S I R')
    dis = ss.Disease(infection=S >> I @ ss.peryear(1)*I, recovery=I >> R @ ss.years(2), init=dict(I=0.05), strains=ss.Strains(['A', 'B']), name='d')
    kw = dict(n_agents=500, dur=ss.years(5), networks='random', verbose=0)
    sim = ss.Sim(dis, analyzers=[ss.infection_log(name='a'), ss.infection_log(name='b'), steps()], **kw).run()
    a, b = sim.analyzers.a, sim.analyzers.b
    assert a.log is b.log is sim.diseases.d.infection_log # Shared, not taken by the first
    df = a.to_df()
    assert sim.analyzers.steps.n == np.bincount(np.searchsorted(sim.t.yearvec, df.t), minlength=sim.t.npts).tolist() # Each step's entries, as they were logged
    assert sim.analyzers.steps.n[1:] == sim.results.d.infection.values[1:].tolist() and set(df.strain) == {'A', 'B'}
    class sources(steps): # An analyzer of its own can ask for the log, with no ss.infection_log in the sim
        needs_infection_log = True
    assert ss.Sim(dis, analyzers=sources(), **kw).run().analyzers.sources.n == sim.analyzers.steps.n
    assert ss.Sim(dis, **kw).run().diseases.d.infection_log is None # Not kept unless asked for
    return sim


if __name__ == '__main__':
    T = sc.timer()
    test_export()
    test_2d_results()
    test_declared_results()
    test_summarize_by()
    test_results_by_group()
    test_destinations()
    test_compare_data()
    test_plot()
    test_infection_log()
    T.toc()
