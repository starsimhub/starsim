"""
Switching demo for the Ghana measles seat: fit, then routine MCV1 at 9 months plus a targeted campaign, 100 runs by district.

Passes when: the fit (with a fittable `away=`) improves the loss; routine starts with newborns; 100 runs fit in memory.
Blocked today by: N21.
Scaled down to 20 synthetic districts (Ghana has 260) and two years (the original runs 2010 to 2028); results memory is projected to full size from the memory per group per step.
"""
import numpy as np
import sciris as sc
import starsim as ss

kw = dict(verbose=0)
ref_first_day  = 8_757/10_000 # Share vaccinated on day one by the spec's "routine at 9 months" (08-interventions.md:91); switching/ghana_measles.md, "The second week" §4
ref_gb_per_run = 4.9    # Results for one age-structured run with a campaign, 2010 to 2028; "The second week" §2
full_districts = 260    # Ghana's districts
full_days      = 6_940  # 2010-01-01 to 2028-12-31, daily
laptop_gb      = 16     # What "fits in memory" means here: 100 runs on a 16 GB laptop
northern = [0, 4, 8, 12, 16] # The synthetic districts standing in for the five northern regions


def make_districts(n=20, seed=1):
    """ A synthetic stand-in for Ghana's districts """
    rng = np.random.default_rng(seed)
    return sc.objdict(names=[f'd{i}' for i in range(n)], popsize=rng.lognormal(np.log(90_000), 0.6, n), lat=rng.uniform(5, 11, n), lon=rng.uniform(-3, 1, n))


def make_measles(d, beta=ss.perday(1.75), away=0.01, report=0.005):
    """ The vignette's model: seasonal SEIR by district with gravity mixing, births, deaths, importation, and reported cases """
    S, E, I, R = ss.states(4)
    infection   = S >> E @ ss.seasonal(mean=beta, amplitude=0.15, peak=59/365) * I
    importation = S >> E @ ss.peryear(0.01/1000) * (S + E + I + R).n / S.n
    onset       = E >> I @ ss.days(6)
    recovery    = I >> R @ ss.days(8)
    births      = ss.birth(S, rate=ss.peryear(0.027*0.3))
    vaccinated  = ss.birth(R, rate=ss.peryear(0.027*0.7))
    deaths      = ss.death(rate=ss.peryear(0.007))
    cases       = ss.outcome(report*onset)
    measles = ss.Disease(infection, importation, onset, recovery, births, vaccinated, deaths, cases, init=dict(R=1 - 1/14), name='measles')
    measles.stratify(district=ss.geo.geography(names=d.names, pop=d.popsize, lat=d.lat, lon=d.lon), mobility=ss.geo.Mobility(ss.geo.gravity(c=1.5), kind='visit', away=away))
    return measles


def make_sim(measles, d, **kwargs):
    return ss.Sim(measles, total_pop=d.popsize.sum(), start='2010-01-01', stop='2011-12-31', unit='day', solver='binomial', **kwargs, **kw)


def monthly(sim):
    """ Reported cases by month and district, as the surveillance data come """
    by = sim.results.measles.cases.by('district')
    return np.add.reduceat(by.values[1:], np.arange(0, len(by.values) - 1, 30), axis=0)


def test_ghana_measles_fit():
    sc.heading('Ghana measles: fit beta, the share of contacts away, and the reporting rate to cases by district and month, with a loss of their own')
    d = make_districts()
    data = monthly(make_sim(make_measles(d, away=0.03, report=0.008), d).run()) # The "surveillance data", from known values
    beta = ss.par(ss.perday(1.75), bounds=[1.0, 2.5])
    away = ss.par(0.01, bounds=[0.001, 0.1])
    report = ss.par(0.005, bounds=[0.001, 0.02])
    sim = make_sim(make_measles(d, beta=beta, away=away, report=report), d)
    gof = lambda sim: np.mean((monthly(sim) - data)**2)
    start = gof(sim.new().run())
    fit = sim.fit(gof=gof, method='optuna', n_trials=10)
    assert gof(fit.sim.new().run()) < start, 'The fit should improve the loss'
    return fit


def test_ghana_measles_routine():
    sc.heading('Ghana measles: routine MCV1 at 9 months starts with newborns')
    ppl = ss.People(10_000, age_data=np.array([np.arange(80), np.exp(-0.035*np.arange(80))]).T)
    S, I, R = ss.states(3)
    sir = ss.Disease(S >> I @ ss.perday(0.1)*I, I >> R @ ss.days(8), ss.birth(S, rate=ss.peryear(0.03)), init=dict(I=0.001), name='sir')
    ri = ss.vaccinate(ss.Vaccine(efficacy=0.85), coverage=0.833, at_age=ss.months(9), name='routine') # laser-measles: MCV1 274 days after birth
    sim = ss.Sim(sir, people=ppl, interventions=ri, start='2021-01-01', stop='2022-01-01', unit='day', **kw).run()
    delivered = sim.results.routine.delivered.values
    assert delivered[:7].sum() < 0.01*len(ppl), f'Routine at 9 months gave {delivered[:7].sum():.0f} doses in the first week (the spec\'s spelling gives {ref_first_day:.0%} of everyone)'
    assert delivered.sum() > 0, 'Children born during the year should reach 9 months and be vaccinated'
    return sim


def test_ghana_measles_memory(monkeypatch):
    sc.heading('Ghana measles: a campaign in the north, with results by age and district; would 100 runs fit in memory?')
    d = make_districts()
    measles = make_measles(d)
    ppl = ss.People(1)
    eligible = (ppl.age >= 0.75) & (ppl.age < 5) & measles.S & measles.district.isin([d.names[i] for i in northern])
    mcv = ss.Vaccine(efficacy=0.85, leaky=False) # All-or-nothing, as laser-measles models it; a move from S to R needs no extra dimension
    sia = ss.vaccinate(mcv, at=['2011-06-01'], coverage=0.8, where=eligible)
    sim = make_sim(measles, d, interventions=sia, stratify=dict(age=[0, 0.75, 5, 15]))
    scale = len(d.names)/full_districts*730/full_days # The memory for results by group is scaled down with the model (two years), so this run keeps them as the full one would (N24)
    monkeypatch.setattr(ss.solvers.Engine, 'group_bytes', ss.solvers.Engine.group_bytes*scale)
    sim.run()
    total = sc.checkmem(sim.results, descend=0).bytesize.iloc[0]
    by_group = sum(getattr(res, key).nbytes for mod in sim.modules for res in mod.results.values() if isinstance(res, ss.Result) for key in ['strata', 'denom'] if getattr(res, key, None) is not None)
    full_gb = 100*(by_group/scale + (total - by_group)*full_days/730)/1e9 # The results by group grow as the budget for them did, and the rest with the days
    assert full_gb < laptop_gb, f'100 full-size runs would keep {full_gb:,.0f} GB of results (the report measured {ref_gb_per_run} GB for one), more than {laptop_gb} GB'
    return sim


if __name__ == '__main__':
    fit = test_ghana_measles_fit()
    ri  = test_ghana_measles_routine()
    sim = test_ghana_measles_memory()
