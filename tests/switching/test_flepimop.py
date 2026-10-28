"""
Switching demo for the flepiMoP seat: the 3,000-place config, fitted with a value per place, output for the hub.

Passes when: the per-place fit runs; Poisson on weekly averages is finite.
Blocked today by: N20, N6, `priors=True`, `to_hub()`.
Scaled down to a config of three counties written by the test (the report's 3,000-place project is generated, not shipped); the speed at 3,000 places (5 s against flepiMoP's 34 to 58 s) is checked by the north-star script (phase 6), not here.
"""
import tempfile
import numpy as np
import pandas as pd
import pytest
import sciris as sc
import starsim as ss
import starsim.fit

ref_n_quantiles = 23 # switching/flepimop.md, "Who this user is": a hub submission is 23 quantiles by scenario, location, target, and week
places = ['10001', '10003', '10005'] # The three counties below
true_Ro = [2.0, 2.3, 2.6] # The value of Ro in each county that the fit should recover
lik = dict(incidHosp=ss.poisson())

config = """
name: three_counties
start_date: 2020-02-01
end_date: 2020-05-31
subpop_setup: {geodata: geodata.csv, mobility: mobility.csv}
initial_conditions: {method: SetInitialConditions, initial_conditions_file: ic.csv, allow_missing_subpops: TRUE, allow_missing_compartments: TRUE}
compartments: {infection_stage: [S, E, I, R]}
seir:
  integration: {method: rk4, dt: 1}
  parameters:
    sigma: {value: 1 / 4}
    gamma: {value: 1 / 5}
    Ro: {value: {distribution: truncnorm, mean: 2.2, sd: 0.1, a: 1.8, b: 2.6}}
  transitions:
    - {source: [S], destination: [E], rate: [Ro * gamma], proportional_to: [[S], [I]], proportion_exponent: ["1", "1"]}
    - {source: [E], destination: [I], rate: [sigma], proportional_to: [E], proportion_exponent: ["1"]}
    - {source: [I], destination: [R], rate: [gamma], proportional_to: [I], proportion_exponent: ["1"]}
outcomes:
  method: delayframe
  outcomes:
    incidHosp: {source: {incidence: {infection_stage: I}}, probability: {value: 0.03}, delay: {value: 7}}
"""
files = dict(
    geodata  = 'subpop,population\n10001,500000\n10003,200000\n10005,100000\n',
    mobility = 'ori,dest,amount\n10001,10003,5000\n10003,10001,5000\n10003,10005,2000\n10005,10003,2000\n',
    ic       = 'subpop,mc_name,amount\n10001,S,499900\n10001,E,100\n10003,S,200000\n10005,S,100000\n',
)


def load_config(folder):
    """ Write the flepiMoP project and import it, as the user would their own """
    sc.savetext(folder/'config.yml', config)
    for key,text in files.items():
        sc.savetext(folder/f'{key}.csv', text)
    return ss.import_from(folder/'config.yml', format='flepimop', verbose=0) # The drawn Ro is imported with its distribution as its prior


def weekly_totals(sim):
    """ Hospitalizations by county as weekly totals, which from an ODE are not whole numbers (data a flow is compared with are its totals over each period, so averages would be a seventh of the size) """
    res = sim.diseases[0].results.incidHosp
    dates = pd.date_range('2020-02-01', periods=len(res.values)) # Through flepiMoP's end_date
    rows = []
    for place in places:
        weekly = pd.Series(np.asarray(res.select(subpop=place).values), index=dates).resample('W-SAT').sum()
        rows += [dict(date=date, subpop=place, incidHosp=val) for date,val in weekly.items()]
    return ss.simdata(pd.DataFrame(rows))


def test_flepimop_import(tmp_path):
    sc.heading('flepiMoP: the config comes in and runs as an ODE by county')
    sim = load_config(tmp_path)
    assert not sim.imported.incomplete and sim.new().init().diseases[0].pars.Ro.prior is not None, 'The drawn Ro should be imported with its distribution as its prior'
    sim.run()
    assert sim.pars.solver == 'ode'
    hosp = sim.diseases[0].results.incidHosp
    for place in places:
        assert hosp.select(subpop=place).values.max() > 0, f'Expected hospitalizations in {place}'
    return sim


def test_flepimop_fit_per_place(tmp_path):
    sc.heading('flepiMoP: one Ro per county, fitted to weekly hospitalizations')
    sim = load_config(tmp_path)
    truth = sim.new(pars=dict(Ro=ss.by(subpop=true_Ro))).run()
    fitsim = sim.new(pars=dict(Ro=ss.par(ss.by(subpop=[2.2]*len(places)), bounds=[1, 4]))) # One fitted value per county
    fitsim.data = weekly_totals(truth)
    assert np.isfinite(ss.fit.loglik(fitsim.new().run(), likelihood=lik)), 'A Poisson likelihood on data that are not whole numbers should be finite'
    fit = fitsim.fit(likelihood=lik, method='l-bfgs-b', verbose=0)
    assert np.allclose(fit.pars.Ro.values, true_Ro, rtol=0.05), f'Expected Ro by county near {true_Ro}, not {fit.pars.Ro}'
    return fit


def test_flepimop_hub(tmp_path):
    sc.heading('flepiMoP: runs with Ro drawn from its prior, written as a hub submission')
    sim = load_config(tmp_path)
    msim = ss.parallel(sim, n_runs=10, priors=True) # Each run draws Ro from the config's truncated normal
    peaks = list(msim.results.values())[0].incidHosp.values.max(axis=1) # Each run's peak
    assert len(set(np.round(peaks, 6))) > 1, 'Expected runs with different draws of Ro to differ'
    hub = ss.io.to_hub(msim, targets=dict(incidHosp='inc hosp'), by='subpop', every='week')
    assert {'origin_date', 'scenario_id', 'location', 'target', 'horizon', 'output_type', 'output_type_id', 'value'} == set(hub.columns)
    assert hub['output_type_id'].nunique() == ref_n_quantiles
    assert hub['location'].nunique() == len(places)
    return hub


if __name__ == '__main__':
    folder = sc.path(tempfile.mkdtemp())
    sim = test_flepimop_import(folder)
    fit = test_flepimop_fit_per_place(folder)
    hub = test_flepimop_hub(folder)
