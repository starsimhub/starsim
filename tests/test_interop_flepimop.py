"""
Test importing flepiMoP configurations
"""

# %% Imports and settings
import pytest
import numpy as np
import pandas as pd
import tempfile
import sciris as sc
import scipy.integrate as spi
import starsim as ss
import paths
from starsim.interop.flepimop import load_flepimop

kw = dict(verbose=0)
examples = paths.lfd/'examples/flepimop' # The originals and their reference results
needs_examples = pytest.mark.skipif(not examples.exists(), reason=paths.no_lfd)


def reference(folder, series):
    """ A series of the original's reference results, in order of time """
    ref = pd.read_csv(folder / 'reference' / 'results.csv')
    return ref[ref.series == series].sort_values('time').value.values


@needs_examples
def test_seir_2pop():
    sc.heading('Testing the flepiMoP two-province SEIR')
    folder = examples / 'seir_2pop'
    sim = load_flepimop(folder / 'original' / 'config_sample_2pop.yml', **kw)
    assert not sim.imported.incomplete
    assert sim.pars.solver == 'ode'
    res = sim.run().results.sample_2pop
    tr = sim.diseases[0].graph.transitions.S_to_E
    assert tr.contact, 'Expected the force of infection to be read as exposure to I'

    # The same numbers as flepiMoP's, in every compartment of every province (both are RK4 with a step of a day)
    for place in ['large_province', 'small_province']:
        for state in 'SEIR':
            ours, theirs = res[f'n_{state}'].select(subpop=place).values, reference(folder, f'{place}_{state}')
            assert len(ours) == len(theirs) + 1 # The sim runs through the end date, as flepiMoP does, so its last number is the day after
            assert np.abs(ours[:-1] - theirs).max() < 1e-6, f'{place} {state} differs from the original'

    # Only the places in subpop_setup.selected
    config = sc.loadyaml(folder / 'original' / 'config_sample_2pop.yml')
    for key in ['geodata', 'mobility']:
        config['subpop_setup'][key] = str(folder / 'original' / config['subpop_setup'][key])
    config['initial_conditions']['initial_conditions_file'] = str(folder / 'original' / config['initial_conditions']['initial_conditions_file'])
    config['subpop_setup']['selected'] = ['large_province']
    with tempfile.TemporaryDirectory() as tmp:
        sc.saveyaml(sc.path(tmp) / 'config.yml', config)
        one = load_flepimop(sc.path(tmp) / 'config.yml', **kw)
    assert not one.imported.incomplete
    assert one.pars.total_pop == 9000 # The large province only
    return sim


@needs_examples
def test_scenarios():
    sc.heading('Testing flepiMoP scenarios and outcomes, with the original\'s draws')
    folder = examples / 'vaccine_scenarios'
    base = load_flepimop(folder / 'original' / 'config_sample_2pop_vaccine_scenarios.yml', **kw)
    assert set(base.imported.scenarios) == {'no_vax', 'pess_vax', 'opt_vax'}
    skipped = dict(base.imported.incomplete)
    assert 'Ro' not in skipped and 'outcomes: incidCase.probability' in skipped, 'Expected the drawn Ro to be imported with its prior, and the drawn outcome probability listed as not translated'
    assert base.new().init().diseases[0].pars.Ro.prior is not None # So ss.parallel(sim, priors=True) draws it

    # Each scenario, run with the values the original drew for each of its ten runs, has the same mean
    draws = pd.read_csv(folder / 'reference' / 'draws.csv')
    for scen in ['no_vax', 'opt_vax']:
        rows = draws[draws.scenario == scen]
        sims = [base.new(pars=dict(Ro=r.Ro, p_incidCase=r.p_case), interventions=base.imported.scenarios[scen], **kw).run() for r in rows.itertuples()]
        res = [sim.results.sample_2pop for sim in sims]
        for place in ['large_province', 'small_province']:
            for key,theirs in [('n_V', 'V'), ('n_I', 'I')]:
                ours = np.mean([r[key].select(subpop=place).values[:-1] for r in res], axis=0)
                assert np.abs(ours - reference(folder, f'{scen}_{place}_{theirs}')).max() < 1e-6
            for key,theirs in [('incidCase', 'incidCase'), ('incidDeath', 'incidDeath'), ('n_incidHosp', 'currHosp')]: # A flow at index k is the number in the step ending on day k; the original's on day k is the number during day k
                ours = np.mean([r[key].select(subpop=place).values[1:] for r in res], axis=0)
                assert np.abs(ours - reference(folder, f'{scen}_{place}_{theirs}')).max() < 1e-6, f'{scen} {place} {key} differs from the original' # Including the end date
    return sims


def test_dimensions(tmp_path=None):
    sc.heading('Testing flepiMoP compartments with two dimensions')
    folder = sc.path(tmp_path or tempfile.mkdtemp())
    beta, gamma, nu, eff = 0.4, 0.1, 0.02, 0.6
    config = dict(
        name = 'sir_vax',
        start_date = '2020-01-01',
        end_date = '2020-05-01',
        compartments = dict(infection_stage=['S', 'I', 'R'], vaccination_stage=['unvaccinated', 'vaccinated']),
        initial_conditions = dict(method='SetInitialConditions', initial_conditions_file='ic.csv'),
        seir = dict(
            integration = dict(method='rk4', dt=1),
            parameters = dict(beta=dict(value=beta), gamma=dict(value=gamma), nu=dict(value=nu), eff=dict(value=eff)),
            transitions = [
                dict(source=[['S'], ['unvaccinated', 'vaccinated']], destination=[['I'], ['unvaccinated', 'vaccinated']], rate=[['beta'], ['1', '1 - eff']],
                     proportional_to=['source', [['I'], [['unvaccinated', 'vaccinated']]]], proportion_exponent=[['1', '1'], ['1', '1']]),
                dict(source=[['I'], ['unvaccinated', 'vaccinated']], destination=[['R'], ['unvaccinated', 'vaccinated']], rate=[['gamma'], ['1']],
                     proportional_to=['source'], proportion_exponent=[['1', '1']]),
                dict(source=[['S'], ['unvaccinated']], destination=[['S'], ['vaccinated']], rate=[['nu'], ['1']],
                     proportional_to=['source'], proportion_exponent=[['1', '1']]),
            ],
        ),
    )
    sc.saveyaml(folder / 'config.yml', config)
    pd.DataFrame(dict(mc_name=['S_unvaccinated', 'I_unvaccinated'], amount=[990, 10])).to_csv(folder / 'ic.csv', index=False)
    sim = load_flepimop(folder / 'config.yml', **kw)
    res = sim.run().results.sir_vax
    assert len(sim.diseases[0].graph.states) == 6

    # The same ODE, written out and solved independently
    def deriv(t, y):
        Su, Sv, Iu, Iv, Ru, Rv = y
        foi = beta*(Iu + Iv)/y.sum()
        return [-foi*Su - nu*Su, nu*Su - (1 - eff)*foi*Sv, foi*Su - gamma*Iu, (1 - eff)*foi*Sv - gamma*Iv, gamma*Iu, gamma*Iv]
    days = np.arange(len(res.timevec))
    sol = spi.solve_ivp(deriv, [0, days[-1]], [990, 0, 10, 0, 0, 0], t_eval=days, rtol=1e-10, atol=1e-8).y
    for i,state in enumerate(['S_unvaccinated', 'S_vaccinated', 'I_unvaccinated', 'I_vaccinated', 'R_unvaccinated', 'R_vaccinated']):
        assert np.abs(res[f'n_{state}'].values - sol[i]).max() < 0.05, f'{state} differs from the ODE'
    return sim


if __name__ == '__main__':
    sim1 = test_seir_2pop()
    sims = test_scenarios()
    sim3 = test_dimensions()
