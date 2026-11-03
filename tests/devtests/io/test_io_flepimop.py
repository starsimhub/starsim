"""
Test importing flepiMoP configurations, and exporting sims as them

The exports are run with flepiMoP from its environment in the lingua_franca_design repo (`_envs/flepimop`; set STARSIM_LFD to say where the repo is), and skipped if it is not there.
"""

# %% Imports and settings
import os
import glob
import subprocess
import numpy as np
import pandas as pd
import tempfile
import pytest
import sciris as sc
import scipy.integrate as spi
import starsim as ss
import paths
from starsim.converters.flepimop import load_flepimop, to_flepimop

kw = dict(verbose=0)
flepimop_bin = paths.lfd/'_envs'/'flepimop'/'bin'


def test_dimensions(tmp_path):
    sc.heading('Testing flepiMoP compartments with two dimensions')
    folder = sc.path(tmp_path)
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
    assert len(sim.diseases[0].states) == 6

    # The same ODE, written out and solved independently
    def deriv(t, y):
        Su, Sv, Iu, Iv, Ru, Rv = y
        foi = beta*(Iu + Iv)/y.sum()
        return [-foi*Su - nu*Su, nu*Su - (1 - eff)*foi*Sv, foi*Su - gamma*Iu, (1 - eff)*foi*Sv - gamma*Iv, gamma*Iu, gamma*Iv]
    days = np.arange(len(res.timevec))
    sol = spi.solve_ivp(deriv, [0, days[-1]], [990, 0, 10, 0, 0, 0], t_eval=days, rtol=1e-10, atol=1e-8).y
    for i,state in enumerate(['S_unvaccinated', 'S_vaccinated', 'I_unvaccinated', 'I_vaccinated', 'R_unvaccinated', 'R_vaccinated']):
        assert np.abs(res[state].values - sol[i]).max() < 0.05, f'{state} differs from the ODE'
    return sim


def export_sim():
    """ An SEIR with dates, a dwell time written as Erlang stages, deaths from I, and births, which flepiMoP has no form for """
    S, E, I, R = ss.states('S E I R')
    seir = ss.Disease(births=ss.birth(S, rate=ss.peryear(0.02)), infection=S >> E @ ss.perday(0.4)*I, onset=E >> I @ ss.days(3),
                      recovery=ss.Transition(I, R, dur=ss.lognorm_ex(ss.days(8), ss.days(3))), deaths=ss.death(I, rate=ss.perday(0.01)), init_prev=0.01, name='seir')
    return ss.Sim(seir, total_pop=1000, unit='day', start='2021-03-01', stop='2021-05-30', verbose=0)


def write(files, folder):
    """ The files of an export, written to a folder; returns the configuration's path """
    for name,text in files.items():
        sc.savetext(folder/name, text)
    return folder/list(files)[0]


def test_flepimop_export(tmp_path):
    sc.heading('Testing an export to flepiMoP, read back')
    sim = export_sim()
    files, summary, lost = to_flepimop(sim)
    assert list(files) == ['seir.yml', 'seir_geodata.csv', 'seir_initial_conditions.csv']
    text = files['seir.yml']
    assert "start_date: '2021-03-01'" in text and "end_date: '2021-05-29'" in text # flepiMoP's output includes its end date
    assert '- - - [I_1, I_2' in text and 'destination: [removed]' in text # Exposure to the sum of the stages; deaths into a compartment
    assert any('"births" into the model' in line for line in lost) and any('"removed"' in line for line in lost)
    back = ss.import_from(write(files, tmp_path), format='flepimop', **kw).run()
    ref = sim.new(solver='ode', substeps=1).run().results.seir # One RK4 step a day, as flepiMoP takes
    res = back.results.seir
    I = sum(res[c].values for c in res.keys() if c.startswith('I_') and c[2:].isdigit())
    assert np.abs(ref.I.values - I).max() < 0.02*ref.I.values.max(), 'Read back, the epidemic should be the same but for the births and deaths, which change the population'
    return back


@pytest.mark.skipif(not (flepimop_bin/'flepimop').exists(), reason='needs flepiMoP (its environment in the lingua_franca_design repo; set STARSIM_LFD)')
def test_flepimop_export_run(tmp_path):
    sc.heading('Testing an export to flepiMoP, run in flepiMoP')
    files, summary, lost = to_flepimop(export_sim())
    config = write(files, tmp_path)
    env = dict(os.environ, PATH=f'{flepimop_bin}{os.pathsep}{os.environ["PATH"]}', NUMBA_NUM_THREADS='1')
    subprocess.run([str(flepimop_bin/'flepimop'), 'simulate', config.name, '--write-csv', '--no-write-parquet', '--jobs', '1', '--out-id', 'test'], check=True, capture_output=True, cwd=tmp_path, env=env)
    out = pd.read_csv(glob.glob(str(tmp_path/'model_output'/'*'/'test'/'seir'/'*.seir.csv'))[0])
    out = out[out.mc_value_type == 'prevalence']
    res = ss.import_from(config, format='flepimop', **kw).run().results.seir # The import, which is the same model
    for state in ['S', 'E', 'I_1', 'I_7', 'R', 'removed']:
        got = out[out.mc_name == state]['all'].values
        assert np.allclose(got, res[state].values[:len(got)], rtol=0, atol=0.05), f'{state} should be the same in flepiMoP' # Not to rounding: flepiMoP's RK4 differs from Starsim's by about 1e-5 of the population for the fast rates of the Erlang stages (the same without them)
    return out


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp:
        sim3 = test_dimensions(tmp)
        test_flepimop_export(sc.path(tmp))
        if (flepimop_bin/'flepimop').exists():
            test_flepimop_export_run(sc.path(tmp))
