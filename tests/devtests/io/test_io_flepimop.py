"""
Test importing flepiMoP configurations
"""

# %% Imports and settings
import numpy as np
import pandas as pd
import tempfile
import sciris as sc
import scipy.integrate as spi
from starsim.converters.flepimop import load_flepimop

kw = dict(verbose=0)


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


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp:
        sim3 = test_dimensions(tmp)
