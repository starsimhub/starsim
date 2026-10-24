"""
Run tests of the Starsim library (starsim.library, i.e. ssl)

These tests check that every library module can be instantiated and run, and that
its headline mechanism actually does something. Detailed tests of individual
modules live in the corresponding test files (e.g. test_diseases.py).
"""

# %% Imports and settings
import inspect
import sciris as sc
import numpy as np
import starsim as ss
import starsim.library as ssl

n_agents = 2_000

# Outbreak diseases specify their natural history in days, so they need a sub-year timestep
outbreak_pars = dict(start=2000, dur=ss.years(1), dt=ss.days(1), n_agents=n_agents, rand_seed=1, verbose=0)
endemic_pars  = dict(start=2000, stop=2030, n_agents=n_agents, rand_seed=1, verbose=0)
mnch_pars     = dict(start=2000, dur=ss.years(3), dt=ss.weeks(1), n_agents=n_agents, rand_seed=1, verbose=0)


@sc.timer()
def test_mnch():
    """ Check that the MNCH modules run and produce their headline outcomes """
    sc.heading('Testing MNCH modules...')

    demographics = lambda: [ss.Pregnancy(fertility_rate=ss.time.freq(100, 'year'), burnin=True), ss.Deaths()]
    networks = lambda: [ss.PrenatalNet(), ss.RandomNet()]
    sims = sc.objdict()

    # Congenital disease: mother-to-child transmission with birth outcomes
    sim = ss.Sim(diseases=ssl.mnch.CongenitalDisease(beta=ss.peryear(20), init_prev=0.2),
                 demographics=demographics(), networks=networks(), **mnch_pars)
    sim.run()
    n_congenital = sim.diseases.congenitaldisease.congenital.sum()
    print(f'  Congenital infections: {n_congenital:n}')
    assert n_congenital > 0, 'Expected some congenital infections'
    sims.congenital = sim

    # Neonatal sepsis: newborns are infected at birth, and some die
    sim = ss.Sim(diseases=ssl.mnch.NeonatalSepsis(), demographics=demographics(),
                 networks=networks(), **mnch_pars)
    sim.run()
    new_inf = sim.results.neonatalsepsis.new_infections.sum()
    nnd = sim.results.pregnancy.neonatal_deaths.sum()
    print(f'  Neonatal sepsis infections: {new_inf:n}; neonatal deaths: {nnd:n}')
    assert new_inf > 0, 'Expected newborns to be infected with sepsis'
    assert nnd > 0, 'Expected some neonatal deaths'
    sims.sepsis = sim

    # Fetal health: infection during pregnancy should restrict growth
    sim = ss.Sim(diseases=ssl.AgentSIR(name='sir', beta=ss.peryear(20)), demographics=demographics(),
                 connectors=ssl.mnch.fetal_infection(), custom=ssl.mnch.FetalHealth(),
                 interventions=ssl.mnch.treat_pregnant(disease='sir', start_year=2001),
                 networks=networks(), **mnch_pars)
    sim.run()
    fh = sim.custom.fetal_health
    n_treated = sim.interventions.treat_pregnant.ti_treated.notnan.sum()
    n_weighed = fh.birth_weight.notnan.sum()
    print(f'  Newborns weighed: {n_weighed:n}; LBW: {fh.lbw.sum():n}; treated: {n_treated:n}')
    assert n_weighed > 0, 'Expected some newborns to have a birth weight'
    assert n_treated > 0, 'Expected some pregnant women to be treated'
    sims.fetal = sim

    return sims


if __name__ == '__main__':
    mnch     = test_mnch()
