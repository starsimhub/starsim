"""
Test Epiverse-TRACE: importing epidemics models, epiparameter's parameters, socialmixr's contacts, and a sim's outbreak as a line list, incidence, and chains
"""
import shutil
import subprocess
import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

kw = dict(verbose=0)
E = ss.converters.epiverse


def has_r_package(pkg):
    """ Whether R is installed with this package """
    if shutil.which('Rscript') is None:
        return False
    return subprocess.run(['Rscript', '-e', f'quit(status = !requireNamespace("{pkg}", quietly = TRUE))'], capture_output=True).returncode == 0


# model_default() with one group, as dump_epidemics.R writes it: the contact matrix [[1]] gives beta*I/N
seir = dict(format='epidemics', version='0.5.0', runs=[dict(model='model_default',
    population=dict(name='one', groups=['all'], contact_matrix=[[1.0]], demography_vector=[1000.0], compartments=list('SEIRV'), initial_conditions=[[0.99, 0, 0.01, 0, 0]]),
    pars=dict(transmission_rate=0.3, infectiousness_rate=0.5, recovery_rate=0.1, time_end=100, increment=1), intervention=None, vaccination=None, time_dependence=None, population_change=False, scenarios=False)])

# The same, by age, with an intervention on contacts and one on transmission, and vaccination, in R
script = '''
library(epidemics)
contact_matrix <- matrix(c(5.0, 2.0, 0.5, 2.0, 7.0, 1.5, 0.5, 1.5, 3.0), 3, 3)
demography <- c(2e6, 5e6, 1.5e6)
rownames(contact_matrix) <- c("[0,20)", "[20,65)", "65+")
names(demography) <- rownames(contact_matrix)
ic <- matrix(c(1 - 1e-6, 0, 1e-6, 0, 0), nrow = 3, ncol = 5, byrow = TRUE)
colnames(ic) <- c("S", "E", "I", "R", "V")
pop <- population(name = "test", contact_matrix = contact_matrix, demography_vector = demography, initial_conditions = ic)
schools <- intervention(name = "schools", type = "contacts", time_begin = 50, time_end = 100, reduction = matrix(c(0.5, 0.01, 0.01)))
masks <- intervention(name = "masks", type = "rate", time_begin = 40, time_end = 120, reduction = 0.1)
vax <- vaccination(name = "vax", time_begin = matrix(30, 3), time_end = matrix(150, 3), nu = matrix(c(0.0001, 0.0005, 0.001)))
out <- model_default(pop, transmission_rate = 1.5/7, infectiousness_rate = 1/3, recovery_rate = 1/7,
  intervention = list(contacts = schools, transmission_rate = masks), vaccination = vax, time_end = 200)
'''


def test_epidemics():
    sc.heading('Testing importing model_default() from epidemics')
    sim = ss.import_from(seir, format='epidemics', **kw)
    assert not sim.imported.incomplete
    res = sim.run().results.seirv

    # By hand
    S, E_, I, R = ss.states('S E I R')
    ref = ss.Disease(S >> E_ @ 0.3*I, E_ >> I @ ss.perday(0.5), I >> R @ ss.perday(0.1), init_n=dict(I=10), name='seir')
    ref = ss.Sim(ref, total_pop=1000, unit='day', stop=100, solver='ode', **kw).run().results.seir
    assert np.allclose(res.I.values, ref.I.values, rtol=1e-6), 'The import should be the SEIR written by hand'
    return sim


@pytest.mark.skipif(not has_r_package('epidemics'), reason='needs R with epidemics')
def test_epidemics_r(tmp_path):
    sc.heading('Testing an epidemics model against epidemics itself')
    sc.savetext(tmp_path/'model.R', script)
    sc.savetext(tmp_path/'run.R', 'source("model.R")\nwrite.csv(out, "out.csv", row.names = FALSE)\n') # Kept apart, since importing runs model.R
    subprocess.run(['Rscript', 'run.R'], cwd=tmp_path, check=True, capture_output=True)
    sim = ss.import_from(tmp_path/'model.R', **kw) # Told from the file, and read by running dump_epidemics.R
    res = sim.run().results.seirv
    r = pd.read_csv(tmp_path/'out.csv')
    for comp,state in dict(susceptible='S', exposed='E', infectious='I', recovered='R', vaccinated='V').items():
        expected = r[r.compartment == comp].groupby('time').value.sum().values
        assert np.allclose(res[state].values, expected, rtol=0, atol=1e-4*expected.max()), f'{comp} should match epidemics'
    return sim


def test_conversions():
    sc.heading('Testing summary statistics to parameters, as epiparameter converts them')
    pars = E.from_summary('gamma', dict(mean=10, sd=5))
    assert np.isclose(pars['shape'], 4) and np.isclose(pars['scale'], 2.5)
    pars = E.from_summary('lnorm', dict(median=5.6, dispersion=1.26))
    assert np.isclose(pars['meanlog'], np.log(5.6)) and np.isclose(pars['sdlog'], np.log(1.26))
    pars = E.from_summary('weibull', dict(mean=6.4, sd=3.05))
    d = ss.weibull(shape=pars['shape'], scale=pars['scale']).mock()
    assert np.isclose(d.mean(), 6.4) and np.isclose(d.std(), 3.05)
    for family in ['lnorm', 'gamma', 'weibull']: # Two percentiles, solved exactly
        pars = E.from_summary(family, dict(quantile_names=['2.5', '5', '95', '97.5'], quantile_values=[3.6, 4.2, 17.3, 19.9]))
        d = E.to_dist(family, pars).mock()
        assert np.allclose(d.ppf(np.array([0.05, 0.95])), [4.2, 17.3]), f'{family} should have the 5th and 95th percentiles (the first symmetric pair, as in R)'
    return


def test_parameter():
    sc.heading('Testing parameters from the epiparameter library')
    try:
        df = E.parameters('ebola')
    except ConnectionError as err:
        pytest.skip(f'the epiparameter library could not be downloaded: {err}')
    assert len(df) and set(df.disease) == {'Ebola Virus Disease'}
    inc = E.parameter('ebola', 'incubation period') # As epiparameter_db(disease = "Ebola", epi_name = "incubation", single_epiparameter = TRUE) gives it
    assert isinstance(inc, ss.gamma) and np.isclose(inc.pars.shape, 1.578, atol=1e-3) and np.isclose(inc.pars.scale, 6.528, atol=1e-3)
    assert 'WHO Ebola Response Team' in inc.ref['citation']
    assert isinstance(E.parameter('ebola', 'offspring'), ss.nbinom)
    with pytest.raises(KeyError):
        E.parameter('not a disease', 'incubation period')
    return inc


def test_contacts(tmp_path):
    sc.heading('Testing contact matrices from socialmixr')
    groups = ['[0,5)', '[5,18)', '[18,65)', '[65,Inf)']
    m = [[1.92, 1.47, 4.84, 0.37], [0.52, 8.86, 6.04, 0.63], [0.45, 1.58, 8.17, 1.03], [0.13, 0.63, 3.98, 1.71]] # POLYMOD, UK, as socialmixr gives it
    pd.DataFrame(m, index=groups, columns=groups).to_csv(tmp_path/'contacts.csv')
    sc.savejson(tmp_path/'contacts.json', dict(matrix=m, participants=[{'age.group': g} for g in groups]))
    for source in [tmp_path/'contacts.csv', str(tmp_path/'contacts.json')]:
        df = E.contact_matrix(source)
        assert list(df.index) == groups and np.allclose(df.values, m)
        net = E.contacts(source)
        assert list(net.edges) == [0, 5, 18, 65]
    S, I, R = ss.states('S I R')
    flu = ss.Disease(S >> I @ 0.05*I, I >> R @ ss.days(5), init_prev=0.001, name='flu').stratify(age=[0, 18, 65])
    sim = ss.Sim(flu, networks=ss.contacts(pd.read_csv(tmp_path/'contacts.csv', index_col=0)), total_pop=1e6, unit='day', stop=50, **kw).run() # The CSV also goes straight in
    assert sim.results.flu.infection.sum() > 0
    return sim


def test_outputs():
    sc.heading('Testing a sim as a line list, incidence, and chains of transmission')
    S, E_, I, R, D = ss.states('S E I R D')
    ebola = ss.Disease(S >> E_ @ 0.15*I, E_ >> I @ ss.days(9), ss.Transition(I, {D: 0.5, R: None}, dur=ss.days(8)), init_prev=0.01, name='ebola')
    sim = ss.Sim(ebola, networks=ss.DegreeNet(n_contacts=4), n_agents=2000, start='2025-01-01', stop='2025-05-01', unit='day',
                 analyzers='infection_log', solver=ss.solvers.abm(keep_times=True), **kw).run()
    res = sim.results.ebola

    ll = E.linelist(sim)
    assert list(ll.columns[:8]) == ['id', 'sex', 'age', 'date_infection', 'date_onset', 'outcome', 'date_outcome', 'source']
    assert len(ll) == res.infection.sum() + sim.diseases.ebola.infection_log.to_df().source.eq(-1).sum() # The infections, and the seeds
    done = ll.dropna(subset=['date_onset', 'outcome'])
    assert (done.date_onset >= done.date_infection).all() and (done.date_outcome >= done.date_onset).all()
    assert set(ll.outcome.dropna()) <= {'died', 'recovered'}
    assert ll.age.between(0, 110).all() and set(ll.sex) <= {'f', 'm'}

    inc = E.incidence(sim, wide=True)
    assert list(inc.columns) == ['date', 'cases', 'deaths'] and inc.cases.sum() == res.infection.sum()
    assert inc.deaths.sum() == (ll.outcome == 'died').sum(), 'Each death should be in the line list'
    long = E.incidence(sim)
    assert list(long.columns) == ['date_index', 'count_variable', 'count'] and len(long) == 2*len(inc)

    tree = E.chains(sim)
    seeds = tree.infector.isna()
    assert (tree.generation[seeds] == 1).all() and tree.chain.nunique() == seeds.sum() and (tree.generation[~seeds] > 1).all()
    off = E.offspring(sim)
    assert off.n_offspring.sum() == (~seeds).sum(), 'Every infection with a source is someone\'s offspring'

    S, I, R = ss.states('S I R') # A compartmental run has incidence, but no people for a line list
    sir = ss.Disease(S >> I @ 0.3*I, I >> R @ ss.days(10), init_prev=0.01, name='sir')
    ode = ss.Sim(sir, unit='day', stop=50, solver='ode', **kw).run()
    assert E.incidence(ode, wide=True).cases.sum() > 0
    with pytest.raises(ValueError):
        E.linelist(ode)
    return sim


if __name__ == '__main__':
    T = sc.timer()
    sim = test_epidemics()
    test_conversions()
    inc = test_parameter()
    out = test_outputs()
    T.toc()
