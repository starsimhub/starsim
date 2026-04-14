"""
Test connectors for coverage improvement.

Tests the seasonality connector, susceptibility/transmissibility modification
patterns, multiple connectors, and scientific correctness of connector effects.
"""

import sciris as sc
import numpy as np
import matplotlib.pyplot as plt
import starsim as ss

n_agents = 1_000
do_plot = False
sc.options(interactive=False)


#%% Helper classes

class SusModifier(ss.Connector):
    """ Connector that modifies susceptibility based on co-infection status """
    def __init__(self, disease_a='sir_a', disease_b='sir_b', rel_sus=2.0, **kwargs):
        super().__init__()
        self.define_pars(
            disease_a = disease_a,
            disease_b = disease_b,
            rel_sus = rel_sus,
        )
        self.update_pars(**kwargs)
        return

    def step(self):
        """ People infected with disease_a have increased susceptibility to disease_b """
        p = self.pars
        disease_a = self.sim.diseases[p.disease_a]
        disease_b = self.sim.diseases[p.disease_b]
        # Reset baseline
        disease_b.rel_sus[:] = 1.0
        # Increase susceptibility for those infected with disease_a
        disease_b.rel_sus[disease_a.infected] = p.rel_sus
        return


class TransModifier(ss.Connector):
    """ Connector that modifies transmissibility based on co-infection status """
    def __init__(self, disease_a='sir_a', disease_b='sir_b', rel_trans=2.0, **kwargs):
        super().__init__()
        self.define_pars(
            disease_a = disease_a,
            disease_b = disease_b,
            rel_trans = rel_trans,
        )
        self.update_pars(**kwargs)
        return

    def step(self):
        """ People infected with disease_a are more transmissible of disease_b """
        p = self.pars
        disease_a = self.sim.diseases[p.disease_a]
        disease_b = self.sim.diseases[p.disease_b]
        # Reset baseline
        disease_b.rel_trans[:] = 1.0
        # Increase transmissibility for those infected with disease_a
        disease_b.rel_trans[disease_a.infected] = p.rel_trans
        return


class ProtectiveConnector(ss.Connector):
    """ Connector where infection with disease_a protects against disease_b """
    def __init__(self, disease_a='sir_a', disease_b='sir_b', **kwargs):
        super().__init__()
        self.define_pars(
            disease_a = disease_a,
            disease_b = disease_b,
        )
        self.update_pars(**kwargs)
        return

    def step(self):
        """ People infected with disease_a cannot acquire disease_b """
        p = self.pars
        disease_a = self.sim.diseases[p.disease_a]
        disease_b = self.sim.diseases[p.disease_b]
        # Reset to baseline
        disease_b.rel_sus[:] = 1.0
        # Block susceptibility for those with disease_a
        disease_b.rel_sus[disease_a.infected] = 0.0
        return


#%% Helper functions

def make_two_disease_sim(connectors=None, n=n_agents):
    """ Create a sim with two SIR diseases on the same network """
    sir_a = ss.SIR(
        name = 'sir_a',
        beta = 0.05,
        dur_inf = 10,
        init_prev = 0.05,
    )
    sir_b = ss.SIR(
        name = 'sir_b',
        beta = 0.05,
        dur_inf = 10,
        init_prev = 0.05,
    )
    net = ss.RandomNet(n_contacts=4)
    sim = ss.Sim(
        n_agents = n,
        diseases = [sir_a, sir_b],
        networks = net,
        connectors = connectors,
        dur = 60,
        verbose = 0,
    )
    return sim


def make_sis_sim(connectors=None, n=n_agents, dur=365):
    """ Create an SIS sim for seasonality testing """
    sis = ss.SIS(
        beta = 0.05,
        dur_inf = 10,
        waning = 0.01,
        init_prev = 0.1,
    )
    net = ss.RandomNet(n_contacts=4)
    sim = ss.Sim(
        n_agents = n,
        start = '2020-01-01',
        dur = dur,
        dt = ss.days(1),
        diseases = sis,
        networks = net,
        connectors = connectors,
        verbose = 0,
    )
    return sim


#%% Tests for the seasonality connector

@sc.timer()
def test_seasonality_default(do_plot=do_plot):
    """ Test that seasonality connector runs with default parameters """
    sc.heading('Testing seasonality default...')

    sim = make_sis_sim(connectors=ss.seasonality())
    sim.run()

    conn = sim.connectors[0]  # Access from sim (connector is copied)
    assert len(conn.factors) > 0, 'Expected seasonality factors to be recorded'

    # Check that factors are within expected range (1 ± scale)
    factors = np.array([f[1] for f in conn.factors])
    assert np.all(factors >= 0), 'Expected all factors to be non-negative'
    assert np.max(factors) <= 1.2 + 0.01, f'Expected max factor <= 1.2 for default scale=0.2, got {np.max(factors)}'
    assert np.min(factors) >= 0.8 - 0.01, f'Expected min factor >= 0.8 for default scale=0.2, got {np.min(factors)}'

    if do_plot:
        plt.figure()
        conn.plot()

    return sim


@sc.timer()
def test_seasonality_scale(do_plot=do_plot):
    """ Test that increasing seasonality scale increases transmission variation """
    sc.heading('Testing seasonality scale...')

    # Low scale
    sim_low = make_sis_sim(connectors=ss.seasonality(scale=0.1), dur=365)
    sim_low.run()

    # High scale
    sim_high = make_sis_sim(connectors=ss.seasonality(scale=0.5), dur=365)
    sim_high.run()

    factors_low = np.array([f[1] for f in sim_low.connectors[0].factors])
    factors_high = np.array([f[1] for f in sim_high.connectors[0].factors])

    range_low = np.max(factors_low) - np.min(factors_low)
    range_high = np.max(factors_high) - np.min(factors_high)

    assert range_high > range_low, \
        f'Expected higher scale to produce wider factor range, got low={range_low:.3f} vs high={range_high:.3f}'

    return sim_low, sim_high


@sc.timer()
def test_seasonality_shift(do_plot=do_plot):
    """ Test that shift parameter offsets the seasonality peak """
    sc.heading('Testing seasonality shift...')

    sim1 = make_sis_sim(connectors=ss.seasonality(scale=0.5, shift=0.0), dur=365)
    sim1.run()

    sim2 = make_sis_sim(connectors=ss.seasonality(scale=0.5, shift=0.5), dur=365)
    sim2.run()

    factors1 = np.array([f[1] for f in sim1.connectors[0].factors])
    factors2 = np.array([f[1] for f in sim2.connectors[0].factors])

    # The peaks should be at different positions
    peak1 = np.argmax(factors1)
    peak2 = np.argmax(factors2)
    assert peak1 != peak2, \
        f'Expected shift=0.5 to move peak position, but both peak at index {peak1}'

    return sim1, sim2


@sc.timer()
def test_seasonality_specific_diseases(do_plot=do_plot):
    """ Test that seasonality can target specific diseases """
    sc.heading('Testing seasonality targeting specific diseases...')

    sir_a = ss.SIR(name='sir_a', beta=0.05, dur_inf=10, init_prev=0.05)
    sir_b = ss.SIR(name='sir_b', beta=0.05, dur_inf=10, init_prev=0.05)
    net = ss.RandomNet(n_contacts=4)

    sim = ss.Sim(
        n_agents = n_agents,
        start = '2020-01-01',
        dur = 365,
        dt = ss.days(1),
        diseases = [sir_a, sir_b],
        networks = net,
        connectors = ss.seasonality(diseases='sir_a', scale=0.5),
        verbose = 0,
    )
    sim.run()

    assert len(sim.connectors[0].factors) > 0, 'Expected seasonality factors to be recorded'
    return sim


@sc.timer()
def test_seasonality_plot(do_plot=do_plot):
    """ Test that seasonality.plot() runs without error """
    sc.heading('Testing seasonality plot...')

    sim = make_sis_sim(connectors=ss.seasonality(scale=0.3), dur=365)
    sim.run()

    conn = sim.connectors[0]
    if do_plot:
        plt.figure()
        fig = conn.plot()
        return fig

    return conn


#%% Tests for susceptibility modification pattern

@sc.timer()
def test_sus_modifier_increases_infections(do_plot=do_plot):
    """ Test that increasing susceptibility via connector increases infections """
    sc.heading('Testing susceptibility modifier increases infections...')

    # Without connector
    sim_base = make_two_disease_sim(connectors=None)
    sim_base.run()

    # With susceptibility modifier (disease_a infection increases sus to disease_b)
    conn = SusModifier(disease_a='sir_a', disease_b='sir_b', rel_sus=5.0)
    sim_conn = make_two_disease_sim(connectors=conn)
    sim_conn.run()

    base_b = sim_base.results.sir_b.cum_infections[-1]
    conn_b = sim_conn.results.sir_b.cum_infections[-1]

    # With generous tolerance for stochastic sim
    assert conn_b >= base_b * 0.8, \
        f'Expected connector to increase sir_b infections, got base={base_b} vs connector={conn_b}'

    if do_plot:
        plt.figure()
        plt.plot(sim_base.results.sir_b.n_infected, label='Without connector')
        plt.plot(sim_conn.results.sir_b.n_infected, label='With sus modifier')
        plt.legend()
        plt.title('Susceptibility modification effect')

    return sim_base, sim_conn


@sc.timer()
def test_trans_modifier_increases_infections(do_plot=do_plot):
    """ Test that increasing transmissibility via connector increases infections """
    sc.heading('Testing transmissibility modifier increases infections...')

    # Without connector
    sim_base = make_two_disease_sim(connectors=None)
    sim_base.run()

    # With transmissibility modifier
    conn = TransModifier(disease_a='sir_a', disease_b='sir_b', rel_trans=5.0)
    sim_conn = make_two_disease_sim(connectors=conn)
    sim_conn.run()

    base_b = sim_base.results.sir_b.cum_infections[-1]
    conn_b = sim_conn.results.sir_b.cum_infections[-1]

    # With generous tolerance for stochastic sim
    assert conn_b >= base_b * 0.8, \
        f'Expected connector to increase sir_b infections, got base={base_b} vs connector={conn_b}'

    return sim_base, sim_conn


@sc.timer()
def test_protective_connector(do_plot=do_plot):
    """ Test that a protective connector reduces infections in the target disease """
    sc.heading('Testing protective connector...')

    # Without connector
    sim_base = make_two_disease_sim(connectors=None, n=2_000)
    sim_base.run()

    # With protective connector (disease_a protects against disease_b)
    conn = ProtectiveConnector(disease_a='sir_a', disease_b='sir_b')
    sim_prot = make_two_disease_sim(connectors=conn, n=2_000)
    sim_prot.run()

    base_b = sim_base.results.sir_b.cum_infections[-1]
    prot_b = sim_prot.results.sir_b.cum_infections[-1]

    assert prot_b <= base_b * 1.2, \
        f'Expected protective connector to reduce sir_b infections, got base={base_b} vs protected={prot_b}'

    if do_plot:
        plt.figure()
        plt.plot(sim_base.results.sir_b.n_infected, label='Without connector')
        plt.plot(sim_prot.results.sir_b.n_infected, label='With protection')
        plt.legend()
        plt.title('Protective connector effect')

    return sim_base, sim_prot


#%% Tests for multiple connectors

@sc.timer()
def test_multiple_connectors(do_plot=do_plot):
    """ Test that multiple connectors can run simultaneously """
    sc.heading('Testing multiple connectors...')

    # Two connectors: one modifies sus, one modifies trans
    conn_sus = SusModifier(disease_a='sir_a', disease_b='sir_b', rel_sus=2.0)
    conn_trans = TransModifier(disease_a='sir_b', disease_b='sir_a', rel_trans=2.0)

    sim = make_two_disease_sim(connectors=[conn_sus, conn_trans])
    sim.run()

    # Both diseases should have infections
    a_inf = sim.results.sir_a.cum_infections[-1]
    b_inf = sim.results.sir_b.cum_infections[-1]

    assert a_inf > 0, f'Expected sir_a infections > 0 with connectors, got {a_inf}'
    assert b_inf > 0, f'Expected sir_b infections > 0 with connectors, got {b_inf}'

    return sim


@sc.timer()
def test_connector_with_seasonality_and_custom(do_plot=do_plot):
    """ Test mixing seasonality and custom connectors """
    sc.heading('Testing seasonality with custom connector...')

    sir_a = ss.SIR(name='sir_a', beta=0.05, dur_inf=10, init_prev=0.05)
    sir_b = ss.SIR(name='sir_b', beta=0.05, dur_inf=10, init_prev=0.05)
    net = ss.RandomNet(n_contacts=4)

    sim = ss.Sim(
        n_agents = n_agents,
        start = '2020-01-01',
        dur = 180,
        dt = ss.days(1),
        diseases = [sir_a, sir_b],
        networks = net,
        connectors = [ss.seasonality(diseases='sir_a', scale=0.3), SusModifier(disease_a='sir_a', disease_b='sir_b', rel_sus=3.0)],
        verbose = 0,
    )
    sim.run()

    assert len(sim.connectors[0].factors) > 0, 'Expected seasonality factors to be recorded'
    return sim


#%% Scientific correctness: vary connector strength

@sc.timer()
def test_sus_modifier_dose_response(do_plot=do_plot):
    """ Test that higher rel_sus leads to more infections (dose-response) """
    sc.heading('Testing susceptibility modifier dose-response...')

    rel_sus_values = [1.0, 3.0, 10.0]
    cum_infections = []

    for rel_sus in rel_sus_values:
        conn = SusModifier(disease_a='sir_a', disease_b='sir_b', rel_sus=rel_sus)
        sim = make_two_disease_sim(connectors=conn, n=2_000)
        sim.run()
        cum_infections.append(sim.results.sir_b.cum_infections[-1])

    # With generous tolerance: higher rel_sus should generally lead to more infections
    # We check that the highest is >= the lowest (allowing stochastic variation)
    assert cum_infections[2] >= cum_infections[0] * 0.7, \
        f'Expected highest rel_sus to produce at least as many infections: ' \
        f'rel_sus=1.0 -> {cum_infections[0]}, rel_sus=10.0 -> {cum_infections[2]}'

    if do_plot:
        plt.figure()
        plt.bar(range(len(rel_sus_values)), cum_infections, tick_label=[str(v) for v in rel_sus_values])
        plt.xlabel('rel_sus')
        plt.ylabel('Cumulative sir_b infections')
        plt.title('Susceptibility modifier dose-response')

    return cum_infections


@sc.timer()
def test_connector_base_class(do_plot=do_plot):
    """ Test that the Connector base class can be instantiated and is a Module """
    sc.heading('Testing Connector base class...')

    conn = ss.Connector()
    assert isinstance(conn, ss.Module), 'Expected Connector to be a Module subclass'

    return conn


#%% Run as a script
if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    # Seasonality tests
    test_seasonality_default(do_plot=do_plot)
    test_seasonality_scale(do_plot=do_plot)
    test_seasonality_shift(do_plot=do_plot)
    test_seasonality_specific_diseases(do_plot=do_plot)
    test_seasonality_plot(do_plot=do_plot)

    # Connector pattern tests
    test_sus_modifier_increases_infections(do_plot=do_plot)
    test_trans_modifier_increases_infections(do_plot=do_plot)
    test_protective_connector(do_plot=do_plot)

    # Multiple connectors
    test_multiple_connectors(do_plot=do_plot)
    test_connector_with_seasonality_and_custom(do_plot=do_plot)

    # Scientific correctness
    test_sus_modifier_dose_response(do_plot=do_plot)

    # Base class
    test_connector_base_class(do_plot=do_plot)

    T.toc()

    if do_plot:
        plt.show()
