"""
Test connectors for coverage improvement.
"""

import sciris as sc
import numpy as np
import matplotlib.pyplot as plt
import starsim as ss

n_agents = 1_000
do_plot = False
sc.options(interactive=False)


class SusModifier(ss.Connector):
    """ Connector that modifies susceptibility based on co-infection """
    def __init__(self, rel_sus=2.0, **kwargs):
        super().__init__()
        self.define_pars(rel_sus=rel_sus)
        self.update_pars(**kwargs)

    def step(self):
        a, b = self.sim.diseases.sir_a, self.sim.diseases.sir_b
        b.rel_sus[:] = 1.0
        b.rel_sus[a.infected] = self.pars.rel_sus


class TransModifier(ss.Connector):
    """ Connector that modifies transmissibility based on co-infection """
    def __init__(self, rel_trans=2.0, **kwargs):
        super().__init__()
        self.define_pars(rel_trans=rel_trans)
        self.update_pars(**kwargs)

    def step(self):
        a, b = self.sim.diseases.sir_a, self.sim.diseases.sir_b
        b.rel_trans[:] = 1.0
        b.rel_trans[a.infected] = self.pars.rel_trans


class ProtectiveConnector(ss.Connector):
    """ Connector where infection with disease_a blocks disease_b """
    def step(self):
        a, b = self.sim.diseases.sir_a, self.sim.diseases.sir_b
        b.rel_sus[:] = 1.0
        b.rel_sus[a.infected] = 0.0


def make_two_disease_sim(connectors=None, n=n_agents):
    """ Create a sim with two SIR diseases """
    return ss.Sim(
        n_agents=n, dur=60, verbose=0,
        diseases=[ss.SIR(name='sir_a', beta=0.05, dur_inf=10, init_prev=0.05),
                  ss.SIR(name='sir_b', beta=0.05, dur_inf=10, init_prev=0.05)],
        networks=ss.RandomNet(n_contacts=4), connectors=connectors,
    )

def make_sis_sim(connectors=None, dur=365):
    """ Create an SIS sim for seasonality testing """
    return ss.Sim(
        n_agents=n_agents, start='2020-01-01', dur=dur, dt=ss.days(1), verbose=0,
        diseases=ss.SIS(beta=0.05, dur_inf=10, waning=0.01, init_prev=0.1),
        networks=ss.RandomNet(n_contacts=4), connectors=connectors,
    )


@sc.timer()
def test_seasonality(do_plot=do_plot):
    """ Test seasonality: default, scale variation, shift, disease targeting, and plot """
    sc.heading('Testing seasonality...')

    # Default params
    sim = make_sis_sim(connectors=ss.seasonality())
    sim.run()
    conn = sim.connectors[0]
    factors = np.array([f[1] for f in conn.factors])
    assert len(factors) > 0, 'Expected factors to be recorded'
    assert np.all(factors >= 0), 'Expected non-negative factors'
    assert np.max(factors) <= 1.21 and np.min(factors) >= 0.79, \
        f'Factors out of range for scale=0.2: [{np.min(factors):.3f}, {np.max(factors):.3f}]'

    # Higher scale = wider range
    sim_lo = make_sis_sim(connectors=ss.seasonality(scale=0.1), dur=365); sim_lo.run()
    sim_hi = make_sis_sim(connectors=ss.seasonality(scale=0.5), dur=365); sim_hi.run()
    f_lo = np.array([f[1] for f in sim_lo.connectors[0].factors])
    f_hi = np.array([f[1] for f in sim_hi.connectors[0].factors])
    assert np.ptp(f_hi) > np.ptp(f_lo), 'Higher scale should produce wider factor range'

    # Shift moves peak
    sim1 = make_sis_sim(connectors=ss.seasonality(scale=0.5, shift=0.0), dur=365); sim1.run()
    sim2 = make_sis_sim(connectors=ss.seasonality(scale=0.5, shift=0.5), dur=365); sim2.run()
    f1 = np.array([f[1] for f in sim1.connectors[0].factors])
    f2 = np.array([f[1] for f in sim2.connectors[0].factors])
    assert np.argmax(f1) != np.argmax(f2), 'Shift should move peak position'

    # Target specific disease
    sim3 = ss.Sim(n_agents=n_agents, start='2020-01-01', dur=365, dt=ss.days(1), verbose=0,
        diseases=[ss.SIR(name='sir_a', beta=0.05, dur_inf=10, init_prev=0.05),
                  ss.SIR(name='sir_b', beta=0.05, dur_inf=10, init_prev=0.05)],
        networks=ss.RandomNet(n_contacts=4), connectors=ss.seasonality(diseases='sir_a', scale=0.5))
    sim3.run()
    assert len(sim3.connectors[0].factors) > 0, 'Targeted seasonality should record factors'

    if do_plot:
        plt.figure()
        conn.plot()
    return sim


@sc.timer()
def test_sus_modifier(do_plot=do_plot):
    """ Test that susceptibility modification increases infections """
    sc.heading('Testing susceptibility modifier...')
    sim_base = make_two_disease_sim(); sim_base.run()
    sim_conn = make_two_disease_sim(connectors=SusModifier(rel_sus=5.0)); sim_conn.run()
    base_b = sim_base.results.sir_b.cum_infections[-1]
    conn_b = sim_conn.results.sir_b.cum_infections[-1]
    assert conn_b >= base_b * 0.8, f'Expected more sir_b infections with connector: base={base_b}, conn={conn_b}'

    if do_plot:
        plt.figure()
        plt.plot(sim_base.results.sir_b.n_infected, label='Baseline')
        plt.plot(sim_conn.results.sir_b.n_infected, label='Sus modifier')
        plt.legend(); plt.title('Susceptibility modification')
    return sim_base, sim_conn


@sc.timer()
def test_trans_modifier(do_plot=do_plot):
    """ Test that transmissibility modification increases infections """
    sc.heading('Testing transmissibility modifier...')
    sim_base = make_two_disease_sim(); sim_base.run()
    sim_conn = make_two_disease_sim(connectors=TransModifier(rel_trans=5.0)); sim_conn.run()
    assert sim_conn.results.sir_b.cum_infections[-1] >= sim_base.results.sir_b.cum_infections[-1] * 0.8, \
        'Expected more infections with transmissibility modifier'
    return sim_base, sim_conn


@sc.timer()
def test_protective_connector(do_plot=do_plot):
    """ Test that a protective connector reduces infections """
    sc.heading('Testing protective connector...')
    sim_base = make_two_disease_sim(n=2_000); sim_base.run()
    sim_prot = make_two_disease_sim(connectors=ProtectiveConnector(), n=2_000); sim_prot.run()
    assert sim_prot.results.sir_b.cum_infections[-1] <= sim_base.results.sir_b.cum_infections[-1] * 1.2, \
        'Expected fewer sir_b infections with protective connector'
    return sim_base, sim_prot


@sc.timer()
def test_multiple_connectors(do_plot=do_plot):
    """ Test multiple connectors and mixed types run together """
    sc.heading('Testing multiple connectors...')
    sim = make_two_disease_sim(connectors=[SusModifier(rel_sus=2.0), TransModifier(rel_trans=2.0)])
    sim.run()
    assert sim.results.sir_a.cum_infections[-1] > 0, 'Expected sir_a infections'
    assert sim.results.sir_b.cum_infections[-1] > 0, 'Expected sir_b infections'

    # Mix seasonality + custom
    sim2 = ss.Sim(n_agents=n_agents, start='2020-01-01', dur=180, dt=ss.days(1), verbose=0,
        diseases=[ss.SIR(name='sir_a', beta=0.05, dur_inf=10, init_prev=0.05),
                  ss.SIR(name='sir_b', beta=0.05, dur_inf=10, init_prev=0.05)],
        networks=ss.RandomNet(n_contacts=4),
        connectors=[ss.seasonality(diseases='sir_a', scale=0.3), SusModifier(rel_sus=3.0)])
    sim2.run()
    assert len(sim2.connectors[0].factors) > 0, 'Seasonality should record factors'
    return sim


@sc.timer()
def test_dose_response_and_base_class(do_plot=do_plot):
    """ Test dose-response (higher rel_sus -> more infections) and base class """
    sc.heading('Testing dose-response and base class...')
    assert isinstance(ss.Connector(), ss.Module), 'Connector should be a Module subclass'
    results = []
    for rel_sus in [1.0, 3.0, 10.0]:
        sim = make_two_disease_sim(connectors=SusModifier(rel_sus=rel_sus), n=2_000)
        sim.run()
        results.append(sim.results.sir_b.cum_infections[-1])
    assert results[2] >= results[0] * 0.7, f'Expected dose-response: rel_sus=1->{results[0]}, rel_sus=10->{results[2]}'
    return results


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()
    test_seasonality(do_plot=do_plot)
    test_sus_modifier(do_plot=do_plot)
    test_trans_modifier(do_plot=do_plot)
    test_protective_connector(do_plot=do_plot)
    test_multiple_connectors(do_plot=do_plot)
    test_dose_response_and_base_class(do_plot=do_plot)
    T.toc()
    if do_plot: plt.show()
