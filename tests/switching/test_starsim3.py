"""
Switching demo for the Starsim 3 seat: their HIV and SIS model unchanged, then v4 verbs on it, then one disease redeclared.

Passes when: v4 verbs work on the model written as code; the port matches within seed noise.
Still open: N9, N10, N29, the migration page (D23 and N19, the verbs on the model written as code, are done).

Scaled down to 1,000 to 5,000 agents over 10 to 15 years. N9 and N10 (routine delivery, the one-step shift) are
for the migration page, and N29 (declared agents 1.4x slower than code) is checked by the north-star script (phase 6).
"""
import warnings
import numpy as np
import sciris as sc
import starsim as ss
import starsim.library as ssl

ref_v3_sis_dict = 21_557 # v3 ss.SIS with beta=dict(randomnet=0.05), monthly steps, 5,000 agents, 10 years; switching/starsim3.md, "The second week", Task 2, trap 1
ref_seed_noise = 0.1 # Two seeds of the same model differ by about 6% (21,287 and 22,652 under v4)

def kw():
    """ The v3 sim's keywords, with a new network each time """
    return dict(start=2000, stop=2010, n_agents=5000, networks=ss.RandomNet(n_contacts=4), verbose=0)


def make_hiv_sis(*interventions, where=None):
    """ The v3 library HIV, as code, with a declared SIS whose susceptibility rises with low CD4 """
    hiv = ssl.HIV(beta={'sexual': [0.008, 0.004]}, init_prev=0.05)
    S, I = ss.states('S I')
    sis = ss.Disease(infection=ss.Transition(S, I, rate=ss.peract(0.05) * I), recovery=ss.Transition(I, S, dur=ss.years(2)), init_prev=0.025, name='sis') # Per act on ss.SexualNet (a rate there is an error); about what ss.peryear(0.6) gave at monthly steps
    where = where if where is not None else hiv.cd4 < 200 # A condition on a v3 array
    intvs = [ssl.ART(year=[2000, 2010], coverage=[0, 0.4])] + [i(hiv) for i in interventions]
    return ss.Sim([hiv, sis], networks=ss.SexualNet(), n_agents=1000, start=2000, stop=2015, dt=ss.years(1/12), interventions=intvs,
                  connectors=ss.modify(sis.rel_sus, multiply=5, where=where), verbose=0)


def test_starsim3_unchanged():
    sc.heading('Starsim 3: the v3 SIS, as code (ssl.AgentSIS, one changed name), gives the v3 answer')
    sis = ssl.AgentSIS(name='sis', beta=dict(randomnet=0.05), dur_inf=ss.years(1), init_prev=0.01, waning=ss.peryear(1e6))
    sim = ss.Sim(diseases=sis, dt=1/12, rand_seed=1, **kw()).run()
    infections = sim.results.sis.infection.cum[-1]
    assert np.isclose(infections, ref_v3_sis_dict, rtol=ref_seed_noise), f'Unchanged v3 code gave {infections:n} infections, not about {ref_v3_sis_dict:n}'
    return sim


def test_starsim3_verbs():
    sc.heading('Starsim 3: v4 verbs on the HIV module written as code')
    halve = lambda hiv: ss.modify(hiv.rel_trans, multiply=0.5, start=2005) # A v4 verb on the v3 disease
    msim = ss.parallel(baseline=make_hiv_sis(), prep=make_hiv_sis(halve), n_runs=2)
    averted = -msim.diff('baseline').prep.hiv.infection.cum.values[:, -1].mean()
    assert averted > 0, f'Halving HIV infectiousness from 2005 should avert infections, not {averted}'
    return msim


def test_starsim3_fit():
    sc.heading('Starsim 3: fit ss.SIS (now declared), with bounds given as Starsim 3 gave the value')
    truth = ss.Sim(diseases=ss.SIS(beta=ss.peryear(0.2)), rand_seed=1, **kw()).run()
    data = truth.to_df()[['time', 'sis_prevalence']].iloc[1:].rename(columns=dict(time='t', sis_prevalence='sis.prevalence'))
    sis = ss.SIS(beta=ss.par(ss.peryear(0.1), bounds=[0.05, 1]))
    fit = ss.Sim(diseases=sis, data=data, **kw()).fit(n_trials=10, rand_seed=1)
    beta = fit.pars.beta.value
    assert np.isclose(beta, 0.2, rtol=0.3), f'The fitted beta, {beta:.3f}, should be near the true 0.2'

    # The same for the SIS written as code: bounds on a parameter of a module written as code are fitted too
    agents = ssl.AgentSIS(name='sis', beta=ss.par(ss.peryear(0.1), bounds=[0.05, 1]), waning=ss.peryear(1e6))
    fit = ss.Sim(diseases=agents, data=data, **kw()).fit(n_trials=10, rand_seed=1)
    beta = fit.pars.beta.value
    assert np.isclose(beta, 0.2, rtol=0.3), f'The fitted beta of the SIS written as code, {beta:.3f}, should be near the true 0.2'
    return fit


def test_starsim3_port():
    sc.heading('Starsim 3: the same SIS redeclared, with the beta dict ported as the migrate-starsim3 skill says')
    skill = sc.loadtext(sc.thispath() / '../../starsim_ai/migrate-starsim3/SKILL.md')
    assert "beta={'random': ss.prob(0.05, 'month')}" in skill # Bare numbers in a beta dict were probabilities per step in Starsim 3; core says nothing about Starsim 3, so the skill carries the rule
    S, I = ss.states('S I')
    sis = ss.Disease(infection=ss.Transition(S, I, rate=dict(randomnet=ss.prob(0.05, 'month')) * I), recovery=ss.Transition(I, S, dur=ss.years(1)), init_prev=0.01, name='sis')
    sim = ss.Sim(sis, dt=ss.years(1/12), rand_seed=1, **kw()).run()
    infections = sim.results.sis.infection.sum()
    assert np.isclose(infections, ref_v3_sis_dict, rtol=ref_seed_noise), f'The port gave {infections:n} infections against v3\'s {ref_v3_sis_dict:n}'
    return sim


if __name__ == '__main__':
    sim   = test_starsim3_unchanged()
    msim  = test_starsim3_verbs()
    fit   = test_starsim3_fit()
    port  = test_starsim3_port()
