"""
Run tests of disease models
"""

# %% Imports and settings
import sciris as sc
import numpy as np
import starsim as ss
import starsim.library as ssl



@sc.timer()
def test_branch_flows():
    sc.heading('Testing the flow to one destination of a split, and as an outcome, as agents and compartments')
    def make(name, beta):
        S, P, CIN, CA = ss.states('S P CIN CA')
        cin = ss.Transition(CIN, {CA: 0.3, S: 0.7}, dur=ss.years(2)) # A split, as HPVsim's CIN that clears or becomes cancer
        return ss.Disease(infection=S >> P @ ss.peryear(beta)*P, progression=P >> CIN @ ss.years(1), cin=cin,
                          cancers=ss.outcome(cin.to(CA)), init_prev=0.1, name=name)
    pars = dict(start=2000, stop=2015, dt=0.25, total_pop=10_000, verbose=0)
    out = sc.obj()
    for solver in ['abm', 'ode']:
        sim = ss.Sim(diseases=[make('a', 3.0), make('b', 2.5)], solver=solver, n_agents=10_000, rand_seed=1, **pars).run()
        res = sim.results
        assert res.a.cin.CA.sum() > 0 and np.allclose(res.a.cancers, res.a.cin.CA) and np.allclose(res.b.cancers, res.b.cin.CA) # The outcome is the flow to CA
        assert np.allclose(res.a.cin.CA + res.a.cin.S, res.a.cin)
        out[solver] = [res.a.cin.CA.sum(), res.a.cin.CA.sum()/res.a.cin.sum()]
    assert np.isclose(out.abm[0], out.ode[0], rtol=0.1) and np.allclose([out.abm[1], out.ode[1]], 0.3, rtol=0.05), out
    return out


@sc.timer()
def test_coded_seeds():
    sc.heading('Testing that the initial infections of a disease written as code are dated at the start, as a declared disease\'s are, so they are infectious for as long as later cases')
    sim = ss.Sim(diseases=ssl.AgentSIR(dur_inf=ss.constant(ss.years(2)), p_death=0), networks='random', n_agents=1000, start=2000, stop=2005, verbose=0).init()
    sir = sim.diseases.agentsir
    seeds = sir.infected.uids
    assert len(seeds) and np.all(sir.ti_infected[seeds] == -1) and np.all(sir.ti_recovered[seeds] == 2) # Dated by the step before the first, so infectious in steps 0 and 1, as a case infected in step 0 is in steps 1 and 2
    sim.run()
    assert np.all(sir.recovered[seeds]) and np.all(sim.results.agentsir.infected[0:2] >= len(seeds))
    return sim


@sc.timer()
def test_coded_modifiers():
    sc.heading('Testing modifiers of rel_sus on a disease written as code that reads them, a property a module adds to the people, and causes of death by hazard')

    class Condition(ss.Disease):
        allow_modifiers = ('rel_sus',)
        def __init__(self, **kwargs):
            super().__init__()
            self.define_pars(p_onset=ss.probperyear(0.2))
            self.define_states(ss.BoolState('affected', label='Affected'), ss.FloatArr('rel_sus', default=1.0))
            self.update_pars(**kwargs)
        def init_pre(self, sim):
            super().init_pre(sim)
            sim.people.add_state(ss.BoolArr('housing', default=ss.bernoulli(0.5))) # Idempotent: each module that reads it adds it
            sim.people.add_state(ss.BoolArr('housing'))
            return
        def step(self):
            uids = (~self.affected).uids
            p = -np.expm1(-self.pars.p_onset.rate*self.t.dt_year*self.current_value('rel_sus', uids))
            self.affected[uids[self.sim.people.housing[uids] | (np.random.default_rng(self.ti).random(len(uids)) < p)]] = True
            return

    cond = Condition(name='cond')
    sim = ss.Sim(diseases=cond, interventions=ss.modify(cond.rel_sus, multiply=0.0, who=ss.person.female), n_agents=2000, start=2000, stop=2005, verbose=0).run()
    ppl, d = sim.people, sim.diseases.cond
    assert np.all(d.affected[ppl.female.uids] == ppl.housing[ppl.female.uids]) # Only those housed: the onset of women is modified away
    assert (d.affected & ppl.male & ~ppl.housing).sum() > 100
    assert np.isclose(ppl.housing.mean(), 0.5, atol=0.05)
    # Two causes of death in the same step, weighted by their hazards; and a declared transition's hazard, with its modifiers
    class Killer(ss.Module):
        def __init__(self, hazard, **kwargs):
            super().__init__(**kwargs)
            self.hazard = hazard
        def step(self):
            if self.ti == 2:
                self.sim.people.request_death(self.sim.people.auids, cause=self, hazard=self.hazard)
    sim = ss.Sim(connectors=[Killer(9, name='a'), Killer(1, name='b')], n_agents=5000, start=2000, stop=2003, verbose=0).run()
    codes = sim.people.cause_of_death.raw
    frac = np.mean(codes[codes >= 0] == sim.people.causes.index('a'))
    assert np.isclose(frac, 0.9, atol=0.02), frac
    S, I = ss.states('S I')
    dis = ss.Disease(S, I, infection=S >> I @ ss.peryear(1), death=I >> ss.dead @ ss.peryear(0.3), name='d')
    sim = ss.Sim(diseases=dis, interventions=ss.modify(dis.death, multiply=0.5, who=ss.person.female), n_agents=500, start=2000, stop=2001, verbose=0).run()
    uids = sim.people.auids
    assert np.allclose(sim.diseases.d.death.hazard(uids), np.where(sim.people.female[uids], 0.15, 0.3))
    return sim



if __name__ == '__main__':
    for name,test in list(globals().items()):
        if name.startswith('test_'):
            test()
