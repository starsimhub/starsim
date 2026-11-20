"""
Test attributes (ss.attr): values for each person drawn or recorded at events, traits, and derived values
"""
import numpy as np
import sciris as sc
import pytest
import starsim as ss
import corpus

kw = dict(verbose=0)


def make_hpv(name, beta=ss.peryear(0.4)):
    """ One genotype: infection, clearance or CIN, and antibodies that seroconversion at clearance gives, the highest kept """
    S, P, C = ss.states('S P C')
    infection = S >> P @ beta*(P + C)
    precin = ss.Transition(P, {C: 0.2, S: None}, dur=ss.lognormal(ss.years(1.5), ss.years(1)))
    cin = ss.Transition(C, S, dur=ss.years(2))
    nab = ss.attr(ss.uniform(0.5, 0.95)*ss.bernoulli(0.6), when=precin.to('S') | cin, combine='max', init=0)
    return ss.Disease(S, P, C, infection=infection, precin=precin, cin=cin, nab=nab, init_prev=0.1, name=name)


def make_nab(dist, combine='max', init=0, rate=ss.peryear(1)):
    """ A disease whose recovery sets an attribute, drawn from a distribution """
    S, I = ss.states('S I')
    rec = ss.Transition(I, S, dur=ss.years(1))
    nab = ss.attr(dist, when=rec, combine=combine, init=init)
    return ss.Disease(S, I, infection=S >> I @ rate*I, rec=rec, nab=nab, init_prev=0.1, name='d')


class TrackNab(ss.Analyzer):
    """ Antibodies never fall, and are 0 or between the bounds of the draw """
    def init_post(self):
        super().init_post()
        self.last = {name:None for name in self.sim.diseases.keys()}

    def step(self):
        for name,d in self.sim.diseases.items():
            vals = d.nab.raw.copy()
            ok = (vals == 0) | ((vals >= 0.5) & (vals <= 0.95)) | np.isnan(vals)
            assert ok[d.nab.auids].all()
            if self.last[name] is not None:
                alive = d.nab.auids[d.nab.auids < len(self.last[name])]
                assert np.all(vals[alive] >= self.last[name][alive] - 1e-12)
            self.last[name] = vals


def test_attr_solvers():
    """ Attributes are kept per agent, so a compartmental solver refuses them and auto runs agents; a trait is run at its mean, with a note """
    hiv, mods = corpus.make_hiv()
    assert isinstance(ss.Sim(hiv, connectors=mods, total_pop=20_000, start=1990, stop=2000, dt=ss.months(1), **kw).init().solver, ss.solvers.abm)
    hiv, mods = corpus.make_hiv()
    with pytest.raises(ss.CapabilityError):
        ss.Sim(hiv, connectors=mods, solver='ode', total_pop=20_000, start=1990, stop=2000, dt=ss.months(1), **kw).init()

    def make(trait=True, **opts):
        S, I, R = ss.states('S I R')
        sus = ss.attr(ss.gamma(shape=2, scale=0.5)) # Mean 1
        dis = ss.Disease(infection=S >> I @ ss.perday(0.3)*(sus if trait else 1)*I, recovery=I >> R @ ss.perday(0.1), init=dict(I=0.001), sus=sus, name='sir')
        return ss.Sim(dis, unit='day', stop=200, total_pop=10_000, **opts, **kw)
    sim = make(solver='ode').run()
    assert np.allclose(sim.results.sir.I, make(trait=False, solver='ode').run().results.sir.I) and any('uses its mean' in note for note in sim.notes)
    assert isinstance(make().init().solver, ss.solvers.abm)
    with pytest.raises(ss.CapabilityError), ss.options.context(approximations='error'):
        make(solver='ode').init()
    return


def test_hpv_nab():
    """ HPV: antibodies at clearance (combine="max") across two genotypes, as cross-immunity """
    m = np.array([[1.0, 0.7], [0.7, 1.0]])
    def run(cross=True):
        hpvs = [make_hpv('hpv16'), make_hpv('hpv18')]
        mods = [ss.modify(h.rel_sus, multiply=1 - np.clip(sum(m[i, j]*hpvs[j].nab for j in range(2)), 0, 1), name=f'nab_{h.name}') for i,h in enumerate(hpvs)] if cross else []
        return ss.Sim(hpvs, connectors=mods, analyzers=TrackNab(), n_agents=10_000, start=2000, stop=2020, dt=ss.months(3), **kw).run()
    sim = run()
    vals = sim.diseases.hpv16.nab.values
    seropos = (vals > 0).mean()
    assert 0.1 < seropos < 1 and (vals > 0).sum() <= sim.results.hpv16.precin.sum() # At most those who ever cleared
    none = run(cross=False)
    for name in ['hpv16', 'hpv18']: # Cross-immunity protects
        assert sim.results[name].infection.sum() < 0.9*none.results[name].infection.sum()
        assert (none.diseases[name].nab.values > 0).mean() > seropos*0.9 # Antibodies are set the same way without the modifier
    return sim


class FirstNab(ss.Analyzer):
    """ The step each person first gets antibodies in, and the value drawn for them then """
    def init_post(self):
        super().init_post()
        self.when = dict()
        self.val = dict()

    def step(self):
        d = self.sim.diseases.d
        for uid in d.nab.auids[d.nab.raw[d.nab.auids] > 0]:
            if uid not in self.when:
                self.when[uid] = self.t.ti
                self.val[uid] = d.nab.raw[uid]


def test_attr_crn():
    """ Reading an attribute in a modifier does not change its draws (common random numbers) """
    def run(rel=None):
        S, I, R = ss.states('S I R')
        recovery = I >> R @ ss.peryear(1)
        nab = ss.attr(ss.uniform(0.5, 0.95), when=recovery, combine='max', init=0)
        d = ss.Disease(S, I, R, infection=S >> I @ ss.peryear(2)*I, recovery=recovery, wane=R >> S @ ss.peryear(0.5), nab=nab, init_prev=0.05, name='d')
        mods = ss.modify(d.rel_sus, multiply=rel(d), name='m') if rel else []
        return ss.Sim(d, connectors=mods, analyzers=FirstNab(), n_agents=2000, start=2000, stop=2010, dt=ss.months(1), rand_seed=1, **kw).run()
    base = run()
    same = run(lambda d: 1 - 0*d.nab) # Reads the attribute, and changes nothing
    assert np.array_equal(base.diseases.d.nab.values, same.diseases.d.nab.values) and np.array_equal(base.results.d.infection, same.results.d.infection)
    less = run(lambda d: 1 - 0.5*d.nab) # Changes who is infected: people who first get antibodies in the same step in both runs get the same draw
    assert less.results.d.infection.sum() < 0.95*base.results.d.infection.sum()
    a, b = base.analyzers[0], less.analyzers[0]
    both = [uid for uid in a.when if b.when.get(uid) == a.when[uid]]
    assert len(both) > 100 and all(a.val[uid] == b.val[uid] for uid in both)
    return


def test_attr_values():
    """ Where an attribute has a value: none read where it has none; init= for newborns, with spare slots not read; a parameter in its distribution, set from the sim """
    with pytest.raises(ValueError): # A case of a derived attribute reads a clock of another state
        A, B = ss.states('A B')
        ss.Disease(A, B, ab=A >> B @ 0.1, x=ss.attr({B: A.time_in}), default_state='A', name='d')
    with pytest.raises(ValueError):
        ss.attr(ss.normal(), combine='mean')

    demog = lambda: [ss.Births(birth_rate=0.025), ss.Deaths(death_rate=0.01)]
    d = make_nab(ss.uniform(0.5, 0.95), rate=ss.peryear(2))
    sim = ss.Sim(d, connectors=ss.modify(d.rel_sus, multiply=1 - d.nab, name='imm'), demographics=demog(), n_agents=1000, start=2000, stop=2005, **kw).run()
    nab, auids = sim.diseases.d.nab, sim.people.auids
    born = auids[auids >= 1000]
    assert len(born) > 50 and not np.isnan(nab.raw[auids]).any() and (nab.raw[born] == 0).any() # Newborns who have not recovered have the init
    d = make_nab(ss.uniform(0.5, 0.95), init=None, rate=ss.peryear(2)) # With no init=, people who have not recovered have no value, which a modifier can't read
    with pytest.raises(ValueError):
        ss.Sim(d, connectors=ss.modify(d.rel_sus, multiply=1 - d.nab, name='imm'), demographics=demog(), n_agents=1000, start=2000, stop=2005, **kw).run()

    means = []
    for sero in [0.2, 0.8]:
        d = make_nab(ss.bernoulli(ss.par(0.6, name='sero')), combine='last', init=None)
        sim = ss.Sim(d, n_agents=2000, start=2000, stop=2005, pars=dict(sero=sero), **kw).run()
        vals = sim.diseases.d.nab.raw[sim.people.auids]
        vals = vals[~np.isnan(vals)] # Those who have recovered
        assert 'sero' in sim.diseases.d.pars and len(vals) > 200
        means.append(vals.mean())
    assert abs(means[0] - 0.2) < 0.07 and abs(means[1] - 0.8) < 0.07, means
    return


if __name__ == '__main__':
    T = sc.timer()
    test_attr_solvers()
    test_hpv_nab()
    test_attr_crn()
    test_attr_values()
    T.toc()
