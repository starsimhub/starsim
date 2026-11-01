"""
Test attributes (ss.attr): values for each person drawn or recorded at events, traits, and derived values; and hooks (ss.func)
"""
import numpy as np
import sciris as sc
import pytest
import starsim as ss

kw = dict(verbose=0)

# HIV (proposal 01 §7.4), with the parameters of the STIsim port
cd4_max, cd4_healthy, art_cd4_growth = 1000, 500, 0.1 # art_cd4_growth is per month
eff_art, time_to_eff, rel_trans_aids, rel_art_mort = 0.99, 0.5, 8.0, 0.25


def make_hiv(init_prev=0.05, init=None, art=ss.peryear(0.2)):
    """ HIV with CD4 as attributes: drawn at infection, recorded when starting and stopping ART, and derived by state """
    class HIV(ss.Disease):
        S        = ss.State()
        acute    = ss.State(rel_trans=5.3)
        latent   = ss.State()
        falling  = ss.State()
        on_art   = ss.State()
        post_art = ss.State()
        infected = acute + latent + falling + on_art + post_art # A sum of states, as a source and for states=
        infection   = S >> acute @ ss.peryear(0.15)*(acute + latent + falling + on_art + post_art)
        latency     = ss.Transition(acute, latent, dur=ss.lognorm_ex(ss.months(1.7), ss.months(1)), stages=3)
        progression = ss.Transition(latent, falling, dur=ss.lognorm_ex(ss.years(10), ss.years(3)))
        aids_death  = ss.Transition(falling, ss.null, dur=ss.lognorm_ex(ss.years(3), ss.years(1)), stages=9)
        start_art   = (acute + latent + falling + post_art) >> on_art @ art
        dropout     = ss.Transition(on_art, post_art, dur=ss.lognorm_ex(ss.years(3), ss.years(1.5)))

        cd4_start     = ss.attr(ss.normal(800, 50), on=infection)
        cd4_latent    = ss.attr(ss.normal(500, 50), on=infection)
        cd4_preart    = ss.attr('cd4', on=start_art) # CD4 just before starting
        cd4_potential = ss.attr(cd4_preart*(1 + (cd4_max - cd4_preart)/cd4_healthy*np.log(cd4_max/cd4_preart)), on=start_art, combine='first') # Set at the first start only
        cd4_stop      = ss.attr('cd4', on=dropout)
        gain = cd4_potential - cd4_preart
        cd4 = ss.attr({
            acute:    cd4_start - (cd4_start - cd4_latent)*acute.progress,
            latent:   cd4_latent,
            falling:  np.maximum(1, cd4_latent*(1 - falling.progress)),
            on_art:   cd4_preart + gain*(2/(1 + np.exp(-art_cd4_growth*on_art.time/ss.months(1))) - 1),
            post_art: np.maximum(1, cd4_stop*(1 - post_art.progress)),
        })
        post_art_death = ss.Transition(post_art, ss.null, dur=np.maximum(np.log(cd4_preart)*cd4_stop/cd4_potential, 0.1)*ss.years(1), stages=10)
        cd4_death      = infected >> ss.null @ ss.by(cd4=ss.peryear([0.30, 0.05, 0.01, 0.005, 0.003]))

    hiv = HIV(init_prev=init_prev, init=init, name='hiv')
    hiv.stratify(cd4=[0, 50, 200, 350, 500], states=hiv.infected) # Derived: no groups
    mods = [
        ss.modify(hiv.cd4_death, multiply=rel_art_mort, where=hiv.on_art, name='art_mort'),
        ss.modify(hiv.rel_trans, multiply=1 - eff_art*np.minimum(hiv.on_art.time/ss.years(time_to_eff), 1), where=hiv.on_art, name='art_trans'),
        ss.modify(hiv.rel_trans, multiply=rel_trans_aids, where=(hiv.cd4 < 200) & (hiv.falling | hiv.post_art), name='aids_trans'),
    ]
    return hiv, mods


def make_hpv(name, beta=ss.peryear(0.4), m_sus=None):
    """ One genotype: infection, clearance or CIN, and antibodies that seroconversion at clearance gives, the highest kept """
    S, P, C = ss.states('S P C')
    infection = S >> P @ beta*(P + C)
    precin = ss.Transition(P, [C, S], p=[0.2, None], dur=ss.lognorm_ex(ss.years(1.5), ss.years(1)))
    cin = ss.Transition(C, S, dur=ss.years(2))
    nab = ss.attr(ss.uniform(0.5, 0.95)*ss.bernoulli(0.6), on=precin.to('S') | cin, combine='max', init=0)
    return ss.Disease(S, P, C, infection=infection, precin=precin, cin=cin, nab=nab, init_prev=0.1, name=name)


class TrackNab(ss.Analyzer):
    """ Antibodies never fall, and are 0 or between the bounds of the draw """
    def init_post(self):
        super().init_post()
        self.last = {name:None for name in self.sim.diseases.keys()}
        return

    def step(self):
        for name,d in self.sim.diseases.items():
            vals = d.nab.raw.copy()
            ok = (vals == 0) | ((vals >= 0.5) & (vals <= 0.95)) | np.isnan(vals)
            assert ok[d.nab.auids].all()
            if self.last[name] is not None:
                n = len(self.last[name])
                alive = d.nab.auids[d.nab.auids < n]
                assert np.all(vals[alive] >= self.last[name][alive] - 1e-12)
            self.last[name] = vals
        return


def test_hiv_cd4_ode():
    sc.heading('HIV CD4 under a compartmental solver: attributes are kept per agent, so it refuses, and auto runs agents (see test_hiv_cd4)')
    kw2 = dict(start=1990, stop=2000, dt=ss.months(1), unit='year', **kw)
    hiv, mods = make_hiv()
    assert ss.Sim(hiv, connectors=mods, total_pop=20_000, **kw2).init().solver == 'abm'
    hiv, mods = make_hiv()
    with pytest.raises(ss.CapabilityError, match='attributes'):
        ss.Sim(hiv, connectors=mods, solver='ode', total_pop=20_000, **kw2).init()
    return


@sc.timer()
def test_hpv_nab():
    sc.heading('HPV: antibodies at clearance (combine="max") across two genotypes, as cross-immunity')
    m = np.array([[1.0, 0.7], [0.7, 1.0]])
    def run(cross=True):
        hpvs = [make_hpv('hpv16'), make_hpv('hpv18')]
        mods = [ss.modify(h.rel_sus, multiply=1 - np.clip(sum(m[i, j]*hpvs[j].nab for j in range(2)), 0, 1), name=f'nab_{h.name}') for i,h in enumerate(hpvs)] if cross else []
        return ss.Sim(hpvs, connectors=mods, analyzers=TrackNab(), n_agents=10_000, start=2000, stop=2020, dt=ss.months(3), unit='year', **kw).run()
    sim = run()
    assert sim.solver == 'abm'
    h = sim.diseases.hpv16
    vals = h.nab.values
    seropos = (vals > 0).mean()
    cleared = sim.results.hpv16.precin.sum() # An upper bound on the people who ever cleared
    assert 0.1 < seropos < 1 and (vals > 0).sum() <= cleared
    none = run(cross=False)
    for name in ['hpv16', 'hpv18']: # Cross-immunity protects
        assert sim.results[name].infection.sum() < 0.9*none.results[name].infection.sum()
        assert (none.diseases[name].nab.values > 0).mean() > seropos*0.9 # Antibodies are set the same way without the modifier
    return sim


def test_hpv_nab_ode():
    sc.heading('HPV antibodies under a compartmental solver: refused, and auto runs agents (see test_hpv_nab)')
    def make(**opts): # One genotype, its own antibodies, combine='max'
        h = make_hpv('hpv', beta=ss.peryear(0.6))
        return ss.Sim(h, connectors=ss.modify(h.rel_sus, multiply=1 - h.nab, name='imm'), start=2000, stop=2020, dt=ss.months(1), unit='year', **opts, **kw)
    with pytest.raises(ss.CapabilityError, match='attributes'):
        make(solver='ode', total_pop=20_000).init()
    sim = make(total_pop=20_000).run()
    assert sim.solver == 'abm' and sim.results.hpv.infection.sum() > 0


@sc.timer()
def test_trait_ode():
    sc.heading('A trait (a distribution drawn once for each person) under a compartmental solver: run at its mean, with a note; auto runs agents')
    def make(trait=True, **opts):
        S, I, R = ss.states('S I R')
        sus = ss.attr(ss.gamma(a=2, scale=0.5)) # Mean 1
        dis = ss.Disease(infection=S >> I @ ss.perday(0.3)*(sus if trait else 1)*I, recovery=I >> R @ ss.perday(0.1), init=dict(I=0.001), sus=sus, name='sir')
        return ss.Sim(dis, unit='day', stop=200, total_pop=10_000, **opts, **kw)
    sim = make(solver='ode').run()
    assert np.allclose(sim.results.sir.I, make(trait=False, solver='ode').run().results.sir.I) and any('uses its mean' in note for note in sim.notes)
    assert make().init().solver == 'abm'
    with pytest.raises(ss.CapabilityError, match='approx=False'):
        make(solver='ode', approx=False).init()
    return


class FirstNab(ss.Analyzer):
    """ The step each person first gets antibodies in, and the value drawn for them then """
    def init_post(self):
        super().init_post()
        self.when = dict()
        self.val = dict()
        return

    def step(self):
        d = self.sim.diseases.d
        vals = d.nab.raw
        for uid in d.nab.auids[vals[d.nab.auids] > 0]:
            if uid not in self.when:
                self.when[uid] = self.t.ti
                self.val[uid] = vals[uid]
        return


@sc.timer()
def test_attr_crn():
    sc.heading('Attributes: reading one in a modifier does not change its draws (common random numbers)')
    def run(rel=None):
        S, I, R = ss.states('S I R')
        recovery = I >> R @ ss.peryear(1)
        nab = ss.attr(ss.uniform(0.5, 0.95), on=recovery, combine='max', init=0)
        d = ss.Disease(S, I, R, infection=S >> I @ ss.peryear(2)*I, recovery=recovery, wane=R >> S @ ss.peryear(0.5), nab=nab, init_prev=0.05, name='d')
        mods = ss.modify(d.rel_sus, multiply=rel(d), name='m') if rel else []
        return ss.Sim(d, connectors=mods, analyzers=FirstNab(), n_agents=2000, start=2000, stop=2010, dt=ss.months(1), unit='year', rand_seed=1, **kw).run()
    base = run()
    same = run(lambda d: 1 - 0*d.nab) # Reads the attribute, and changes nothing
    assert np.array_equal(base.diseases.d.nab.values, same.diseases.d.nab.values)
    assert np.array_equal(base.results.d.infection, same.results.d.infection)

    # A modifier that changes who is infected: people who first get antibodies in the same step in both runs get the same draw
    less = run(lambda d: 1 - 0.5*d.nab)
    assert less.results.d.infection.sum() < 0.95*base.results.d.infection.sum()
    a, b = base.analyzers[0], less.analyzers[0]
    both = [uid for uid in a.when if b.when.get(uid) == a.when[uid]]
    assert len(both) > 100
    assert all(a.val[uid] == b.val[uid] for uid in both)
    return


@sc.timer()
def test_attr_errors():
    sc.heading('Attributes: read where they have no value')
    with pytest.raises(ValueError): # A case of a derived attribute reads a clock of another state
        A, B = ss.states('A B')
        ss.Disease(A, B, ab=A >> B @ 0.1, x=ss.attr({B: A.time}), first='A', name='d')
    S, I = ss.states('S I')
    infection = S >> I @ ss.perday(0.1)*I
    rec = ss.attr(ss.normal(10, 1), on=infection)
    d = ss.Disease(S, I, infection=infection, back=S >> I @ ss.perday(0.01)*rec/10, rec=rec, init_prev=0.01, name='d')
    with pytest.raises(ValueError): # Read by a rate from S, where people who were never infected have none
        ss.Sim(d, n_agents=1000, stop=20, unit='day', **kw).run()
    with pytest.raises(ValueError):
        ss.attr(ss.normal(), combine='mean')
    return


def make_nab(dist, combine='max', init=0, rate=ss.peryear(1)):
    """ A disease whose recovery sets an attribute, drawn from a distribution """
    S, I = ss.states('S I')
    rec = ss.Transition(I, S, dur=ss.years(1))
    nab = ss.attr(dist, on=rec, combine=combine, init=init)
    return ss.Disease(S, I, infection=S >> I @ rate*I, rec=rec, nab=nab, init_prev=0.1, name='d')


def test_attr_dist_par():
    """ A distribution with a named parameter (ss.par) can be drawn in an attribute, and the parameter can be set on the sim """
    ss.Sim(make_nab(ss.bernoulli(ss.par(0.6, name='sero'))), n_agents=1000, start=2000, stop=2005, **kw).run() # Raised a TypeError when drawn
    means = []
    for sero in [0.2, 0.8]:
        d = make_nab(ss.bernoulli(ss.par(0.6, name='sero')), combine='last', init=None)
        sim = ss.Sim(d, n_agents=2000, start=2000, stop=2005, pars=dict(sero=sero), **kw).run()
        d = sim.diseases.d
        assert 'sero' in d.pars
        vals = d.nab.raw[sim.people.auids]
        vals = vals[~np.isnan(vals)] # Those who have recovered
        assert len(vals) > 200
        means.append(vals.mean())
    assert abs(means[0] - 0.2) < 0.07 and abs(means[1] - 0.8) < 0.07, means
    return


def test_attr_births():
    """ An attribute with init= read by a modifier, in a sim with births: newborns have the init, and spare slots in the arrays are not read """
    d = make_nab(ss.uniform(0.5, 0.95), rate=ss.peryear(2))
    demog = lambda: [ss.Births(birth_rate=25), ss.Deaths(death_rate=10)]
    sim = ss.Sim(d, connectors=ss.modify(d.rel_sus, multiply=1 - d.nab, name='imm'), demographics=demog(), n_agents=1000, start=2000, stop=2005, **kw).run() # Raised ValueError at the first step
    nab = sim.diseases.d.nab
    auids = sim.people.auids
    born = auids[auids >= 1000]
    assert len(born) > 50
    assert not np.isnan(nab.raw[auids]).any()
    assert (nab.raw[born] == 0).any() # Newborns who have not recovered have the init

    # With no init=, people who have not recovered have no value, which the error names
    d = make_nab(ss.uniform(0.5, 0.95), init=None, rate=ss.peryear(2))
    with pytest.raises(ValueError, match='nab'):
        ss.Sim(d, connectors=ss.modify(d.rel_sus, multiply=1 - d.nab, name='imm'), demographics=demog(), n_agents=1000, start=2000, stop=2005, **kw).run()
    return


@sc.timer()
def test_attr_grown_by_code():
    """ A trait of people made by code that sets their age afterwards (people.grow() then people.age[...] = ..., as Starsim 3's pregnancy did) is drawn from the age it set (it was drawn from a default adult age) """
    class Maker(ss.Demographics):
        def step(self):
            new = self.sim.people.grow(20)
            self.sim.people.age[new] = -0.5
    S, I = ss.states('S I')
    d = ss.Disease(S, I, inf=S >> I @ ss.peryear(0), age0=ss.attr(ss.constant(v=lambda sim, uids: sim.people.age[uids])), name='d')
    sim = ss.Sim(d, demographics=Maker(), n_agents=1000, start=2000, stop=2002, unit='year', **kw).run()
    new = sim.people.auids[sim.people.auids >= 1000]
    assert len(new) and np.all(sim.diseases.d.age0.raw[new] == -0.5)
    return

if __name__ == '__main__':
    sc.options(interactive=False)
    test_hiv_cd4_ode()
    test_hpv_nab()
    test_hpv_nab_ode()
    test_trait_ode()
    test_attr_crn()
    test_attr_errors()
    test_attr_dist_par()
    test_attr_births()
    test_attr_grown_by_code()
