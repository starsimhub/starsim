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


class CheckCD4(ss.Analyzer):
    """ Every infected agent has a CD4 count at every step """
    def step(self):
        hiv = self.sim.diseases.hiv
        inf = (hiv.acute | hiv.latent | hiv.falling | hiv.on_art | hiv.post_art).uids
        assert not np.isnan(hiv.engine.attr_values(hiv.graph.attrs.cd4, inf)).any()
        return


@sc.timer()
def test_hiv_cd4():
    sc.heading('HIV: CD4 drawn at infection, recorded at ART, derived by state; deaths by CD4 with no analyzer')
    kw2 = dict(start=1990, stop=2010, dt=ss.months(1), unit='year', **kw)
    hiv, mods = make_hiv()
    sim = ss.Sim(hiv, connectors=mods, analyzers=CheckCD4(), n_agents=5000, **kw2).run()
    assert sim.solver == 'abm'
    r = sim.results.hiv
    by = r.cd4_death.by('cd4')
    assert by.columns == ['0-49', '50-199', '200-349', '350-499', '500+']
    assert np.allclose(by.values.sum(axis=1), r.cd4_death.values) and r.cd4_death.sum() > 50
    assert by.values[:, 0].sum() > 0 and by.values[:, -1].sum() > 0 # Deaths at low CD4 (the highest rate) and high (on ART)
    assert np.allclose(r.on_art.by('cd4').values.sum(axis=1), r.on_art.values) and r.on_art.select(cd4='500+')[-1] > r.on_art.select(cd4='0-49')[-1]
    assert np.allclose(r.cd4_death.cum.by('cd4').values[-1], by.values.sum(axis=0))
    h = sim.diseases.hiv
    ever = ~np.isnan(h.cd4_preart.raw[h.cd4_preart.auids]) # Recorded at the start of ART: in the range of CD4; the potential is above it
    pre, pot = h.cd4_preart.values[ever], h.cd4_potential.values[ever]
    assert ever.sum() > 100 and pre.min() >= 1 and pre.max() < 1000 and np.all(pot >= pre - 1e-9)
    on = h.on_art.uids
    assert np.all(h.cd4[on] >= h.cd4_preart[on] - 1e-9) # Recovering on ART
    return sim


@sc.timer()
def test_attr_rules():
    sc.heading('Attributes: counts, first and min, traits, events at placement, modifiers, and a function each step')
    calls = []
    def hook(sim): # Called each step: a function given to the disease replaces a hook on an event (ss.func)
        d = sim.diseases.sis
        uids = d.infection.uids # Who was infected this step
        assert np.all(d.I[uids] | np.isin(uids, d.recovery.uids)) # In the new state, unless they have left it already (a hook at the event saw them as they moved)
        assert np.all(d.n_inf[uids] >= 1) # After the attributes the event sets
        calls.append(len(uids))
    def make(births=False):
        S, I = ss.states('S I')
        infection = S >> I @ ss.perday(0.2)*I
        recovery = ss.Transition(I, S, dur=ss.days(10))
        extra = dict(birth=ss.null >> S @ ss.perday(5)) if births else {}
        return ss.Disease(S, I, infection=infection, recovery=recovery,
            n_inf   = ss.attr(1, on=infection, combine='sum'),           # A count of events
            first   = ss.attr(ss.normal(5, 1), on=infection, combine='first'),
            low     = ss.attr(ss.normal(5, 1), on=infection, combine='min', init=10),
            avg     = ss.attr(I.time, on=recovery, combine=lambda old, new: np.where(np.isnan(old), new, (old + new)/2)), # Recorded: the time just completed
            rel     = ss.attr(ss.uniform(0.5, 1.5)),                     # A trait
            dx      = hook,
            init_prev=0.01, name='sis', **extra)
    kw2 = dict(start=0, stop=100, unit='day', **kw)
    sim = ss.Sim(make(), n_agents=5000, **kw2).run()
    d = sim.diseases.sis
    r = sim.results.sis
    n = d.n_inf.values
    seeds = 50
    assert n.sum() == r.infection.sum() + seeds # Each infection counted once, and the seeds count as infected at the start
    assert sum(calls) == r.infection.sum() # The function sees each infection in the run
    had = n > 0
    assert np.all(np.isnan(d.first.values[~had])) and not np.isnan(d.first.values[had]).any()
    assert np.all(d.low.values[~had] == 10) and np.all(d.low.values[had] < 10)
    avg = d.avg.values[~np.isnan(d.avg.values)]
    assert len(avg) > 100 and 7 < avg.mean() < 13 # Times in I of about 10 days
    rel = d.rel.values
    assert rel.min() >= 0.5 and rel.max() <= 1.5 and abs(rel.mean() - 1) < 0.02

    # Traits of people born during the run; a modifier of an attribute, read by a rate
    sim = ss.Sim(make(births=True), n_agents=2000, **kw2).run()
    d = sim.diseases.sis
    assert len(d.rel.auids) > 2000 and not np.isnan(d.rel.values).any()
    def faster(rel_mult):
        S, I = ss.states('S I')
        rel = ss.attr(ss.uniform(0.5, 1.5))
        d = ss.Disease(S, I, onset=S >> I @ ss.perday(0.002)*rel, rel=rel, name='d')
        mods = ss.modify(d.rel, multiply=rel_mult, where=d.S, name='m')
        return ss.Sim(d, connectors=mods, n_agents=2000, **kw2).run().results.d.onset.sum()
    assert faster(2.0) > 1.5*faster(1.0)
    return sim


@sc.timer()
def test_burnin():
    sc.heading('Seeds part way through their natural history: a sim that starts earlier (burn-in), which replaces ss.spread')
    kw2 = dict(stop=1996, dt=ss.months(1), unit='year', n_agents=10_000, **kw)
    out = sc.objdict()
    for label,start in [('plain', 1990), ('burnin', 1980)]:
        hiv, mods = make_hiv(art=ss.peryear(0))
        r = ss.Sim(hiv, connectors=mods, start=start, **kw2).run().results.hiv
        i = (1990 - start)*12 # The step of 1990
        out[label] = sc.objdict(falling=r.falling[i], prog=r.progression[i:i+37].sum(), aids=r.aids_death[i:i+37].sum())
    p, b = out.plain, out.burnin
    assert p.prog == 0 and p.falling == 0 # Without it, nobody progresses for years
    assert b.falling > 50 and b.prog > 100 and b.aids > 50, out
    return out


if __name__ == '__main__':
    sc.options(interactive=False)
    test_hiv_cd4()
    test_attr_rules()
    test_burnin()
