"""
Sidebar: genotype as a strain dimension of ONE disease (instead of one disease per genotype, as port.py).
Checks which genotype-specific intervention features work. Run: conda run -n starsim python genotype_dim.py
"""
import numpy as np
import starsim as ss

ppl = ss.People(1)
G = ['hpv16', 'hpv18', 'hi5']

def make():
    S, I, CIN, CA = ss.states('S I CIN CA')
    hpv = ss.Disease(
        S, I, CIN, CA,
        infection = ss.transmission('S -> I', beta=dict(mfnet=0.15), source=I | CIN | CA),
        clear     = ss.progression('I -> S', p=0.75, dur=ss.years(1.5)),
        dysplasia = ss.progression('I -> CIN', p=0.25, dur=ss.years(2), where=ppl.female),
        regress   = ss.progression('CIN -> S', p=0.9, dur=ss.years(4)),
        invasion  = ss.progression('CIN -> CA', p=0.1, dur=ss.years(12)),
        cancer_death = ss.death('CA ->', rate=ss.peryear(0.2)),
        births = ss.birth('-> S', rate=ss.peryear(0.025)), deaths = ss.death(rate=ss.peryear(0.01)),
        init_prev = 0.15, name = 'hpv',
    )
    return hpv.stratify(genotype=G, split=[0.4, 0.4, 0.2])

kw = lambda: dict(n_agents=5000, networks=ss.MFNet(duration=ss.lognorm_ex(mean=ss.years(1), std=ss.years(1)), n_contacts=ss.poisson(0.8), concurrency=0.15),
                  start=1990, stop=2030, dt=ss.years(0.25), verbose=0, rand_seed=1)

def tryit(label, fn):
    try:
        print(f'OK   {label}: {fn()}')
    except Exception as e:
        print(f'FAIL {label}: {type(e).__name__}: {str(e)[:300]}')

def infections_by_genotype(sim):
    return sim.results.hpv.infection.by('genotype').values[sim.t.yearvec >= 2005].sum(axis=0).round()

base = ss.Sim(make(), **kw()).run()
tryit('baseline infections 2005-30 by genotype', lambda: infections_by_genotype(base))

# 1. Test sensitivity by state and genotype: works (one disease)
def t_test():
    hpv = make()
    dna = ss.Test(sensitivity=dict(I=ss.by(genotype=[0.9, 0.9, 0.5]), CIN=0.9, CA=0.9), specificity=0.98)
    via = ss.Test(sensitivity=dict(CIN=0.75, CA=1.0), specificity=0.85)
    abl = ss.Treatment(efficacy=0.936, transition='regress') # one treatment covers every genotype
    s = ss.screen(dna, coverage=0.3, interval=ss.years(1), start=2020, eligible=ppl.female & (ppl.age >= 30) & (ppl.age < 50),
                  on_positive=ss.screen(via, coverage=0.9, on_positive=ss.treat(abl, coverage=0.8)), name='s')
    sim = ss.Sim(hpv, interventions=s, **kw()).run()
    r = sim.results
    return dict(screened=r.s.delivered.sum(), dna_pos=r.s.diagnosed.sum(), via_pos=r.s_positive.diagnosed.sum(), treated=r.s_positive_positive.delivered.sum())
tryit('DNA (genotype-specific) -> VIA (stage-specific) -> ablation, all declarative', t_test)

# 2. Vaccine efficacy by genotype: accepted, but keyed on the PERSON's genotype level, which for a susceptible is stale
def t_vx():
    hpv = make()
    vx = ss.Vaccine(efficacy=ss.by(genotype=[1.0, 0.0, 0.0])) # meant: protects against hpv16 only
    v = ss.vaccinate(vx, coverage=1.0, years=[2005], eligible=ppl.female, name='v')
    sim = ss.Sim(hpv, interventions=v, **kw()).run()
    d = sim.diseases.hpv
    holds = sim.people.has_vx.raw.astype(bool)
    lvl = sim.people.genotype.raw if 'genotype' in sim.people.states else None
    after = d.ti_I.raw > 4*15 # infected since 2005 (last infection)
    out = dict(infections_by_genotype=infections_by_genotype(sim))
    if lvl is not None:
        out['holders infected after vx, by their genotype level'] = [int((holds & after & (lvl == k)).sum()) for k in range(3)]
    return out
tryit('Vaccine(efficacy=ss.by(genotype=[1,0,0]))', t_vx)

# 3. The faithful form: vaccination as a transfer along a vx dimension, adjust infection by (vx x genotype)
def t_transfer2d():
    hpv = make()
    hpv.stratify(vx=[False, True], split=[1, 0], adjust={'infection': ss.by([[1, 1, 1], [0, 0, 0.5]], dims=['vx', 'genotype'])})
    sim = ss.Sim(hpv, interventions=ss.transfer(vx=True, coverage=1.0, years=[2005], eligible=ppl.female), **kw()).run()
    return infections_by_genotype(sim)
tryit('stratify(vx) + adjust infection by ss.by(dims=[vx, genotype])', t_transfer2d)
