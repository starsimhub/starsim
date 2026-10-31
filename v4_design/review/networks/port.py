"""
STIsim networks on Starsim v4: a stress test.

A. StructuredSexual as far as v4's declarative network options go (ss.MFNet layers)
B. StructuredSexual as a v4 network subclass, reusing STIsim's own code
C. Per-act transmission, directional beta, condoms
D. Network-derived conditions: partner notification, ">=2 partners in the last year"
E. solver='auto' and describe()
F. Timing: 10k agents, 20 years, monthly

Run: conda run -n starsim python port.py
"""
import warnings
import numpy as np
import sciris as sc
import starsim as ss
import stisim as sti

warnings.filterwarnings('ignore', message='.*non-initialized Arr.*')
kw = dict(start=2000, stop=2020, dt=ss.months(1), verbose=0)
results = sc.objdict()


def attempt(label, func):
    """ Run something that may fail; record the error rather than stopping """
    try:
        out = func()
        print(f'  [ok]  {label}' + (f': {out}' if out is not None else ''))
        return out
    except Exception as e:
        msg = f'{type(e).__name__}: {str(e).splitlines()[0][:250]}'
        print(f'  [ERR] {label}: {msg}')
        results[label] = msg
        return None


#%% Shared disease: HIV-like, declared (acute -> latent -> AIDS -> death), per-act beta per direction
def make_hiv(nets=('structuredsexual',), m2f=0.008, f2m=0.004, extra=None, **kwargs):
    """ beta must name every network in the sim (E403 otherwise), so bookkeeping networks get 0 """
    beta = {n: [m2f, f2m] for n in nets}
    beta.update(extra or {})

    class HIV(ss.Disease):
        S = ss.State()
        A = ss.State(infectious=True, rel_trans=6.0) # Acute
        L = ss.State(infectious=True)                # Latent
        D = ss.State(infectious=True, rel_trans=3.0) # AIDS
        infection = ss.transmission('S -> A', beta=beta)
        acute     = ss.progression('A -> L', dur=ss.months(3))
        latent    = ss.progression('L -> D', dur=ss.lognorm_ex(mean=ss.years(8), std=ss.years(3)))
        aids      = ss.death('D ->', dur=ss.years(2))
        infected  = A + L + D # A result: number infected

    return HIV(init_prev=0.05, eff_condom=0.9, name='hiv', **kwargs)


#%% A. Declarative: StructuredSexual approximated by ss.MFNet layers
print('\nA. StructuredSexual from declarative ss.MFNet layers')

def sex_dep(f, m):
    """ A sex-specific value, as a callable distribution parameter """
    return lambda self, sim, uids: np.where(sim.people.female[uids], f, m)

def make_layers(condoms=True):
    debut = ss.lognorm_ex(mean=sex_dep(20, 21), std=3) # STIsim: debut_f, debut_m
    acts  = ss.lognorm_ex(mean=ss.freqperyear(80), std=ss.freqperyear(30))
    layers = [
        ss.MFNet(name='stable', n_contacts=ss.bernoulli(0.55), max_contacts=1,
                 duration=ss.lognorm_ex(mean=ss.years(10), std=ss.years(3)), debut=debut, acts=acts),
        ss.MFNet(name='casual', n_contacts=0.15, concurrency=0.03, participation=ss.bernoulli(p=sex_dep(0.15, 0.2)), # participation ~ medium+high risk
                 duration=ss.lognorm_ex(mean=ss.years(1), std=ss.years(1)), debut=debut, acts=acts),
        ss.MFNet(name='sw', n_contacts=1.0, participation=ss.bernoulli(p=sex_dep(0.05, 0.12)), # FSW/clients: degree target is the same for both sexes
                 duration=ss.months(1), debut=debut, acts=ss.poisson(lam=ss.freqperyear(24))),
    ]
    return layers

use = dict(stable=(0.05, 0.2), casual=(0.2, 0.5), sw=(0.4, 0.8)) # Condom use by layer, 2000 and 2020

def run_layers(condoms=None):
    layers = make_layers()
    if condoms == 'expr': # Condom use as a per-act beta multiplier, an expression of a time series: beta*(1 - eff*use(t))
        ts = {k: ss.timeseries({2000: a, 2020: b}, mode='linear') for k,(a,b) in use.items()}
        beta = {k: [0.008*(1 - 0.9*u), 0.004*(1 - 0.9*u)] for k,u in ts.items()}
    elif condoms == 'precomputed': # The same, worked out by hand: one time series per layer and direction
        ts = lambda b0,a,b: ss.timeseries({2000: b0*(1 - 0.9*a), 2020: b0*(1 - 0.9*b)}, mode='linear')
        beta = {k: [ts(0.008,a,b), ts(0.004,a,b)] for k,(a,b) in use.items()}
    else:
        beta = {k: [0.008, 0.004] for k in ['stable', 'casual', 'sw']}
    hiv = make_hiv(nets=(), extra=beta)
    sim = ss.Sim(hiv, networks=layers, n_agents=5000, demographics=[ss.Births(birth_rate=ss.peryear(25)), ss.Deaths(death_rate=ss.peryear(10))], **kw).run()
    for name in ['stable', 'casual', 'sw']:
        r = sim.networks[name].results
        print(f'    {name:7s} mean degree {r.mean_degree[-60:].mean():.2f}, concurrency {r.concurrency[-60:].mean():.3f}, edges {len(sim.networks[name])}')
    sw = sim.networks.sw
    uids = sw.people_seeking(sim.people)
    deg, fem = sw.degrees(uids), sim.people.female[uids]
    print(f'    sw degree: FSW {deg[fem].mean():.2f} clients/step, clients {deg[~fem].mean():.2f} FSW/step (STIsim: FSW see many clients, clients ~1)')
    return sim

simA0 = attempt('A1 MFNet layers, no condoms', lambda: run_layers())
attempt('A2a condoms: beta = 0.008*(1 - 0.9*ss.timeseries(...))', lambda: run_layers('expr') and 'ran')
simA = attempt('A2b condoms: beta = ss.timeseries of beta*(1-eff*use), per layer and direction', lambda: run_layers('precomputed'))
for sim_,lab in [(simA0, 'no condoms'), (simA, 'condoms')]:
    if sim_ is not None:
        print(f'    HIV prevalence 2020 ({lab}): {sim_.results.hiv.prevalence[-1]:.3f}')

# Things StructuredSexual does that MFNet has no option for: record the errors
attempt('A3 MFNet(match=dict(risk=0.9)) (risk-assortative)', lambda: ss.Sim(make_hiv(nets=('mf',)), networks=ss.MFNet(name='mf', n_contacts=0.7, match=dict(risk=0.9)), n_agents=1000, **kw).init() and 'accepted')
attempt('A4 MFNet n_contacts by risk group (callable mean)',
        lambda: ss.Sim(make_hiv(nets=('mf',)), networks=ss.MFNet(name='mf', n_contacts=ss.poisson(lam=lambda self, sim, uids: np.full(len(uids), 0.7))), n_agents=1000, **kw).init() and 'accepted')
attempt('A5 PartnerNet as MF (no sex): match=dict(female=0.0)',
        lambda: (lambda s: f'share of same-sex partnerships {np.mean(s.people.female.raw[s.networks.pn.edges.p1] == s.people.female.raw[s.networks.pn.edges.p2]):.2f}')(
            ss.Sim(make_hiv(nets=('pn',)), networks=ss.PartnerNet(name='pn', n_contacts=0.7, dur=ss.years(2), match=dict(female=0.0)), n_agents=2000, **kw).run()))

if simA0 is not None:
    attempt('A6 net.check() on the casual layer', lambda: '\n' + str(simA0.networks.casual.check(n_runs=2, output=True)))


#%% B. StructuredSexual as a v4 network subclass, reusing STIsim
print('\nB. sti.StructuredSexual subclassed for v4 declared diseases')

class StructuredSexual(sti.StructuredSexual):
    """
    STIsim's StructuredSexual, unchanged except:
      - net_beta reads eff_condom from a declared disease (an ss.par) and undoes the engine's
        conversion of a per-act probability into 1 - exp(-beta*dt) (see NOTES, bug 1)
      - partners_12m: current partners + partnerships ended in the last year (needs PriorPartners)
    """
    def __init__(self, pars=None, **kwargs):
        kwargs.setdefault('name', 'structuredsexual')
        super().__init__(pars=pars, **kwargs)
        self.define_states(ss.FloatArr('partners_12m', default=0))
        return

    def net_beta(self, disease_beta=None, uids=None, disease=None):
        engine = getattr(disease, 'engine', None)
        if engine is not None: # Declared disease: the engine passed 1 - exp(-beta*dt); recover the per-act beta
            disease_beta = -np.log1p(-disease_beta)/engine.dt
        eff = disease.pars.get('eff_condom', 0)
        eff = eff.value if isinstance(eff, ss.par) else eff
        if uids is None: uids = Ellipsis
        c = self.edges.condoms[uids]
        acts = self.edges.acts[uids]
        p_none = (1 - disease_beta*(1 - eff))**(acts*c) * (1 - disease_beta)**(acts*(1 - c))
        return (1 - p_none)*self.edges.beta[uids]

    def step(self):
        super().step()
        n = len(self.sim.people.uid.raw)
        deg = np.bincount(self.edges.p1, minlength=n)[:n] + np.bincount(self.edges.p2, minlength=n)[:n]
        prior = self.sim.networks.get('priorpartners')
        if prior is not None and len(prior):
            deg = deg + np.bincount(prior.edges.p1, minlength=n)[:n] + np.bincount(prior.edges.p2, minlength=n)[:n]
        self.partners_12m.raw[:] = deg
        return


condom_data = {(0,0): dict(year=[2000, 2020], val=[0.05, 0.2]), (1,1): dict(year=[2000, 2020], val=[0.2, 0.5]),
               (0,1): dict(year=[2000, 2020], val=[0.2, 0.5]), (1,0): dict(year=[2000, 2020], val=[0.2, 0.5]),
               ('fsw','client'): dict(year=[2000, 2020], val=[0.4, 0.8])}

def make_nets(sub=True):
    cls = StructuredSexual if sub else sti.StructuredSexual
    return [cls(condom_data=condom_data, recall_prior=True), sti.PriorPartners()]

# Unmodified v3 class with a declared disease: fails
attempt('B1 unmodified sti.StructuredSexual + declared disease (no eff_condom)',
        lambda: ss.Sim(ss.Disease(infection=ss.transmission('S -> I', beta=dict(structuredsexual=[0.008, 0.004])), init_prev=0.05, name='hiv'), networks=sti.StructuredSexual(), n_agents=1000, **kw).run() and 'ran')
attempt('B2 unmodified sti.StructuredSexual + declared disease with eff_condom=0.9',
        lambda: ss.Sim(make_hiv(), networks=sti.StructuredSexual(), n_agents=1000, **kw).run() and 'ran')
attempt('B3 beta dict without the bookkeeping network priorpartners',
        lambda: ss.Sim(make_hiv(), networks=make_nets(), n_agents=1000, **kw).run() and 'ran')

simB = attempt('B4 subclass + declared HIV', lambda: ss.Sim(make_hiv(extra=dict(priorpartners=0)), networks=make_nets(), n_agents=5000,
                                                           demographics=[ss.Births(birth_rate=ss.peryear(25)), ss.Deaths(death_rate=ss.peryear(10))], **kw).run())
if simB is not None:
    net = simB.networks.structuredsexual
    et = {v:k for k,v in net.edge_types.items()}
    types = np.unique(net.edges.edge_type, return_counts=True)
    print('    edges by type:', {et[int(k)]: int(v) for k,v in zip(*types)})
    print(f'    mean condom use on edges 2020: {net.edges.condoms.mean():.2f}; HIV prevalence 2020: {simB.results.hiv.prevalence[-1]:.3f}')
    print(f'    share of sexually active with >=2 partners in last 12m: {np.mean(net.partners_12m[net.over_debut.uids] >= 2):.3f}')


#%% C. Per-act beta: is a bare beta used as a per-act probability?
print('\nC. Per-act beta on a sexual network (bug check)')

def passed_beta(unit, beta=0.5):
    seen = {}
    class Spy(ss.MFNet):
        def net_beta(self, disease_beta=None, inds=None, disease=None):
            seen['b'] = disease_beta
            return super().net_beta(disease_beta, inds, disease)
    d = ss.Disease(infection=ss.transmission('S -> I', beta=dict(mf=[beta, beta])), init_prev=0.05, name='d')
    sim = ss.Sim(d, networks=Spy(name='mf'), n_agents=500, start='2000-01-01', stop='2001-01-01', dt=ss.months(1), unit=unit, verbose=0).run()
    return f'beta=0.5 per act reaches net_beta as {seen["b"]:.4f} (sim unit {sim.unit})'

attempt('C1 unit=month', lambda: passed_beta('month'))
attempt('C2 unit=year ', lambda: passed_beta('year'))

def prev_by_unit(unit):
    d = ss.Disease(infection=ss.transmission('S -> I', beta=dict(mf=[0.008, 0.004])), init_prev=0.05, name='d')
    net = ss.MFNet(name='mf', n_contacts=0.7, duration=ss.lognorm_ex(mean=ss.years(5), std=ss.years(3)))
    sim = ss.Sim(d, networks=net, n_agents=5000, unit=unit, rand_seed=1, start='2000-01-01', stop='2020-01-01', dt=ss.months(1), verbose=0).run()
    return f'{sim.results.d.prevalence[-1]:.3f} ({len(sim.results.d.prevalence)} points)' # Bare start=2000 with unit='month' would mean month 2000
attempt('C3 SI prevalence 2020, same model, unit=month', lambda: prev_by_unit('month'))
attempt('C4 SI prevalence 2020, same model, unit=year ', lambda: prev_by_unit('year'))


#%% D. Network-derived conditions: PN and ">=2 partners in last 12 months"
print('\nD. Network-derived conditions for interventions')

def make_ng():
    """ Gonorrhea-like SIS. 'cure' and 'pn_cure' never happen by themselves (rate 0): they exist only so each
    treatment has its own event, which on= can select (clearance I -> S would include natural clearance) """
    return ss.Disease(
        infection = ss.transmission('S -> I', beta=dict(structuredsexual=[0.4, 0.3], priorpartners=0)),
        clearance = ss.progression('I -> S', dur=ss.months(6)),
        cure      = ss.progression('I -> S', rate=ss.peryear(0)),
        pn_cure   = ss.progression('I -> S', rate=ss.peryear(0)),
        init_prev=0.05, eff_condom=0.9, name='ng')

def run_pn(pn=None, eligible=None, label=''):
    ng = make_ng()
    drug = ss.Treatment(efficacy=0.95, transition='cure')
    test = ss.Test(sensitivity=0.9, specificity=0.98)
    intvs = [ss.screen(test, coverage=0.02, interval=ss.months(1), eligible=eligible,
                       on_positive=ss.treat(drug, name='tx'), name='screening')]
    trace = ['structuredsexual', 'priorpartners'] # Current partners + partnerships ended in the last year
    if pn == 'shared': # PN triggered by the index cure, but partners are cured by the same transition: they trigger PN in turn
        intvs.append(ss.treat(drug, on=ng.cure, trace=trace, coverage=0.5, name='pn'))
    elif pn == 'separate': # A second dummy transition for partners stops the cascade
        intvs.append(ss.treat(ss.Treatment(efficacy=0.95, transition='pn_cure'), on=ng.cure, trace=trace, coverage=0.5, name='pn'))
    sim = ss.Sim(ng, networks=make_nets(), interventions=intvs, n_agents=5000, **kw).run()
    r = sim.results
    out = f'{label}: NG prevalence 2020 {r.ng.n_I[-1]/(r.ng.n_I[-1]+r.ng.n_S[-1]):.3f}, index diagnosed {r.screening.diagnosed.sum():.0f}'
    if pn: out += f', partners treated {r.pn.delivered.sum():.0f} ({r.pn.delivered.sum()/r.screening.diagnosed.sum():.1f} per index)'
    return out

attempt('D1 baseline (no PN)', lambda: run_pn(label='no PN'))
attempt('D2 PN, on=index cure, same cure transition', lambda: run_pn('shared', label='PN shared'))
attempt('D3 PN, separate dummy transition for partners', lambda: run_pn('separate', label='PN separate'))
def pn_list():
    ng = make_ng()
    drug = ss.Treatment(efficacy=0.95, transition='cure')
    scr = ss.screen(ss.Test(sensitivity=0.9), coverage=0.02, interval=ss.months(1),
                    on_positive=[ss.treat(drug, name='tx'), ss.treat(drug, trace='structuredsexual', coverage=0.5, name='pn')])
    return ss.Sim(ng, networks=make_nets(), interventions=scr, n_agents=2000, **kw).run() and 'ran'
attempt('D4 on_positive=[treat index, treat partners] (a list)', pn_list)

net_tmp = StructuredSexual()
attempt('D5 eligible=net.partners_12m >= 2 (lazy condition on a network array)',
        lambda: run_pn(eligible=net_tmp.partners_12m >= 2, label='lazy cond'))
attempt('D6 eligible=function of sim', lambda: run_pn(eligible=lambda sim: sim.networks.structuredsexual.partners_12m >= 2, label='>=2 partners, function'))


#%% E. Solver and describe
print('\nE. solver="auto" and describe()')
if simB is not None:
    print(f'    simB.solver = {simB.solver!r}; reason: {simB.solver_reason}')
    txt = simB.describe(output=True)
    for line in txt.splitlines():
        if any(k in line.lower() for k in ['network', 'structuredsexual', 'priorpartners', 'beta', 'infection', 'mix', 'route', 'solver', 'agents']):
            print('    |', line[:160])
attempt('E1 solver="auto", total_pop (no n_agents), StructuredSexual',
        lambda: (lambda s: f'solver={s.solver!r}, reason: {s.solver_reason}')(ss.Sim(make_hiv(extra=dict(priorpartners=0)), networks=make_nets(), total_pop=5000, **kw).init()))
attempt('E2 solver="ode" with StructuredSexual', lambda: ss.Sim(make_hiv(extra=dict(priorpartners=0)), networks=make_nets(), total_pop=5000, solver='ode', **kw).run() and 'ran')
attempt('E3 solver="ode" with MFNet layers', lambda: (lambda s: f'ran; notes: {s.notes}')(ss.Sim(make_hiv(nets=('stable', 'casual', 'sw')), networks=make_layers(), total_pop=5000, solver='ode', **kw).run()))


#%% F. Timing
print('\nF. Timing: 10k agents, 2000-2020, monthly')
tk = dict(n_agents=10_000, start=2000, stop=2020, dt=ss.months(1), verbose=0)
dem = lambda: [ss.Births(birth_rate=ss.peryear(25)), ss.Deaths(death_rate=ss.peryear(10))]
def timed(label, make):
    sim = make()
    T = sc.timer()
    sim.run()
    t = T.toc(output=True)
    print(f'    {label:55s} {t:5.1f} s')
    return t
attempt('F0a', lambda: timed('networks only: StructuredSexual subclass + PriorPartners', lambda: ss.Sim(networks=make_nets(), demographics=dem(), **tk)))
attempt('F0b', lambda: timed('networks only: 3 MFNet layers', lambda: ss.Sim(networks=make_layers(), demographics=dem(), **tk)))
attempt('F1', lambda: timed('v3: sti.HIV + sti.StructuredSexual (legacy ss.Infection path)',
                            lambda: ss.Sim(diseases=sti.HIV(init_prev=0.05), networks=[sti.StructuredSexual(condom_data=condom_data)], demographics=dem(), **tk)))
attempt('F2', lambda: timed('v4: declared HIV + StructuredSexual subclass',
                            lambda: ss.Sim(make_hiv(extra=dict(priorpartners=0)), networks=make_nets(), demographics=dem(), **tk)))
attempt('F3', lambda: timed('v4: declared HIV + 3 MFNet layers',
                            lambda: ss.Sim(make_hiv(nets=('stable', 'casual', 'sw')), networks=make_layers(), demographics=dem(), **tk)))
attempt('F4', lambda: timed('v4: declared HIV + one targeted MFNet',
                            lambda: ss.Sim(make_hiv(nets=('mf',)), networks=ss.MFNet(name='mf', n_contacts=0.7, concurrency=0.05, duration=ss.lognorm_ex(mean=ss.years(5), std=ss.years(3))), demographics=dem(), **tk)))

print('\nErrors recorded:')
for k,v in results.items():
    print(f'  {k}: {v}')
