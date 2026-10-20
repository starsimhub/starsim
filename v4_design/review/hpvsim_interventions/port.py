"""
HPVsim v3 interventions and products, ported to Starsim v4.

Natural history is deliberately simplified (someone else ports the real one):
three genotypes, each its own ss.Disease (as in HPVsim v3, one module per genotype,
so coinfection is allowed):  S -> I -> CIN -> CA -> dead, with clearance I -> S and
regression CIN -> S. 'excised' (CIN -> I, rate 0) exists only as the target of
treatment: the lesion is removed, the infection is not.

Programs: routine girls' vaccination (9-14, coverage scale-up), catch-up campaign,
1 vs 2 doses, WHO screen-and-treat (a) HPV DNA -> ablation-eligible? -> ablation
else excision, (b) HPV DNA -> VIA triage -> treat, (c) screen every 5 years ages
30-50 (memory of last screen), loss to follow-up at each step, capacity (dropped vs
queued), radiation, therapeutic vaccine. Products come from HPVsim's CSVs.

Run: conda run -n starsim python port.py
"""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

DATA = '/Users/robynstuart/gf/hpvsim/hpvsim/data'
G = ['hpv16', 'hpv18', 'hi5']      # genotypes; names match the HPVsim CSVs
ppl = ss.People(1)                 # for lazy conditions on people
women = lambda lo, hi: ppl.female & (ppl.age >= lo) & (ppl.age < hi)
T0 = 2020                          # programs start


#%% Simplified natural history

nh = dict( # p_cin, dur_cin_onset, p_invasion, dur_invasion
    hpv16 = (0.30, 2.0, 0.08, 12.0),
    hpv18 = (0.25, 2.0, 0.06, 14.0),
    hi5   = (0.15, 3.0, 0.03, 18.0),
)

def make_genotype(name, beta=0.05, demography=False):
    p_cin, d_cin, p_ca, d_ca = nh[name]
    # ss.Births()/ss.Deaths() leave newborns in NO state of a declared disease (never infected), so births and
    # background deaths are declared on one genotype; newborns then enter S of every genotype.
    dem = dict(births=ss.birth('-> S', rate=ss.peryear(0.025)), deaths=ss.death(rate=ss.peryear(0.01))) if demography else {}
    S, I, CIN, CA = ss.states('S I CIN CA')
    return ss.Disease(
        S, I, CIN, CA,
        infection    = ss.transmission('S -> I', beta=dict(mfnet=beta), source=I | CIN | CA),
        clear        = ss.progression('I -> S', p=1-p_cin, dur=ss.years(1.5)),
        dysplasia    = ss.progression('I -> CIN', p=p_cin, dur=ss.years(d_cin), where=ppl.female),
        regress      = ss.progression('CIN -> S', p=1-p_ca, dur=ss.years(4)),
        invasion     = ss.progression('CIN -> CA', p=p_ca, dur=ss.years(d_ca)),
        cancer_death = ss.death('CA ->', rate=ss.peryear(0.2)),
        excised      = ss.Transition(CIN, I, rate=ss.peryear(0)), # only moved along by treatment
        init_prev    = 0.08,
        name         = name,
        **dem,
    )


#%% Products, from HPVsim's CSVs

csv = sc.objdict({k: pd.read_csv(f'{DATA}/products_{k}.csv') for k in ['vx', 'dx', 'tx', 'txvx']})
state_map = dict(susceptible='S', precin='I', cin='CIN', cancerous='CA') # HPVsim v3 states -> these ('latent' has no counterpart)


def vaccine(name, primary='hpv16', sterilizing_p=0.95):
    """
    A prophylactic vaccine: one ss.Vaccine for the primary genotype, plus one ss.modify per
    other genotype keyed on holding it (a v4 product acts on exactly one disease).
    v3's all-or-nothing 'take' (sterilizing_p) has no v4 counterpart, so it is folded into a leaky efficacy.
    """
    rel = csv.vx[csv.vx.name == name].set_index('genotype').rel_imm
    eff = lambda g: float(rel[g])*(sterilizing_p + (1 - sterilizing_p)*sterilizing_p)
    vx = ss.Vaccine(efficacy=eff(primary), name=name, cost=5.0)
    holds = ss.Cond('people', f'has_{name}') # no public spelling for "holds this product"
    cross = [ss.modify(f'{g}.rel_sus', multiply=1 - eff(g), where=holds, name=f'{name}_{g}') for g in G if g != primary]
    return vx, cross


def p_positive(name):
    """
    HPVsim's dx table as a function of the sim: P(positive) per agent = 1 - prod(1 - p[state, genotype])
    over every (genotype, state) the agent is in. This is v3's "hierarchy-min across genotypes".
    v4 ss.Test sensitivity dicts are per state of ONE disease, so a cross-genotype test must be a function.
    """
    df = csv.dx[(csv.dx.name == name) & (csv.dx.result == 'positive')]
    table = {(r.genotype, state_map[r.state]): r.probability for r in df.itertuples() if r.state in state_map}
    def prob(sim):
        n = len(sim.people.uid.raw)
        p_neg = np.ones(n)
        susc_all = np.ones(n, bool)
        for g in G:
            dis = sim.diseases[g]
            susc_all &= dis.S.raw
            for s in ['I', 'CIN', 'CA']:
                p = table.get((g, s), table.get(('all', s), 0.0))
                p_neg[dis[s].raw] *= 1 - p
        p_neg[susc_all] *= 1 - table.get(('all', 'S'), 0.0)
        return 1 - p_neg
    return prob

everyone = lambda sim: np.ones(len(sim.people.uid.raw), bool) # so the Test's sensitivity function applies to all

def dx(name):
    return ss.Test(sensitivity=p_positive(name), detects=everyone, name=name)


def tx_efficacy(name, g, state='cin'):
    df = csv.tx[(csv.tx.name == name) & (csv.tx.state == state) & csv.tx.genotype.isin([g, 'all'])]
    return float(df.efficacy.iloc[0])


#%% Delivery helpers (the parts v4 could not say declaratively)

def treat_all_genotypes(product_name, coverage=1.0):
    """
    One ss.treat per genotype (a v4 Treatment cures along one transition of one disease), plus a
    follow-up function that hands the people a test sent to all of them. Uses Delivery internals
    (triggered, receive, selector) and looks deliveries up in the sim, since ss.Sim copies its modules.
    """
    txs = []
    for g in G:
        tx = ss.Treatment(efficacy=tx_efficacy(product_name, g), transition='excised', name=f'{product_name}_{g}_tx', cost=None)
        txs.append(ss.treat(tx, disease=g, eligible=ss.Cond('state', g, 'CIN'), coverage=coverage, name=f'{product_name}_{g}'))
        txs[-1].triggered = True # otherwise it also runs on its own, treating everyone infected
    def followup(sim, uids):
        for d in txs:
            mine = sim.interventions[d.name]
            mine.receive(mine.selector.to_selection(uids))
    followup.deliveries = txs
    return followup


class screen_mem(ss.screen):
    """ ss.screen that remembers when each person was last tested (v4 deliveries keep no such record) """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.define_states(ss.FloatArr('ti_screened', label='Time last screened'))

    def test(self, chosen):
        self.ti_screened[self.uids(chosen)] = self.sim.ti
        return super().test(chosen)


def due_for_screen(name, lo=30, hi=50, every=5):
    """ Eligible: women lo-hi not screened by `name` in the last `every` years. Returns UIDs. """
    def eligible(sim):
        p = sim.people
        u = p.auids
        last = sim.interventions[name].ti_screened[u]
        since = (sim.ti - last)*sim.t.dt_year
        ok = p.female[u] & (p.age[u] >= lo) & (p.age[u] < hi) & (np.isnan(last) | (since >= every))
        return u[ok]
    return eligible


class Queue(ss.Intervention):
    """ FIFO waiting list, like v3's treat_num: on_positive=queue.add, then deliver=queue.take """
    def __init__(self, per_step, still_eligible=None, **kwargs):
        super().__init__(**kwargs)
        self.per_step = per_step # in agents
        self.still_eligible = still_eligible
        self.waiting = []
        self.n_waiting = []
        return

    def add(self, sim, uids):
        q = sim.interventions[self.name]
        new = set(q.waiting)
        q.waiting += [int(u) for u in uids if int(u) not in new]

    def take(self, sim):
        q = sim.interventions[self.name]
        alive = sim.people.alive.raw
        q.waiting = [u for u in q.waiting if alive[u]]
        out, q.waiting = q.waiting[:q.per_step], q.waiting[q.per_step:]
        return ss.uids(np.array(out, dtype=int))

    def step(self):
        self.n_waiting.append(len(self.waiting))


#%% Programs

def vaccination(product='bivalent', doses=1, start=T0, catchup=True, dose2_boost=0.0):
    """
    Routine girls 9-14 with coverage scaled up 2020->2025, a one-off catch-up 15-25, and an optional
    second dose 6 months later. v3 'routine' (annual prob among eligible) = v4 rounds every year;
    v4 routine (no rounds) offers each girl once only, at the first step she is eligible.
    """
    vx, cross = vaccine(product)
    scaleup = ss.timeseries({start: 0.2, start + 5: 0.8}, mode='linear')
    intvs = [ss.vaccinate(vx, eligible=women(9, 15), coverage=scaleup, interval=ss.years(1), start=start, disease='hpv16', name='routine_vx')]
    if catchup:
        intvs.append(ss.vaccinate(vx, eligible=women(15, 26), coverage=0.5, years=[start], disease='hpv16', name='catchup_vx'))
    if doses == 2:
        # The same product cannot be given twice (v4 never delivers to holders), so dose 2 is another product
        dose2 = ss.Vaccine(efficacy=dose2_boost, name=f'{product}_dose2', cost=5.0)
        def due(sim):
            u = sim.people.auids
            got = sim.people[f'has_{product}'][u].astype(bool)
            t1 = sim.interventions.routine_vx.product.ti_delivered[u]
            return u[got & ((sim.ti - t1)*sim.t.dt_year >= 0.5)]
        intvs.append(ss.vaccinate(dose2, eligible=due, coverage=0.9, start=start, disease='hpv16', name='dose2_vx'))
    return intvs, cross


def algo_a(coverage=0.3, screen_capacity=1000, tx_capacity=None, queue=None):
    """ (a) HPV DNA -> ablation-eligible? -> ablation (else excision). LTFU: 80% attend treatment. """
    ablate, excise = treat_all_genotypes('ablation'), treat_all_genotypes('excision')
    eligible_for_ablation = ss.Test(sensitivity=0.9, specificity=0.9, detects=~cancer_any(), name='abl_elig') # positive = can be ablated
    tx_visit = ss.screen(eligible_for_ablation, coverage=0.8, capacity=tx_capacity, on_positive=ablate, on_negative=excise, disease='hpv16', name='tx_visit')
    if queue is not None: # waiting list instead of drop
        tx_visit = ss.screen(eligible_for_ablation, deliver=queue.take, on_positive=ablate, on_negative=excise, disease='hpv16', name='tx_visit')
    primary = ss.screen(dx('hpv'), coverage=coverage, interval=ss.years(1), start=T0, eligible=women(30, 50),
                        capacity=screen_capacity, on_positive=queue.add if queue else tx_visit, disease='hpv16', name='primary')
    intvs = [primary, tx_visit] + ([queue] if queue else []) + ablate.deliveries + excise.deliveries
    return intvs


def algo_b(coverage=0.3):
    """ (b) HPV DNA -> VIA triage (90% attend) -> ablation (80% attend) """
    ablate = treat_all_genotypes('ablation', coverage=0.8)
    via = ss.screen(dx('via'), coverage=0.9, on_positive=ablate, disease='hpv16', name='via_triage')
    primary = ss.screen(dx('hpv'), coverage=coverage, interval=ss.years(1), start=T0, eligible=women(30, 50),
                        on_positive=via, disease='hpv16', name='primary')
    return [primary] + ablate.deliveries


def algo_c(annual_prob=0.3):
    """ (c) HPV DNA every 5 years, ages 30-50: per-step uptake among those not screened in the last 5 years -> ablation """
    ablate = treat_all_genotypes('ablation', coverage=0.8)
    p_step = 1 - (1 - annual_prob)**0.25
    primary = screen_mem(dx('hpv'), coverage=p_step, interval=ss.years(0.25), start=T0, eligible=due_for_screen('primary'),
                         on_positive=ablate, disease='hpv16', name='primary')
    return [primary] + ablate.deliveries


def cancer_any():
    out = ss.Cond('state', G[0], 'CA')
    for g in G[1:]:
        out = out | ss.Cond('state', g, 'CA')
    return out


def radiation(coverage=0.5):
    """ v3 extends time to cancer death by ~18 months; v4: a held treatment that scales the death rate """
    out = []
    for g in G:
        rad = ss.Treatment(efficacy=dict(mortality=0.5), name=f'radiation_{g}')
        out.append(ss.treat(rad, eligible=ss.Cond('state', g, 'CA'), coverage=coverage, start=T0, disease=g, name=f'rad_{g}'))
    return out


def txvx(year=2030, coverage=0.5):
    """
    Therapeutic vaccine (v3 txvx2): clears CIN with genotype-specific efficacy, and protects against
    progression. Per genotype: a vaccine (efficacy against 'dysplasia', a named transition) delivered
    first, then a treatment to those who now hold it and have CIN, in the same step.
    """
    rel = csv.txvx[csv.txvx.name == 'txvx2'].set_index('genotype').rel_imm
    vx16 = ss.Vaccine(efficacy=dict(dysplasia=0.5*rel['hpv16'], invasion=0.5*rel['hpv16']), name='txvx')
    out = [ss.vaccinate(vx16, eligible=women(30, 50), coverage=coverage, years=[year], disease='hpv16', name='txvx_vx')]
    holds = ss.Cond('people', 'has_txvx')
    cross = [ss.modify(f'{g}.{tr}', multiply=1 - 0.5*rel[g], where=holds, name=f'txvx_{g}_{tr}') for g in G[1:] for tr in ['dysplasia', 'invasion']]
    for g in G:
        cure = ss.Treatment(efficacy=tx_efficacy('txvx2', g), transition='excised', name=f'txvx_tx_{g}')
        out.append(ss.treat(cure, eligible=holds & ss.Cond('state', g, 'CIN'), years=[year], disease=g, name=f'txvx_cure_{g}'))
    return out, cross


#%% Sim

def make_sim(interventions=None, connectors=None, label=None, n_agents=5000, seed=1):
    net = ss.MFNet(duration=ss.lognorm_ex(mean=ss.years(1), std=ss.years(1)), n_contacts=ss.poisson(0.8), concurrency=0.15)
    return ss.Sim(
        diseases=[make_genotype(g, demography=(g == G[0])) for g in G], networks=net,
        interventions=interventions, connectors=connectors, n_agents=n_agents, total_pop=100_000,
        start=1990, stop=2050, dt=ss.years(0.25), rand_seed=seed, verbose=0, label=label,
    )


def annual(res, years=range(T0, 2026)):
    a = res.annualize()
    yv = np.floor([float(getattr(t, 'year', t)) for t in a.timevec])
    return [round(float(a.values[yv == y][0])) if any(yv == y) else None for y in years]


def cancers(sim, after=T0):
    i = sim.t.yearvec >= after
    return sum(sim.results[g].invasion[i].sum() for g in G)


def run_safely(label, fn):
    try:
        return fn()
    except Exception as e:
        print(f'FAIL {label}: {type(e).__name__}: {str(e)[:300]}')
        return None


if __name__ == '__main__':
    T = sc.timer()
    sims = sc.objdict()

    # Vaccination
    sims.base = make_sim(label='baseline')
    for key, prod, doses in [('vx1', 'bivalent', 1), ('vx2', 'bivalent', 2), ('vx9', 'nonavalent', 1)]:
        v, c = vaccination(prod, doses=doses)
        sims[key] = make_sim(v, c, label=f'{prod}, {doses} dose(s), routine+catch-up')

    # Screen and treat. "Screen 1000/yr" = capacity per yearly round; treatment capped at 5/week so it binds at this size
    sims.a = make_sim(algo_a(), label='(a) HPV -> abl-eligible? -> abl/exc')
    sims.a_rate = make_sim(algo_a(screen_capacity=ss.peryear(1000)), label='(a) screen cap as ss.peryear(1000)')
    sims.a_cap = make_sim(algo_a(tx_capacity=ss.perweek(5)), label='(a) + treat cap 5/wk, capacity= (drop)')
    q = Queue(per_step=max(1, round(5*13/20)), name='queue') # 5/week x 13 weeks/step / 20 people per agent
    sims.a_queue = make_sim(algo_a(queue=q), label='(a) + treat cap 5/wk, FIFO queue')
    sims.b = make_sim(algo_b(), label='(b) HPV -> VIA -> ablation')
    sims.c = make_sim(algo_c(), label='(c) HPV if not screened in 5y, 30-50')
    sims.rad = make_sim(algo_b() + radiation(), label='(b) + radiation')
    tv, tc = txvx()
    sims.txvx = make_sim(tv, tc, label='therapeutic vaccine 2030')

    for k, sim in sims.items():
        run_safely(k, sim.run)
    print(f'Ran {len(sims)} sims in {T.toc(output=True):.1f} s; solver {sims.base.solver}\n')

    deaths = lambda sim: sum(sim.results[g].cancer_death[sim.t.yearvec >= T0].sum() for g in G)
    inf = lambda sim, g: sim.results[g].infection[sim.t.yearvec >= T0].sum()
    b = sims.base
    print(f'{"scenario":42s} {"cancers":>8s} {"vs base":>8s} {"ca deaths":>9s}  infections 2020-50 vs base (16/18/hi5)')
    for k, sim in sims.items():
        ca = cancers(sim)
        rel = ' '.join(f'{inf(sim, g)/inf(b, g) - 1:+5.0%}' for g in G)
        print(f'{sim.label:42s} {ca:8.0f} {ca/cancers(b) - 1:+8.0%} {deaths(sim):9.0f}  {rel}')

    i20 = sc.findnearest(b.t.yearvec, 2020)
    print('\nBaseline 2020: n_I/(n_alive/2) by genotype:', {g: round(float(b.results[g].n_I[i20]/(b.results.n_alive[i20]/2)), 3) for g in G})

    yrs = list(range(T0, 2026))
    print(f'\nBy year {yrs[0]}-{yrs[-1]} (people, scaled x20):')
    for k, keys in dict(vx1=['routine_vx', 'catchup_vx'], vx2=['routine_vx', 'dose2_vx']).items():
        for key in keys:
            print(f'  {k:8s} {key:24s}', annual(sims[k].results[key].delivered, yrs))
    for k in ['a', 'a_rate', 'a_cap', 'a_queue', 'b', 'c']:
        r = sims[k].results
        print(f'  {k:8s} {"screened":24s}', annual(r.primary.delivered, yrs))
        print(f'  {k:8s} {"HPV positive":24s}', annual(r.primary.diagnosed, yrs))
        if 'tx_visit' in r:
            print(f'  {k:8s} {"treated (abl+exc visits)":24s}', annual(r.tx_visit.delivered, yrs))
        if 'via_triage' in r:
            print(f'  {k:8s} {"VIA triaged":24s}', annual(r.via_triage.delivered, yrs))
            print(f'  {k:8s} {"VIA positive":24s}', annual(r.via_triage.diagnosed, yrs))
        print(f'  {k:8s} {"CIN lesions ablated":24s}', [sum(x) for x in zip(*[annual(r[f'ablation_{g}'].delivered, yrs) for g in G])])
    print('  a_queue  waiting list (agents) at start of each year:', sims.a_queue.interventions.queue.n_waiting[4*(T0-1990)::4][:6])
    r = sims.vx2.results
    print(f'  vx2 doses cost 2020-2050: ${float(r.routine_vx.cost.sum() + r.catchup_vx.cost.sum() + r.dose2_vx.cost.sum()):,.0f}')
