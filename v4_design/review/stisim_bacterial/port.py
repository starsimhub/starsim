"""
STIsim bacterial STIs (chlamydia, gonorrhea, trichomoniasis) ported to the Starsim v4 declarative API.

v3 sources: stisim/diseases/sti.py (SEIS), chlamydia.py, gonorrhea.py, trichomoniasis.py,
stisim/interventions/base_interventions.py, gonorrhea_interventions.py.

Run: conda run -n starsim python port.py
Each section prints what it shows; failures are caught and printed, not hidden.
"""
import traceback
import numpy as np
import sciris as sc
import starsim as ss

ppl = ss.People(1) # Only for writing lazy conditions on people (ppl.age, ppl.female)
F, M = 0, 1        # Level of the 'sex' stratification
BETA_SCALE = 1.0   # Betas below are uncalibrated (v3's are per act on StructuredSexual, with care in place)

def try_run(label, func):
    """ Run one section, printing the error (and keeping going) if it fails """
    sc.heading(label)
    try:
        return func()
    except Exception as e:
        tb = traceback.format_exc().splitlines()
        print(f'FAILED: {type(e).__name__}: {e}')
        print('\n'.join(tb[-14:]))
        return None


#%% Parameters (v3 STIPars/CTPars/NGPars, trichomoniasis.py); units are months

def wk(x):  return x/4.345 # weeks -> months, for the lognormal pars, which have no units here (see NOTES: lognorm_ex(mean=ss.weeks(...)) is fine, but per-sex values can't be typed)

pars = dict(
    ng = dict(
        beta_m2f=0.09, rel_beta_f2m=0.5, dur_exp=0.0, # months; v3 beta 0.06
        p_symp=[0.35, 0.65],
        dur_presymp=[[wk(1), wk(12)], [wk(0.25), wk(1)]],     # [mean, std] for women, men
        p_symp_care=[0.66, 0.83],
        dur_symp2care=[[2, 1], [wk(1), wk(2)]],
        dur_asymp2clear=[[8, 2], [6, 3]],
        dur_symp2clear=[[9, 2], [6, 3]],
        p_pid=0.0, dur_prepid=[1.5, 3], p_pid_care=0.1, dur_pid2care=[wk(2), wk(4)], dur_pid2clear=[12, 1.15],
        p_persist=0.0,
        init_prev=0.01,
    ),
    ct = dict(
        beta_m2f=0.04, rel_beta_f2m=0.5, dur_exp=wk(1), # v3 beta 0.06
        p_symp=[0.20, 0.54],
        dur_presymp=[[wk(1), wk(10)], [wk(0.25), wk(1)]],
        p_symp_care=[0.42, 0.83],
        dur_symp2care=[[2, 1], [wk(1), wk(2)]],
        dur_asymp2clear=[[18, 1], [12, 1]],
        dur_symp2clear=[[18, 1], [12, 1]],
        p_pid=0.2, dur_prepid=[1.5, 3], p_pid_care=0.1, dur_pid2care=[wk(2), wk(4)], dur_pid2clear=[12, 1.15], # v3 CT default p_pid=0; 0.2 (the STIPars default) here so PID is exercised
        p_persist=0.0,
        init_prev=0.01,
    ),
    tv = dict(
        beta_m2f=0.025, rel_beta_f2m=0.5, dur_exp=0.0, # v3 beta 0.1
        p_symp=[0.4, 0.5],
        dur_presymp=[[wk(1), wk(12)], [wk(0.25), wk(1)]],
        p_symp_care=[0.39, 0.27],
        dur_symp2care=[[2, 1], [wk(1), wk(2)]],
        dur_asymp2clear=[[48, 6], [wk(26), wk(4)]],
        dur_symp2clear=[[wk(20), wk(4)], [wk(18), wk(4)]],
        p_pid=0.0, dur_prepid=[1.5, 1], p_pid_care=0.0, dur_pid2care=[1, 1], dur_pid2clear=[12, 1],
        p_persist=0.9, dur_persist=1200, # 90% of asymptomatic women do not clear (v3: p_clear=0.1, dur_persist=100 years)
        init_prev=0.01,
    ),
)


#%% The disease: an SEIS with sex-specific branches and durations

def by_sex(f, m):
    """ A per-agent value by sex, as a function of the sim. ss.by(sex=[f, m]) can't be used: see NOTES (sex stratification vs people.female) """
    return lambda sim: np.where(sim.people.female, f, m)

def lognorm(ms):
    return ss.lognorm_ex(mean=ms[0], std=ms[1])

def sexed(src, dst, p, durs, prefix):
    """
    A branch out of `src` into `dst` taken with probability p[sex], after a lognormal dwell durs[sex].

    v4 has no way to give a dwell-time *distribution* per sex (ss.by of a Dist, or a Dist with ss.by
    parameters, both fail), so each sex gets its own transition, and the other sex's probability is 0.
    """
    return {
        f'{prefix}_f': ss.Transition(src, dst, p=by_sex(p[0], 0.0), dur=lognorm(durs[0])),
        f'{prefix}_m': ss.Transition(src, dst, p=by_sex(0.0, p[1]), dur=lognorm(durs[1])),
    }

def make_sti(name, p, eff=0.9, fail=None, direct=False, extra=None):
    """
    v3 SEIS (sti.py:430-743) as declared states and transitions.

    States: S susceptible; E exposed (latent); A asymptomatic (never symptomatic); P presymptomatic
    (will become symptomatic); Y symptomatic; C symptomatic and seeking care (the clinic queue);
    N symptomatic, gave up seeking care; PID (women); T just treated (resolved immediately into cure or failure).
    """
    S, E, A, P, Y, C, N, PID, T = ss.states('S E A P Y C N PID T')
    pf, pm = p['p_pid'], 0.0
    trs = dict(
        infection = S >> E @ dict(mf=[BETA_SCALE*p['beta_m2f'], BETA_SCALE*p['beta_m2f']*p['rel_beta_f2m']]) * (A + P + Y + C + N + PID),
    )

    # Exposed -> presymptomatic or asymptomatic, by sex (v3 set_exposure + set_symptoms)
    if p['dur_exp'] > 0:
        trs['latency'] = ss.progression('E -> P | A', p=by_sex(*p['p_symp']), dur=ss.constant(p['dur_exp']))
    else: # A branch with no dwell happens on entry (at the next step). ss.Transition(E, [P, A], p=...) with no dur crashes (see NOTES), so one transition per branch
        ps = p['p_symp']
        trs['latency'] = ss.Transition(E, P, p=by_sex(ps[0], ps[1]))
        trs['latency_a'] = ss.Transition(E, A, p=by_sex(1 - ps[0], 1 - ps[1]))

    # Presymptomatic -> symptomatic, after a sex-specific lognormal delay
    trs.update({
        'onset_f': ss.Transition(P, Y, dur=lognorm(p['dur_presymp'][0]), where=ppl.female),
        'onset_m': ss.Transition(P, Y, dur=lognorm(p['dur_presymp'][1]), where=~ppl.female),
    })

    # Asymptomatic: PID (women), persistence (TV women), or clearance (v3 set_pid, set_duration, Trichomoniasis.set_duration)
    q = 1 - pf
    trs['pid_a'] = ss.Transition(A, PID, p=by_sex(pf, 0.0), dur=lognorm(p['dur_prepid']))
    trs.update(sexed(A, S, [q*(1 - p['p_persist']), 1.0], p['dur_asymp2clear'], 'clear_a'))
    if p['p_persist']:
        trs['persist'] = ss.Transition(A, S, p=by_sex(q*p['p_persist'], 0.0), dur=ss.constant(p['dur_persist']))

    # Symptomatic: PID (women), care seeking after a delay, or clearance (v3 set_care_seeking, set_duration)
    pc = p['p_symp_care']
    trs['pid_y'] = ss.Transition(Y, PID, p=by_sex(pf, 0.0), dur=lognorm(p['dur_prepid']))
    trs.update(sexed(Y, C, [q*pc[0], pc[1]], p['dur_symp2care'], 'seek'))
    trs.update(sexed(Y, S, [q*(1 - pc[0]), 1 - pc[1]], p['dur_symp2clear'], 'clear_y'))

    # PID: care seeking or clearance (v3 set_pid_care_seeking)
    trs['seek_pid'] = ss.Transition(PID, C, p=p['p_pid_care'], dur=lognorm(p['dur_pid2care']))
    trs['clear_pid'] = ss.Transition(PID, S, p=1 - p['p_pid_care'], dur=lognorm(p['dur_pid2clear']))

    # Waiting for care: people give up after a mean of a month (a hazard, so it competes with being treated)
    trs['balk'] = C >> N @ ss.months(1)
    trs.update(sexed(N, S, [1.0, 1.0], p['dur_symp2clear'], 'clear_n'))

    # Treatment: from any infected state into T, which resolves at once into cure or failure (v3 STITreatment.administer/change_states)
    trs['treat'] = (E | A | P | Y | C | N | PID) >> T @ ss.permonth(0) # Only ever moved along by ss.Treatment(transition='treat')
    trs['cure'] = ss.Transition(T, S, p=eff)
    trs['fail'] = ss.Transition(T, A, p=1 - eff if fail is None else fail) # A failure stays infected, without symptoms or care seeking. p=None should be "the rest" (spec), but is E209
    for key,make in (extra or {}).items(): # Transitions that need this disease's states
        trs[key] = make(locals())
    if direct: # The other way: cure straight to S, and ss.Treatment(efficacy=...) leaves failures where they were
        trs['cure_direct'] = (E | A | P | Y | C | N | PID) >> S @ ss.permonth(0)

    infected = E + A + P + Y + C + N + PID + T
    dis = ss.Disease(**trs, name=name, init=dict(A=p['init_prev']),
                     infected=infected, prev=infected/(S + infected))
    return dis


def sync_sex(sim):
    """ Make the 'sex' stratification agree with people.female, which MFNet uses (see NOTES) """
    sim.init()
    people = sim.people
    uids = people.auids
    people.sex[uids] = np.where(people.female.raw[uids], F, M)
    sim.groups_changed(uids)
    return sim

def make_net():
    return ss.MFNet(n_contacts=0.8, concurrency=0.15, duration=ss.lognorm_ex(mean=ss.years(1), std=ss.years(1)), name='mf')

base = dict(n_agents=4000, total_pop=100_000, start=2000, stop=2015, dt=ss.months(1), verbose=0, # NB: adding unit='month' silently reads start/stop as months (see NOTES)
            stratify=[dict(sex=['female', 'male'], split=[0.5, 0.5]), dict(risk=['low', 'high'], split=[0.8, 0.2])])


def summarize(sim, names=('ng', 'ct', 'tv'), last=12):
    """ The results we want by sex: new infections, prevalence, treated, overtreated """
    out = sc.objdict()
    for name in names:
        if name not in sim.diseases:
            continue
        r = sim.results[name]
        inf = r.infection.by('sex').values[-last:].sum(axis=0)
        prev = r.prev.by('sex').values[-1]
        out[name] = sc.objdict(new_inf_f=inf[F], new_inf_m=inf[M], prev_f=prev[F], prev_m=prev[M],
                               cured=r.cure.values.sum(), failed=r.fail.values.sum())
        print(f'{name}: last-year infections F/M {inf[F]:.0f}/{inf[M]:.0f}; prevalence F/M {prev[F]:.3f}/{prev[M]:.3f}; '
              f'cured {r.cure.values.sum():.0f}, failed {r.fail.values.sum():.0f}, PID {sum(r[k].values.sum() for k in ["pid_a", "pid_y"]):.0f}')
    return out


#%% 1. Natural history only

def natural_history():
    diseases = [make_sti(k, v) for k,v in pars.items()]
    sim = sync_sex(ss.Sim(diseases=diseases, networks=make_net(), **base))
    sim.run()
    print('solver:', sim.solver, '|', sim.solver_reason)
    summarize(sim)
    ng = sim.results.ng
    toP = ng.latency.by('sex').values.sum(axis=0)
    toA = ng.latency_a.by('sex').values.sum(axis=0)
    print(f'NG share symptomatic by sex: F {toP[F]/(toP[F]+toA[F]):.2f} (0.35), M {toP[M]/(toP[M]+toA[M]):.2f} (0.65)')
    ct = sim.results.ct
    try:
        toP = ct.latency.select(to='P').by('sex').values.sum(axis=0) # The spec's way to get one branch of a branching transition
    except KeyError as e:
        print(f'  latency.select(to="P") fails: {e}')
        toP = (ct.onset_f.by('sex').values + ct.onset_m.by('sex').values).sum(axis=0) # Instead: what leaves P (lags by the presymptomatic period)
    lat = ct.latency.by('sex').values.sum(axis=0)
    print(f'CT share symptomatic by sex: F {toP[F]/lat[F]:.2f} (0.20), M {toP[M]/lat[M]:.2f} (0.54)')
    return sim

nh = try_run('1. Natural history: CT, NG, TV on an MFNet, no care', natural_history)


#%% 2. Treatment of a disease with several infected states

def treatment_semantics():
    def sim_with(tx, **kw):
        ng = make_sti('ng', pars['ng'], direct=True)
        return sync_sex(ss.Sim(diseases=ng, networks=make_net(), interventions=tx, **sc.mergedicts(base, dict(stop=2005), kw)))

    # (a) The spec's default: efficacy is the chance of leaving the infected states by "the recovery transition"
    try:
        sim_with(ss.treat(ss.Treatment(efficacy=0.9), interval=ss.months(1), coverage=0.1)).run()
        print('(a) ss.Treatment(efficacy=0.9) with no transition=: ran')
    except Exception as e:
        print(f'(a) ss.Treatment(efficacy=0.9) with no transition=:\n    {type(e).__name__}: {str(e)[:400]}')

    # (b) A multi-source transition straight to S: failures stay in whatever state they were in
    drug = ss.Treatment(efficacy=0.9, transition='cure_direct')
    sim = sim_with(ss.treat(drug, interval=ss.months(1), coverage=0.1, name='tx')).run()
    r = sim.results
    print(f'(b) cure_direct, efficacy=0.9: delivered {r.tx.delivered.sum():.0f}, cured {r.ng.cure_direct.sum():.0f} '
          f'(failures = delivered - cured, among the infected; not a result)')

    # (c) Everyone treated goes to T, which resolves into cure (p=efficacy) or fail: failures are a flow
    drug = ss.Treatment(efficacy=1.0, transition='treat')
    sim = sim_with(ss.treat(drug, interval=ss.months(1), coverage=0.1, name='tx')).run()
    r = sim.results
    print(f'(c) treat -> T -> cure|fail: delivered {r.tx.delivered.sum():.0f}, treated {r.ng.treat.sum():.0f}, cured {r.ng.cure.sum():.0f}, failed {r.ng.fail.sum():.0f}')
    print(f'    default eligible = the "infected" states; delivered by sex F/M {r.tx.delivered.by("sex").values.sum(0)}')

    # (d) Which states does the library count as infected (default eligibility, what a test detects)?
    graph = sim.diseases.ng.graph
    print('    graph.infected:', [s.name for s in graph.infected], '(T is missing: it is not infectious)')
    return sim

tx = try_run('2. Treatment that moves people out of any infected state, with failure', treatment_semantics)


#%% 3. Symptomatic care seeking with a capacity of 100 a week

def capacity():
    big = dict(total_pop=1_000_000, stop=2006) # Enough care seekers (about 700 a month) that 100 a week (434 a month) binds
    def run(label, make_tx, quiet=False, **kw):
        ng = make_sti('ng', pars['ng'])
        tx = make_tx(ng)
        sim = sync_sex(ss.Sim(diseases=ng, networks=make_net(), interventions=tx, **sc.mergedicts(base, big, kw))).run()
        r = sim.results
        name = sc.tolist(tx)[0].name
        d = r[name].delivered.values
        seek = sum(r.ng[k].values for k in ['seek_f', 'seek_m', 'seek_pid'])
        if not quiet:
            print(f'{label}: per month: start seeking care {seek[12:].mean():.0f}, treated {d[12:].mean():.0f}, '
                  f'in queue (n_C) {r.ng.n_C[12:].mean():.0f}, gave up {r.ng.balk.values[12:].mean():.0f}')
        return sim

    drug = lambda: ss.Treatment(efficacy=1.0, transition='treat', name='drug')
    cap = ss.perweek(100)

    # (a) Routine (no interval): each person is offered treatment once in their life, so reinfections are never treated
    run('(a) routine, eligible=C, no cap        ', lambda ng: ss.treat(drug(), eligible=ng.C, name='clinic'))

    # (b) Rounds every step: everyone in C is offered it each month
    run('(b) interval=1 month, no cap           ', lambda ng: ss.treat(drug(), eligible=ng.C, interval=ss.months(1), name='clinic'))

    # (c) Capacity: unmet demand stays in C (a queue), drained only by balking (mean wait a month)
    sim = run('(c) interval=1 month, cap 100/week     ', lambda ng: ss.treat(drug(), eligible=ng.C, interval=ss.months(1), capacity=cap, name='clinic'))

    # (d) Event-triggered: only those who started seeking care this step; demand over the cap is lost (they stay in C, untreated)
    def on_seek(ng):
        try:
            seeking = ng.seek_f | ng.seek_m | ng.seek_pid
        except TypeError as e:
            print(f'    on=ng.seek_f | ng.seek_m: {type(e).__name__}: {e}')
            seeking = ss.conditions.as_cond(ng.seek_f) | ng.seek_m | ng.seek_pid # Works, through an internal function
        return ss.treat(drug(), on=seeking, capacity=cap, name='clinic')
    run('(d) on=seek_*, cap 100/week            ', on_seek)

    # (e) Priority, symptomatic first. deliver= returns UIDs, and capacity caps them; is the order used?
    def make_priority(reverse):
        def priority(sim):
            ng = sim.diseases.ng
            first, second = ng.C.uids, (ng.A & sim.people.female).uids # Asymptomatic women stand in for screen-positives
            if reverse:
                first, second = second, first
            return ss.uids(np.concatenate([first, second]))
        return priority
    cap25 = ss.perweek(25)
    runs = [run('', lambda ng, rev=rev: ss.treat(drug(), deliver=make_priority(rev), capacity=cap25, name='clinic'), quiet=True, total_pop=100_000) for rev in [False, True]]
    same = np.array_equal(runs[0].results.ng.n_C.values, runs[1].results.ng.n_C.values)
    print(f'(e1) deliver= in priority order vs the reverse, cap 25/week: identical results = {same} (the cap chooses at random)')

    # (e2) Two deliveries in loop order; the second's capacity is what the first left, given as a function
    def leftover(sim):
        used = sim.results.symp.delivered[sim.ti]*sim.current_scale # Mid-run, results are in agents, not people: they are scaled at the end
        return max(0.0, 25*4.345 - used) # People per round (it runs in monthly rounds)
    def two(ng):
        shared = drug() # One product for both deliveries (two products both named 'drug' are an error)
        symp = ss.treat(shared, eligible=ng.C, interval=ss.months(1), capacity=cap25, name='symp')
        screened = ss.treat(shared, eligible=ng.A & ppl.female, interval=ss.months(1), capacity=leftover, name='screened')
        return [symp, screened]
    sim = run('(e2) symp first, then leftover capacity', two, quiet=True, total_pop=100_000) # 25 agents per person, so the cap (4.3 agents a month) is not lost to rounding
    r = sim.results
    tot = r.symp.delivered.values + r.screened.delivered.values
    print(f'(e2) symp first, then leftover, first year: symptomatic treated {r.symp.delivered.values[1:13].mean():.0f}/month, '
          f'screened {r.screened.delivered.values[1:13].mean():.0f}/month, total {tot[1:13].mean():.0f} (cap 109; max in a month {tot.max():.0f})')
    return sim

cap = try_run('3. Care seeking, capacity, queues, and priority', capacity)


#%% 4. Screening asymptomatic people by age, sex and risk, with a test delay and loss to follow-up

def screening():
    ng = make_sti('ng', pars['ng'])
    test = ss.Test(sensitivity=0.9, specificity=0.98, delay=ss.weeks(2), name='naat')
    drug = ss.Treatment(efficacy=1.0, transition='treat', name='ctx')
    young_women = ppl.female & (ppl.age >= 15) & (ppl.age < 25)
    young_women_fn = lambda sim: sim.people.female & (sim.people.age >= 15) & (sim.people.age < 25)
    try:
        eligible = young_women | (ppl.risk == 'high') # Risk is a stratification of the sim
    except Exception as e:
        print(f'ppl.risk == "high": {type(e).__name__}: {e}')
        try:
            eligible = young_women | (ng.risk == 'high')
            print('  ng.risk == "high" (on the disease, for a stratification of the sim): accepted')
        except Exception as e:
            print(f'  ng.risk == "high": {type(e).__name__}: {e}')
            high = lambda sim: sim.people.risk == 1 # Level index, not label
            eligible = lambda sim: young_women_fn(sim) | high(sim)
    linked = ss.treat(drug, coverage=0.8, name='linked') # 20% lost to follow-up between result and treatment
    screen = ss.screen(test, eligible=eligible, coverage=0.3, interval=ss.years(1), start=2002, on_positive=linked, name='screen')
    sim = sync_sex(ss.Sim(diseases=ng, networks=make_net(), interventions=screen, **sc.mergedicts(base, dict(stop=2008)))).run()
    r = sim.results
    print(f'screened {r.screen.delivered.sum():.0f}, positive {r.screen.diagnosed.sum():.0f}, treated {r.linked.delivered.sum():.0f} '
          f'({r.linked.delivered.sum()/max(r.screen.diagnosed.sum(), 1):.2f} linked)')
    ti_s = np.flatnonzero(r.screen.delivered.values); ti_d = np.flatnonzero(r.screen.diagnosed.values)
    print(f'rounds at steps {ti_s.tolist()}, results at {ti_d.tolist()} (delay of 2 weeks, monthly steps)')
    infected_treated = r.ng.treat.sum()
    print(f'treated who were infected {infected_treated:.0f}; overtreated (false positives, or cleared while waiting) {r.linked.delivered.sum() - infected_treated:.0f}')
    print(f'screened by sex F/M {r.screen.delivered.by("sex").values.sum(0)}, by risk low/high {r.screen.delivered.by("risk").values.sum(0)}')
    return sim

scr = try_run('4. Screening, delay, loss to follow-up', screening)


#%% 5. Syndromic management across NG, CT and TV: overtreatment

def syndromic():
    diseases = [make_sti(k, v) for k,v in pars.items()]
    ng, ct, tv = diseases
    seeking = ng.C | ct.C | tv.C # Conditions on three diseases' states of the same name, each keeping its own
    # v3 SyndromicManagement: discharge is treated for NG+CT, and women also with metronidazole (TV). One delivery per drug
    # and disease, since a delivery acts on one disease; they cannot share one clinic capacity (see NOTES)
    deliveries = []
    for dis, eligible in [(ng, seeking), (ct, seeking), (tv, seeking & ppl.female)]:
        drug = ss.Treatment(efficacy=1.0, transition='treat', name=f'{dis.name}_drug')
        deliveries.append(ss.treat(drug, eligible=eligible, interval=ss.months(1), disease=dis.name, name=f'{dis.name}_tx'))
    sim = sync_sex(ss.Sim(diseases=diseases, networks=make_net(), interventions=deliveries, **sc.mergedicts(base, dict(stop=2008)))).run()
    r = sim.results
    for name in ['ng', 'ct', 'tv']:
        d = r[f'{name}_tx'].delivered.by('sex').values.sum(0)
        t = r[name].treat.by('sex').values.sum(0)
        over = d - t
        print(f'{name}: treated F/M {d[F]:.0f}/{d[M]:.0f}; infected F/M {t[F]:.0f}/{t[M]:.0f}; overtreated F/M {over[F]:.0f}/{over[M]:.0f}')
    print(f'NG and CT deliveries should see the same visits, but delivered {r.ng_tx.delivered.sum():.0f} vs {r.ct_tx.delivered.sum():.0f}: '
          f'the first one moves its own C into T, so the second no longer sees them')
    summarize(sim)

    # Workaround: decide who is at the clinic once per step, before any delivery, in a custom module
    class Clinic(ss.Intervention):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_states(ss.BoolState('visit', label='At the clinic this step'))
        def step(self):
            d = self.sim.diseases
            self.visit[:] = False
            self.visit[(d.ng.C | d.ct.C | d.tv.C).uids] = True
    diseases = [make_sti(k, v) for k,v in pars.items()]
    at_clinic = lambda sim: sim.interventions.clinic.visit
    women_at_clinic = lambda sim: sim.interventions.clinic.visit & sim.people.female
    deliveries = [Clinic(name='clinic')]
    for dis, eligible in zip(diseases, [at_clinic, at_clinic, women_at_clinic]):
        drug = ss.Treatment(efficacy=1.0, transition='treat', name=f'{dis.name}_drug')
        deliveries.append(ss.treat(drug, eligible=eligible, interval=ss.months(1), disease=dis.name, name=f'{dis.name}_tx'))
    sim = sync_sex(ss.Sim(diseases=diseases, networks=make_net(), interventions=deliveries, **sc.mergedicts(base, dict(stop=2008)))).run()
    r = sim.results
    print('With a Clinic module that flags visits first:')
    for name in ['ng', 'ct', 'tv']:
        d = r[f'{name}_tx'].delivered.by('sex').values.sum(0)
        t = r[name].treat.by('sex').values.sum(0)
        print(f'  {name}: treated F/M {d[F]:.0f}/{d[M]:.0f}; overtreated F/M {d[F]-t[F]:.0f}/{d[M]-t[M]:.0f}')
    return sim

syn = try_run('5. Syndromic management and overtreatment', syndromic)


#%% 6. Gonorrhea AMR: treatment efficacy that falls over time, and resistance acquired through treatment failure

def amr():
    """
    v3 (gonorrhea_interventions.py): a continuous per-agent rel_treat, inherited from the infector, multiplied down by 1% on each
    failure and 0.5% on each unnecessary treatment; efficacy = 0.96*rel_treat. v4 has no per-agent continuous state that is
    inherited through transmission, so resistance is a strain: 'sens' and 'res', passed on by the infector.
    """
    ts = lambda a, b: ss.timeseries({2000: a, 2010: b}, mode='linear')
    cure = ss.by(strain=[ts(0.96, 0.90), 0.30]) # Base efficacy falls over time; resistant infections mostly fail
    fail = ss.by(strain=[ts(0.03, 0.075), 0.70]) # Every branch spelled out: p=None for "the rest" is an error, and ss.by has no 1 - x with time series in it
    acquire = lambda st: ss.Transition(st['T'], st['A'], p=ts(0.01, 0.025), move=dict(strain='sens -> res')) # Only sensitive infections; resistant ones take the other branches
    def make():
        ng = make_sti('ng', pars['ng'], eff=cure, fail=fail, extra=dict(acquire=acquire))
        return ng.stratify(strain=['sens', 'res'], split=[0.95, 0.05], states=['E', 'A', 'P', 'Y', 'C', 'N', 'PID', 'T'])
    ng = make()
    drug = ss.Treatment(efficacy=1.0, transition='treat', name='ctx')
    tx = ss.treat(drug, eligible=ng.C, interval=ss.months(1), name='tx')
    screen = ss.screen(ss.Test(sensitivity=0.9, specificity=0.98, name='naat'), eligible=lambda sim: sim.people.female & (sim.people.age < 25),
                       coverage=0.3, interval=ss.years(1), on_positive=ss.treat(drug, coverage=0.8, name='linked'), name='screen')
    sim = sync_sex(ss.Sim(diseases=ng, networks=make_net(), interventions=[tx, screen], **sc.mergedicts(base, dict(stop=2015)))).run()
    r = sim.results.ng
    by_strain = r.infection.by('strain').values
    share = by_strain[:, 1]/np.maximum(by_strain.sum(1), 1)
    print(f'share of new infections resistant: 2001 {share[12:24].mean():.2f}, 2014 {share[-12:].mean():.2f}; acquired through failure {r.acquire.sum():.0f}')
    c = r.cure.by('strain').values.sum(0); f = r.fail.by('strain').values.sum(0)
    print(f'cure share by strain: sens {c[0]/(c[0]+f[0]):.2f} (n={c[0]+f[0]:.0f}), res {c[1]/(c[1]+f[1]):.2f} (n={c[1]+f[1]:.0f}; 0.30 expected, 0.29 pooled over 4 seeds)')
    cy = r.cure.values; fy = r.fail.values + r.acquire.values
    print(f'cure share overall, first vs last 3 years: {cy[:36].sum()/(cy[:36].sum()+fy[:36].sum()):.3f} vs {cy[-36:].sum()/(cy[-36:].sum()+fy[-36:].sum()):.3f}')
    return sim

res = try_run('6. Gonorrhea AMR as a strain, with efficacy that changes over time', amr)


#%% 7. ChlamydiaBL: infectiousness from a per-agent bacterial load (v3 chlamydia.py:89-177)

def bacterial_load():
    """ Load grows to a peak at 2 months then decays; rel_trans = 2/(1 + exp(-0.5*log10(load))) - 1. A modifier whose amount is a function of the sim """
    ct = make_sti('ct', pars['ct'])
    def rel_trans(sim):
        d = sim.diseases.ct
        u = sim.people.auids
        start = np.where(np.isnan(d.ti_E.raw[u]), d.ti_A.raw[u], d.ti_E.raw[u]) # Time of infection: there is none, so time of entering E, or A for the people seeded there (they never entered E)
        t = np.nan_to_num(np.maximum(sim.ti - start, 0), nan=0.0) # Months
        load = np.where(t < 2, np.exp(np.log(1e8)*t/2), 1e8*np.exp(-np.log(2)/0.58*(t - 2)))
        return 2/(1 + np.exp(-0.5*np.log10(np.maximum(load, 1)))) - 1
    load = ss.modify(ct.rel_trans, multiply=rel_trans, name='load')
    sim = sync_sex(ss.Sim(diseases=ct, networks=make_net(), connectors=load, **sc.mergedicts(base, dict(stop=2008)))).run()
    base_sim = sync_sex(ss.Sim(diseases=make_sti('ct', pars['ct']), networks=make_net(), **sc.mergedicts(base, dict(stop=2008)))).run()
    print(f'CT infections 2000-2008: {base_sim.results.ct.infection.sum():.0f} without the load, {sim.results.ct.infection.sum():.0f} with it')
    return sim

bl = try_run('7. Chlamydia bacterial load as a modifier of rel_trans', bacterial_load)


#%% 8. Results by sex, all care in place (the headline numbers)

def headline():
    diseases = [make_sti(k, v) for k,v in pars.items()]
    seeking = diseases[0].C | diseases[1].C | diseases[2].C
    intvs = []
    for dis in diseases:
        drug = ss.Treatment(efficacy=1.0, transition='treat', name=f'{dis.name}_drug')
        intvs.append(ss.treat(drug, eligible=seeking, interval=ss.months(1), capacity=ss.perweek(100), disease=dis.name, name=f'{dis.name}_tx'))
    sim = sync_sex(ss.Sim(diseases=diseases, networks=make_net(), interventions=intvs, **base)).run()
    summarize(sim)
    for dis in diseases:
        d = sim.results[f'{dis.name}_tx'].delivered.by('sex').values[-12:].sum(0)
        t = sim.results[dis.name].treat.by('sex').values[-12:].sum(0)
        print(f'{dis.name} last year: treated F/M {d[F]:.0f}/{d[M]:.0f}, overtreated F/M {d[F]-t[F]:.0f}/{d[M]-t[M]:.0f}')
    print('Auto results for ng:', sorted(k for k in sim.results.ng.keys() if not k.startswith('cum_'))[:40])
    return sim

hl = try_run('8. Headline results by sex', headline)
