"""
Keeling and Rohani (2008), chapter 6: stochastic dynamics (programs 6.1 to 6.6)

The references are the book's equations integrated with scipy (for the deterministic part) and an exact event-driven (Gillespie) simulation written here in numpy, which is what program 6.3 is. Stochastic solvers are compared with it distributionally: final size, peak, peak time, and the probability of a minor outbreak, by two-sample KS tests. Runs are serial, to keep the machine load at one process.

Usage: python kr_ch6.py [closed|ccs|noise|all]
"""
import sys
import time
import warnings
import numpy as np
import sciris as sc
import starsim as ss
from scipy import stats
from common import capture, series

warnings.simplefilter('ignore')


#%% The exact reference: event-driven SIR, with optional births, deaths, and imports (program 6.3; 6.5 adds imports)

def exact_sir(N, beta, gamma, I0, T, mu=0.0, eps=0.0, S0=None, seed=0):
    """
    Gillespie's direct method for SIR with frequency-dependent transmission, births into S at mu*N, deaths at mu per person, and imports (an extra force of infection eps per susceptible)

    Returns the number infected at each whole day (0..T) and the number of infections.
    """
    rng = np.random.default_rng(seed)
    S = N - I0 if S0 is None else S0
    I = I0
    R = N - S - I
    t = 0.0
    out = np.zeros(T+1); out[0] = I
    day = 1
    n_inf = 0
    while day <= T:
        n = S + I + R
        rates = np.array([beta*S*I/max(n, 1) + eps*S, gamma*I, mu*N, mu*S, mu*I, mu*R])
        total = rates.sum()
        if total <= 0:
            out[day:] = I
            break
        t += rng.exponential(1/total)
        while day <= T and day <= t:
            out[day] = I; day += 1
        if day > T:
            break
        e = np.searchsorted(np.cumsum(rates), rng.random()*total, side='right')
        if e == 0:   S -= 1; I += 1; n_inf += 1
        elif e == 1: I -= 1; R += 1
        elif e == 2: S += 1
        elif e == 3: S -= 1
        elif e == 4: I -= 1
        else:        R -= 1
    return out, n_inf


#%% The v4 model

def sir_model(beta, gamma, I0, mu=0.0, eps=0.0, S0=None, N=None):
    """ SIR as the book writes it; births and deaths if mu > 0, imports if eps > 0 """
    S, I, R = ss.states(3)
    infection = S >> I @ ss.perday(beta) * I
    if eps:
        infection = S >> I @ ss.perday(beta) * I + ss.perday(eps) # Imports: a force of infection that does not depend on I
    trans = dict(infection=infection, recovery=I >> R @ ss.perday(gamma))
    if mu:
        trans['births'] = ss.null >> S @ ss.perday(mu) * (S + I + R).n
        trans['deaths'] = S + I + R >> ss.null @ ss.perday(mu)
    init = dict(I=I0) if S0 is None else dict(S=S0, I=I0, R=N-S0-I0)
    return ss.Disease(**trans, init=init, name='sir')


def v4_runs(solver, n, N, beta, gamma, I0, T, **kw):
    """ n serial runs of the v4 model under a solver: n_I per day, and infections per run """
    out = []
    for seed in range(n):
        simkw = dict(n_agents=N) if solver == 'abm' else dict(total_pop=N)
        sim = ss.Sim(sir_model(beta, gamma, I0, N=N, **kw), unit='day', stop=T, solver=solver, rand_seed=seed+1, verbose=0, **simkw)
        sim.run()
        out.append((series(sim, 'sir.n_I'), series(sim, 'sir.infection').sum()))
    return out


def summarize(runs, N):
    """ Final size, peak, peak time, and minor outbreaks (final size under 10% of N) """
    I = np.array([r[0] for r in runs])
    final = np.array([r[1] for r in runs])
    return sc.objdict(final=final, peak=I.max(axis=1), tpeak=I.argmax(axis=1), minor=np.mean(final < 0.1*N))


def compare(a, b):
    """ KS p-values for final size, peak, and peak time among major outbreaks, and the minor-outbreak shares """
    out = sc.objdict(minor=(a.minor, b.minor))
    for k in ['final', 'peak', 'tpeak']:
        ma, mb = a.final >= 0.1*N_closed, b.final >= 0.1*N_closed
        out[k] = stats.ks_2samp(a[k][ma], b[k][mb]).pvalue
        out[k+'_mean'] = (a[k][ma].mean(), b[k][mb].mean())
    return out


#%% 6.3 and 6.4: a closed SIR, exact against every stochastic solver

N_closed, beta_c, gamma_c, I0_c, T_c = 1000, 0.4, 0.2, 5, 200

def closed(n=300):
    """ R0 = 2 in 1,000 people, 5 infected at the start: exact Gillespie in numpy against gillespie, tau, binomial, sde, and abm """
    T = sc.timer()
    ref = summarize([exact_sir(N_closed, beta_c, gamma_c, I0_c, T_c, seed=s) for s in range(n)], N_closed)
    print(f'exact (numpy): minor {ref.minor:.3f}, mean final {ref.final[ref.final >= 100].mean():.1f}, mean peak {ref.peak[ref.final >= 100].mean():.1f}, mean peak day {ref.tpeak[ref.final >= 100].mean():.1f} ({T.tocout(output=True):.1f} s)')
    rows = {}
    for solver in ['gillespie', 'tau', 'binomial', 'sde', 'abm']:
        T = sc.timer()
        runs, _, err = capture(lambda: v4_runs(solver, n, N_closed, beta_c, gamma_c, I0_c, T_c))
        if err:
            print(f'{solver:10s} error: {err[:300]}')
            continue
        res = summarize(runs, N_closed)
        c = compare(res, ref)
        rows[solver] = c
        print(f"{solver:10s} minor {c.minor[0]:.3f} vs {c.minor[1]:.3f}; KS p final {c.final:.3f}, peak {c.peak:.3f}, tpeak {c.tpeak:.3f}; means final {c.final_mean[0]:.1f} vs {c.final_mean[1]:.1f}, peak {c.peak_mean[0]:.1f} vs {c.peak_mean[1]:.1f}, tpeak {c.tpeak_mean[0]:.1f} vs {c.tpeak_mean[1]:.1f} ({T.tocout(output=True):.1f} s)")
    return rows


#%% 6.5 and 6.6: persistence and the critical community size, with births and deaths, and with imports

beta_m, gamma_m, mu_m = 17/13, 1/13, 1/(70*365) # R0 about 17, a 13-day infectious period, 70-year lifespan

def endemic(N):
    R0 = beta_m/(gamma_m + mu_m)
    S0 = int(round(N/R0))
    I0 = max(1, int(round(N*mu_m*(R0 - 1)/beta_m)))
    return S0, I0

def fadeout(I):
    """ Whether the infection died out at some point (no imports) """
    return bool((I[1:] == 0).any())

def ccs(sizes=(5e3, 2e4, 1e5, 5e5), n=20, years=10, solvers=('exact', 'gillespie', 'binomial', 'abm')):
    """ The share of runs in which an SIR started at its endemic equilibrium dies out within `years`, by population size and solver """
    T_ = years*365
    print(f'persistence over {years} years, {n} runs each; endemic I* = N x {mu_m*(beta_m/(gamma_m+mu_m)-1)/beta_m:.2e}')
    for N in sizes:
        N = int(N)
        S0, I0 = endemic(N)
        line = f'N={N:>8,} (I0={I0:>4}): '
        for solver in solvers:
            if solver == 'exact' and N > 2e4:
                line += 'exact   -    '; continue
            if solver == 'abm' and N > 1e4:
                line += 'abm   -    '; continue
            T = sc.timer()
            if solver == 'exact':
                runs = [exact_sir(N, beta_m, gamma_m, I0, T_, mu=mu_m, S0=S0, seed=s)[0] for s in range(n)]
            else:
                runs, _, err = capture(lambda: [r[0] for r in v4_runs(solver, n, N, beta_m, gamma_m, I0, T_, mu=mu_m, S0=S0)])
                if err:
                    line += f'{solver} error ({err[:80]}) '; continue
            line += f'{solver} {np.mean([fadeout(I) for I in runs]):.2f} ({T.tocout(output=True):.0f}s)  '
        print(line, flush=True)


def imports(N=5000, eps=1e-6, n=20, years=10):
    """ 6.6: imports keep re-seeding a population below the critical community size: the number of separate epidemics """
    T_ = years*365
    S0, I0 = endemic(N)
    def n_epidemics(I):
        on = I > 0
        return int(np.sum(on[1:] & ~on[:-1]))
    ex = [n_epidemics(exact_sir(N, beta_m, gamma_m, I0, T_, mu=mu_m, eps=eps, S0=S0, seed=s)[0]) for s in range(n)]
    print(f'imports eps={eps}: exact: mean re-introductions {np.mean(ex):.2f}')
    for solver in ['gillespie', 'binomial']:
        runs, _, err = capture(lambda: [r[0] for r in v4_runs(solver, n, N, beta_m, gamma_m, I0, T_, mu=mu_m, eps=eps, S0=S0)])
        if err:
            print(f'  {solver}: error {err[:300]}'); continue
        print(f'  {solver}: mean re-introductions {np.mean([n_epidemics(I) for I in runs]):.2f}')


#%% 6.1 and 6.2: noise added to the ODE

def noise():
    """ 6.1 (constant additive noise) has no v4 spelling; 6.2 (noise scaled as demographic noise) is the chemical Langevin equation, which `solver='sde'` should be """
    # 6.1: is there any way to add a noise term to an ODE? Try the obvious spellings
    S, I, R = ss.states(3)
    for label, fn in [
        ("ss.Sim(..., solver='sde', noise=1e-3)", lambda: ss.Sim(sir_model(beta_c, gamma_c, 50), total_pop=1000, unit='day', stop=50, solver='sde', noise=1e-3, verbose=0).run()),
        ("ss.Sim(..., solver='ode', integrator='sde')", lambda: ss.Sim(sir_model(beta_c, gamma_c, 50), total_pop=1000, unit='day', stop=50, integrator='sde', verbose=0).run()),
    ]:
        _, _, err = capture(fn)
        print(f'6.1 {label}: {"error " + err[:200] if err else "ran"}')
    # 6.2: sde against exact, at a size where the CLE should be good (N=10,000, I0=50)
    n, N, I0, T = 200, 10_000, 50, 150
    ex = summarize([exact_sir(N, beta_c, gamma_c, I0, T, seed=s) for s in range(n)], N)
    for solver in ['sde', 'gillespie']:
        runs, _, err = capture(lambda: v4_runs(solver, n, N, beta_c, gamma_c, I0, T))
        if err:
            print(f'6.2 {solver} error {err[:300]}'); continue
        res = summarize(runs, N)
        for k in ['final', 'peak', 'tpeak']:
            print(f'6.2 {solver:9s} {k:6s} mean {res[k].mean():9.1f} vs exact {ex[k].mean():9.1f}; sd {res[k].std():7.1f} vs {ex[k].std():7.1f}; KS p {stats.ks_2samp(res[k], ex[k]).pvalue:.3f}')


#%% 6.4: tau-leaping, which v4 reaches only as gillespie's switch above about a million events

def tau(N=10_000_000, I0=100, n=20, T=120):
    """ A closed SIR in ten million people (16 million events, above the 1 s budget): gillespie (which switches to tau-leaping, with a note) against gillespie with approx=False (exact), and the ODE """
    from common import ode
    t = np.arange(T+1.)
    y = ode(lambda t, y: [-beta_c*y[0]*y[1]/N, beta_c*y[0]*y[1]/N - gamma_c*y[1]], [N-I0, I0], t)
    print(f'6.4 ode: peak {y[1].max():.0f} on day {y[1].argmax()}, final R {N - y[0][-1]:.0f}')
    for label, kw in [('gillespie (default)', dict()), ('gillespie approx=False', dict(approx=False))]:
        T_ = sc.timer()
        pk, tp, fin, notes = [], [], [], set()
        for seed in range(n):
            sim, text, err = capture(lambda: ss.Sim(sir_model(beta_c, gamma_c, I0), total_pop=N, unit='day', stop=T, solver='gillespie', rand_seed=seed+1, **kw).run())
            if err:
                print(f'  {label}: error {err[:300]}'); break
            notes |= {l.strip()[:160] for l in text.splitlines() if 'tau' in l.lower()}
            I = series(sim, 'sir.n_I'); pk.append(I.max()); tp.append(I.argmax()); fin.append(series(sim, 'sir.infection').sum())
        if pk:
            print(f'6.4 {label}: peak {np.mean(pk):.0f} (sd {np.std(pk):.0f}), day {np.mean(tp):.1f} (sd {np.std(tp):.1f}), final {np.mean(fin):.0f} (sd {np.std(fin):.0f}), {T_.tocout(output=True):.1f} s; notes: {sorted(notes)[:2]}')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('tau', 'all'): tau()
    if what in ('closed', 'all'): closed()
    if what in ('noise', 'all'):  noise()
    if what in ('ccs', 'all'):    ccs()
    if what in ('imports', 'all'): imports()
