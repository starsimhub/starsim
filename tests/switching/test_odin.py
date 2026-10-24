"""
Switching demo for the odin seat: the stochastic SIR from R, fitted by PMCMC, then as agents.

Passes when: distinct runs; the filter within 3x of dust per particle-step.
Blocked today by: N17, N5, the odin importer (N13).
The model is the report's age-structured stochastic SEIR (4 ages, a contact matrix, seasonal forcing, a quarter-day step).
The step "from R" is tests/test_r.py (about 20 s, so not repeated here): building a model, and ss.parallel after init(), from R now work.
"""
import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import starsim.fit
import starsim.interop

ref_final_R = 532_176 # Mean final R over 1,000 odin 1.2.7 runs: switching/odin.md, "The second week" §2
ref_step_us = 1.35 # odin's single-threaded forward run per particle-step, in µs: switching/odin.md, "The second week" §3
ref_cases = [3, 1, 12, 10, 32, 73, 116, 129, 246, 390, 740, 1094, 1950, 1599, 2598, 3189, 4885, 4012, 5096, 5655, 7789, 4031, 9381, 8058,
             8006, 5767, 5989, 3812, 4955, 3666, 3992, 3093, 5952, 4071, 3806, 4024, 2205, 3734, 3202, 1549] # Weekly cases from one odin run (cases.csv): switching/odin.md §3

ref_delay = dict(peak=6_185_515, day=125, S=334_138, E=1_305_913, I=2_792_751, R=5_567_199) # odin's seir_array (a 14-day delay() solved by deSolve::dede): the peak of I and its day, and each state on day 365 (LFD examples/odin/seir_array/reference)
ref_delay_run = 0.002 # odin's run of seir_array, in s (the same reference's timing.json); with 200 Erlang stages for the delay, Starsim took 1.9 s and was 1.6% off

m = [[12, 4, 2, 1], [4, 8, 3, 1], [2, 3, 6, 2], [1, 1, 2, 3]]
pop = np.array([200_000, 300_000, 350_000, 150_000])
ages = [0, 5, 18, 65]
odin_model = '\n'.join([ # The report's age_seir.R, without its weekly-cases line
    'n_age <- 4', 'dt <- user(0.25)', 'beta <- user(0.012)', 'amp <- user(0.3)', 'S0[] <- user()', 'I0[] <- user()', 'm[, ] <- user()',
    'N[] <- S[i] + E[i] + I[i] + R[i]', 'seas <- 1 + amp * cos(2 * 3.141592653589793 * step * dt / 365)',
    's_ij[, ] <- m[i, j] * I[j] / N[j]', 'lambda[] <- beta * seas * sum(s_ij[i, ])',
    'n_SE[] <- rbinom(S[i], 1 - exp(-lambda[i] * dt))', 'n_EI[] <- rbinom(E[i], 1 - exp(-0.2 * dt))', 'n_IR[] <- rbinom(I[i], 1 - exp(-0.1 * dt))',
    'update(S[]) <- S[i] - n_SE[i]', 'update(E[]) <- E[i] + n_SE[i] - n_EI[i]', 'update(I[]) <- I[i] + n_EI[i] - n_IR[i]', 'update(R[]) <- R[i] + n_IR[i]',
    'initial(S[]) <- S0[i]', 'initial(E[]) <- 0', 'initial(I[]) <- I0[i]', 'initial(R[]) <- 0',
    'dim(S0) <- n_age', 'dim(I0) <- n_age', 'dim(m) <- c(n_age, n_age)', 'dim(N) <- n_age', 'dim(s_ij) <- c(n_age, n_age)', 'dim(lambda) <- n_age',
    'dim(n_SE) <- n_age', 'dim(n_EI) <- n_age', 'dim(n_IR) <- n_age', 'dim(S) <- n_age', 'dim(E) <- n_age', 'dim(I) <- n_age', 'dim(R) <- n_age',
])


def make_seir(fitted=False):
    """ The odin model by hand: 10 lines against odin's 46; with fitted=True, beta, gamma, and reporting are fitted to weekly cases """
    beta  = ss.par(ss.perday(0.012), bounds=[0.005, 0.05]) if fitted else ss.seasonal(mean=ss.perday(0.012), amplitude=0.3, peak=0)
    gamma = ss.par(ss.perday(0.1), bounds=[0.05, 0.3]) if fitted else ss.perday(0.1)
    rho   = ss.par(0.3, bounds=[0.05, 1])
    S, E, I, R = ss.states(4)
    infection = S >> E @ beta * I
    cases = ss.outcome(rho * infection, likelihood=ss.nbinom(n=20)) # Compared with weekly data by its total over each week
    seir = ss.Disease(infection, E >> I @ ss.perday(0.2), I >> R @ gamma, cases, init_n=dict(I=ss.by(age=[10, 0, 0, 0])), name='seir')
    seir.stratify(age=ages, split=pop/pop.sum())
    data = pd.DataFrame(dict(time=7*np.arange(1, 41), cases=ref_cases)) if fitted else None
    return ss.Sim(seir, networks=ss.contacts(m, ages=ages), total_pop=pop.sum(), unit='day', dt=0.25, stop=280 if fitted else 365,
                  solver='binomial', use_aging=False, data=data, verbose=0)


def test_odin_import(tmp_path):
    sc.heading('Importing the odin model the user already has')
    path = tmp_path/'age_seir.R'
    path.write_text(odin_model)
    sim = ss.interop.load(path, format='odin', pars=dict(S0=pop - [10, 0, 0, 0], I0=[10, 0, 0, 0], m=m), stop=365)
    final_R = np.asarray(sim.run(n_runs=20).results.age_seir.R.values)[:, -1].mean()
    assert np.isclose(final_R, ref_final_R, rtol=0.03), f'Imported model: mean final R {final_R:,.0f}, odin {ref_final_R:,}'
    return sim


def test_odin_runs():
    sc.heading('Running the hand-written model many times, after looking at it')
    sim = make_seir()
    sim.init() # As an odin user calls gen$new() before $run()
    msim = ss.parallel(sim, n_runs=20)
    final_R = np.asarray(msim.results.seir.R.values)[:, -1]
    assert len(np.unique(final_R)) == 20, f'Expected 20 distinct runs after init(), got {len(np.unique(final_R))} (N5)'
    assert np.isclose(final_R.mean(), ref_final_R, rtol=0.03), f'Mean final R {final_R.mean():,.0f}, odin {ref_final_R:,}'
    return msim


def test_odin_filter():
    sc.heading('Timing the particle filter that PMCMC uses, per particle-step')
    sim = make_seir(fitted=True).init()
    pf = ss.fit.ParticleFilter(sim)
    pf(n_particles=10, seed=0) # Compile
    n = 50
    T = sc.timer()
    loglik = pf(n_particles=n, seed=1)
    step_us = T.toc(output=True)/(n*sim.t.npts)*1e6
    assert np.isfinite(loglik)
    assert step_us < 3*ref_step_us, f'Filter {step_us:.1f} µs per particle-step, against 3 × {ref_step_us} µs for dust'
    # The fit itself, fit = sim.fit(method='pmcmc'), is 4 chains of particle MCMC at 1,000 particles; it is too slow for CI until this passes
    return step_us


def test_odin_delay():
    sc.heading('The age-structured SEIR with a latent period of exactly 14 days, a delay() in odin and a fixed delay here')
    b = 1/50 # Births and deaths, per year
    aging = np.array([1, 1/4, 1/10, 1/15, 0]) # One over the width of each age group, per year
    split = b*np.cumprod(np.append(1, aging[:-1])/(aging + b)) # The stable age distribution
    surv = np.exp(-b*14/365) # The share of the latent who live 14 days
    S, E, I, R = ss.states(4)
    seir = ss.Disease(
        infection = S >> E @ ss.perday(1) * I,
        onset     = E >> I @ ss.constant(ss.days(14)), # A conveyor of 14 steps, not stiff Erlang stages
        recovery  = I >> R @ ss.days(30),
        waning    = R >> S @ ss.days(60),
        births    = ss.birth(S, rate=ss.peryear(b)),
        deaths    = ss.death(rate=ss.peryear(b)),
        history   = ss.null >> I @ ss.timeseries({0: ss.perday(surv), 14: 0}), # odin's delay() takes the first 14 days from before the start (see the vignette's README)
        init_n    = dict(I=1),
        name      = 'seir',
    )
    seir.stratify(age=[0, 1, 5, 15, 30], split=split)
    sim = ss.Sim(seir, total_pop=1e7, unit='day', stop=365, verbose=0).init()
    sim.run()
    res = sim.results.seir
    n_I = res.I.values
    assert n_I.argmax() == ref_delay['day'] and np.isclose(n_I.max(), ref_delay['peak'], rtol=0.003), f'Peak of {n_I.max():,.0f} on day {n_I.argmax()}, against odin\'s {ref_delay["peak"]:,} on day {ref_delay["day"]}'
    for state in 'SEIR':
        assert np.isclose(res[state][-1], ref_delay[state], rtol=0.005), f'{state} on day 365: {res[state][-1]:,.0f}, against odin\'s {ref_delay[state]:,}'
    T = sc.timer()
    sim.restart()
    sim.run()
    seconds = T.toc(output=True)
    assert seconds < 50*ref_delay_run, f'A run took {seconds:.3f} s, against {ref_delay_run} s for odin'
    return sim


@pytest.mark.xfail(reason='SW8: a seed below one agent rounds to zero (now with a note), so I=ss.by(age=[10, 0, 0, 0]) for 1M people gives no infections with 20,000 agents; whether to round stochastically or raise is open', strict=False)
def test_odin_agents():
    sc.heading('The same model as agents')
    agents = make_seir().new(solver='abm', n_agents=20_000, dt=1, stop=120).run()
    n_I = agents.results.seir.I.values
    assert n_I[0] > 0 and n_I.max() > 100*n_I[0], f'Expected an epidemic among the agents; infected {n_I[0]:.0f} at the start and {n_I.max():.0f} at most'
    return agents


if __name__ == '__main__':
    import tempfile
    sim   = test_odin_import(sc.path(tempfile.mkdtemp()))
    msim  = test_odin_runs()
    step  = test_odin_filter()
    delay = test_odin_delay()
    agent = test_odin_agents()
