"""
Switching demo for the odin seat: the stochastic SIR from R, fitted by PMCMC, then as agents.

Passes when: distinct runs; the filter within 3x of dust per particle-step.
Blocked today by: N17, N5, N27, the odin importer (N13).
The model is the report's age-structured stochastic SEIR (4 ages, a contact matrix, seasonal forcing, a quarter-day step).
The step "from R" is tests/test_r.py (about 20 s, so not repeated here): building a model, and ss.parallel after init(), from R now work.
"""
import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

ref_final_R = 532_176 # Mean final R over 1,000 odin 1.2.7 runs: switching/odin.md, "The second week" §2
ref_step_us = 1.35 # odin's single-threaded forward run per particle-step, in µs: switching/odin.md, "The second week" §3
ref_cases = [3, 1, 12, 10, 32, 73, 116, 129, 246, 390, 740, 1094, 1950, 1599, 2598, 3189, 4885, 4012, 5096, 5655, 7789, 4031, 9381, 8058,
             8006, 5767, 5989, 3812, 4955, 3666, 3992, 3093, 5952, 4071, 3806, 4024, 2205, 3734, 3202, 1549] # Weekly cases from one odin run (cases.csv): switching/odin.md §3

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
    seir = ss.Disease(infection, E >> I @ ss.perday(0.2), I >> R @ gamma, cases, init=dict(I=ss.by(age=[10, 0, 0, 0])), name='seir')
    seir.stratify(age=ages, split=pop/pop.sum())
    data = pd.DataFrame(dict(time=7*np.arange(1, 41), cases=ref_cases)) if fitted else None
    return ss.Sim(seir, networks=ss.contacts(m, bands=ages), total_pop=pop.sum(), unit='day', dt=0.25, stop=280 if fitted else 365,
                  solver='binomial', use_aging=False, data=data, verbose=0)


def test_odin_import(tmp_path):
    sc.heading('Importing the odin model the user already has')
    path = tmp_path/'age_seir.R'
    path.write_text(odin_model)
    sim = ss.load(path, format='odin', pars=dict(S0=pop - [10, 0, 0, 0], I0=[10, 0, 0, 0], m=m), stop=365)
    final_R = np.asarray(sim.run(n_runs=20).results.age_seir.n_R.values)[:, -1].mean()
    assert np.isclose(final_R, ref_final_R, rtol=0.03), f'Imported model: mean final R {final_R:,.0f}, odin {ref_final_R:,}'
    return sim


def test_odin_runs():
    sc.heading('Running the hand-written model many times, after looking at it')
    sim = make_seir()
    sim.init() # As an odin user calls gen$new() before $run()
    msim = ss.parallel(sim, n_runs=20)
    final_R = np.asarray(msim.results.seir.n_R.values)[:, -1]
    assert len(np.unique(final_R)) == 20, f'Expected 20 distinct runs after init(), got {len(np.unique(final_R))} (N5)'
    assert np.isclose(final_R.mean(), ref_final_R, rtol=0.03), f'Mean final R {final_R.mean():,.0f}, odin {ref_final_R:,}'
    return msim


@pytest.mark.xfail(reason='N27: the particle filter is serial, about 11 µs (23 µs on a loaded machine) per particle-step against odin\'s 1.35 µs', strict=False)
def test_odin_filter():
    sc.heading('Timing the particle filter that PMCMC uses, per particle-step')
    sim = make_seir(fitted=True).init()
    pf = ss.ParticleFilter(sim)
    pf(n_particles=10, seed=0) # Compile
    n = 50
    T = sc.timer()
    loglik = pf(n_particles=n, seed=1)
    step_us = T.toc(output=True)/(n*sim.t.npts)*1e6
    assert np.isfinite(loglik)
    assert step_us < 3*ref_step_us, f'Filter {step_us:.1f} µs per particle-step, against 3 × {ref_step_us} µs for dust'
    # The fit itself, fit = sim.calibrate(), is 4 chains of particle MCMC at 1,000 particles; it is too slow for CI until this passes
    return step_us


@pytest.mark.xfail(reason='New: a whole-number seed below one agent rounds to zero with no note, so I=ss.by(age=[10, 0, 0, 0]) for 1M people gives no infections with 20,000 agents', strict=False)
def test_odin_agents():
    sc.heading('The same model as agents')
    agents = make_seir().new(solver='abm', n_agents=20_000, dt=1, stop=120).run()
    n_I = agents.results.seir.n_I.values
    assert n_I[0] > 0 and n_I.max() > 100*n_I[0], f'Expected an epidemic among the agents; infected {n_I[0]:.0f} at the start and {n_I.max():.0f} at most'
    return agents


if __name__ == '__main__':
    import tempfile
    sim   = test_odin_import(sc.path(tempfile.mkdtemp()))
    msim  = test_odin_runs()
    step  = test_odin_filter()
    agent = test_odin_agents()
