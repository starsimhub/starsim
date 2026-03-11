#!/usr/bin/env python3
"""
Benchmark a few representative simulation scenarios.

Examples:
    ./benchmark_examples.py
    ./benchmark_examples.py repeats=2
    ./benchmark_examples.py case=complex_30
"""
import sciris as sc
import starsim as ss
import starsim_examples as sse

sc.options(interactive=False)


def make_simple_setup():
    """A tiny run where setup costs dominate the total runtime."""
    return ss.Sim(
        n_agents=50_000,
        dur=1,
        dt=1,
        rand_seed=1,
        verbose=0,
        diseases=ss.SIS(),
        networks=ss.RandomNet(),
        label='simple_setup',
    )


def make_complex_30():
    """A shorter run with substantially more model structure."""
    hiv = sse.HIV()
    hiv.pars['beta'] = {'mf': [0.15, 0.10], 'maternal': [0.2, 0], 'random': [0, 0]}

    return ss.Sim(
        n_agents=80_000,
        dur=15,
        dt=0.5,  # 30 timesteps
        rand_seed=1,
        verbose=0,
        diseases=[ss.SIS(), hiv],
        networks=[ss.RandomNet(), ss.MFNet(), ss.MaternalNet()],
        demographics=ss.Pregnancy(),
        label='complex_30',
    )


def make_moderate():
    """A moderately complex run pushed by many timesteps."""
    return ss.Sim(
        n_agents=45_000,
        dur=30,
        dt=0.1,  # 300 timesteps
        rand_seed=1,
        verbose=0,
        diseases=[ss.SIR(), ss.SIS()],
        networks=[ss.RandomNet(), ss.MFNet()],
        demographics=True,
        label='moderate_300',
    )


CASES = sc.objdict(
    simple_setup=sc.objdict(
        label='Simple setup-dominated simulation',
        timesteps=1,
        repeats=7,
        factory=make_simple_setup,
    ),
    complex=sc.objdict(
        label='High-complexity simulation with 30 timesteps',
        timesteps=30,
        repeats=2,
        factory=make_complex_30,
    ),
    moderate=sc.objdict(
        label='Moderate-complexity simulation with 300 timesteps',
        timesteps=100,
        repeats=3,
        factory=make_moderate,
    ),
)


def benchmark_case(name, case, repeats=None):
    repeats = sc.ifelse(repeats, case.repeats)
    init_times = []
    run_times = []

    for r in range(repeats):
        print(f'  Repeat {r + 1}/{repeats}')

        sim = case.factory()

        t0 = sc.tic()
        sim.init()
        init_times.append(sc.toc(t0, output=True))

        t0 = sc.tic()
        sim.run()
        run_times.append(sc.toc(t0, output=True))

    row = sc.objdict(
        case=name,
        label=case.label,
        n_agents=int(sim.pars.n_agents),
        timesteps=case.timesteps,
        init_best=round(min(init_times), 3),
        init_mean=round(sum(init_times) / len(init_times), 3),
        run_best=round(min(run_times), 3),
        run_mean=round(sum(run_times) / len(run_times), 3),
        total_best=round(min(i + r for i, r in zip(init_times, run_times)), 3),
        total_mean=round((sum(init_times) + sum(run_times)) / repeats, 3),
    )
    return row


def get_cases(selection):
    if selection in [None, 'all']:
        return CASES
    if selection not in CASES:
        errormsg = f'Invalid case "{selection}"; choices are: {sc.strjoin(CASES.keys())}, all'
        raise ValueError(errormsg)
    return sc.objdict({selection: CASES[selection]})


def benchmark_examples(repeats=None, case='all'):
    selected = get_cases(case)
    rows = []

    for name, spec in selected.items():
        sc.heading(f'Benchmarking: {spec.label}')
        rows.append(benchmark_case(name, spec, repeats=repeats))
        print()

    df = sc.dataframe(rows)
    df = df[['case', 'n_agents', 'timesteps', 'init_best', 'init_mean', 'run_best', 'run_mean', 'total_best', 'total_mean']]
    print('Summary (seconds):')
    print(df.round(3).to_string(index=False))
    return df


if __name__ == '__main__':
    args = sc.argparse(repeats=None, case='all')
    benchmark_examples(repeats=args.repeats, case=args.case)
