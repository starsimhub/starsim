"""
Run a Starsim port and compare its results with the original's, exactly or statistically

Both sides are long CSV files with columns `time`, `series`, `value`, and either `run`
(one row per run, for replicates) or `low` and `high` (a summary of many runs, e.g.
the 5% and 95% quantiles), as written by `lingua_franca_design/examples/tools.py`.

    # Run the port's `sim` (or `make_sim()`) from a Python file, 100 times, and write its results
    python compare.py run model.py port.csv --runs 100

    # Compare: exactly if both sides are one run each, statistically otherwise
    python compare.py compare original.csv port.csv --map I=sir.n_I R=sir.n_R

Needs NumPy, pandas, and SciPy (and Starsim for `run`).
"""
import sys
import argparse
import runpy
import numpy as np
import pandas as pd
from scipy import stats


#%% Running the port

def load_sim(filename):
    """ The sim defined by a Python file: its `sim`, or what its `make_sim()` returns """
    ns = runpy.run_path(filename, run_name='__validate__') # Not '__main__', so the file's own run does not happen
    if 'make_sim' in ns:
        return ns['make_sim']()
    elif 'sim' in ns:
        return ns['sim']
    errormsg = f'{filename} defines neither "sim" nor "make_sim()"'
    raise ValueError(errormsg)


def times(sim, mode='auto'):
    """ The time of each result: in the sim's unit from its start, or in years if it has dates, or the step ('index') """
    tv = sim.timevec
    if mode == 'index':
        return np.arange(len(tv), dtype=float)
    try:
        return np.array(tv, dtype=float)
    except (TypeError, ValueError):
        return np.array([t.years for t in tv], dtype=float)


def results(sim, series=None, time='auto', runs=None):
    """ The sim's results as long rows: one per time and series, named <module>.<result>; with runs (a MultiSim's results), one per run too """
    import starsim as ss
    t = times(sim, time)
    rows = []
    for modname,res in sim.results.items():
        if not isinstance(res, ss.Results):
            continue
        for key,r in res.items():
            name = f'{modname}.{key}'
            if not isinstance(r, ss.Result) or np.ndim(r.values) != 1 or (series and name not in series):
                continue
            if runs is None:
                rows += [dict(time=ti, series=name, value=float(v)) for ti,v in zip(t, r.values)]
            else: # Each run's values, as rows of an array
                for i,vals in enumerate(getattr(getattr(runs, modname), key).values):
                    rows += [dict(run=i, time=ti, series=name, value=float(v)) for ti,v in zip(t, vals)]
    return rows


def run(filename, out, n_runs=1, seed=1, series=None, time='auto', n_cpus=None):
    """ Run a port n_runs times (with seeds seed, seed+1, ...), on at most n_cpus processes, and write its results """
    import starsim as ss
    sim = load_sim(filename)
    sim.pars.verbose = 0
    sim.pars.rand_seed = seed
    rows = []
    if n_runs == 1:
        rows = [dict(run=0, **row) for row in results(sim.run(), series, time)]
    else:
        msim = ss.parallel(sim, n_runs=n_runs, n_cpus=n_cpus, verbose=0)
        if len(msim.sims) < n_runs: # Restarted in this process: one sim, with every run in msim.results
            rows = results(msim.sims[0], series, time, runs=msim.results)
        else:
            for i,s in enumerate(msim.sims):
                rows += [dict(run=i, **row) for row in results(s, series, time)]
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f'Wrote {len(df)} rows ({n_runs} run{"s" if n_runs > 1 else ""}, {df.series.nunique()} series) to {out}')
    return df


#%% Comparing

def read(filename):
    """ Read a long CSV of results """
    df = pd.read_csv(filename)
    missing = {'time', 'series', 'value'} - set(df.columns)
    if missing:
        errormsg = f'{filename} has no column(s) {sorted(missing)}; it needs time, series, value, and run or low/high'
        raise ValueError(errormsg)
    df['series'] = df.series.astype(str)
    return df


def kind(df):
    """ 'runs' (replicates), 'summary' (a mean with low and high), or 'single' (one run) """
    if 'run' in df.columns and df['run'].nunique() > 1:
        return 'runs'
    elif {'low', 'high'} <= set(df.columns) and df['low'].notna().any():
        return 'summary'
    return 'single'


def matrix(df, name, t):
    """ A series as an array of runs by times, at the times t (interpolated where they differ) """
    sub = df[df.series == name]
    groups = sub.groupby('run') if 'run' in sub.columns else [(0, sub)]
    out = []
    for _,g in groups:
        g = g.sort_values('time')
        out.append(np.interp(t, g.time.values, g.value.values))
    return np.array(out)


def summaries(m, t):
    """ The peak, the time of the peak, and the final value of each run """
    return dict(peak=m.max(axis=1), peak_time=t[m.argmax(axis=1)], final=m[:, -1])


quantiles = [0.05, 0.95] # What low and high are, in a summary of many runs


def compare_series(a, b, name_a, name_b, rtol=None, alpha=0.01, shift=0):
    """ Compare one series: the original (a) and the port (b) """
    ka, kb = kind(a), kind(b)
    ta = np.sort(a[a.series == name_a].time.unique())
    tb = np.sort(b[b.series == name_b].time.unique()) + shift
    t = ta[(ta >= tb.min()) & (ta <= tb.max())] # The original's times that the port covers
    bb = b.assign(time=b.time + shift)
    ma, mb = matrix(a, name_a, t), matrix(bb, name_b, t)
    scale = max(np.abs(ma).max(), np.abs(mb).max(), 1e-12)
    row = dict(series=name_a if name_a == name_b else f'{name_a} ~ {name_b}', times=len(t))

    if ka == 'single' and kb == 'single': # Exact: both deterministic
        rtol = 1e-3 if rtol is None else rtol
        err = np.abs(ma[0] - mb[0])
        row.update(mode='exact', max_abs=err.max(), max_rel=err.max()/scale, tol=rtol, ok=bool(err.max()/scale <= rtol))
        return row

    if ka == 'summary': # The original is a mean and a band: the port's mean and the same quantiles of its runs, like for like
        sub = a[a.series == name_a].sort_values('time')
        low, high = [np.interp(t, sub.time.values, sub[k].values) for k in ['low', 'high']]
        qlow, qhigh = [np.quantile(mb, q, axis=0, method='lower') for q in quantiles] # As tools.py's summaries: the sorted value at int(q*(n - 1))
        rtol = 0.1 if rtol is None else rtol
        row.update(mode='band', max_rel=np.abs(ma[0] - mb.mean(axis=0)).max()/scale, low_rel=np.abs(low - qlow).max()/scale, high_rel=np.abs(high - qhigh).max()/scale, tol=rtol)
        row['ok'] = bool(max(row['max_rel'], row['low_rel'], row['high_rel']) <= rtol)
        return row

    # Statistical: runs on both sides (or a single run of a deterministic original against the port's runs)
    mean_a, mean_b = ma.mean(axis=0), mb.mean(axis=0)
    var_a = ma.var(axis=0, ddof=1)/len(ma) if len(ma) > 1 else 0 # A deterministic original has no spread
    se = np.sqrt(var_a + mb.var(axis=0, ddof=1)/len(mb))
    z = np.divide(mean_a - mean_b, se, out=np.zeros_like(se), where=se > 0)
    row.update(mode='statistical', runs=f'{len(ma)}/{len(mb)}', max_rel=np.abs(mean_a - mean_b).max()/scale, frac_z3=np.mean(np.abs(z) > 3))
    ok = row['frac_z3'] <= 0.05
    if len(ma) > 1:
        sa, sb = summaries(ma, t), summaries(mb, t)
        for key in sa:
            p = stats.ks_2samp(sa[key], sb[key]).pvalue if np.ptp(np.r_[sa[key], sb[key]]) > 0 else 1.0
            row[f'ks_{key}'] = p
            ok = ok and p >= alpha
    row['ok'] = bool(ok)
    return row


def compare(file_a, file_b, mapping=None, rtol=None, alpha=0.01, shift=0, plot=None):
    """ Compare every matched series of the original (file_a) and the port (file_b) """
    a, b = read(file_a), read(file_b)
    mapping = dict(mapping or {})
    if not mapping: # Series with the same name, or the same last part (I ~ sir.n_I is not guessed: give --map)
        names_b = set(b.series)
        mapping = {s:s for s in a.series.unique() if s in names_b}
    if not mapping:
        errormsg = f'No series in common; give --map ORIGINAL=PORT. The original has {sorted(a.series.unique())[:10]}, the port {sorted(b.series.unique())[:10]}'
        raise ValueError(errormsg)
    rows = [compare_series(a, b, na, nb, rtol=rtol, alpha=alpha, shift=shift) for na,nb in mapping.items()]
    df = pd.DataFrame(rows)
    with pd.option_context('display.width', 200, 'display.max_columns', 20, 'display.float_format', '{:.3g}'.format):
        print(df.to_string(index=False))
    print(f'\n{"All series agree" if df.ok.all() else f"{(~df.ok).sum()} of {len(df)} series differ"} (original {kind(a)}, port {kind(b)})')
    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(mapping), 1, figsize=(8, 2.5*len(mapping)), squeeze=False)
        for ax,(na,nb) in zip(axes[:, 0], mapping.items()):
            for df_,name,label,dt in [(a, na, 'original', 0), (b, nb, 'port', shift)]:
                sub = df_[df_.series == name].assign(time=lambda d: d.time + dt)
                g = sub.groupby('time').value
                ax.plot(g.mean().index, g.mean().values, label=label)
                if 'low' in sub and sub['low'].notna().any():
                    ax.fill_between(sub.time, sub['low'], sub['high'], alpha=0.2)
                elif sub.get('run') is not None and sub['run'].nunique() > 1:
                    ax.fill_between(g.mean().index, g.quantile(0.05).values, g.quantile(0.95).values, alpha=0.2)
            ax.set_title(na if na == nb else f'{na} ~ {nb}')
            ax.legend()
        fig.tight_layout()
        fig.savefig(plot)
        print(f'Plot saved to {plot}')
    return df


#%% Command line

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('run', help='Run a Starsim port and write its results')
    p.add_argument('model', help='a Python file defining sim or make_sim()')
    p.add_argument('out', help='the CSV to write')
    p.add_argument('--runs', type=int, default=1)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--cpus', type=int, help='the most processes to run on (default: every core)')
    p.add_argument('--series', nargs='*', help='the results to keep, e.g. sir.n_I (default: all)')
    p.add_argument('--time', default='auto', choices=['auto', 'index'], help="the time column: the sim's time, or the step")
    p = sub.add_parser('compare', help='Compare the original (first) with the port (second)')
    p.add_argument('original')
    p.add_argument('port')
    p.add_argument('--map', nargs='*', default=[], help='ORIGINAL=PORT pairs of series names')
    p.add_argument('--rtol', type=float, help='tolerance: relative to the largest value (default 1e-3 exact, 0.1 against a band)')
    p.add_argument('--alpha', type=float, default=0.01, help='significance for the Kolmogorov-Smirnov tests')
    p.add_argument('--shift', type=float, default=0, help="added to the port's times, e.g. 1 if it labels each value one step earlier")
    p.add_argument('--plot', help='save a plot of each series to this file')
    args = parser.parse_args(argv)
    if args.cmd == 'run':
        run(args.model, args.out, n_runs=args.runs, seed=args.seed, series=args.series, time=args.time, n_cpus=args.cpus)
    else:
        df = compare(args.original, args.port, mapping=dict(m.split('=', 1) for m in args.map), rtol=args.rtol, alpha=args.alpha, shift=args.shift, plot=args.plot)
        return 0 if df.ok.all() else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
