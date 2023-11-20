"""
Exploring multisim (on vs off) for a model HIV system sweeping coverage of either ART or PrEP.
"""

# %% Imports and settings
import os
import stisim as ss
import sciris as sc
import pandas as pd
import seaborn as sns
import numpy as np

# Suppress warning from seaborn
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

n = 100 # Agents
n_rand_seeds = 25
intv_cov_levels = [0.01, 0.10, 0.25, 0.73] + [0] # Must include 0 as that's the baseline

figdir = os.path.join(os.getcwd(), 'figs', 'ART')
sc.path(figdir).mkdir(parents=True, exist_ok=True)

def run_sim(n, idx, intv_cov, rand_seed, multirng):

    print(f'Starting sim {idx} with rand_seed={rand_seed} and intv_cov={intv_cov}, multirng={multirng}')

    ppl = ss.People(n)

    ppl.networks = ss.ndict(
        ss.embedding(pars={'dur': ss.lognormal(5, 3)}),
        ss.maternal()
        )

    hiv_pars = {
        'beta': {'embedding': [0.2, 0.15], 'maternal': [0.3, 0]},
        'init_prev': np.maximum(10/n, 0.01),
        'art_efficacy': 0.96,
    }
    hiv = ss.HIV(hiv_pars)

    pregnancy = ss.Pregnancy()
    deaths = ss.background_deaths()

    pars = {
        'start': 1980,
        'end': 2070,
        'rand_seed': rand_seed,
        'verbose': 0,
        'remove_dead': True,
    }

    if intv_cov > 0:
        pars['interventions'] = [ ss.hiv.ART(t=[0, 10, 20], coverage=[0, intv_cov/3, intv_cov]) ]

    sim = ss.Sim(people=ppl, diseases=[hiv], demographics=[pregnancy, deaths], pars=pars, label=f'Sim with {n} agents and intv_cov={intv_cov}')
    sim.initialize()
    sim.run()

    df = pd.DataFrame( {
        'ti': sim.tivec,
        #'hiv.n_infected': sim.results.hiv.n_infected, # Optional, but mostly redundant with prevalence
        'hiv.prevalence': sim.results.hiv.prevalence,
        'hiv.cum_deaths': sim.results.hiv.new_deaths.cumsum(),
        'pregnancy.cum_births': sim.results.pregnancy.births.cumsum(),
    })
    df['intv_cov'] = intv_cov
    df['rand_seed'] = rand_seed
    df['multirng'] = multirng

    print(f'Finishing sim {idx} with rand_seed={rand_seed} and intv_cov={intv_cov}, multirng={multirng}')

    return df

def run_scenarios():
    results = []
    times = {}
    for multirng in [True, False]:
        ss.options(multirng=multirng)
        cfgs = []
        for rs in range(n_rand_seeds):
            for intv_cov in intv_cov_levels:
                cfgs.append({'intv_cov':intv_cov, 'rand_seed':rs, 'multirng':multirng, 'idx':len(cfgs)})
        T = sc.tic()
        results += sc.parallelize(run_sim, kwargs={'n': n}, iterkwargs=cfgs, die=True, serial=False)
        times[f'multirng={multirng}'] = sc.toc(T, output=True)

    print('Timings:', times)

    df = pd.concat(results)
    df.to_csv(os.path.join(figdir, 'result.csv'))
    return df

def plot_scenarios(df):
    d = pd.melt(df, id_vars=['ti', 'rand_seed', 'intv_cov', 'multirng'], var_name='channel', value_name='Value')
    d['baseline'] = d['intv_cov']==0
    bl = d.loc[d['baseline']]
    scn = d.loc[~d['baseline']]
    bl = bl.set_index(['ti', 'channel', 'rand_seed', 'intv_cov', 'multirng'])[['Value']].reset_index('intv_cov')
    scn = scn.set_index(['ti', 'channel', 'rand_seed', 'intv_cov', 'multirng'])[['Value']].reset_index('intv_cov')
    mrg = scn.merge(bl, on=['ti', 'channel', 'rand_seed', 'multirng'], suffixes=('', '_ref'))
    mrg['Value - Reference'] = mrg['Value'] - mrg['Value_ref']
    mrg = mrg.sort_index()

    fkw = {'sharey': False, 'sharex': 'col', 'margin_titles': True}

    ## TIMESERIES
    g = sns.relplot(kind='line', data=d, x='ti', y='Value', hue='intv_cov', col='channel', row='multirng',
        height=5, aspect=1.2, palette='Set1', errorbar='sd', lw=2, facet_kws=fkw)
    g.set_titles(col_template='{col_name}', row_template='multirng: {row_name}')
    g.set_xlabels(r'$t_i$')
    g.fig.savefig(os.path.join(figdir, 'timeseries.png'), bbox_inches='tight', dpi=300)

    ## DIFF TIMESERIES
    for ms, mrg_by_ms in mrg.groupby('multirng'):
        g = sns.relplot(kind='line', data=mrg_by_ms, x='ti', y='Value - Reference', hue='intv_cov', col='channel', row='intv_cov',
            height=3, aspect=1.0, palette='Set1', estimator=None, units='rand_seed', lw=0.5, facet_kws=fkw) #errorbar='sd', lw=2, 
        g.set_titles(col_template='{col_name}', row_template='Coverage: {row_name}')
        g.fig.suptitle('MultiRNG' if ms else 'SingleRNG')
        g.fig.subplots_adjust(top=0.88)
        g.set_xlabels(r'Timestep $t_i$')
        g.fig.savefig(os.path.join(figdir, 'diff_multi.png' if ms else 'diff_single.png'), bbox_inches='tight', dpi=300)

    ## FINAL TIME
    tf = df['ti'].max()
    mtf = mrg.loc[tf]
    g = sns.displot(data=mtf.reset_index(), kind='kde', fill=True, rug=True, cut=0, hue='intv_cov', x='Value - Reference', col='channel', row='multirng',
        height=5, aspect=1.2, facet_kws=fkw, palette='Set1')
    g.set_titles(col_template='{col_name}', row_template='multirng: {row_name}')
    g.set_xlabels(f'Value - Reference at $t_i={{{tf}}}$')
    g.fig.savefig(os.path.join(figdir, 'final.png'), bbox_inches='tight', dpi=300)

    print('Figures saved to:', os.path.join(os.getcwd(), figdir))

    return


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plot', help='plot from a cached CSV file', type=str)
    args = parser.parse_args()

    if args.plot:
        print('Reading CSV file', args.plot)
        df = pd.read_csv(args.plot, index_col=0)
    else:
        print('Running scenarios')
        df = run_scenarios()

    plot_scenarios(df)

    print('Done')