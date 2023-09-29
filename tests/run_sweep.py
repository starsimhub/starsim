"""
A simple beta sweep with multisim on and off using HIV as an example pathogen
"""

# %% Imports and settings
import os
import stisim as ss
import sciris as sc
import pandas as pd
import seaborn as sns

n = 1_000 # Agents
n_rand_seeds = 250
x_beta_levels = [0.5, 0.95, 1.05, 2] + [1] # Must include 1 as that's the baseline

figdir = os.path.join(os.getcwd(), 'figs', 'BetaSweep')
sc.path(figdir).mkdir(parents=True, exist_ok=True)

def run_sim(n, x_beta, rand_seed, multistream):
    ppl = ss.People(n)

    net_pars = {'multistream': multistream}
    ppl.networks = ss.ndict(ss.simple_embedding(pars=net_pars), ss.maternal(pars=net_pars))

    hiv_pars = {
        'beta': {'simple_embedding': [x_beta * 0.10, x_beta * 0.08], 'maternal': [x_beta * 0.2, 0]},
        'initial': 10,
        'multistream': multistream,
    }
    hiv = ss.HIV(hiv_pars)

    preg_pars = {'multistream': multistream}
    pregnancy = ss.Pregnancy(preg_pars)

    pars = {
        'start': 1980,
        'end': 2010,
        'rand_seed': rand_seed,
        'multistream': multistream,
    }
    sim = ss.Sim(people=ppl, modules=[hiv, pregnancy], pars=pars, label=f'Sim with {n} agents and x_beta={x_beta}')
    sim.initialize()
    sim.run()

    df = pd.DataFrame( {
        'ti': sim.tivec,
        #'hiv.n_infected': sim.results.hiv.n_infected,
        'hiv.prevalence': sim.results.hiv.prevalence,
        'hiv.cum_deaths': sim.results.hiv.new_deaths.cumsum(),
        'pregnancy.cum_births': sim.results.pregnancy.births.cumsum(),
    })
    df['x_beta'] = x_beta
    df['rand_seed'] = rand_seed
    df['multistream'] = multistream

    return df

def run_scenarios():
    results = []
    times = {}
    for multistream in [True, False]:
        cfgs = []
        for rs in range(n_rand_seeds):
            for x_beta in x_beta_levels:
                cfgs.append({'x_beta':x_beta, 'rand_seed':rs, 'multistream':multistream})
        T = sc.tic()
        results += sc.parallelize(run_sim, kwargs={'n': n}, iterkwargs=cfgs, die=True)
        times[f'Multistream={multistream}'] = sc.toc(T, output=True)

    print('Timings:', times)

    df = pd.concat(results)
    df.to_csv(os.path.join(figdir, 'result.csv'))
    return df


def plot_scenarios(df):
    d = pd.melt(df, id_vars=['ti', 'rand_seed', 'x_beta', 'multistream'], var_name='channel', value_name='Value')
    d['baseline'] = d['x_beta']==1
    bl = d.loc[d['baseline']]
    scn = d.loc[~d['baseline']]
    bl = bl.set_index(['ti', 'channel', 'rand_seed', 'x_beta', 'multistream'])[['Value']].reset_index('x_beta')
    scn = scn.set_index(['ti', 'channel', 'rand_seed', 'x_beta', 'multistream'])[['Value']].reset_index('x_beta')
    mrg = scn.merge(bl, on=['ti', 'channel', 'rand_seed', 'multistream'], suffixes=('', '_ref'))
    mrg['Value - Reference'] = mrg['Value'] - mrg['Value_ref']
    mrg = mrg.sort_index()

    fkw = {'sharey': False, 'sharex': 'col', 'margin_titles': True}

    ## TIMESERIES
    g = sns.relplot(kind='line', data=d, x='ti', y='Value', hue='x_beta', col='channel', row='multistream',
        height=5, aspect=1.2, palette='Set1', errorbar='sd', lw=2, facet_kws=fkw)
    g.set_titles(col_template='{col_name}', row_template='Multistream: {row_name}')
    g.set_xlabels(r'$t_i$')
    g.fig.savefig(os.path.join(figdir, 'timeseries.png'), bbox_inches='tight', dpi=300)

    ## DIFF TIMESERIES
    for ms, mrg_by_ms in mrg.groupby('multistream'):
        g = sns.relplot(kind='line', data=mrg_by_ms, x='ti', y='Value - Reference', hue='x_beta', col='channel', row='x_beta',
            height=3, aspect=1.0, palette='Set1', estimator=None, units='rand_seed', lw=0.5, facet_kws=fkw) #errorbar='sd', lw=2, 
        g.set_titles(col_template='{col_name}', row_template='Coverage: {row_name}')
        g.fig.suptitle('Multistream' if ms else 'Centralized')
        g.fig.subplots_adjust(top=0.88)
        g.set_xlabels(r'Value - Reference at $t_i$')
        g.fig.savefig(os.path.join(figdir, 'diff_multistream.png' if ms else 'diff_centralized.png'), bbox_inches='tight', dpi=300)

    ## FINAL TIME
    tf = df['ti'].max()
    mtf = mrg.loc[tf]
    g = sns.displot(data=mtf.reset_index(), kind='kde', fill=True, rug=True, cut=0, hue='x_beta', x='Value - Reference', col='channel', row='multistream',
        height=5, aspect=1.2, facet_kws=fkw, palette='Set1')
    g.set_titles(col_template='{col_name}', row_template='Multistream: {row_name}')
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