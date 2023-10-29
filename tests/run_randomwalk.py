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

n = 1 # Agents
n_rand_seeds = 100
p_levels = [0.51, 0.55, 0.75, 1] + [0.5] # Must include 0.5 as that's the baseline

figdir = os.path.join(os.getcwd(), 'figs', 'RandomWalk')
sc.path(figdir).mkdir(parents=True, exist_ok=True)

class RandomWalk(ss.Disease):
    def __init__(self, pars=None):
        super().__init__(pars)

        self.position = ss.State('position', int, 0)
        self.rng_choose = ss.Stream('choose')
        self.rng_step = ss.Stream('step')

        self.pars = ss.omerge({
            'p': 0.5, # Probability of stepping to the right
        }, self.pars)
        return

    def update_states(self, sim):
        alive_ids = ss.true(sim.people.alive)
        #self.position[alive_ids] += 2*self.rng_step.bernoulli(alive_ids, prob=self.pars['p']) - 1

        # Pick individuals to move with probabilty p
        move_ids = self.rng_choose.bernoulli_filter(uids=alive_ids, prob=0.5) # self.pars['p_move']
        self.position[move_ids] += 2*self.rng_step.bernoulli(move_ids, prob=self.pars['p']) - 1

        return

    def init_results(self, sim):
        super().init_results(sim)
        self.results += ss.Result(self.name, 'mean', sim.npts, dtype=int)
        return

    def update_results(self, sim):
        super(RandomWalk, self).update_results(sim)
        self.results['mean'][sim.ti] = sim.people.randomwalk.position.mean() 
        return

    def make_new_cases(self, sim):
        pass # No new cases

    def set_prognoses(self, sim, uids, from_uids=None):
        pass # No prognoses


def run_sim(n, idx, p, rand_seed, multistream):

    print(f'Starting sim {idx} with rand_seed={rand_seed} and p={p}, multistream={multistream}')

    ppl = ss.People(n)

    ppl.networks = ss.ndict()

    rw_pars = {
        'p': p,
        'init_prev': 0,
    }
    rw = RandomWalk(rw_pars)

    pars = {
        'start': 1980,
        'end': 2070,
        'rand_seed': rand_seed,
        'verbose': 0,
        'remove_dead': True,
        'n_agents': len(ppl), # TODO
    }

    sim = ss.Sim(people=ppl, diseases=[rw], demographics=None, pars=pars, label=f'Sim with {n} agents and p={p}')
    sim.initialize()
    sim.run()

    df = pd.DataFrame( {
        'ti': sim.tivec,
        'rw.mean': sim.results.randomwalk.mean,
    })
    df['p'] = p
    df['rand_seed'] = rand_seed
    df['multistream'] = multistream

    print(f'Finishing sim {idx} with rand_seed={rand_seed} and p={p}, multistream={multistream}')

    return df

def run_scenarios():
    results = []
    times = {}
    for multistream in [False, True]:
        ss.options(multistream=multistream)
        cfgs = []
        for rs in range(n_rand_seeds):
            for p in p_levels:
                cfgs.append({'p':p, 'rand_seed':rs, 'multistream':multistream, 'idx':len(cfgs)})
        T = sc.tic()
        results += sc.parallelize(run_sim, kwargs={'n': n}, iterkwargs=cfgs, die=True, serial=False)
        times[f'Multistream={multistream}'] = sc.toc(T, output=True)

    print('Timings:', times)

    df = pd.concat(results)
    df.to_csv(os.path.join(figdir, 'result.csv'))
    return df

def plot_scenarios(df):
    d = pd.melt(df, id_vars=['ti', 'rand_seed', 'p', 'multistream'], var_name='channel', value_name='Value')
    d['baseline'] = d['p'] == 0.5
    bl = d.loc[d['baseline']]
    scn = d.loc[~d['baseline']]
    bl = bl.set_index(['ti', 'channel', 'rand_seed', 'p', 'multistream'])[['Value']].reset_index('p')
    scn = scn.set_index(['ti', 'channel', 'rand_seed', 'p', 'multistream'])[['Value']].reset_index('p')
    mrg = scn.merge(bl, on=['ti', 'channel', 'rand_seed', 'multistream'], suffixes=('', '_ref'))
    mrg['Value - Reference'] = mrg['Value'] - mrg['Value_ref']
    mrg = mrg.sort_index()

    fkw = {'sharey': False, 'sharex': 'col', 'margin_titles': True}

    ## TIMESERIES
    g = sns.relplot(kind='line', data=d, x='ti', y='Value', hue='p', col='channel', row='multistream',
        height=5, aspect=1.2, palette='Set1', errorbar='sd', lw=2, facet_kws=fkw)
    g.set_titles(col_template='{col_name}', row_template='Multistream: {row_name}')
    g.set_xlabels(r'$t_i$')
    g.fig.savefig(os.path.join(figdir, 'timeseries.png'), bbox_inches='tight', dpi=300)

    ## DIFF TIMESERIES
    for ms, mrg_by_ms in mrg.groupby('multistream'):
        g = sns.relplot(kind='line', data=mrg_by_ms, x='ti', y='Value - Reference', hue='p', col='channel', row='p',
            height=3, aspect=1.0, palette='Set1', estimator=None, units='rand_seed', lw=0.5, facet_kws=fkw) #errorbar='sd', lw=2, 
        g.set_titles(col_template='{col_name}', row_template='Coverage: {row_name}')
        g.fig.suptitle('Multistream' if ms else 'Centralized')
        g.fig.subplots_adjust(top=0.88)
        g.set_xlabels(r'Timestep $t_i$')
        g.fig.savefig(os.path.join(figdir, 'diff_multistream.png' if ms else 'diff_centralized.png'), bbox_inches='tight', dpi=300)

    ## FINAL TIME
    tf = df['ti'].max()
    mtf = mrg.loc[tf]
    g = sns.displot(data=mtf.reset_index(), kind='kde', fill=True, rug=True, cut=0, hue='p', x='Value - Reference', col='channel', row='multistream',
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