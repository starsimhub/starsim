"""
Disability-adjusted life years (DALYs), and incremental cost-effectiveness ratios (ICERs).

Used with: agents and compartments; declared and coded modules
"""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt
from .costs import get_result, years, in_years

__all__ = ['DALYs', 'ICER']

# Years of life left at each age, from the reference life table of the Global Burden of Disease 2019 study
life_table = sc.obj(
    ages = np.array([0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]),
    left = np.array([88.9, 88.0, 84.0, 79.0, 74.1, 69.1, 64.1, 59.2, 54.3, 49.3, 44.4, 39.6, 34.9, 30.3, 25.7, 21.3, 17.1, 13.2, 10.0, 7.6, 5.9]),
)


class DALYs(ss.Analyzer):
    """
    The disability-adjusted life years of a sim: the years lived with disability, plus the years of life lost

    Years lived with disability (`yld`) are the time people spend in a state, multiplied
    by the disability weight of the state. Years of life lost (`yll`) are the years that
    each person who dies of a disease would otherwise have lived. Every result is per
    step, and `dalys` is their sum.

    Args:
        weights (dict): the disability weight of a result that counts people, by its path, e.g. `{'hiv.I': 0.2}`
        life_lost (float): the years of life lost for each death; by default, with agents, those left at each person's age at death in a standard life table (from the Global Burden of Disease 2019 study). Without agents, ages at death aren't known, so give this
        deaths (list): the paths of the results that count deaths, needed only with `life_lost` (default: each disease's `death` or `new_deaths`)

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        sir = ss.SIR(p_death=0.01)
        sim = ss.Sim(diseases=sir, networks='random', analyzers=ssl.DALYs(weights={'sir.I': 0.1}))
        sim.run()
        print(sim.results.dalys.dalys.sum())
        ```
    """
    def __init__(self, weights=None, life_lost=None, deaths=None, **kwargs):
        super().__init__(**kwargs)
        self.weights = sc.mergedicts(weights)
        self.life_lost = life_lost
        self.deaths = sc.tolist(deaths) if deaths is not None else None
        return

    def init_results(self):
        super().init_results()
        if self.life_lost is None and self.sim.solver != 'abm':
            errormsg = f'ssl.DALYs() has no ages at death to work from with the "{self.sim.solver}" solver: give the years of life lost for each death, with life_lost='
            raise ValueError(errormsg)
        self.define_results(*[ss.Result(name, kind='flow') for name in ['yld', 'yll', 'dalys']])
        return

    def death_results(self):
        """ The paths of the results that count deaths """
        if self.deaths is not None:
            return self.deaths
        res = self.sim.results
        keys = {dis.name:[key for key in ['death', 'new_deaths'] if key in res[dis.name]] for dis in self.sim.diseases.values()}
        return [f'{name}.{found[0]}' for name,found in keys.items() if found]

    def fill(self, ind=slice(None)):
        """ Work out the years lived with disability and, given `life_lost`, the years of life lost, at some time points """
        res = self.results
        sim = self.sim
        res.yld[ind] = sum((weight*np.asarray(get_result(sim.results, path))[ind]*self.t.dt_year for path,weight in self.weights.items()), start=np.zeros_like(res.yld.values[ind]))
        if self.life_lost is not None:
            res.yll[ind] = sum((self.life_lost*np.asarray(get_result(sim.results, path))[ind] for path in self.death_results()), start=np.zeros_like(res.yll.values[ind]))
        return

    def step(self):
        """ The years of life lost by those who died in this step, and the results so far, for what runs during the sim, such as `ssl.ICER` """
        ti = self.ti
        self.fill(ti)
        if self.life_lost is None:
            people = self.sim.people
            for dis in self.sim.diseases.values():
                died = people.died_of(dis)
                self.results.yll[ti] += np.interp(people.age.raw[died], life_table.ages, life_table.left).sum()
        self.results.dalys[ti] = self.results.yld[ti] + self.results.yll[ti]
        return

    def finalize_results(self):
        super().finalize_results()
        self.fill()
        self.results.dalys[:] = self.results.yld.values + self.results.yll.values
        return


class ICER(ss.Analyzer):
    """
    Compare the costs and effects of scenarios: incremental cost-effectiveness ratios

    Each scenario's cost and outcome are totaled (discounted if `rate` is given), and the
    outcome averted is that of the baseline less the scenario's. The scenarios are ordered
    by cost, and each is compared with the next cheaper one that isn't dominated. A
    scenario is dominated if it costs more and averts no more than a cheaper one, and
    extendedly dominated if a mix of two others would avert more for the same cost.
    With several runs of each scenario, their means are compared.

    This works in two ways. After the scenarios have run, give them to `ssl.ICER()`, and
    read `icer.df`. Or add it to a sim as an analyzer, after `ssl.Costs` and `ssl.DALYs`:
    the sim then counts as one more scenario, compared at each step with the scenarios
    given over the same period, so that what runs in the sim can act on the comparison,
    for example an intervention that follows `icer.best(wtp=1000)`.

    Args:
        sims (MultiSim/dict/list): the sims of each scenario: a run MultiSim, a dict of run sims by label, or a list of run sims
        outcome (str): the path of the result to avert (default: the total of `ssl.DALYs`), e.g. 'measles.infection'
        cost (str): the path of the cost result (default: the total of `ssl.Costs`)
        baseline (str): the label of the scenario to compare with (default: the first)
        rate (float): the discount rate per year for both costs and outcomes (see `ssl.discount()`)
        start (float/str): when discounting starts
        window (int): in a sim, the number of steps to compare, ending with the latest (default: all of them so far); 1 compares the last step alone

    Attributes:
        df (dataframe): a row for each scenario, with columns `cost`, `outcome`, `averted`, `incremental cost`, `incremental averted`, `ICER`, `dominated`, and `extended`

    Examples:
        ```python
        # After the runs
        msim = ss.parallel({'Nothing': base, 'Vaccine': base.new(interventions=vx)}, n_runs=20)
        icer = ssl.ICER(msim, rate=0.03, start=2010)
        icer.df
        icer.plot()
        icer.plot_ceac()
        icer.best(wtp=1000)

        # During a run, compared with a finished sim
        icer = ssl.ICER(dict(Nothing=base.run()), window=1)
        sim = ss.Sim(..., analyzers=[ssl.Costs(), ssl.DALYs(weights=...), icer])
        ```
    """
    def __init__(self, sims=None, outcome='dalys.dalys', cost='costs.total', baseline=None, rate=0, start=None, window=None, **kwargs):
        super().__init__(**kwargs)
        self.outcome = outcome
        self.cost = cost
        self.baseline = baseline
        self.rate = rate
        self.start = start
        self.window = window
        self.t_years = None
        self.scenarios = sc.obj() if sims is None else self.collect(sims) # By label, the cost and outcome of each run at each time point, so the sims themselves aren't kept
        return

    def collect(self, sims):
        """ The cost and outcome of each scenario, as arrays of runs by time points """
        try:
            if isinstance(sims, ss.MultiSim):
                labels = np.array(sims.runs['scenario'])
                cost, outcome = [np.atleast_2d(np.asarray(get_result(sims.results, path))) for path in [self.cost, self.outcome]]
                self.t_years = years(get_result(sims.results, self.cost))
                return sc.obj({label:sc.obj(cost=cost[labels == label], outcome=outcome[labels == label]) for label in dict.fromkeys(labels.tolist())})
            sims = sims if isinstance(sims, dict) else {sim.label:sim for sim in sims}
            out = sc.obj()
            for label,sim in sims.items():
                cost, outcome = [np.asarray(get_result(sim.results, path), dtype=float)[None, :] for path in [self.cost, self.outcome]]
                out[label] = sc.obj(cost=cost, outcome=outcome)
                self.t_years = years(get_result(sim.results, self.cost))
            return out
        except KeyError as E:
            errormsg = f'ssl.ICER() compares the results "{self.cost}" and "{self.outcome}", which were not found: add ssl.Costs() and ssl.DALYs() to the sims, or give cost= and outcome='
            raise KeyError(errormsg) from E

    def step(self):
        """ Count this sim as a scenario, as it stands """
        ti = self.ti
        res = self.sim.results
        label = self.sim.label or 'This sim'
        if self.t_years is None:
            self.t_years = np.asarray(self.t.yearvec, dtype=float)
        if label not in self.scenarios:
            n = len(self.t_years)
            self.scenarios[label] = sc.obj(cost=np.full((1, n), np.nan), outcome=np.full((1, n), np.nan))
        now = self.scenarios[label]
        now.cost[0, ti], now.outcome[0, ti] = [float(np.asarray(get_result(res, path))[ti]) for path in [self.cost, self.outcome]]
        self._ti = ti
        return

    def totals(self):
        """ The discounted total cost and outcome of each scenario, as arrays with one value per run, by label """
        t = self.t_years
        start = t[0] if self.start is None else in_years(ss.date(self.start))
        weights = (1 + self.rate) ** -np.maximum(t - start, 0)
        stop = getattr(self, '_ti', len(t) - 1) + 1
        first = 0 if self.window is None else max(stop - self.window, 0)
        out = sc.obj()
        total = lambda arr: np.nansum((arr*weights)[:, first:stop], axis=1)
        for label,data in self.scenarios.items():
            out[label] = sc.obj(cost=total(data.cost), outcome=total(data.outcome))
        return out

    @property
    def df(self):
        """ The table of the scenarios (see the class docstring) """
        totals = self.totals()
        df = pd.DataFrame(dict(cost=[v.cost.mean() for v in totals.values()], outcome=[v.outcome.mean() for v in totals.values()]), index=pd.Index(totals.keys(), name='scenario'))
        baseline = df.index[0] if self.baseline is None else self.baseline
        df['averted'] = df.outcome[baseline] - df.outcome
        df = df.sort_values(['cost', 'averted'], ascending=[True, False])
        df['incremental cost'] = df['incremental averted'] = df['ICER'] = np.nan
        df['dominated'] = df['extended'] = False

        # Simple dominance: a cheaper scenario averts as much
        keep = []
        for label,row in df.iterrows():
            if keep and row.averted <= df.averted[keep[-1]]:
                df.loc[label, 'dominated'] = True
            else:
                keep.append(label)

        # Extended dominance: the ratios along the rest should rise, and one that is above the next is left out
        ratio = lambda a,b: (df.cost[b] - df.cost[a])/(df.averted[b] - df.averted[a])
        found = True
        while found:
            found = False
            for a,b,c in zip(keep, keep[1:], keep[2:]):
                if ratio(a, b) > ratio(b, c):
                    df.loc[b, 'extended'] = True
                    keep.remove(b)
                    found = True
                    break

        for a,b in zip(keep, keep[1:]):
            df.loc[b, 'incremental cost'] = df.cost[b] - df.cost[a]
            df.loc[b, 'incremental averted'] = df.averted[b] - df.averted[a]
            df.loc[b, 'ICER'] = ratio(a, b)
        return df

    def net_benefit(self, wtp):
        """ The net benefit of each scenario at a willingness to pay per outcome averted: a dataframe with a row for each run and a column for each scenario """
        totals = self.totals()
        n = max(len(v.cost) for v in totals.values())
        return pd.DataFrame({label:-wtp*np.resize(v.outcome, n) - np.resize(v.cost, n) for label,v in totals.items()})

    def best(self, wtp):
        """
        The scenario with the highest mean net benefit at a willingness to pay

        Args:
            wtp (float): the willingness to pay per outcome averted
        """
        return self.net_benefit(wtp).mean().idxmax()

    def plot(self, **kwargs):
        """
        Plot the cost-effectiveness plane: each scenario's cost against the outcome it averts, with each run, and the line through those that aren't dominated

        Args:
            kwargs (dict): passed to `ss.plot_args()`
        """
        kw = ss.plot_args(kwargs)
        totals = self.totals()
        df = self.df
        base = df.outcome[df.index[0] if self.baseline is None else self.baseline]
        with ss.style(**kw.style):
            fig, ax = plt.subplots(**kw.fig)
            for label,v in totals.items():
                ax.scatter(base - v.outcome, v.cost, alpha=0.3 if len(v.cost) > 1 else 1, label=label)
            front = df[~df.dominated & ~df.extended]
            ax.plot(front.averted, front.cost, c='k', lw=1)
            ax.set_xlabel(f'Averted ({self.outcome})')
            ax.set_ylabel(f'Cost ({self.cost})')
            ax.legend(frameon=False)
        return ss.return_fig(fig, **kw.return_fig)

    def plot_ceac(self, wtp=None, **kwargs):
        """
        Plot the cost-effectiveness acceptability curve: the share of runs in which each scenario is the best, by willingness to pay

        Args:
            wtp (array): the values of willingness to pay (default: 50 up to twice the largest ICER)
            kwargs (dict): passed to `ss.plot_args()`
        """
        kw = ss.plot_args(kwargs)
        if wtp is None:
            top = self.df.ICER.max()
            wtp = np.linspace(0, 2*top if np.isfinite(top) and top > 0 else 1, 50)
        shares = pd.DataFrame([self.net_benefit(w).idxmax(axis=1).value_counts(normalize=True) for w in wtp], index=wtp).fillna(0)
        with ss.style(**kw.style):
            fig, ax = plt.subplots(**kw.fig)
            for label in shares.columns:
                ax.plot(wtp, shares[label], label=label)
            ax.set_xlabel('Willingness to pay per outcome averted')
            ax.set_ylabel('Share of runs in which it is the best')
            ax.set_ylim(0, 1)
            ax.legend(frameon=False)
        return ss.return_fig(fig, **kw.return_fig)
