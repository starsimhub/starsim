"""
Costs, discounting, and incremental cost-effectiveness ratios (ICERs).
"""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

__all__ = ['Costs', 'discount', 'icer']


def get_result(results, path):
    """ A result by its path, e.g. 'measles.infection' for `results.measles.infection`; the key after the module's name may itself have a dot, as for a transition with several destinations """
    mod, _, key = path.partition('.')
    return results[mod][key] if key else results[mod]


def years(res):
    """ The time of each point of a result, in years """
    tv = res.timevec
    return np.asarray(tv.years if hasattr(tv, 'years') else tv, dtype=float)


class Costs(ss.Analyzer):
    """
    The costs of a sim, from its results, with their total

    Each delivery made with a `cost=` (e.g. `ss.vaccinate(..., cost=5)`) already has a
    `cost` result; this adds them up, along with a cost for each event of any other
    result (e.g. care for each infection) and a fixed cost per year. Every cost is a
    result of this analyzer, named after where it came from, and `total` is their sum.

    Args:
        unit_costs (dict): the cost of each event of a result, by the result's path, e.g. `{'measles.infection': 50}` for $50 of care for each infection, or `{'routine.delivered': 5}` for $5 a dose of a delivery made without its own `cost=`
        fixed (float): a cost per year that does not depend on any result, such as running a program (staff, cold chain, training)
        start (float/str): when the fixed cost starts (default: the start of the sim)
        deliveries (bool): whether to include the `cost` result of every module that has one

    Examples:
        ```python
        import starsim as ss
        import starsim.library as ssl

        vx = ss.vaccinate(efficacy=0.9, coverage=0.5, where=ss.ppl.age < 5, interval=ss.years(1), cost=5)
        costs = ssl.Costs(unit_costs={'sir.infection': 50}, fixed=1000)
        sim = ss.Sim(diseases='sir', networks='random', interventions=vx, analyzers=costs)
        sim.run()
        sim.results.costs.plot()
        print(sim.results.costs.total.sum())
        ```
    """
    def __init__(self, unit_costs=None, fixed=0, start=None, deliveries=True, **kwargs):
        super().__init__(**kwargs)
        self.unit_costs = sc.mergedicts(unit_costs)
        self.fixed = fixed
        self.start = start
        self.deliveries = deliveries
        return

    def init_results(self):
        super().init_results()
        names = [mod.name for mod in self.sim.modules if self.deliveries and mod is not self and 'cost' in mod.results]
        names += [path.replace('.', '_') for path in self.unit_costs]
        names += ['fixed'] if self.fixed else []
        self.define_results(*[ss.Result(name, scale=False) for name in names + ['total']]) # Each is made from results that are already scaled
        return

    def step(self):
        """ Costs are worked out once the sim has run, in `finalize_results()` """
        pass

    def finalize_results(self):
        super().finalize_results()
        res = self.results
        for mod in self.sim.modules:
            if mod.name in res and mod is not self:
                res[mod.name][:] = mod.results.cost
        for path, unit_cost in self.unit_costs.items():
            res[path.replace('.', '_')][:] = unit_cost * get_result(self.sim.results, path)
        if self.fixed:
            t = self.t.yearvec
            start = t[0] if self.start is None else ss.date(self.start).years
            width = np.diff(t, append=t[-1]) # The years from each time point to the next
            res.fixed[:] = self.fixed * width * (t >= start)
        res.total[:] = sum(r.values for k, r in res.items() if isinstance(r, ss.Result) and k != 'total')
        return


def discount(values, rate, start=None, t=None):
    """
    Discount values over time at a yearly rate

    Each value at time t (in years) is weighted by `(1+rate)**-(t-start)`, so values
    before `start` are not discounted.

    Args:
        values (Result/array): a result (or the results of the runs of a MultiSim), or an array of values at the times `t`
        rate (float): the discount rate per year, e.g. 0.03
        start (float/str): when discounting starts (default: the first time point)
        t (array): the time of each value in years (default: the result's own times)

    Returns:
        The discounted values, of the same type as `values` (so a result can be summed, plotted, etc.)

    Examples:
        ```python
        cost = ssl.discount(sim.results.costs.total, 0.03, start=2010).sum()
        ```
    """
    t = years(values) if t is None else np.asarray(t, dtype=float)
    start = t[0] if start is None else ss.date(start).years
    weights = (1 + rate) ** -np.maximum(t - start, 0)
    return values * weights # The time axis is last, so this works for the runs of a MultiSim too


def icer(sims, outcome, baseline=None, cost='costs.total', rate=0, start=None):
    """
    Compare the costs and effects of scenarios: incremental cost-effectiveness ratios

    Each scenario's cost and outcome are totaled over the run (discounted if `rate` is
    given), and the outcome averted is that of the baseline less the scenario's. The
    scenarios are ordered by cost, and each is compared with the next cheaper one that
    it is not dominated by: a scenario that costs more and averts no more than a cheaper
    one is dominated, and has no ratio. With several runs of each scenario (a MultiSim
    run with `n_runs`), their means are compared.

    Args:
        sims (MultiSim/dict/list): the sims of each scenario, labeled by scenario: a run MultiSim, a dict of run sims by label, or a list of run sims
        outcome (str): the path of the result to avert, e.g. 'measles.infection' or 'hiv.new_deaths'
        baseline (str): the label of the scenario to compare with (default: the first)
        cost (str): the path of the cost result (default: the total of `ssl.Costs`)
        rate (float): the discount rate per year for both costs and outcomes (see `ssl.discount()`)
        start (float/str): when discounting starts

    Returns:
        A dataframe with a row for each scenario and columns `cost`, `outcome`, `averted`, `incremental cost`, `incremental averted`, `ICER`, and `dominated`

    Examples:
        ```python
        msim = ss.parallel({'Nothing': base, 'Vaccine': base.new(interventions=vx)})
        ssl.icer(msim, outcome='measles.infection', rate=0.03, start=2010)
        ```
    """
    def total(res):
        return np.asarray(discount(res, rate, start)).sum(axis=-1)

    # Totals of each scenario, as the mean of their runs
    if isinstance(sims, ss.MultiSim):
        scenarios = list(sims.runs['scenario'])
        costs, outcomes = [total(get_result(sims.results, path)) for path in [cost, outcome]]
    else:
        sims = sims if isinstance(sims, dict) else {sim.label:sim for sim in sims}
        scenarios = list(sims.keys())
        costs, outcomes = [[total(get_result(sim.results, path)) for sim in sims.values()] for path in [cost, outcome]]
    df = pd.DataFrame(dict(scenario=scenarios, cost=costs, outcome=outcomes)).groupby('scenario', sort=False).mean()
    baseline = df.index[0] if baseline is None else baseline
    df['averted'] = df.outcome[baseline] - df.outcome
    df = df.sort_values(['cost', 'averted'], ascending=[True, False])

    # Compare each scenario with the next cheaper one that is not dominated (simple dominance)
    df['incremental cost'] = df['incremental averted'] = df['ICER'] = np.nan
    df['dominated'] = False
    last = None
    for label, row in df.iterrows():
        if last is not None and row.averted <= df.averted[last]:
            df.loc[label, 'dominated'] = True
            continue
        if last is not None:
            df.loc[label, 'incremental cost'] = row.cost - df.cost[last]
            df.loc[label, 'incremental averted'] = row.averted - df.averted[last]
            df.loc[label, 'ICER'] = df.loc[label, 'incremental cost'] / df.loc[label, 'incremental averted']
        last = label
    return df
