"""
Costs and discounting.

Used with: agents and compartments; declared and coded modules
"""
import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['Costs', 'discount']


def get_result(results, path):
    """ A result by its path, e.g. 'measles.infection' for `results.measles.infection`; the key after the module's name may itself have a dot, as for a transition with several destinations """
    mod, _, key = path.partition('.')
    return results[mod][key] if key else results[mod]


def years(res):
    """ The time of each point of a result, in years """
    tv = res.timevec
    return np.asarray(tv.years if hasattr(tv, 'years') else tv, dtype=float)


def in_years(time):
    """ A time on a sim's timeline (a date or a duration, as `ss.connectors.to_time()` gives), in years """
    return time.years if isinstance(time, ss.date) else time.to('year').value


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
        start (float/str): when the fixed cost starts, read as any time is (see `ss.connectors.to_time()`: a number from 1900 on is a year; default: the start of the sim)
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
        self.define_results(*[ss.Result(name, kind='flow') for name in names + ['total']]) # Costs in each step; filled in again after the sim's results are scaled
        return

    def fill(self, ind=slice(None)):
        """ Work out the costs of some time points """
        res = self.results
        for mod in self.sim.modules:
            if mod.name in res and mod is not self:
                res[mod.name][ind] = mod.results.cost[ind]
        for path, unit_cost in self.unit_costs.items():
            res[path.replace('.', '_')][ind] = unit_cost * get_result(self.sim.results, path)[ind]
        if self.fixed:
            t = self.t.yearvec
            start = t[0] if self.start is None else in_years(ss.connectors.to_time(self.sim, self.start))
            width = np.diff(t, append=t[-1]) # The years from each time point to the next
            res.fixed[ind] = (self.fixed * width * (t >= start))[ind]
        res.total[ind] = sum(np.asarray(r.values)[ind] for k, r in res.items() if isinstance(r, ss.Result) and k not in ['total', 'timevec'])
        return

    def step(self):
        """ The costs so far, for what runs during the sim, such as `ssl.ICER`; the results are final after `finalize_results()` """
        self.fill(self.ti)
        return

    def finalize_results(self):
        super().finalize_results()
        self.fill()
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
