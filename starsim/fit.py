"""
Fitting sims to data: likelihoods, particle filters, samplers, and what `sim.fit()` runs

These exist so that a model can be fitted, with its uncertainty, in the same way whichever solver runs it. Parameters are chosen by giving them bounds (`ss.par`), in declared modules or in the parameters of modules written as code. Some methods are for some solvers only: gradients come from JAX, and particle filters need a stochastic compartmental solver.

Imported when it is first used (by `ss.fit.X`, `sim.fit()`, or `sim.sensitivity()`), not by `import starsim`.

Used with: agents and compartments; declared and coded modules
"""
import os
import inspect
import sys
import tempfile
import warnings
import functools
import numpy as np
import pandas as pd
import scipy.stats as sps
import scipy.special as spsp
import sciris as sc
import starsim as ss
from .results import compare, lineup, match_column, match_name, parse_column, column_levels, flow_at_data

__all__ = ['mismatch', 'loglik', 'prior_draws', 'sensitivity', 'profile', 'optimize', 'compare_models', 'split_data', 'diagnose', 'fit_stages',
           'fit_scipy', 'fit_optuna', 'fit_if2', 'fit_pmcmc', 'fit_abc_smc', 'fit_nuts', 'abc_smc', 'pmmh', 'nuts',
           'Fit', 'Optimization', 'ModelComparison', 'ParticleFilter', 'Swarm']


#%% Data and fitting for models with declared parameters

def mismatch(sim, df=None, weights=None, gof=None):
    """
    The goodness of fit between a sim and its data (smaller is better)

    By default, for each data series, the normalized absolute error: the mean of
    |model − data| divided by the mean of the data; the series are then averaged.
    A series whose data are all zero is divided by the mean of the model instead,
    so it adds at most 1 (not raw counts, which would swamp the other series).

    Args:
        sim (`ss.Sim`): a sim that has been run
        df (df): the data, if not the sim's
        weights (dict): the relative weight of each data series (default: equal)
        gof (str/func): 'nae' (default), 'rmse', or a function of the sim returning a number
    """
    check_gof(gof)
    if callable(gof):
        return gof(sim)
    total = 0.0
    wsum = 0.0
    for col,(expected,model) in compare(sim, df).items():
        scale = np.abs(expected).mean() or np.abs(model).mean() or 1.0 # An all-zero series by the model's own size, so it cannot swamp the others
        diff = model - expected
        value = np.sqrt(np.mean(diff**2))/scale if gof == 'rmse' else np.mean(np.abs(diff))/scale
        weight = series_weight(weights, col)
        total += weight*value
        wsum += weight
    return total/wsum if wsum else np.nan


def series_weight(weights, col):
    """ The weight of a data series: given for its column, or for its result (which applies to every stratum of it) """
    weights = weights or {}
    return weights.get(col, weights.get(parse_column(col)[0], 1.0))


def likelihoods(sim, df=None, likelihood=None):
    """ The likelihood of each data series: given at calibration time (by column or result name), else declared on the outcome, else None """
    df = sim.data if df is None else df
    if likelihood is not None and not isinstance(likelihood, dict):
        cols = [] if df is None else list(df.columns)
        errormsg = f'likelihood= takes the likelihood of each data series by name, e.g. likelihood=dict({cols[0] if cols else "cases"}={likelihood}), not one {type(likelihood).__name__}'
        raise TypeError(errormsg)
    out = sc.obj()
    for col in ([] if df is None else df.columns): # No data, with a loss that is a function of the sim
        name = parse_column(col)[0] # A stratum of a result has the likelihood of the result
        res = match_name(sim, name)
        if res is None: # A column that matches no result, which is not compared (see ss.results.check_data())
            continue
        given = (likelihood or {}).get(col, (likelihood or {}).get(name, (likelihood or {}).get(res.name)))
        if given is None:
            for mod in sim.declared:
                if mod.results.get(res.name) is res and res.name in mod.graph.outcomes: # By the result itself, since its module is a label
                    given = mod.graph.outcomes[res.name].likelihood
        out[col] = given
    return out


def lik_pars(lik):
    """ The parameters of a likelihood: a distribution's, or none for a function """
    return lik.pars if isinstance(lik, ss.Dist) else {}


def logpdf(dist, observed, model, xp=np, pars=None):
    """
    The log-likelihood of each observed value, when the model's value is the mean of the distribution

    The distribution's other parameters are taken as given (a parameter that is an
    `ss.par` takes its current value). Supported: `ss.poisson`, `ss.nbinom` (with
    `n` as the dispersion), `ss.binom` (with `n` the number of trials, and the model's
    value the probability), `ss.bernoulli` (an observation of 0 or 1, with the model's
    value its probability), `ss.normal` (with `scale`), and `ss.lognorm_ex` (with `std`).
    The counts of a Poisson or negative binomial need not be whole numbers (e.g. weekly
    averages): the factorials are taken as gamma functions, which gives the same values
    for whole numbers.

    A likelihood can also be a function of the observed and the model's values,
    `fn(observed, model)`, returning the log-likelihood of each, for one the list above
    lacks (e.g. a normal whose variance grows with the mean). It is worked out with
    NumPy, so the log-density in JAX (for NUTS) does not cover it, and NUTS samples
    with Python in the loop instead.

    Args:
        dist (`ss.Dist`/func): the likelihood
        observed (array): the data
        model (array): the model's values at the data
        xp (module): the array namespace: NumPy, or `jax.numpy` for the log-density in JAX (see `ss.gradients.Density`)
        pars (dict): the distribution's parameters, if not their current values (the log-density in JAX gives those that are fitted as JAX values)
    """
    if not isinstance(dist, ss.Dist):
        if not callable(dist):
            errormsg = f'A likelihood is a distribution (e.g. ss.poisson()) or a function fn(observed, model) returning the log-likelihood of each value, not {type(dist).__name__}'
            raise TypeError(errormsg)
        if xp is not np:
            raise ss.gradients.Unsupported(f'the likelihood {getattr(dist, "__name__", "")}(), a function')
        out = np.asarray(dist(np.asarray(observed), np.asarray(model)), dtype=float)
        return np.broadcast_to(out, np.broadcast_shapes(np.shape(observed), np.shape(model)))
    jsp = None if xp is np else ss.gradients.jax().scipy
    gammaln = spsp.gammaln if jsp is None else jsp.special.gammaln
    norm = sps.norm if jsp is None else jsp.stats.norm
    if pars is None:
        pars = {key:(val.raw if isinstance(val, ss.par) else val) for key,val in dist.pars.items()}
    mean = xp.maximum(model, 1e-9)
    if isinstance(dist, ss.poisson):
        out = observed*xp.log(mean) - mean - gammaln(observed + 1)
        return xp.where(observed < 0, -np.inf, out) # A count cannot be negative
    elif isinstance(dist, ss.nbinom):
        n = pars['n']
        out = gammaln(observed + n) - gammaln(n) - gammaln(observed + 1) + n*xp.log(n/(n + mean)) + observed*xp.log(mean/(n + mean))
        return xp.where(observed < 0, -np.inf, out)
    elif isinstance(dist, ss.binom): # The positives among n, with the model's value as the probability
        n = pars['n']
        p = xp.clip(model, 1e-12, 1 - 1e-12)
        out = gammaln(n + 1) - gammaln(observed + 1) - gammaln(n - observed + 1) + observed*xp.log(p) + (n - observed)*xp.log1p(-p)
        return xp.where((observed < 0) | (observed > n), -np.inf, out)
    elif isinstance(dist, ss.bernoulli): # Whether something happened, with the model's value as its probability
        p = xp.clip(model, 1e-12, 1 - 1e-12)
        return observed*xp.log(p) + (1 - observed)*xp.log1p(-p)
    elif isinstance(dist, ss.normal):
        return norm.logpdf(observed, model, pars['scale'])
    elif isinstance(dist, ss.lognorm_ex): # The log of the value is normal
        sigma2 = xp.log1p(pars['std']**2/mean**2)
        logobs = np.log(np.where(observed > 0, observed, 1.0))
        return xp.where(observed > 0, norm.logpdf(logobs, xp.log(mean) - sigma2/2, xp.sqrt(sigma2)) - logobs, -np.inf)
    errormsg = f'{type(dist).__name__} cannot be used as a likelihood; use ss.poisson(), ss.nbinom(n=...), ss.binom(n=...), ss.bernoulli(), ss.normal(scale=...), or ss.lognorm_ex(std=...)'
    raise TypeError(errormsg)


def loglik(sim, df=None, likelihood=None, weights=None, by_series=False):
    """
    The log-likelihood of a sim's data (larger is better): the sum over the series, which are treated as independent

    Args:
        sim (`ss.Sim`): a sim that has been run
        df (df): the data, if not the sim's
        likelihood (dict): the likelihood of each series, if not declared on its outcome
        weights (dict): a multiplier on each series' log-likelihood (default 1)
        by_series (bool): return a dict of the log-likelihood of each series instead of the total
    """
    liks = likelihoods(sim, df, likelihood)
    out = sc.obj()
    for col,(expected,model) in compare(sim, df).items():
        if liks[col] is None:
            errormsg = f'The data series "{col}" has no likelihood; declare one on its outcome (ss.outcome(..., likelihood=ss.poisson())) or pass likelihood=dict({col}=...)'
            raise ValueError(errormsg)
        out[col] = series_weight(weights, col)*logpdf(liks[col], expected, model).sum()
    return out if by_series else float(sum(out.values()))


def logprior(par, x, xp=np):
    """
    The log-density of a parameter's prior at a value: its `prior=` if it has one, else uniform over its bounds

    With `xp` as `jax.numpy`, in JAX (for the log-density in JAX), which covers normal, lognormal, gamma, beta, exponential, and uniform priors.
    """
    prior = par.prior
    low, high = [b.value if isinstance(b, ss.TimePar) else b for b in par.bounds] # Uniform over the bounds is a constant, so the unit of typed bounds does not matter
    if prior is None:
        return -np.log(high - low)
    stats = sps if xp is np else ss.gradients.jax().scipy.stats
    if isinstance(prior, ss.uniform):
        return stats.uniform.logpdf(x, prior.pars.low, prior.pars.high - prior.pars.low)
    frozen = sc.dcp(prior).init(force=True).dist
    if frozen is None:
        errormsg = f'The prior of "{par.name}" ({type(prior).__name__}) has no density to use; use e.g. ss.normal(), ss.lognorm_ex(), ss.gamma(), ss.beta_dist(), or ss.uniform()'
        raise TypeError(errormsg)
    elif xp is np:
        return frozen.logpdf(x)
    name = getattr(getattr(frozen, 'dist', None), 'name', None) # In JAX: SciPy's distribution, by name
    kw = dict(loc=0.0, scale=1.0) | frozen.kwds
    args = frozen.args
    if name == 'norm':
        return stats.norm.logpdf(x, kw['loc'], kw['scale'])
    elif name == 'lognorm':
        s = args[0] if args else kw['s']
        return stats.norm.logpdf(xp.log(x - kw['loc']), xp.log(kw['scale']), s) - xp.log(x - kw['loc'])
    elif name == 'gamma':
        a = args[0] if args else kw['a']
        return stats.gamma.logpdf(x, a, kw['loc'], kw['scale'])
    elif name == 'beta':
        a, b = args if args else (kw['a'], kw['b'])
        return stats.beta.logpdf(x, a, b, kw['loc'], kw['scale'])
    elif name == 'expon':
        return stats.expon.logpdf(x, kw['loc'], kw['scale'])
    raise ss.gradients.Unsupported(f'the prior of "{par.name}" ({type(prior).__name__})')


class ParticleFilter(sc.prettyobj):
    """
    An estimate of the likelihood of a stochastic sim's data, from a bootstrap particle filter (see `sim.fit()` with `method='pmcmc'`)

    Many runs of the sim (the particles, `ss.solvers.Particles`) go forward together. At each data point, each is
    weighted by the likelihood of the data given its results, and when the weights become
    uneven, the runs are drawn again in proportion to them, so that runs that have strayed
    from the data are dropped and those that follow it are copied. The product over the data
    of the mean weight is an unbiased estimate of the likelihood (Andrieu, Doucet & Holenstein 2010),
    which is what particle MCMC needs.

    Each declared disease is filtered on its own, since the diseases of a sim that the kernels run
    whole do not affect each other, except those tracked jointly (see `ss.solvers.Crossing`), which are
    filtered together, as the runs of their program. An outcome is compared with the data as its expected number
    (rather than drawn, as a run of the sim draws it), as the likelihood would compare it.

    Each particle has random numbers of its own, so the particles are shared between threads
    (see `ss.options.threads`; one in the workers of `ss.parallel()` and of a calibration), and the
    estimate does not depend on how many there are.

    Args:
        sim (`ss.Sim`): the sim, initialized; it is restarted with each set of values before the filter is called
        likelihood (dict): the likelihood of each data series, if not declared on its outcome
        weights (dict): a multiplier on each series' log-likelihood (default 1)
    """
    def __init__(self, sim, likelihood=None, weights=None):
        liks = likelihoods(sim, likelihood=likelihood)
        missing = [col for col,lik in liks.items() if lik is None]
        why = self.reason(sim) or (f'the data series {sc.strjoin(missing)} have no likelihood' if missing else None)
        if why:
            errormsg = f'The particle filter cannot run this sim, since {why}; use method="optuna" instead'
            raise ValueError(errormsg)
        self.sim = sim
        lined = lineup(sim)
        mods = list(sim.declared)
        self.series = []
        for col in sim.data.columns:
            base, dims = parse_column(col)
            res = match_name(sim, base)
            j = [j for j,mod in enumerate(mods) if mod.results.get(res.name) is res][0]
            engine = mods[j].engine
            keep = None
            if dims: # The groups of a stratum
                keep = np.ones(sim.groups.n_groups)
                for dim,level in column_levels(sim, dims).items():
                    d = sim.groups.dim_index(dim)
                    keep = keep*(sim.groups.group_levels[:, d] == sim.groups.strats[d].index(level))
            W, expected, rows = lined[col]
            spans = [(np.flatnonzero(w).min(initial=len(w)) if w.any() else 0, np.flatnonzero(w).max(initial=0) + 1) for w in W] # The steps that each data point is worked out from
            read = self.reader(engine, res.name)
            back = 0 if read is not None else self.reach(engine, res.name)
            self.series.append(sc.obj(col=col, j=j, name=res.name, read=read, back=back, keep=keep, W=W, spans=spans, expected=expected, rows=rows,
                                          lik=liks[col], weight=series_weight(weights, col)))
        self.engines = {j:mods[j].engine for j in sorted({s.j for s in self.series})}
        return

    @staticmethod
    def reason(sim):
        """ Why the particle filter cannot run a sim (apart from any data series without a likelihood), if it cannot """
        if sim.solver not in ['binomial', 'gillespie', 'tau']:
            return f'it has solver="{sim.solver}", not a stochastic compartmental one'
        elif not sim.kernel_only():
            return f'it is not run in one call to the kernels ({sim.kernel_reason()})'
        elif sim.data is None:
            return 'it has no data'
        for col in sim.data.columns:
            base, dims = parse_column(col)
            res = match_name(sim, base)
            mods = [mod for mod in sim.declared if mod.results.get(res.name) is res]
            if not mods:
                return f'the data series "{col}" is not a result of a declared disease'
            if dims and ParticleFilter.reader(mods[0].engine, res.name) is None:
                return f'the data series "{col}" is a stratum of "{res.name}", which is not a state or a transition'
        return None

    @staticmethod
    def reader(engine, name):
        """ Where a result is in the stocks or flows of a run, if it is read straight from them: ('stocks' or 'flows', indices to sum) """
        graph = engine.graph
        if name in graph.states:
            return 'stocks', [list(graph.states).index(name)]
        flows = engine.tr_index | engine.transfer_index
        if name in flows:
            return 'flows', [flows[name]]
        return None

    @staticmethod
    def reach(engine, name):
        """ For a result worked out from the others: how many steps before a time point its value there depends on the run, or None if on the whole run before it """
        outcomes = engine.outcomes
        if name in engine.graph.exprs or name == 'prevalence': # Of the stocks and flows at the time point
            return 0
        elif name not in outcomes: # E.g. the number in an outcome, which depends on how long people have stayed in it
            return None
        info = outcomes[name]
        reaches = [ParticleFilter.reach(engine, key) if key in outcomes else 0 for key,weight in info.sources]
        if None in reaches:
            return None
        return len(info.kernel) - 1 + max(reaches) # Its delay, after those of the outcomes it is worked out from

    @staticmethod
    def sources(engine, name):
        """ An outcome, and the outcomes it is worked out from """
        return {name}.union(*[ParticleFilter.sources(engine, key) for key,weight in engine.outcomes[name].sources if key in engine.outcomes])

    def derive(self, s, stocks, flows):
        """ The values of a result worked out from the others, at the steps given: an outcome from the flows (and just the outcomes it needs), or anything else as a run works it out """
        engine = self.engines[s.j]
        if s.name in engine.outcomes and flows.shape[-1] == 1:
            new = sc.obj({name:flows[:, :, ind, 0] for name,ind in engine.tr_index.items()})
            return engine.derived_outcomes(new, expected=True, names=self.sources(engine, s.name))[s.name]
        return engine.collect(stocks, flows, expected=True)[s.name]

    def __call__(self, n_particles=100, seed=None, swarm=None):
        """
        The estimate of the log-likelihood, for the sim as it was last restarted

        Args:
            n_particles (int): the number of particles
            seed (int): the random seed
            swarm (`Swarm`): for IF2, each particle's parameter values, which are perturbed before each data point (and at the start), and drawn again with the particles
        """
        rng = np.random.default_rng(seed)
        total = 0.0
        for j,engine in self.engines.items():
            cross = engine.cross
            if cross is None:
                total += self.filter([j], n_particles, rng, swarm=swarm)
            elif j == min(i for i,e in self.engines.items() if e.cross is cross): # Once for the diseases tracked jointly
                total += self.filter([i for i,e in self.engines.items() if e.cross is cross], n_particles, rng, cross=cross, swarm=swarm)
        return total

    def filter(self, js, n, rng, cross=None, swarm=None):
        """ The estimate of the log-likelihood of the data of one disease (or of the diseases of a crossing, run by its program) """
        series = [s for s in self.series if s.j in js]
        by_group = any(s.keep is not None for s in series)

        def view(j, arrays):
            """ The stocks or flows of disease j, as its engine has them """
            if cross is None:
                return arrays
            k = cross.engines.index(self.engines[j])
            return (cross.engine_stocks(k, arrays[0], by_group), cross.engine_flows(k, arrays[1], by_group))

        # The data points in order, with the last step each needs and the first step that any later one needs
        points = sc.ddict(list)
        for s in series:
            for i,row in enumerate(s.rows):
                points[row].append((s, i))
        order = sorted(points)
        last = np.maximum.accumulate([max(s.spans[i][1] - 1 for s,i in points[row]) for row in order])
        first = [min(max(s.spans[i][0] - s.back, 0) if s.back is not None else 0 for s,i in points[row]) for row in order]
        runs = ss.solvers.Particles(self.engines[js[0]] if cross is None else cross, n, rng, by_group=by_group)
        npts = runs.npts
        needed = np.append(np.minimum.accumulate(first[::-1])[::-1][1:], npts) # For each point, the earliest step of the runs that the points after it read

        tables = shares = None # The engine's own, the same for every particle (unless each has its own, for IF2)
        if swarm is not None:
            swarm.perturb(rng)
        logw = np.full(n, -np.log(n)) # The normalized log-weight of each particle
        total = 0.0
        row0 = 0
        for k,row in enumerate(order):
            if swarm is not None: # The rows of the tables (of those that vary in time) for the steps to this data point
                swarm.perturb(rng)
                row0 = runs.done
                tables = swarm.tables(rows=slice(row0, max(last[k], runs.done)))
                shares = swarm.shares() if swarm.moving else None
            runs.advance(last[k], tables, shares, row0)
            stocks, flows = runs.stocks, runs.flows
            obs = np.zeros(n)
            for s,i in points[row]:
                lo, hi = s.spans[i]
                if s.read is None: # A result worked out from the others, from the steps it depends on (or the whole run)
                    start = max(lo - s.back, 0) if s.back is not None else 0
                    end = hi if s.back is not None else npts
                    vals = self.derive(s, *view(s.j, (stocks[:, start:end], flows[:, start:end])))[:, lo-start:hi-start]
                else:
                    kind, inds = s.read
                    arr = view(s.j, (stocks[:, lo:hi], flows[:, lo:hi]))[0 if kind == 'stocks' else 1][:, :, inds, :].sum(axis=2)
                    vals = (arr*s.keep).sum(axis=-1) if s.keep is not None else arr.sum(axis=-1)
                obs += s.weight*logpdf(s.lik, s.expected[i], vals @ s.W[i, lo:hi])
            obs = np.where(np.isnan(obs), -np.inf, obs)
            joint = logw + obs
            top = joint.max()
            if not np.isfinite(top) and swarm is not None: # For IF2, a data point that no run can give is skipped (as camdl does), rather than ending the pass
                continue
            elif not np.isfinite(top): # No run can give this data
                runs.restore()
                return -np.inf
            norm = top + np.log(np.exp(joint - top).sum())
            total += norm
            logw = joint - norm
            weights = np.exp(logw)
            if (1/(weights**2).sum() < n/2 and k < len(order) - 1) or swarm is not None: # Too uneven (or for IF2, always): draw the runs again (systematically), and keep the steps that are still needed
                cum = np.cumsum(weights)
                picks = np.minimum(np.searchsorted(cum, (rng.random() + np.arange(n))/n*cum[-1]), n - 1)
                runs.select(picks, keep=needed[k])
                if swarm is not None:
                    swarm.resample(picks)
                logw = np.full(n, -np.log(n))
        runs.restore()
        return total


class Swarm(sc.prettyobj):
    """
    The parameter values of each particle of IF2, iterated filtering, and the rates of the run that they give (see `sim.fit()` with method='if2')

    Each particle carries values of its own, on the unbounded scale that the fit searches on, which a
    random walk perturbs at the start of each pass of the filter and before each data point, by steps that
    shrink geometrically, to `cooling` of their first size after 50 passes (as pomp's `mif2()` and camdl
    do; Ionides et al. 2015). The rates are worked out from each particle's values by the model's own
    expressions, so the kernels run each particle with its own, and they are checked against the sim's own.

    Args:
        sim (`ss.Sim`): the sim, initialized, with the data of one declared disease
        fitted (dict): the fitted parameters, by name
        set_pars (func): restarts the sim with an array of values
        to_x (func): gives the values (an array, or particles by values) from the search scale, and their places between the bounds
        z0 (array): the values the sim is restarted with now, on the search scale
        sd (float): the first size of a step of the random walk, on the search scale
        cooling (float): the share of the first size that a step has after 50 passes
        n_points (int): the data points in a pass
    """
    def __init__(self, sim, fitted, set_pars, to_x, z0, sd=0.05, cooling=0.5, n_points=1):
        self.sim = sim
        self.fitted = list(fitted.values())
        self.names = list(fitted.keys())
        self.to_x = to_x
        self.sd = sd
        self.factor = cooling**(1/(50*(1 + n_points))) # Each pass perturbs the values at the start and before each data point
        self.steps = 0
        self.z = None
        mods = [mod for mod in sim.declared if mod.engine.cross is None]
        engines = {id(mod.engine):mod.engine for mod in sim.declared}
        if len(engines) != 1 or not mods:
            errormsg = 'IF2 fits a sim with one declared disease (not tracked jointly with another); use method="pmcmc"'
            raise ValueError(errormsg)
        self.engine = engine = mods[0].engine
        self.index, modified = ss.gradients.fitted_index(self.names, self.fitted, [engine.mod], sim) # The parameter objects that stand for each fitted value, as in ss.gradients.Density
        if modified:
            errormsg = f'IF2 cannot fit "{modified[0]}", which a modifier changes; use method="pmcmc"'
            raise ValueError(errormsg)
        self.consts = {k:self.index[id(par)] for k,par in enumerate(engine.prog_pars) if id(par) in self.index}
        self.flows = []
        for f,flow in enumerate(engine.flows): # The flows whose rate is worked out from fitted values alone, as in ss.gradients.Density
            if flow.dynamic or flow.kind == 'conveyor': # The flow out of a conveyor has no rate of its own (see ss.kernels.convey())
                continue
            dw = engine.dwells.get(flow.tr.name) if flow.expr is None else None
            leaves = flow.expr.leaves if flow.expr is not None else [dw.par] if dw is not None and dw.dist is None else []
            if dw is not None and dw.dist is not None and any(id(leaf) in self.index for leaf in dw.par.leaves):
                errormsg = f'IF2 cannot fit a parameter of the distribution of a dwell time ("{flow.tr.name}"); use method="pmcmc"'
                raise ValueError(errormsg)
            if any(id(leaf) in self.index for leaf in leaves):
                self.flows.append((f, flow, dw, np.broadcast_to(engine.flow_mult(flow), (engine.n_groups,)).astype(float)))

        # Each fitted value must set the rates of the run, or the time share between places, and nothing else: the rest is the same for every particle
        self.x0 = x0 = to_x(z0)[0]
        set_pars(x0)
        base = ss.gradients.capture(engine, compact=False)[2]
        self.base = [base[key] for key in ss.gradients.table_fields]
        self.sh0 = base['sh_val'] # The time share at x0, and its change with each value (see shares())
        self.sh_slopes = np.zeros((len(self.names), len(self.sh0)))
        self.moving = False
        for i,name in enumerate(self.names):
            x = to_x(z0 + 0.5*np.eye(len(z0))[i])[0] # Another value of this one
            set_pars(x)
            X = ss.gradients.capture(engine, compact=False)[2]
            changed = [key for key in X if not np.array_equal(X[key], base[key])]
            others = [key for key in changed if key not in ['fixed', 'consts', 'sh_val']]
            if others or not changed:
                why = f'it sets {sc.strjoin(others)} of the run, not only its rates' if others else 'it is not a parameter of the rates of the run (e.g. it is one of how the data are observed)'
                errormsg = f'IF2 cannot fit "{name}" yet, since {why}; fit it with method="pmcmc", or fix it'
                raise ValueError(errormsg)
            got = self.tables(x[None])
            if not all(np.allclose(g[0], X[key], rtol=1e-9, atol=1e-12) for g,key in zip(got, ss.gradients.table_fields)):
                errormsg = f'IF2 cannot fit "{name}" yet, since the rates worked out from it differ from the sim\'s own; fit it with method="pmcmc"'
                raise ValueError(errormsg)
            if 'sh_val' in changed: # The movement between places, worked out for each particle from a straight line (as away= gives), which a third value checks
                self.sh_slopes[i] = (X['sh_val'] - self.sh0)/(x[i] - x0[i])
                self.moving = True
                x = to_x(z0 - 0.5*np.eye(len(z0))[i])[0]
                set_pars(x)
                if not np.allclose(self.shares(x[None])[0], ss.gradients.capture(engine)[2]['sh_val'], rtol=1e-9, atol=1e-12):
                    errormsg = f'IF2 cannot fit "{name}" yet, since the movement between places does not change in proportion to it (as away= does); fit it with method="pmcmc", or fix it'
                    raise ValueError(errormsg)
        set_pars(x0)
        return

    def shares(self, x=None):
        """ The time share between places of every particle, as `ss.kernels.advance()` takes it (particles by entries; default, the swarm's) """
        x = self.to_x(self.z)[0] if x is None else x
        return np.ascontiguousarray(self.sh0 + (x - self.x0) @ self.sh_slopes)

    def convert(self, i, val):
        """ A fitted value as the engine uses it: in the sim's unit, as `ss.gradients.Density.convert()` """
        return ss.gradients.convert_fitted(self.fitted[i].raw, val, self.sim.unit_base)

    def start(self, z, n):
        """ Start every one of n particles at the same values (on the search scale) """
        self.z = np.tile(np.asarray(z, dtype=float), (n, 1))
        self.steps = 0
        return

    def perturb(self, rng):
        """ A step of the random walk for every particle """
        self.z = self.z + rng.normal(size=self.z.shape)*self.sd*self.factor**self.steps
        self.steps += 1
        return

    def resample(self, picks):
        """ The values of the particles drawn again """
        self.z = self.z[picks]
        return

    def estimate(self):
        """ The values at the middle of the swarm (its mean on the search scale, as pomp's `coef()` of a `mif2()` fit) """
        return self.to_x(self.z.mean(axis=0))[0]

    def tables(self, x=None, rows=slice(None)):
        """ The parts of the program that vary in time, as `ss.kernels.advance()` takes them: for every particle, with its own values (particles by values; default, the swarm's), at some rows (of those that vary in time) """
        x = self.to_x(self.z)[0] if x is None else x
        n = len(x)
        ctx = ParticleCtx(self, self.engine, x.T[:, :, None, None], rows) # Each value as particles by 1 by 1, so a value by row and group broadcasts
        out = [(table[rows] if len(table) > 1 else table)[None] for table in self.base]
        if self.flows:
            out[0] = np.repeat(out[0], n, axis=0)
            for f,flow,dw,mult in self.flows:
                out[0][:, :, f, :] = ss.gradients.fixed_rate(flow, dw, ctx)*mult
        if self.consts:
            out[2] = np.repeat(out[2], n, axis=0)
            for k,i in self.consts.items():
                out[2][:, :, k, :] = self.convert(i, x[:, i])[:, None, None]
        return out


class ParticleCtx(ss.gradients.Context):
    """ The context for the rates of each particle of IF2: the fitted values as arrays over the particles, the others as the engine has them (a value that varies in time, at some rows of the tables) """
    def __init__(self, swarm, engine, x, rows):
        self.density = swarm
        self.engine = engine
        self.x = x
        self.rows = rows
        self.n = None
        self.series = True
        self.xp = np
        return

    def value(self, p):
        kept = self.engine.series.get(id(p)) if id(p) not in self.density.index else None
        return kept[1][self.rows] if kept is not None else super().value(p)


def abc_smc(distance, logprior, z0, particles=200, generations=8, quantile=0.5, tolerance=None, min_acceptance=0.02, seed=None, verbose=True):
    """
    Approximate Bayesian computation by sequential Monte Carlo (Beaumont et al. 2009): a posterior without a likelihood

    A population of particles is drawn from the prior, and in each generation a new one is drawn
    from the last (each particle perturbed by a Gaussian kernel twice as wide as the population)
    and kept only if its run is within the tolerance of the data; the tolerance is the median
    distance of the last population, so it shrinks as the population moves toward the data. Weights
    correct for the particles not being drawn from the prior.

    Args:
        distance (func): the distance of a run from the data, given a point (on the sampler's scale) and a seed
        logprior (func): the log of the prior density of a point, on the sampler's scale
        z0 (array): a point on the sampler's scale (only its length is used: the first generation is drawn uniformly on that scale and weighted by the prior)
        particles (int): the size of the population
        generations (int): the most generations; fewer if the tolerance or the acceptance rate says to stop
        quantile (float): the share of the last population's distances below the next tolerance
        tolerance (float): stop once the tolerance is this small
        min_acceptance (float): stop once a generation accepts fewer of its proposals than this
        seed (int): the random seed
        verbose (bool): whether to say what is being done

    Returns the population (particles × values, on the sampler's scale), its normalized weights and
    distances, the tolerance and acceptance rate of each generation, and the number of runs.
    """
    rng = np.random.default_rng(seed)
    d = len(z0)
    seeds = iter(rng.integers(2**31, size=10**8))
    # Generation 0: from the prior, by importance sampling of uniform draws on the sampler's scale (each value's logit between its bounds)
    z = rng.uniform(-6, 6, size=(particles, d)) # Within 0.25% of each bound
    dist = np.array([distance(zi, next(seeds)) for zi in z])
    logw = np.array([logprior(zi) for zi in z])
    w = np.exp(logw - logw.max())
    w /= w.sum()
    tolerances, acceptances = [np.inf], [1.0]
    n_runs = particles
    for g in range(generations):
        eps = np.quantile(dist[np.isfinite(dist)], quantile) if np.isfinite(dist).any() else np.inf
        cov = 2*np.atleast_2d(np.cov(z.T, aweights=w)) + 1e-12*np.eye(d)
        chol = np.linalg.cholesky(cov)
        inv = np.linalg.inv(cov)
        new_z, new_dist, tried = [], [], 0
        while len(new_z) < particles and tried < 50*particles:
            zi = z[rng.choice(particles, p=w)] + chol @ rng.standard_normal(d)
            tried += 1
            di = distance(zi, next(seeds))
            if di <= eps:
                new_z.append(zi)
                new_dist.append(di)
        n_runs += tried
        if len(new_z) < particles: # Too few accepted: keep the last population
            break
        new_z = np.array(new_z)
        kern = np.exp(-0.5*np.einsum('ijk,kl,ijl->ij', new_z[:, None] - z[None], inv, new_z[:, None] - z[None])) @ w # The density each new particle was proposed with (up to a constant)
        logw = np.array([logprior(zi) for zi in new_z]) - np.log(np.maximum(kern, 1e-300))
        w = np.exp(logw - logw.max())
        w /= w.sum()
        z, dist = new_z, np.array(new_dist)
        tolerances.append(eps)
        acceptances.append(particles/tried)
        if verbose:
            print(f'ABC-SMC generation {g+1}: tolerance {eps:.4g}, {acceptances[-1]:.0%} of proposals accepted, effective sample size {1/np.sum(w**2):.0f}')
        if (tolerance is not None and eps <= tolerance) or acceptances[-1] < min_acceptance:
            break
    return z, w, dist, tolerances[1:], acceptances[1:], n_runs


def pmmh(logpost, z0, draws=1000, warmup=1000, chains=4, particles=None, seed=None, parallel=True, verbose=True):
    """
    Sample from a posterior by particle marginal Metropolis-Hastings: a random-walk Metropolis sampler on a log-density that is estimated by a particle filter

    Since the estimate of the likelihood is unbiased, the chains sample from the exact
    posterior (Andrieu, Doucet & Holenstein 2010), however noisy it is; more noise only
    makes them move more slowly. In the warm-up, the proposal's covariance is learned from
    the chain so far (Haario et al. 2001), and its scale is tuned toward accepting 15% of
    proposals (about the best when the estimate is noisy; Sherlock et al. 2015). In the first
    half of the warm-up, the estimate where the chain is is made again at each step, so that
    a chain far from the posterior, where the estimate is noisiest, cannot be held by a lucky one.

    Args:
        logpost (func): called with a point (an array, on an unbounded scale), a random number generator (`np.random.Generator`) to draw any seeds from, and the number of particles; returns an estimate of the log of the posterior density there
        z0 (array): the point to start each chain from
        draws (int): the draws to keep from each chain
        warmup (int): the draws for the warm-up of each chain, which are not kept
        chains (int): the number of chains
        particles (int): the number of particles; by default, each chain uses 100 for the first half of its warm-up, and then enough that the variance of the estimate at the middle of where it has been lately is about 1 (Doucet et al. 2015)
        seed (int): the random seed
        parallel (bool): whether to run the chains in parallel processes
        verbose (bool): whether to say what is being done

    Returns:
        The draws (chains × draws × values), the estimate of the log-density at each (chains × draws), the share of proposals accepted after the warm-up in each chain, and the number of particles that each chain used
    """
    z0 = np.asarray(z0, dtype=float)
    d = len(z0)

    def one_chain(chain_seed):
        rng = np.random.default_rng(chain_seed)
        n = particles or 100
        out = np.zeros((draws, d))
        logdens = np.zeros(draws)
        accepted = 0.0
        z = z0.copy()
        lp = logpost(z, rng, n)
        log_scale = np.log(2.38**2/d)
        cov = np.eye(d)*0.01
        history = []
        for i in range(warmup + draws):
            if i < warmup//2: # Early in the warm-up, the estimate where the chain is is made again each time, so a lucky one cannot hold it (the draws are not kept)
                lp = logpost(z, rng, n)
            elif particles is None and i == warmup//2: # Near the posterior by now, so the number of particles it needs there (at the middle of where the chain has been lately) can be found
                center = np.mean(history[len(history)//2:], axis=0)
                spread = np.var([logpost(center, rng, n) for k in range(10)])
                n = int(np.clip(np.ceil(n*spread), 50, 5000)) if np.isfinite(spread) else 5000
                lp = logpost(z, rng, n)
            prop = rng.multivariate_normal(z, np.exp(log_scale)*cov)
            lp_prop = logpost(prop, rng, n)
            accept = np.log(rng.random()) < lp_prop - lp
            if accept:
                z, lp = prop, lp_prop
            if i < warmup: # Learn the shape of the posterior, and tune the scale
                history.append(z)
                log_scale += (float(accept) - 0.15)/np.sqrt(i + 1)
                if i >= 50 and i % 10 == 0:
                    cov = np.cov(np.array(history[len(history)//2:]).T).reshape(d, d) + 1e-8*np.eye(d)
            else:
                out[i - warmup] = z
                logdens[i - warmup] = lp
                accepted += accept/draws
        return out, logdens, accepted, n

    T = sc.timer()
    seeds = np.random.default_rng(seed).integers(2**31, size=chains)
    if parallel and chains > 1 and 'jax' in sys.modules: # Forking a process that has started JAX can deadlock
        parallel = False
        if verbose:
            print('Particle MCMC: running the chains one after another, since JAX has been started in this process')
    if parallel and chains > 1:
        method = ss.kernels.start_method()
        with ss.kernels.starting_workers(method):
            results = sc.parallelize(ss.kernels.worker(one_chain), iterarg=list(seeds), start_method=method)
    else:
        results = [one_chain(chain_seed) for chain_seed in seeds]
    out, logdens, accepted, n = [np.array(v) for v in zip(*results)]
    if verbose:
        print(f'Particle MCMC: {chains} chain{"s" if chains != 1 else ""}{" in parallel" if parallel and chains > 1 else ""}, each of {warmup} warm-up draws and then {draws} draws, '
              f'in {T.tt(output=True):.1f} s; particles: {sc.strjoin(n)}; proposals accepted: {sc.strjoin([f"{a:.0%}" for a in accepted])}')
    return out, logdens, accepted, n


def nuts(logdensity, z0, draws=500, warmup=500, chains=4, traced=False, forward=True, seed=None, verbose=True):
    """
    Sample from a posterior with the No-U-Turn sampler (NUTS, from BlackJAX), with Stan's warm-up of the step size and the mass matrix

    The log-density is given in one of two ways. By default, as a function of NumPy arrays that
    returns its value and gradient, so it can be anything (here, a run of the sim, differentiated
    by `ss.gradients.gradient()`); it is called back from JAX, and the chains are run one after
    another. With `traced=True`, as a JAX function (see `ss.gradients.Density`), so that the whole
    of the sampling is compiled, with the chains run together (each on cores of its own); its
    gradient is taken in forward mode (one direction per parameter), which for a few parameters
    is much cheaper than reverse mode through the steps of an ODE, or with `forward=False` in
    reverse mode (one pass whatever their number, for many parameters).

    Args:
        logdensity (func): called with a point (an array, on an unbounded scale); returns the log of the posterior density there and its gradient (or, with `traced=True`, the log-density alone, as a JAX function)
        z0 (array): the point to start each chain near, or a point for each chain (chains × values)
        draws (int): the draws to keep from each chain
        warmup (int): the draws for the warm-up of each chain, which are not kept
        chains (int): the number of chains
        traced (bool): whether `logdensity` is a JAX function
        forward (bool): for a JAX function, whether its gradient is taken in forward mode (else reverse mode, for which it has to be differentiable)
        seed (int): the random seed
        verbose (bool): whether to say what is being done

    Returns:
        The draws (chains × draws × values), the log-density at each (chains × draws), and whether each followed a divergent trajectory (chains × draws)
    """
    jax = ss.gradients.jax()
    import blackjax
    jnp = jax.numpy
    starts = np.broadcast_to(np.asarray(z0, dtype=float), (chains, np.shape(z0)[-1])) # Each chain's start
    d = starts.shape[1]
    x64 = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True) # For the whole run, since the callbacks are made from other threads, which jax.enable_x64() does not reach
    try:
        key = jax.random.key(seed if seed is not None else 0)
        if traced:
            return nuts_traced(jax, blackjax, logdensity, starts, key, draws, warmup, verbose, forward=forward)
        shape = (jax.ShapeDtypeStruct((), jnp.float64), jax.ShapeDtypeStruct((d,), jnp.float64))

        def host(z):
            value, grad = logdensity(np.asarray(z, dtype=float))
            if not np.isfinite(value):
                return np.float64(-np.inf), np.zeros(d)
            return np.float64(value), np.asarray(grad, dtype=np.float64)

        @jax.custom_vjp
        def logp(z):
            return jax.pure_callback(host, shape, z)[0]

        def forward(z):
            value, grad = jax.pure_callback(host, shape, z)
            return value, grad

        def backward(grad, cotangent):
            return (cotangent*grad,)

        logp.defvjp(forward, backward)
        out = np.zeros((chains, draws, d))
        logdens = np.zeros((chains, draws))
        divergent = np.zeros((chains, draws), dtype=bool)
        for c in range(chains):
            key, start_key, warm_key, run_key = jax.random.split(key, 4)
            start = jnp.asarray(starts[c]) + 0.01*jax.random.normal(start_key, (d,)) # Each a little apart, even from the same point
            if verbose:
                print(f'NUTS chain {c+1} of {chains}: {warmup} warm-up draws, then {draws} draws')
            ss.solvers.backends.compiling('the NUTS sampler', 'JAX', 1, 10)
            adapt = blackjax.window_adaptation(blackjax.nuts, logp)
            (state, params), _ = adapt.run(warm_key, start, num_steps=warmup)
            step = jax.jit(blackjax.nuts(logp, **params).step)
            for i,draw_key in enumerate(jax.random.split(run_key, draws)):
                state, info = step(draw_key, state)
                out[c, i] = np.asarray(state.position)
                logdens[c, i] = float(state.logdensity)
                divergent[c, i] = bool(info.is_divergent)
        return out, logdens, divergent
    finally:
        jax.config.update('jax_enable_x64', x64)


def nuts_traced(jax, blackjax, logdensity, starts, key, draws, warmup, verbose, forward=True):
    """ `nuts()` with `traced=True`: the whole of the sampling compiled, with the chains run together """
    jnp = jax.numpy
    chains, d = starts.shape
    logdensity = jax.jit(logdensity) # Traced once, rather than wherever the sampler uses it

    if forward:
        @jax.custom_vjp
        def logp(z):
            return logdensity(z)

        def slopes(z): # The value and every partial derivative, in forward mode
            values, slopes = jax.vmap(lambda e: jax.jvp(logdensity, (z,), (e,)))(jnp.eye(d))
            return values[0], slopes

        def backward(slopes, cotangent):
            return (cotangent*slopes,)

        logp.defvjp(slopes, backward)
    else: # In reverse mode, as JAX takes it
        logp = logdensity

    def one_chain(key, start):
        warm_key, run_key = jax.random.split(key)
        adapt = blackjax.window_adaptation(blackjax.nuts, logp)
        (state, params), _ = adapt.run(warm_key, start, num_steps=warmup)
        step = blackjax.nuts(logp, **params).step
        def body(state, key):
            state, info = step(key, state)
            return state, (state.position, state.logdensity, info.is_divergent)
        _, out = jax.lax.scan(body, state, jax.random.split(run_key, draws))
        return out

    start_key, chain_key = jax.random.split(key)
    starts = jnp.asarray(starts) + 0.01*jax.random.normal(start_key, (chains, d)) # Each a little apart, even from the same point
    keys = jax.random.split(chain_key, chains)
    devices = jax.devices('cpu')
    n_dev = max(k for k in range(1, min(len(devices), chains) + 1) if chains % k == 0) # Each device takes the same number of chains
    shape = lambda arr: arr.reshape((n_dev, chains//n_dev) + arr.shape[1:])
    T = sc.timer()
    ss.solvers.backends.compiling('the NUTS sampler', 'JAX', 1, 10)
    if n_dev > 1: # Each on cores of its own
        compiled = jax.pmap(jax.vmap(one_chain), devices=devices[:n_dev]).lower(shape(keys), shape(starts)).compile()
        args = (shape(keys), shape(starts))
    else:
        compiled = jax.jit(jax.vmap(one_chain)).lower(keys, starts).compile()
        args = (keys, starts)
    compiling = T.tt(output=True)
    out = [np.asarray(v).reshape((chains,) + v.shape[2 if n_dev > 1 else 1:]) for v in compiled(*args)]
    if verbose:
        print(f'NUTS: {chains} chains on {n_dev} core{"s" if n_dev != 1 else ""}, each of {warmup} warm-up draws and then {draws} draws: compiled in {compiling:.1f} s, sampled in {T.tt(output=True):.1f} s')
    return out


def diagnose(draws):
    """
    The R-hat (near 1 if the chains agree) and the effective sample size of each value, from draws (chains × draws × values)

    Both are BlackJAX's, rank-normalized and from split chains (Vehtari et al. 2021, as in Stan and ArviZ): the R-hat is the
    larger of that of the draws and that of their distance from the median, and the effective sample size is of the bulk.
    Both are NaN for chains of fewer than 4 draws, too short to tell, and without BlackJAX (`pip install "starsim[jax]"`).

    Args:
        draws (array): the draws, with shape (chains, draws, values)
    """
    if draws.shape[1] < 4:
        nans = np.full(draws.shape[2], np.nan)
        return nans, nans.copy()
    try:
        import blackjax.diagnostics as bd
    except ImportError:
        ss.note('the R-hat and effective sample size need JAX, which is not installed, so they are NaN: use pip install "starsim[jax]"')
        nans = np.full(draws.shape[2], np.nan)
        return nans, nans.copy()
    jax = ss.gradients.jax()
    with jax.enable_x64(True): # Otherwise in single precision
        return np.array(bd.rhat(draws)).reshape(draws.shape[2]), np.array(bd.ess_bulk(draws)).reshape(draws.shape[2]) # BlackJAX drops an axis of one value


class Fit(sc.prettyobj):
    """
    The result of fitting a sim to its data; see `sim.fit()`

    Attributes:
        sim (`ss.Sim`): the calibrated sim: a copy of the original with the best-fit values, already run
        pars (dict): the best-fit value of each fitted parameter; can be passed to `ss.Sim(pars=...)`
        table (df): parameter, initial value, best fit, and bounds (and with `method='if2'`, the lowest and highest of the chains' estimates, as `spread_lower` and `spread_upper`)
        before (`ss.Sim`): the sim with the initial values, already run
        mismatch (float): the goodness of fit of the calibrated sim, or its negative log-likelihood if the fit used likelihoods
        measure (str): what `mismatch` measures
        mismatch_before (float): the same, before the fit
        converged (bool): whether the searches from the best few points of the scan that are apart end at the same value (for SciPy; with a single search, SciPy's own stopping rule), stopped improving (for Optuna: by less than 1% over the last quarter of the trials), or has chains that agree (for NUTS and particle MCMC: an R-hat of at most 1.01, and no divergent draws; for IF2: chains from different starts that end within 2 of the best log-likelihood)
        why (str): what `converged` is based on
        scores (list): the score of the best-fit values in each run (one for a deterministic sim); `mismatch` is their mean, so it is the score that was minimized
        trials (df): every evaluation: the values, and the `objective` that the method minimized (what `measure` says; with `method='if2'`, also each pass of each chain: the middle of the swarm's values, and minus the log-likelihood of the pass, whose values were perturbed)
        samples (df): with `method='nuts'`, `'pmcmc'`, or `'abc-smc'`, the draws from the posterior: one row per draw, with its chain, its number, and a column for each parameter
        study (`optuna.Study`): with `method='optuna'`, Optuna's study, e.g. for its plots (`optuna.visualization.plot_param_importances(fit.study)`)
        diagnostics (dict): what else the method reports: `divergences` (with `method='nuts'`, the number of draws that followed a divergent trajectory, which suggests the sampler missed part of the posterior), `acceptance` (with `method='pmcmc'`, the share of proposals that each chain accepted after its warm-up), and `band` (with a posterior, for each data series, the 2.5th and 97.5th percentiles of the model's value over draws from it, at each step, or for a flow with data at a longer period than the step, of its totals over the data's periods; with `'pmcmc'`, each from a run with a seed of its own, so the band includes chance)
    """
    def __init__(self, sim, before, pars, table, trials, method, mismatch, mismatch_before, measure, samples, diagnostics, converged, why, scores, study=None):
        self.sim = sim
        self.before = before
        self.pars = pars
        self.table = table
        self.trials = trials
        self.method = method
        self.mismatch = mismatch
        self.mismatch_before = mismatch_before
        self.measure = measure
        self.samples = samples
        self.diagnostics = diagnostics
        self.converged = converged
        self.why = why
        self.scores = scores
        self.study = study
        return

    def project(self, scenarios=None, n=None, rand_seed=None, **kwargs):
        """
        Run the fitted model forward under each scenario, with the uncertainty in the fitted values: one run per draw from the posterior, each with a seed of its own (or runs of the best fit, each with its own seed, for a method that gives none), with the same draws and seeds in every scenario

        Args:
            scenarios (dict): for each scenario, what changes: None, an intervention or a list of them, or a dict of arguments for `sim.new()` (default: the fitted sim alone)
            n (int): the number of runs of each scenario (default: 100; but one for the best fit of a deterministic sim (an ODE), whose runs would all be the same)
            rand_seed (int): the seed of the draws
            kwargs (dict): passed to `sim.new()` for every scenario, e.g. a later `stop=`

        Returns an `ss.MultiSim`, whose results have a run dimension labeled by scenario, with the draws as runs of each scenario (see `msim.to_df(quantiles=...)`, whose quantiles are over the draws).

        Examples:
            ```python
            fit = sim.fit(method='abc-smc')
            proj = fit.project(dict(baseline=None, vaccine=ss.vaccinate(...)), stop='2021-01-01')
            df = proj.to_df(quantiles=[0.05, 0.5, 0.95])
            ```
        """
        names = list(self.table['par']) # With a parameter by level, one name for each level
        sims = []
        for label,change in (scenarios or dict(fit=None)).items():
            if change is None:
                change = {}
            elif not isinstance(change, dict):
                change = dict(interventions=change)
            sims.append(self.sim.new(label=label, **sc.mergedicts(dict(pars=dict(self.pars)), kwargs, change)))
        if self.samples is None:
            n = sc.ifelse(n, 1 if self.sim.solver == 'ode' else 100) # With no posterior, the runs differ only by their seeds
            return ss.parallel(*sims, n_runs=n)
        n = sc.ifelse(n, 100)
        draws = self.samples.sample(n=n, replace=len(self.samples) < n, random_state=rand_seed)[names]
        return ss.parallel(*sims, priors=draws) # Draws, not a sweep: runs of each scenario

    def summarize(self):
        """ Print the goodness of fit before and after, and return the estimates (a dataframe), which are printed too in a script (see `ss.utils.return_table()`) """
        if self.samples is None:
            print(f'Fitted {len(self.pars)} parameter(s) by {self.method}, with {len(self.trials)} evaluations')
        else: # The evaluations in JAX are not recorded
            print(f'Fitted {len(self.pars)} parameter(s) by {self.method}')
        ss.utils.return_table(self.table, index=False) # Printed in a script; in a notebook, shown once, as what is returned
        print(f'{self.measure.capitalize()}: {self.mismatch_before:.4g} before, {self.mismatch:.4g} after' + (f' (the mean of {len(self.scores)} runs)' if self.scores is not None and len(self.scores) > 1 else ''))
        if self.converged is not None:
            print(f'{"Converged" if self.converged else "Not converged"}: {self.why}')
        if 'spread_lower' in self.table:
            spread = "the chains' estimates" if self.method == 'if2' else 'the best 10% of the trials'
            print(f'spread_lower and spread_upper: the range of {spread}, an indication of the uncertainty only, not an interval')
        if self.samples is not None:
            n_chains = self.samples.chain.nunique()
            print(f'Posterior: {len(self.samples):,} draws from {n_chains} chain{"s" if n_chains != 1 else ""} (best: the draw with the highest posterior density; median, lower, and upper: the median and 95% interval of the posterior)')
            acceptance, divergences = self.diagnostics.acceptance, self.diagnostics.divergences
            if acceptance is not None:
                print(f'Proposals accepted by each chain: {sc.strjoin([f"{a:.0%}" for a in acceptance])}')
            unsettled = self.table.par[self.table.rhat > 1.01].tolist() if 'rhat' in self.table else []
            if unsettled or divergences:
                ss.warn(f'{f"the chains disagree about {sc.strjoin(unsettled)} (R-hat above 1.01)" if unsettled else ""}{"; " if unsettled and divergences else ""}{f"{divergences} draws diverged" if divergences else ""}; '
                        f'try more warm-up (warmup=), or check that the parameters are identifiable', ss.errors.NotConverged)
        return self.table

    def to_df(self):
        """ Every evaluation: the parameter values tried, and the objective at each """
        return self.trials

    def plot(self, residuals=False, **kwargs):
        """
        Plot the results that have data against it, before and after fitting; a flow with data at a longer period than the step (e.g. weekly cases) is shown as its totals over the data's periods, as the fit compares them

        Args:
            residuals (bool): whether to add the residuals of the fitted model (the data minus the model, as the fit compares them) in axes below each result
            kwargs (dict): see `ss.plot_args()`, e.g. `figsize=(8, 4)`

        Example:
            ```python
            fit = sim.fit()
            fit.plot(residuals=True) # A pattern in the residuals points to the structure of the model
            ```
        """
        cols = list(self.sim.data.columns)
        times = plot_times(self.sim.data)
        kw = ss.plot_args(kwargs, alpha=0.8, data_color='k')
        k = 2 if residuals else 1 # Rows of axes for each row of results
        nrows, ncols = sc.getrowscols(len(cols))
        with ss.style(**kw.style):
            fig, axs = sc.getrowscols(k*nrows*ncols, nrows=k*nrows, make=True, squeeze=False, remove_extra=False, **(dict(height_ratios=[2, 1]*nrows) if residuals else {}), **kw.fig)
            main, below = axs[::k].flatten(), axs[1::2].flatten() if residuals else [] # The axes of the results, and of their residuals
            for ax in [*main[len(cols):], *below[len(cols):]]:
                fig.delaxes(ax)
            if residuals:
                lined, compared = lineup(self.sim), compare(self.sim)
                for ax,col in zip(below, cols):
                    expected, model = compared[col]
                    ax.scatter(times[lined[col][2]], expected - model, s=15, **kw.data)
                    ax.axhline(0, color='k', lw=0.5)
                    ax.sharex(main[cols.index(col)])
                    ax.set_title(f'{col}: data minus model')
            for ax,col in zip(main, cols):
                for sim,label in [(self.before, 'Before'), (self.sim, 'After')]:
                    res = match_column(sim, col)
                    summed = flow_at_data(sim, col, res)
                    x, y = (times[summed[0]], summed[1]) if summed is not None else (res.timevec, res.values)
                    ax.plot(x, y, label=label, **kw.plot)
                if self.diagnostics.band is not None:
                    ax.fill_between(x, *self.diagnostics.band[col], alpha=0.3, label='95% of the posterior')
                ax.scatter(times, self.sim.data[col], s=15, label='Data', zorder=10, **kw.data)
                ax.set_title(col + (' (total over each period of the data)' if summed is not None else ''))
                ax.legend(frameon=False)
            sc.figlayout(fig=fig)
        return ss.return_fig(fig, **kw.return_fig)


def plot_times(df):
    """ The times of data, as a sim's results have them (dates as an `ss.DateArray`), so they share an axis, with Starsim's date ticks """
    x = df.index
    return ss.DateArray([ss.date(t) for t in x]) if len(x) and isinstance(x[0], pd.Timestamp) else np.asarray(x)


def vector_labels(sim, by, key):
    """ The labels of the levels of a parameter with a value for each level of one stratification """
    if by.dim is None:
        raise ss.CapabilityError(f'the parameter "{key}" has a value for each combination of {sc.strjoin(by.dims)}; a fit takes a value for each level of one stratification, e.g. ss.par(ss.by(patch=[0.2, 0.3]), bounds=[0.1, 1])')
    strat = next((strat for strat in sim.groups.strats if strat.name == by.dim), None)
    return [str(label) for label in strat.labels] if strat is not None else [str(i) for i in range(len(sc.tolist(by.values)))]


def product_pairs(sim, names):
    """
    The pairs of fitted parameters that appear in the model only multiplied together, or only one divided by the other, so that data can determine only their product (or ratio)

    Args:
        sim (`ss.Sim`): the sim, initialized
        names (list): the names of the fitted parameters of the model ("beta", or "sir.beta")

    Returns a list of (name, name, 'product' or 'ratio').
    """
    found = {}
    for name in names:
        modname, _, key = name.rpartition('.')
        found[name] = [mod.graph.pars[key] for mod in sim.declared if key in mod.graph.pars and modname in ['', mod.name]]
    uses = [] # Every expression that a parameter can appear in
    for mod in sim.declared:
        graph = mod.graph
        for tr in graph.transitions.values():
            uses += [tr.rate, tr.dur] + sc.tolist(tr.p)
        for tf in graph.transfers.values():
            uses += [tf.rate, tf.dur] + sc.tolist(tf.p)
        for oc in graph.outcomes.values():
            uses += [oc.source, oc.delay, oc.duration]
        uses += list(graph.exprs.values()) + [init for state in graph.states.values() for init in [state.init, state.init_n]]
    uses += [m.amount for mods in sim.modifiers.values() for m in mods]
    uses = [use for use in uses if isinstance(use, ss.expressions.Expr)]

    def signs(expr, sign=1, out=None):
        """ The parameters that are factors of an expression: +1 for one it is multiplied by, -1 for one it is divided by """
        out = {} if out is None else out
        if isinstance(expr, ss.expressions.BinOp) and expr.op in ['*', '/']:
            signs(expr.a, sign, out)
            signs(expr.b, sign if expr.op == '*' else -sign, out)
        elif isinstance(expr, ss.par):
            out[id(expr)] = out.get(id(expr), 0) + sign
        return out

    out = []
    for i,a in enumerate(names):
        for b in names[i+1:]:
            relation = None
            for use in uses:
                ids = [id(leaf) for leaf in use.leaves]
                n_a = sum(ids.count(id(p)) for p in found[a])
                n_b = sum(ids.count(id(p)) for p in found[b])
                if n_a == 0 and n_b == 0:
                    continue
                factors = signs(use)
                sa = sum(factors.get(id(p), 0) for p in found[a])
                sb = sum(factors.get(id(p), 0) for p in found[b])
                if n_a != 1 or n_b != 1 or abs(sa) != 1 or abs(sb) != 1 or relation not in [None, sa*sb]: # Each once, as a factor, the same way everywhere
                    relation = 0
                    break
                relation = sa*sb
            if relation:
                out.append((a, b, 'product' if relation > 0 else 'ratio'))
    return out


def correlated_pairs(names, values, threshold):
    """ The pairs of parameters whose values (one row each, e.g. draws from a posterior) are correlated beyond a threshold: (name, name, r) """
    out = []
    if len(names) > 1 and values.shape[1] > 2:
        r = np.corrcoef(values)
        for i in range(len(names)):
            for k in range(i+1, len(names)):
                if np.isfinite(r[i, k]) and abs(r[i, k]) > threshold:
                    out.append((names[i], names[k], r[i, k]))
    return out


gof_names = ['nae', 'rmse'] # The goodness-of-fit measures that mismatch() knows
max_scan = 100 # The most points a scan before a search tries by default: 10 per parameter up to 10 parameters
scipy_methods = ['nelder-mead', 'powell', 'l-bfgs-b', 'slsqp'] # With and without gradients, all keeping to the bounds


class Optimization(sc.prettyobj):
    """
    The result of allocating spending to maximize (or minimize) a result; see `ss.fit.optimize()`

    Attributes:
        alloc (dict): the best allocation, by program
        value (float): the result with it
        total (float): its total (with `target=`, the least that reaches the target)
        alloc_before (dict): the allocation before
        value_before (float): the result with it
        sim (`ss.Sim`): the sim run with the best allocation
        n_runs (int): the number of runs the search took
    """
    def __init__(self, **kwargs):
        for key,val in kwargs.items():
            setattr(self, key, val)
        return

    def summarize(self):
        """ Print the result before and after, and return the allocation before and after (a dataframe), which is printed too in a script (see `ss.utils.return_table()`) """
        df = pd.DataFrame(dict(before=self.alloc_before, after=self.alloc))
        ss.utils.return_table(df, float_format=lambda v: f'{v:,.0f}')
        print(f'{self.what}: {self.value_before:,.6g} before, {self.value:,.6g} after ({self.n_runs} runs)')
        if self.target is not None:
            print(f'The least total that reaches {self.target:,.6g}: {self.total:,.0f} (from {sum(self.alloc_before.values()):,.0f})')
        return df


def optimize(sim, total=None, maximize=None, minimize=None, at=None, start=None, spend=None, bounds=None, target=None, maxiter=50, verbose=None):
    """
    Allocate spending between programs to maximize (or minimize) a result, keeping the total fixed; or find the least total that reaches a target

    The spending of each program is a parameter of the model: one per program of an imported program book
    (`sim.imported.programs`), or any parameters, given by name with `spend=`. The search (SciPy's SLSQP) is
    over the shares of the total, and each point is a run (restarted, if the sim is nothing but compartments).
    With `target=`, the total is doubled until its best allocation reaches the target, then found by bisection
    between a total that reaches it and one that does not (to within 1%), assuming that more money never does worse.

    Args:
        sim (`ss.Sim`): the sim, not yet run
        total (float): the total to allocate (default: the current total); with `target=`, the total the search starts from
        maximize (str/func): what to make as large as possible: a result of the disease (e.g. `'all_vs'`, or `'hiv.n_vs'`), or a function of the run sim
        minimize (str/func): or what to make as small as possible
        at (float/date): the time of the result (default: the end), read as any time is (see `ss.connectors.to_time()`: a number from 1900 on is a year); without it, a function of the sim decides
        start (float/date): when the new allocation takes effect (default: the start of the sim), read as `at` is
        spend (dict/list): the spending parameters: names, or {program: name} (default: the imported programs)
        bounds (tuple/dict): the least and most a program can have, as amounts (default: 0 and the total); a dict gives them by program, e.g. `dict(testing=(1e5, None))` (None for no limit), and a program not in it has the default
        target (float): the value of the result to reach (at most this with `minimize=`, at least with `maximize=`) with the least total
        maxiter (int): the most iterations of the search (of each total's, with `target=`)
        verbose (bool): whether to print progress

    Returns an `ss.fit.Optimization`.

    Examples:
        ```python
        sim = ss.import_from('hiv_framework.xlsx', format='atomica', databook='hiv_databook.xlsx', progbook='hiv_progbook.xlsx', programs_start=2020)
        best = ss.fit.optimize(sim, start=2020, maximize='all_vs', at=2030)
        best.summarize()
        least = ss.fit.optimize(sim, start=2020, maximize='all_vs', at=2030, target=1.1*best.value) # The least money that reaches it
        ```
    """
    import scipy.optimize as spo
    verbose = sc.ifelse(verbose, ss.options.verbose)
    if (maximize is None) == (minimize is None):
        errormsg = 'ss.fit.optimize() takes one of maximize= or minimize=: a result (e.g. maximize="all_vs") or a function of the sim'
        raise ValueError(errormsg)
    if spend is None:
        spend = getattr(sim, 'imported', {}).get('programs') if isinstance(getattr(sim, 'imported', None), dict) else None
        if not spend:
            errormsg = 'ss.fit.optimize() needs the parameters of spending: give spend= (their names, or {program: name}), or import a program book'
            raise ValueError(errormsg)
    spend = dict(spend) if isinstance(spend, dict) else {name:name for name in sc.tolist(spend)}
    labels, names = list(spend.keys()), list(spend.values())
    probe = sim.new(verbose=0).init()
    current = {}
    for name in names:
        found = [mod.pars[name] for mod in probe.declared if name in mod.pars]
        if not found:
            raise sc.KeyNotFoundError(f'ss.fit.optimize() has spend="{name}", which is not a parameter of the model')
        current[name] = found[0].raw if isinstance(found[0], ss.par) else found[0]
    to_time = lambda time: ss.connectors.to_time(probe, time) # On the sim's timeline: a date, or a duration on its own axis
    as_year = lambda time: time.years if isinstance(time, ss.date) else time.to('year').value
    t0 = probe.t.tvec[0]
    begin = None if start is None else to_time(start)
    def level(raw):
        """ A spending parameter's value at the start of the new allocation """
        if isinstance(raw, ss.timeseries):
            return float(raw.at(as_year(sc.ifelse(begin, t0)), probe))
        return float(raw.value if isinstance(raw, ss.TimePar) else raw)
    before = np.array([level(current[name]) for name in names])
    total = float(sc.ifelse(total, before.sum()))
    by_program = bounds if isinstance(bounds, dict) else {}
    unknown = set(by_program) - set(labels)
    if unknown:
        raise ValueError(f'ss.fit.optimize() has bounds for {sc.strjoin(sorted(unknown))}, which {"is not a program" if len(unknown) == 1 else "are not programs"}; the programs are {sc.strjoin(labels)}')
    def limits(total):
        """ The least and most each program can have, of a total """
        pairs = [by_program.get(label, (None, None)) if isinstance(bounds, dict) else (bounds or (None, None)) for label in labels]
        lo = np.array([0.0 if a is None else float(a) for a,b in pairs])
        hi = np.array([total if b is None else min(float(b), total) for a,b in pairs])
        return lo, hi

    def values(amounts):
        """ The parameters for an allocation: from the start of the new allocation, if given """
        out = {}
        for name,val in zip(names, amounts):
            raw = current[name]
            if begin is None or begin <= t0:
                out[name] = float(val)
            else: # Keyed by times on the sim's timeline, so a sim with dates reads them as it should
                old = {k:v for k,v in zip(raw.times, raw.values) if to_time(k) < begin} if isinstance(raw, ss.timeseries) else {}
                old = old or {t0:float(raw.at(as_year(t0), probe)) if isinstance(raw, ss.timeseries) else level(raw)} # The value before the new allocation
                out[name] = ss.timeseries({**old, begin:float(val)})
        return out

    what = maximize if maximize is not None else minimize
    sign = -1.0 if maximize is not None else 1.0
    def read(run):
        if callable(what):
            return float(what(run))
        modname, _, key = str(what).rpartition('.')
        mod = run.diseases[modname] if modname else run.declared[0]
        vals = np.asarray(mod.results[key].values)
        i = len(vals) - 1 if at is None else int(np.argmin(abs(np.asarray(run.t.yearvec) - as_year(to_time(at)))))
        return float(vals[i])

    runner = sim.new(pars=values(before), verbose=0).init() # Built with the spending as it will be given, so a restart can change it
    restart = runner.kernel_only()
    n_runs = [0]
    def run(amounts):
        n_runs[0] += 1
        if restart:
            return runner.restart(pars=values(amounts)).run()
        return sim.new(pars=values(amounts), verbose=0).run()
    value_before = read(run(before))
    scale = max(abs(value_before), 1e-12)

    def allocate(total, x0):
        """ The best allocation of a total (as amounts), its value, its run, and SciPy's result, from a starting allocation (as shares) """
        lo, hi = limits(total)
        if total <= 0:
            best = run(np.zeros(len(names)))
            return np.zeros(len(names)), read(best), best, None
        objective = lambda x: sign*read(run(x*total))/scale
        res = spo.minimize(objective, x0, method='SLSQP', bounds=list(zip(lo/total, hi/total)),
                           constraints=[dict(type='eq', fun=lambda x: x.sum() - 1)], options=dict(maxiter=maxiter))
        x = np.clip(res.x, lo/total, hi/total)
        x = x/x.sum()
        best = run(x*total)
        return x*total, read(best), best, res

    T = sc.timer()
    shares = before/before.sum() if before.sum() > 0 else np.full(len(names), 1/len(names)) # The current shares, which the search starts from
    if target is None:
        alloc, value, best, res = allocate(total, shares)
        scaled = shares*total # The current allocation, scaled to the total
        lo, hi = limits(total)
        if total > 0 and np.all((lo <= scaled) & (scaled <= hi)): # The search can end worse than it started
            start_value = value_before if np.allclose(scaled, before) else read(run(scaled))
            if sign*value > sign*start_value:
                alloc, value, best = scaled, start_value, run(scaled)
    else:
        reached = lambda value: sign*value <= sign*target
        most = limits(np.inf)[1].sum() # The most that can be spent, if every program has an upper bound
        least = limits(0)[0].sum() # And the least
        tried = lambda tot, value: print(f'  Total {tot:,.6g}: {what} {value:,.6g} at best ({"reaches" if reached(value) else "does not reach"} {target:,.6g})') if verbose else None
        high = min(max(total, least), most)
        out = allocate(high, shares)
        tried(high, out[1])
        low = least
        for i in range(10): # Double the total until it reaches the target
            if reached(out[1]):
                break
            if high >= most:
                break
            low, high = high, min(2*high, most)
            out = allocate(high, out[0]/out[0].sum() if out[0].sum() > 0 else shares)
            tried(high, out[1])
        if not reached(out[1]):
            raise ValueError(f'ss.fit.optimize() could not reach {what} = {target:,.6g}: the best allocation of {high:,.6g} gives {out[1]:,.6g}')
        for i in range(30): # Halve the interval between a total that does not reach the target and one that does
            if high - low <= 0.01*high:
                break
            mid = (low + high)/2
            trial = allocate(mid, out[0]/out[0].sum())
            tried(mid, trial[1])
            if reached(trial[1]):
                high, out = mid, trial
            else:
                low = mid
        alloc, value, best, res = out
    if verbose:
        print(f'Optimized {what}: {value_before:,.6g} to {value:,.6g}' + (f', with a total of {alloc.sum():,.6g}' if target is not None else '') + f', in {n_runs[0]} runs ({T.toc(output=True):.1f} s)')
    return Optimization(alloc=dict(zip(labels, alloc)), value=value, total=float(alloc.sum()), target=target, alloc_before=dict(zip(labels, before)), value_before=value_before,
                        sim=best, n_runs=n_runs[0], what=str(getattr(what, '__name__', what)), converged=bool(res.success) if res is not None else True, why=str(res.message) if res is not None else 'nothing to allocate')


def bound_value(b, raw):
    """ A bound of a parameter in the unit of its value: a typed bound (e.g. `ss.days(3)`) is converted to the type of the parameter """
    return (type(raw)(b) if isinstance(raw, ss.TimePar) else b).value if isinstance(b, ss.TimePar) else b


def bounded_pars(probe, pars=None, unused=()):
    """
    The parameters with bounds in an initialized sim, as `sim.fit()` and `sensitivity()` vary them

    Args:
        probe (Sim): an initialized sim
        pars (list): the names to take (default: all with bounds)
        unused (set): the ids of parameters to leave out (those of likelihoods not used)

    Returns the parameters by name, and the names of those that set the mixing when the sim is built: of networks that a compartmental run turns into contact rates (those of the movement between places are worked out again on a restart: see `ss.Sim.restart()`).
    """
    fitted = sc.obj()
    for mod in probe.declared:
        for name,p in mod.graph.pars.items():
            if id(p) in unused: # A parameter of a likelihood that is not used in this fit
                continue
            if p.bounds is not None and (pars is None or name in pars or f'{mod.name}.{name}' in pars) and not any(p is q for q in fitted.values()): # A parameter shared by several modules is fitted once
                same = [m.graph.pars[name] for m in probe.declared if name in m.graph.pars]
                key = name if all(q is same[0] for q in same) else f'{mod.name}.{name}'
                fitted[key] = p
    mixed = getattr(probe, 'mixed', []) if probe.solver != 'abm' else [] # Networks that a compartmental run turns into contact rates
    contacts = [] # Their parameters that are fitted, which set the contact rates when the sim is built
    coded = [(mod, name, p) for mod in list(probe.modules) + mixed if isinstance(mod.pars, ss.Pars) for name,p in mod.pars.bounded().items()] # A parameter that a module's code reads (written as code, or the coded part of a declared one), given as ss.par with bounds, e.g. ssl.HIV(beta=ss.par(0.01, bounds=[0.001, 0.1])); see ss.Pars.bounded()
    for mod,name,p in coded:
        if any(p is q for q in fitted.values()): # One ss.par shared by several modules (e.g. a beta for every genotype) is fitted once
            continue
        if p.bounds is not None and (pars is None or name in pars or f'{mod.name}.{name}' in pars):
            taken = name in fitted or any(name in m.pars for m in probe.declared if m is not mod) or any(n == name and q is not p for m,n,q in coded) # Named with its module if the name alone is ambiguous, e.g. "hpv16.rel_beta"
            fitted[f'{mod.name}.{name}' if taken else name] = p
            if any(mod is net for net in mixed):
                contacts.append(f'{mod.name}.{name}')
    for name,p in probe.pars.bounded().items(): # A sim setting given as ss.par, e.g. ss.Sim(n_agents=ss.par(5000, bounds=[1000, 20000])): each value is a new sim (see ss.Sim.new())
        if p.bounds is not None and (pars is None or name in pars) and name not in fitted:
            fitted[name] = p
    return fitted, contacts


def expand(probe, fitted):
    """
    Each parameter that has several values as one entry for each, as a fit fits them and a gradient is taken with respect to them

    A value for each level of one stratification (`ss.par(ss.by(patch=[...]), bounds=)`) is "beta[a]", "beta[b]", ...,
    with the same bounds and prior; an array with bounds for each element (`ss.par(np.array([...]), bounds=[lows, highs])`)
    is "knots[0]", "knots[1]", ..., except the elements whose bounds are equal (fixed); and a time series
    (`ss.par(ss.timeseries(...), bounds=)`) is "R[0]", "R[7]", ..., by its times. A parameter without bounds has every element.

    Returns the parameters by name, with those as their entries; the position of each level, by name; and the parameter's name,
    the position, and the bounds of each element (of an array or a time series), by name
    """
    out, levels, elements = sc.obj(), {}, {}
    for key,p in fitted.items():
        raw = p.raw
        if isinstance(raw, ss.by):
            for i,label in enumerate(vector_labels(probe, raw, key)):
                out[f'{key}[{label}]'] = p
                levels[f'{key}[{label}]'] = i
        elif isinstance(raw, (np.ndarray, ss.timeseries)):
            n = len(raw.values) if isinstance(raw, ss.timeseries) else raw.size
            bounds = p.bounds if p.bounds is not None else [-np.inf, np.inf]
            lows, highs = [np.broadcast_to(np.asarray([bound_value(v, element(raw, 0)) for v in np.ravel(b)] if isinstance(b, (list, np.ndarray)) else bound_value(b, element(raw, 0)), dtype=float), n) for b in bounds]
            labels = [str(t) for t in raw.times] if isinstance(raw, ss.timeseries) else range(n)
            for i,label in zip(np.flatnonzero(lows < highs), np.asarray(labels, dtype=object)[lows < highs]):
                out[f'{key}[{label}]'] = p
                elements[f'{key}[{label}]'] = (key, i, lows[i], highs[i])
        else:
            out[key] = p
    return out, levels, elements


def element(raw, i):
    """ Element i of an array, or value i of a time series """
    return raw.values[i] if isinstance(raw, ss.timeseries) else raw.flat[i]


def with_element(raw, i, val, whole=None):
    """ An array or a time series with element i set to val: in `whole` if given (a copy made by an earlier call), else in a copy of raw """
    if whole is None:
        if isinstance(raw, ss.timeseries):
            whole = ss.timeseries(list(raw.times), list(raw.values), mode=raw.mode, period=raw.period)
        else:
            whole = np.array(raw, dtype=float)
    if isinstance(whole, ss.timeseries):
        whole.values[i] = val
    else:
        whole.flat[i] = val
    return whole


def to_pars(names, fitted, x, types, elements=None, external=None):
    """ The values of the model's parameters, for `sim.new(pars=...)`: a level by its name ("beta[a]"), an array or a time series whole (with the elements not fitted as they were); the values of a likelihood's own parameters are set on them """
    out = {}
    for k,(name,p,val) in enumerate(zip(names, fitted.values(), x)):
        cls = types[k]
        val = cls(val) if cls is not None else float(val)
        if external is not None and external[k]:
            p.value = val
        elif elements and name in elements:
            key, i = elements[name][:2]
            out[key] = with_element(p.raw, i, val, out.get(key))
        else:
            out[name] = val
    return out


def entry(name, p, levels=None, elements=None):
    """ The value of an entry of `expand()`, as the parameter holds it (e.g. typed) """
    if levels and name in levels:
        return sc.tolist(p.raw.values)[levels[name]]
    elif elements and name in elements:
        return element(p.raw, elements[name][1])
    return p.raw


def start_values(raws):
    """ The values of entries of `expand()` as numbers, and the type of each typed one (to keep its unit), or None """
    x0 = np.array([raw.value if isinstance(raw, ss.TimePar) else raw for raw in raws], dtype=float)
    return x0, [type(raw) if isinstance(raw, ss.TimePar) else None for raw in raws]


def sensitivity(sim, pars=None, results=None, n=100, rand_seed=None, verbose=None):
    """
    Which parameters matter: Latin hypercube samples over the bounds of the parameters, with the partial rank correlation of each with a summary of each result

    Needs no data. A result is summarized by its peak and its final value (a flow by its peak and its
    total). A partial rank correlation near 1 or -1 means the result goes up or down with the parameter,
    with the others held fixed; near 0, that it does not matter, so it cannot be fitted either.

    Args:
        sim (`ss.Sim`): the sim, which is not changed
        pars (list): the names of the parameters (default: all that have bounds)
        results (list): the results, e.g. `['I']` (default: every count of a state, and every flow)
        n (int): the number of samples
        rand_seed (int): the seed of the samples
        verbose (bool): whether to print the table

    Returns a dataframe of the partial rank correlations, with a row for each parameter and a column for each summary of a result.
    """
    probe = sim.new(verbose=0).init()
    bounded = bounded_pars(probe, pars)[0]
    if not bounded:
        raise ValueError('sensitivity() needs parameters with bounds, e.g. beta = ss.par(0.3, bounds=[0.1, 0.5]).')
    names = list(bounded.keys())
    lows, highs = [np.array([float(p.bounds[k]) for p in bounded.values()]) for k in [0, 1]]
    logs = (lows > 0) & (highs/np.maximum(lows, 1e-300) >= 10) # Wide positive bounds are sampled on a log scale, as fit() searches them
    unit = sps.qmc.LatinHypercube(d=len(names), seed=rand_seed if rand_seed is not None else 1).random(n)
    lo, hi = np.where(logs, np.log(lows), lows), np.where(logs, np.log(highs), highs)
    X = lo + unit*(hi - lo)
    X = np.where(logs, np.exp(X), X)
    msim = ss.parallel(sim.new(verbose=0), pars={name:X[:, j].tolist() for j,name in enumerate(names)}, verbose=False) # The samples are not scenarios to list
    summaries = sc.obj()
    counts = set() # The counts of a state, by module and name
    for mod in probe.modules:
        graph = getattr(mod, 'graph', None)
        states = [s.name for s in mod.auto_state_list] + (list(graph.states) if graph is not None else [])
        counts |= {(mod.name, state) for state in states}
    for modname,key,res in msim.results.walk():
        if modname is None or res.columns is not None:
            continue
        if results is not None and key not in results:
            continue
        if results is None and not ((modname, key) in counts or res.kind == 'flow'):
            continue
        vals = np.asarray(res.values)
        summaries[f'{modname}.{key} peak'] = vals.max(axis=1)
        summaries[f'{modname}.{key} {"total" if res.kind == "flow" else "final"}'] = vals.sum(axis=1) if res.kind == 'flow' else vals[:, -1]
    ranks = lambda v: sps.rankdata(v)
    RX = np.column_stack([ranks(X[:, j]) for j in range(len(names))])
    table = sc.dataframe(index=names)
    for label,y in summaries.items():
        ry = ranks(y)
        col = []
        for j in range(len(names)): # The correlation of what is left of the parameter and the result once the other parameters are regressed out
            others = np.column_stack([np.ones(n), np.delete(RX, j, axis=1)])
            res_x = RX[:, j] - others @ np.linalg.lstsq(others, RX[:, j], rcond=None)[0]
            res_y = ry - others @ np.linalg.lstsq(others, ry, rcond=None)[0]
            denom = np.sqrt((res_x**2).sum()*(res_y**2).sum())
            col.append(float(res_x @ res_y/denom) if denom > 0 else 0.0)
        table[label] = col
    if sc.ifelse(verbose, ss.options.verbose):
        print(f'Partial rank correlations, from {n} samples over the bounds:')
        print(table.round(2).to_string())
    return table


def profile(sim, par, values=None, level=0.95, **kwargs):
    """
    The profile likelihood of one parameter: at each of its values, the others refitted by maximum likelihood (`sim.fit(pars=...)`), and the confidence interval it gives

    The interval is the values whose negative log-likelihood is within half the chi-squared quantile with one
    degree of freedom (1.92 for 95%) of the lowest. Unlike the spread of a point estimate, it also shows whether
    the data determine the parameter at all: a profile that stays flat to a bound does not end within it.

    For a deterministic sim (an ODE) whose data series all have a likelihood, and parameters with no priors
    (so each refit is the maximum likelihood); a stochastic sim's estimates of the likelihood are noisy, and
    need a smoothed profile, which is not yet supported.

    Args:
        sim (`ss.Sim`): the sim, with data, which is not changed
        par (str): the name of the parameter to profile, one with a single value
        values (list): its values (default: the best fit of every parameter, 10 points spread over the bounds of this one, on the scale it is searched on, then 10 more at a time over the interval and the points next to it, until 5 are within it, at most 4 times)
        level (float): the confidence level of the interval
        kwargs (dict): passed to `sim.fit()` for each fit, e.g. `likelihood=` or `pars=` (the parameters to refit; default: all that have bounds)

    Returns an objdict with `table` (a dataframe with a row for each value: the value, the refitted values of the
    others, the negative log-likelihood `nll`, and whether the value is in the interval), `best` (the value with
    the lowest), `interval` (its low and high ends, where the profile crosses the threshold, interpolated between
    the values; an end that the values do not reach is the outermost of them, with a warning), and `threshold`.

    Examples:
        ```python
        prof = ss.fit.profile(sim, 'beta', likelihood=dict(cases=ss.poisson()))
        prof.interval # The 95% confidence interval of beta
        prof.table.plot('beta', 'nll')
        ```
    """
    kwargs.setdefault('verbose', 0)
    problem = Problem(sim, **{k:kwargs[k] for k in ['pars', 'data', 'weights', 'likelihood', 'gof', 'penalty', 'n_runs', 'reseed', 'rand_seed', 'nopython'] if k in kwargs}, verbose=0)
    name_of = lambda key: key.rpartition('[')[0] if key in problem.levels else key
    if par not in problem.names[:problem.n_model] and values is None:
        errormsg = f'profile() takes a parameter of the model that is fitted (one with bounds), or its values=; "{par}" is not one of {sc.strjoin(dict.fromkeys(map(name_of, problem.names)))}'
        raise ValueError(errormsg)
    if par in problem.levels or any(name_of(key) == par for key in problem.levels):
        errormsg = f'profile() takes a parameter with a single value, but "{par}" has a value for each level'
        raise ValueError(errormsg)
    if problem.stochastic:
        errormsg = f'profile() needs a deterministic sim (solver="ode"), not solver="{problem.runner.solver}": the likelihoods of a stochastic sim are estimates, whose noise a profile would need to smooth over'
        raise ValueError(errormsg)
    others = [key for key in dict.fromkeys(map(name_of, problem.names)) if key != par]
    if not problem.use_lik or any(has for key,has in zip(problem.names, problem.priors) if name_of(key) in others):
        why = 'not every data series has a likelihood' if not problem.use_lik else 'a parameter refitted has a prior, so a fit would maximize the posterior'
        raise ValueError(f'profile() maximizes the likelihood, but {why}')
    calibrate = dict(kwargs, pars=others)
    cls = problem.types[problem.names.index(par)] if par in problem.names else None # Keep the unit of a typed value

    def point(value):
        """ The others refitted with the parameter fixed at this value """
        fixed = sim.new(pars={par:cls(value) if cls is not None else float(value)}, verbose=0)
        fixed.data = kwargs.get('data', sim.data)
        if others:
            with warnings.catch_warnings(): # Far from the best, the others are expected to end at their bounds
                warnings.simplefilter('ignore', ss.errors.AtBound)
                fit = fixed.fit(**calibrate)
            return dict({par:float(value)}, **dict(zip(fit.table.par, fit.table.best)), nll=fit.mismatch)
        return {par:float(value), 'nll':-loglik(fixed.run(), likelihood=kwargs.get('likelihood'), weights=kwargs.get('weights'))}

    threshold = sps.chi2.ppf(level, 1)/2
    if values is not None:
        rows = [point(v) for v in values]
    else:
        fit = sim.fit(**sc.mergedicts(kwargs, dict(pars=others + [par]) if 'pars' in kwargs else {}))
        best = dict(zip(fit.table.par, fit.table.best))
        rows = [dict({par:float(best[par])}, **{k:v for k,v in best.items() if k != par}, nll=fit.mismatch)]
        i = problem.names.index(par)
        lo, hi = problem.encode(problem.lows)[i], problem.encode(problem.highs)[i]
        decode = lambda z: float(np.exp(z) if problem.logs[i] else z)
        rows += [point(decode(z)) for z in np.linspace(lo, hi, 10)]
        for j in range(4): # Then 10 more over the interval and the points next to it, until there are 5 within it
            df = pd.DataFrame(rows).sort_values(par)
            within = np.flatnonzero(df.nll.values - df.nll.min() <= threshold)
            if len(within) >= 5:
                break
            zs = np.log(df[par].values) if problem.logs[i] else df[par].values
            zlo, zhi = zs[max(within[0] - 1, 0)], zs[min(within[-1] + 1, len(df) - 1)]
            rows += [point(decode(z)) for z in np.linspace(zlo, zhi, 12)[1:-1]] # Between the points already run
    table = pd.DataFrame(rows).sort_values(par).drop_duplicates(par).reset_index(drop=True)
    x, nll = table[par].values, table.nll.values
    table['within'] = nll - nll.min() <= threshold
    k = int(np.argmin(nll))
    ends = []
    for side in [-1, 1]: # Outward from the best, to where the profile first rises above the threshold
        j = k
        while 0 <= j + side < len(x) and nll[j + side] - nll[k] <= threshold:
            j += side
        if 0 <= j + side < len(x):
            a, b = nll[j] - nll[k], nll[j + side] - nll[k]
            ends.append(float(x[j] + (threshold - a)/(b - a)*(x[j + side] - x[j])))
        else:
            ss.warn(f'the profile of "{par}" stays within {threshold:.3g} of its lowest up to {x[j]:n}, the {"lowest" if side < 0 else "highest"} value, so the interval may extend beyond it; consider wider bounds or more values', ss.errors.AtBound)
            ends.append(float(x[j]))
    return sc.obj(table=table, best=float(x[k]), interval=tuple(ends), threshold=float(threshold))


def prior_draws(sim, n, seed=None):
    """
    Values of the model's parameters drawn from their priors (`ss.par(prior=...)`, or uniform over the bounds of one with bounds and no prior), for `ss.parallel(sim, priors=True)`

    Returns a dict with a list of `n` values for each parameter that has a prior or bounds;
    a parameter with a value for each level (`ss.par(ss.by(age=[...]), prior=...)`) has a draw
    for each level, as "beta[a]", "beta[b]", ..., as `fit()` fits it.
    """
    rng = np.random.default_rng(seed)
    probe = sim.new(verbose=0).init()
    out = sc.obj()
    for mod in probe.declared:
        for name,p in mod.graph.pars.items():
            if name in out or (p.prior is None and p.bounds is None):
                continue
            keys = [f'{name}[{label}]' for label in vector_labels(probe, p.raw, name)] if isinstance(p.raw, ss.by) else [name]
            for key in keys:
                if p.prior is None:
                    vals = rng.uniform(*p.bounds, size=n)
                else:
                    prior = sc.dcp(p.prior)
                    prior.strict = False
                    vals = prior.init(trace=f'prior_{key}', seed=int(rng.integers(2**31)), force=True).rvs(n)
                    if p.bounds is not None:
                        vals = np.clip(vals, *p.bounds)
                out[key] = [float(v) for v in vals]
    if not out:
        errormsg = 'priors=True needs parameters with a prior or bounds, e.g. ss.par(0.3, prior=ss.normal(0.3, 0.05)) or ss.par(0.3, bounds=[0.1, 0.5])'
        raise ValueError(errormsg)
    return out


def fit_stages(sim, stages, verbose=None, **kwargs):
    """
    Staged fitting: run each method in turn, each starting from the best fit of the one before; see `sim.fit()` with `method=` a list

    Each stage is a method's name, or a dict with `method=` and that method's settings. A stage
    that does not converge stops the sequence (with a warning), and the fit returned is that
    of the last stage that did, or the first stage's best attempt if it was the one that failed.

    Args:
        sim (Sim): the sim to fit
        stages (list): the stages, each a method's name, or a dict with `method=` and its settings
        verbose (bool): whether to say what is being done (default: `ss.options.verbose`)
        kwargs (dict): passed to `sim.fit()` for every stage, except that a setting of a fitting method goes only to the stages whose method takes it

    Examples:
        ```python
        import starsim as ss

        beta = ss.par(0.3, bounds=[0.1, 1])
        S, I, R = ss.states(3)
        infection = S >> I @ beta*I
        sir = ss.Disease(infection, I >> R @ ss.days(8), init=dict(I=0.01), name='sir')
        pars = dict(total_pop=10_000, unit='day', stop=60)
        data = ss.Sim(sir, pars=dict(beta=0.5), **pars).run().to_data('I')
        fit = ss.Sim(sir, data=data, **pars).fit(method=['optuna', dict(method='nelder-mead', maxiter=100)], n_trials=30) # A search, then a polish
        print(fit.pars)
        ```
    """
    verbose = sc.ifelse(verbose, ss.options.verbose)
    stages = [dict(method=stage) if isinstance(stage, str) else dict(stage) for stage in stages]
    settings = [key for key in kwargs if key not in inspect.signature(ss.Sim.fit).parameters] # Those of the fitting methods
    takes = {stage['method'].lower():method_options(stage['method'].lower()) for stage in stages if stage['method'].lower() in methods}
    known = {key for options in takes.values() for key in options} # A SciPy method takes what no other stage does, which SciPy checks (see fit_scipy())
    unused = [key for key in settings if key not in known and not any(method in scipy_methods for method in takes)]
    if unused:
        raise TypeError(f'fit(method={list(takes)}) takes {sc.strjoin([f"{key}=" for key in sorted(known)])}, not {sc.strjoin(unused)}')
    fit, good = None, None
    current = sim
    for i,stage in enumerate(stages):
        method = stage.pop('method')
        if verbose:
            print(f'Stage {i+1} of {len(stages)}: {method}' + (f', started from stage {i}' if i else ''))
        shared = {key:val for key,val in kwargs.items() if key not in settings or key in takes.get(method.lower(), []) or (key not in known and method.lower() in scipy_methods)}
        fit = current.fit(method=method, verbose=verbose, **sc.mergedicts(shared, stage))
        if not fit.converged:
            if i < len(stages) - 1:
                ss.warn(f'stage {i+1} ({method}) did not converge ({fit.why}), so the stages after it were not run; returning {"stage " + str(i) if good is not None else "its best attempt"}')
            return good if good is not None else fit
        good = fit
        current = fit.sim.new() # With the best-fit values
    return fit


def check_gof(gof):
    """ Raise an error for a goodness of fit that is not known (rather than using the default in its place) """
    if gof is not None and not callable(gof) and gof not in gof_names:
        errormsg = f'Unknown goodness of fit gof="{gof}"; the choices are {sc.strjoin(gof_names)}, or a function of the sim that returns a number'
        raise ValueError(errormsg)
    return


def default_method(sim, use_lik, no_grad=None, n_pars=None):
    """
    The fitting method that suits a sim, and why: always a point estimate (a posterior is asked for by name), so that what is installed does not change what a fit gives; see `sim.fit()`

    Args:
        sim (`ss.Sim`): the sim, initialized
        use_lik (bool): whether every data series has a likelihood (the fit is then the maximum likelihood, or the most probable values if there are priors)
        no_grad (str): why there are no gradients, if the fit knows a reason the sim does not (e.g. a parameter that sets contact rates)
        n_pars (int): the number of fitted parameters; a point estimate of one or two uses Nelder-Mead even with gradients, since it is faster there
    """
    no_grad = no_grad or ss.solvers.why_not(sim, 'jax')
    if sim.solver != 'ode':
        method, why = 'optuna', 'a stochastic sim'
        posterior = 'pmcmc' if use_lik and ParticleFilter.reason(sim) is None else 'abc-smc'
    elif no_grad:
        method, why = 'nelder-mead', f'no gradients, since {no_grad}'
        posterior = 'abc-smc'
    else:
        few = n_pars is not None and n_pars <= 2
        method = 'nelder-mead' if few else 'l-bfgs-b'
        why = f'{n_pars} fitted parameter{"s" if n_pars != 1 else ""}, for which it is faster than gradients' if few else 'gradients by automatic differentiation'
        posterior = 'nuts'
    if use_lik: # Every series has a likelihood, so a posterior can be asked for
        why += f'; for a posterior, method="{posterior}"'
    return method, why


class Problem(sc.prettyobj):
    """
    What a fit works on, built once: the fitted parameters (with their bounds, priors, scales, and levels), the data and how it is scored, and how the sim is run; see `sim.fit()`

    Each fitting method (see `methods`) is a function of a problem. Values are on one of three scales:
    the values themselves (`x`); the scale an optimizer searches on (`encode()`, the log of a value with wide
    positive bounds); and the scale a sampler or random walk moves on, on which every value is allowed
    (`to_z()`, the logit of each value's place between its bounds, of its log for wide bounds).

    Args: as for `sim.fit()`
    """
    def __init__(self, sim, pars=None, data=None, weights=None, likelihood=None, gof=None, penalty=None, n_runs=None, reseed=None, rand_seed=None, nopython=True, verbose=None):
        base = sim.new(verbose=0)
        self.nopython = nopython
        self.rand_seed = sc.ifelse(rand_seed, base.pars.rand_seed) # A fit is reproducible, as a sim is
        if data is not None:
            base.data = data
        probe = sim.new(verbose=0) # To find the parameters and check the data
        if data is not None:
            probe.data = data
        probe.init()
        if probe.data is None and not callable(gof): # A loss that is a function of the sim brings its own data
            raise ValueError('fit() needs data: attach it to the sim with ss.Sim(..., data=...), with one column named like each result to fit; or give gof= as a function of the sim that returns the loss.')
        self.verbose = verbose = sc.ifelse(verbose, ss.options.verbose)
        check_gof(gof)
        if probe.data is not None and not callable(gof): # Every column is compared with its result, so one that matches none is an error; a loss that is a function of the sim reads what it needs
            ss.results.check_data(probe, probe.data, die=True)
        self.base, self.likelihood, self.weights, self.gof = base, likelihood, weights, gof

        # The parameters
        liks = likelihoods(probe, likelihood=likelihood)
        self.use_lik = use_lik = all(lik is not None for lik in liks.values()) and gof is None
        used = [id(val) for lik in liks.values() if use_lik and lik is not None for val in lik_pars(lik).values()]
        unused = [id(val) for mod in probe.declared for oc in mod.graph.outcomes.values() if oc.likelihood is not None for val in lik_pars(oc.likelihood).values() if id(val) not in used]
        fitted, contacts = bounded_pars(probe, pars, unused)
        settings = [key for key in fitted if key in probe.pars.bounded()] # Sim settings, for which the sim is built again
        fitted, self.levels, self.elements = expand(probe, fitted) # A value for each level, an array, or a time series: each value fitted on its own
        levels, elements = self.levels, self.elements
        self.n_model = len(fitted)
        if use_lik:
            for col,lik in liks.items(): # A parameter of a likelihood given at calibration time, with bounds, is fitted too
                for key,val in lik_pars(lik).items():
                    if isinstance(val, ss.par) and val.module is None and val.bounds is not None and (pars is None or val.name in sc.tolist(pars)):
                        fitted[val.name or f'{col}_{key}'] = val
            if verbose:
                print(f'Using the likelihood: the sum of the log-likelihoods of {sc.strjoin(liks.keys())}, treated as independent' + (' (weighted)' if weights else ''))
        elif any(lik is not None for lik in liks.values()) and verbose:
            why = f'gof="{gof}" was given' if gof is not None else 'not every data series has a likelihood'
            print(f'Note: {why}, so the likelihoods of {sc.strjoin([k for k,v in liks.items() if v is not None])} are not used')
        if not fitted:
            raise ValueError('fit() found no parameter with bounds; give the parameters to fit bounds, e.g. beta = ss.par(0.1, bounds=[0.01, 0.5]).')
        self.fitted = fitted
        self.names = names = list(fitted.keys())
        self.products = product_pairs(probe, names[:self.n_model]) # Parameters that the data cannot tell apart, from the model
        self.penalty_fn = penalty
        raws = [entry(name, p, levels, elements) for name,p in fitted.items()]
        self.lows, self.highs = [np.array([elements[name][2+k] if name in elements else bound_value(p.bounds[k], raw) for name,p,raw in zip(names, fitted.values(), raws)], dtype=float) for k in [0, 1]]
        self.x0, self.types = start_values(raws) # Keep the unit of typed values
        self.logs = logs = (self.lows > 0) & (self.highs/np.maximum(self.lows, 1e-300) >= 10) # Wide positive bounds are searched on a log scale
        if verbose and logs.any():
            print(f'Searching on a log scale: {sc.strjoin(np.array(names)[logs])}')
        self.external = [i >= self.n_model for i in range(len(fitted))] # Parameters of a likelihood given at calibration time
        self.priors = [p.prior is not None for p in fitted.values()]
        self.use_map = False # Set with the method; see choose()
        self.base_lows = np.where(logs, np.log(np.where(logs, self.lows, 1.0)), self.lows) # The scale for sampling and random walks: see to_x()
        self.span = np.where(logs, np.log(np.where(logs, self.highs, 1.0)), self.highs) - self.base_lows

        # How it is run and scored
        self.stochastic = probe.solver not in ['ode']
        self.n_runs = sc.ifelse(n_runs, 4 if self.stochastic and probe.solver != 'abm' else 1) # A stochastic compartmental sim is scored over several runs, so the fit is not to one draw of chance; agents, whose runs are slow, by one run a point, as Optuna's search averages over nearby points
        self.seeds = [sc.ifelse(base.pars.rand_seed, 1) + i for i in range(self.n_runs)] # The same seeds at every point (common random numbers), so the score changes only with the values
        self.reseed = reseed
        self.fresh = False # Whether each evaluation takes seeds of its own; see choose()
        self.runner = runner = base.new() # If the sim is nothing but compartments, it is built once and restarted for each evaluation
        runner.data = base.data
        self.can_restart = runner.init().kernel_only() and not contacts and not settings # The mixing is worked out when the sim is built, as are its settings, so a new value needs a new sim
        self.no_grad = ss.solvers.why_not(runner, 'jax') or (f'{contacts[0]} sets the mixing when the sim is built' if contacts else None) or (f'{settings[0]} is a setting of the sim' if settings else None) # Why there are no gradients, if there are not
        self.trials = [] # Every evaluation
        self._density = None # The log-likelihood in JAX, once built: see jax_density()
        return

    def choose(self, method=None):
        """ The fitting method: the one given, checked against what the sim allows, or the default (see `default_method()`) """
        verbose = self.verbose
        if method is None:
            method, why = default_method(self.runner, self.use_lik, no_grad=self.no_grad, n_pars=len(self.fitted))
            if verbose:
                print(f'Fitting by {method} ({why}; override with method=)')
        self.use_map = self.use_lik and any(self.priors) and method not in sampling_methods and method != 'if2' # A posterior method uses the priors itself, and IF2 finds the maximum likelihood
        self.fresh = self.stochastic and sc.ifelse(self.reseed, method == 'optuna') # Optuna's search learns from nearby points, so with fresh seeds at each it averages over chance, rather than fitting one seed's realization; a local method needs the same seeds, or its steps follow the noise
        if verbose and self.stochastic and method not in ['pmcmc', 'abc-smc', 'if2']: # Which take a run of their own for each point
            runs = f'the mean of {self.n_runs} runs' if self.n_runs > 1 else 'one run'
            seeds = 'fresh seeds at each point' if self.fresh else ('the same seeds at every point' if self.n_runs > 1 else 'the same seed at every point')
            print(f'Scoring each point by {runs}, with {seeds} (override with n_runs= and reseed=)')
        if verbose and self.use_map:
            print(f'Minimizing the negative log-posterior, since {sc.strjoin(np.array(self.names)[self.priors])} {"has a prior" if sum(self.priors) == 1 else "have priors"}')
        filtered = ParticleFilter.reason(self.runner)
        if self.penalty_fn is not None and (method in sampling_methods or method == 'if2'):
            raise ValueError(f'penalty= is added to what a best fit minimizes, but method="{method}" does not minimize; for a posterior, give the parameters priors (ss.par(..., prior=...))')
        elif method == 'nuts' and (not self.use_lik or self.no_grad):
            why = 'not every data series has a likelihood' if not self.use_lik else f'it has no gradients, since {self.no_grad}'
            raise ValueError(f'method="nuts" gives a posterior, which needs a likelihood for every data series and the gradients of a sim run in JAX, but {why}.')
        elif method == 'pmcmc' and (not self.use_lik or filtered):
            why = 'not every data series has a likelihood' if not self.use_lik else filtered
            raise ValueError(f'method="pmcmc" gives a posterior, which needs a likelihood for every data series and a stochastic compartmental sim that the kernels run whole, but {why}.')
        elif method == 'if2' and (not self.use_lik or filtered or self.levels or self.elements or self.n_model < len(self.fitted)):
            why = 'not every data series has a likelihood' if not self.use_lik else filtered or ('a parameter has several values' if self.levels or self.elements else 'a parameter of a likelihood is fitted')
            raise ValueError(f'method="if2" needs a likelihood for every data series and a stochastic compartmental sim that the kernels run whole, and fits the parameters of the model\'s rates, but {why}; use method="pmcmc"')
        return method

    def to_pars(self, x):
        """ The values of the model's parameters, for `sim.new(pars=...)`: see `to_pars()` """
        return to_pars(self.names, self.fitted, x, self.types, self.elements, self.external)

    def restart(self, x):
        """ The runner, restarted with these values """
        return self.runner.restart(pars=self.to_pars(x))

    def run(self, x, seed=None, fresh=False):
        """ A run of the sim with these values: the runner restarted if it can be, else a new sim """
        if self.can_restart and not fresh:
            return self.runner.restart(pars=self.to_pars(x), rand_seed=seed).run()
        kw = {} if seed is None else dict(rand_seed=seed)
        new = self.base.new(pars=self.to_pars(x), **kw)
        new.data = self.base.data
        return new.run()

    def measure(self, sim):
        """ The score of a run: minus its log-likelihood, or its goodness of fit """
        return -loglik(sim, likelihood=self.likelihood, weights=self.weights) if self.use_lik else mismatch(sim, weights=self.weights, gof=self.gof)

    def logprior(self, x):
        """ The log of the priors' density at these values """
        return sum(logprior(p, val) for p,val in zip(self.fitted.values(), x))

    def penalty(self, x):
        """ Minus the log of the priors, so that with likelihoods an optimizer finds the most probable values (the maximum a posteriori) rather than ignoring the priors; plus the penalty given on the values (`penalty=`) """
        out = -sum(logprior(p, val) for p,val,has in zip(self.fitted.values(), x, self.priors) if has) if self.use_map else 0.0
        return out + (float(self.penalty_fn(self.to_pars(x))) if self.penalty_fn is not None else 0.0)

    def objective(self, x, number=None):
        """ What an optimizer minimizes: the mean score over the runs (plus the penalty of the priors); each evaluation is recorded. With fresh seeds, evaluation `number` (default: the next) takes seeds after those of the evaluations before it """
        seeds = self.seeds
        if self.fresh: # Seeds of this evaluation's own
            seeds = [seed + self.n_runs*sc.ifelse(number, len(self.trials)) for seed in seeds]
        values = [self.measure(self.run(x, seed=seed)) for seed in seeds]
        value = float(np.mean(values)) + self.penalty(x)
        self.trials.append(dict(zip(self.names, x), objective=value))
        return value

    def jax_density(self):
        """
        Minus the log-likelihood and its gradient as one compiled function in JAX (see `ss.gradients.Density`), or None if the
        model is not covered (or not every series has a likelihood); built the first time it is asked for. With many values it is
        differentiated in reverse mode, in one pass, rather than in one direction per value
        """
        if self._density is None:
            self._density = False
            if self.use_lik:
                try:
                    density = ss.gradients.Density(self.runner, self.fitted, self.x0, self.restart, likelihood=self.likelihood, weights=self.weights, external=self.external, bounds=(self.lows, self.highs))
                except ss.gradients.Unsupported as E:
                    self.python_in_loop(f'the log-likelihood in JAX does not cover {E} yet', 'the gradients')
                    return None
                density.priors = False
                jax = ss.gradients.jax()
                d = len(self.x0)
                reverse = d*sum(X['y'].size for X in density.X0) >= reverse_min # Forward mode takes about as long as the values times the compartments; reverse mode, about the same whatever the number of values
                adaptive = [run for run in density.runs if not run.n_sub] # The runs whose substeps are worked out as the run goes
                compiled = []
                if reverse: # Each step takes a fixed number of substeps, as reverse mode needs: as many as the run at the start needs (an optimizer, unlike a sampler, needs no one function over the bounds), and more if a step needs them (see call())
                    with jax.enable_x64(True):
                        need = float(jax.jit(lambda x: density.evaluate(x)[1])(jax.numpy.asarray(self.x0, dtype=float)))
                    density.reverse()
                    for run in adaptive:
                        run.n_max = int(2**np.ceil(np.log2(max(need, 1))))
                    def fn(x):
                        (value, need), grad = jax.value_and_grad(lambda x: density.evaluate(x), has_aux=True)(x)
                        return -value, -grad, need
                else:
                    def fn(x): # Every partial derivative, in forward mode
                        (values, needs), (slopes, _) = jax.vmap(lambda e: jax.jvp(density.evaluate, (x,), (e,)))(jax.numpy.eye(d))
                        return -values[0], -slopes, 0
                def call(x):
                    with jax.enable_x64(True):
                        while True:
                            if not compiled:
                                ss.solvers.backends.compiling('the gradient of the log-likelihood', 'JAX', 0.5, 5)
                                compiled.append(jax.jit(fn))
                            value, grad, need = compiled[0](jax.numpy.asarray(x, dtype=float))
                            if not reverse or not adaptive or need <= min(run.n_max for run in adaptive):
                                return float(value), np.array(grad, dtype=float) # A copy, since JAX's is read-only
                            for run in adaptive: # Taken again with more substeps, rounded up so that it is compiled again only a few times
                                run.n_max = max(run.n_max, int(2**np.ceil(np.log2(float(need)))))
                            compiled.clear()
                self._density = call
        return self._density or None

    def python_in_loop(self, why, what):
        """ Take something (e.g. the gradients) with Python in the loop, which is much slower, only if `nopython=False`; otherwise say why it can't be taken in JAX """
        if self.nopython:
            errormsg = f'The fit would take {what} with Python in the loop, which is much slower, since {why}; to fit this way anyway, use sim.fit(..., nopython=False)'
            raise ss.CapabilityError(errormsg)
        if self.verbose:
            print(f'Note: taking {what} with Python in the loop, which is slower, since {why}')
        return

    def with_gradient(self, x):
        """ The objective and its gradient, through the run in JAX: the log-likelihood as one function in JAX if the model allows (see `jax_density()`), else the run differentiated with Python either side (see `ss.gradients.gradient()`) """
        density = self.jax_density()
        if density is not None:
            value, derivs = density(x)
        else:
            value, derivs = ss.gradients.gradient(self.runner, self.restart, x, self.measure)
            derivs = np.array(derivs, dtype=float)
        if self.use_map or self.penalty_fn is not None: # The slope of the priors and the penalty, numerically
            hs = 1e-6*np.maximum(np.abs(x), 1e-6)
            derivs += np.array([(self.penalty(x + h*e) - self.penalty(x - h*e))/(2*h) for h,e in zip(hs, np.eye(len(x)))])
        value = float(value) + self.penalty(x)
        self.trials.append(dict(zip(self.names, x), objective=value))
        return value, derivs

    def start(self):
        """ The objective at the initial values, which has to be finite """
        self.mismatch_before = value = self.objective(self.x0)
        if not np.isfinite(value): # Nothing can be compared with it, so the fit would wander
            series = loglik(self.run(self.x0), likelihood=self.likelihood, weights=self.weights, by_series=True) if self.use_lik else {}
            bad = [col for col,val in series.items() if not np.isfinite(val)]
            errormsg = f'The fit cannot start: its {"negative log-likelihood" if self.use_lik else "mismatch"} at the starting values is {value}'
            errormsg += f', from the data series {sc.strjoin(bad)}' if bad else ''
            raise ValueError(errormsg + '; check the data (e.g. for negative or missing values) and the likelihoods')
        return value

    def encode(self, x):
        """ Values on the scale an optimizer searches on """
        return np.where(self.logs, np.log(np.maximum(x, 1e-300)), x)

    def decode(self, z):
        """ Values from the scale an optimizer searches on """
        return np.clip(np.where(self.logs, np.exp(np.minimum(z, 700)), z), self.lows, self.highs)

    def to_x(self, z):
        """ Values from the scale a sampler moves on, and each one's place between its bounds """
        share = np.clip(1/(1 + np.exp(-z)), 1e-12, 1 - 1e-12)
        pos = self.base_lows + self.span*share
        return np.where(self.logs, np.exp(pos), pos), share

    def share_of(self, x):
        """ The place of each value between its bounds, on the scale it is searched on """
        return np.clip((np.where(self.logs, np.log(np.maximum(x, 1e-300)), x) - self.base_lows)/self.span, 1e-9, 1 - 1e-9)

    def to_z(self, x):
        """ Values on the scale a sampler moves on """
        share = self.share_of(x)
        return np.log(share/(1 - share))

    def logjac(self, x, share):
        """ The log of dx/dz, which turns a density of the values into one on the scale a sampler moves on """
        return np.log(self.span*share*(1 - share)*np.where(self.logs, x, 1.0)).sum(axis=-1)

    def spread(self, n):
        """ Points spread over the bounds (by a Latin hypercube, on the scale searched on) """
        sampler = sps.qmc.LatinHypercube(d=len(self.names), seed=self.rand_seed if self.rand_seed is not None else 1)
        lo, hi = self.encode(self.lows), self.encode(self.highs)
        return self.decode(lo + sampler.random(n)*(hi - lo))

    def scan_size(self, n=None):
        """ The number of points of a scan: `n`, or by default 10 per parameter, but at most `max_scan`, since a random scan of many parameters costs more runs than it finds (with a note when capped) """
        if n is None:
            n = min(10*len(self.names), max_scan)
            if n < 10*len(self.names):
                ss.note(f'the scan before the search tries {n} points, not 10 for each of the {len(self.names)} parameters; set scan= to change it', verbose=self.verbose)
        return n

    def scan(self, n=None, top=None):
        """ The best of the initial values and `n` points spread over the bounds (default: see `scan_size()`), since a fit can have several local optima; with `top`, the best few of them that are apart (at least a tenth of the bounds in some parameter), best first """
        n = self.scan_size(n)
        points = np.vstack([self.x0, self.spread(n)]) if n else self.x0[None]
        values = [self.mismatch_before] + [self.objective(x) for x in points[1:]]
        order = np.argsort(values, kind='stable') # The initial values first among equals
        if top is None:
            return points[order[0]]
        chosen = []
        for i in order:
            if all(np.abs(self.share_of(points[i]) - self.share_of(points[j])).max() >= 0.1 for j in chosen):
                chosen.append(i)
            if len(chosen) == top:
                break
        return points[chosen]

    def par_table(self, best):
        """ The table of the fitted parameters: each one's starting value, best value, and bounds, with its label if any has one """
        table = pd.DataFrame(dict(par=self.names, initial=self.x0, best=best, low=self.lows, high=self.highs))
        labels = [getattr(p, 'label', None) or '' for p in self.fitted.values()]
        if any(labels):
            table.insert(1, 'label', labels)
        return table

    def posterior(self, method, out):
        """ The draws of a sampler, as the best values, a table, and the samples, with the R-hat and effective sample size; and the band of the results over them """
        names = self.names
        zs = out.zs
        xs = np.array([[self.to_x(z)[0] for z in chain] for chain in zs])
        logjac = self.logjac(xs, self.share_of(xs))
        out.best = xs.reshape(-1, len(names))[np.argmax((out.logdens - logjac).reshape(-1))] # The draw with the highest posterior density (of the values themselves, not on the sampler's scale)
        n_chains, n_draws = zs.shape[:2]
        rhat, n_eff = diagnose(xs)
        out.samples = samples = pd.DataFrame(xs.reshape(-1, len(names)), columns=names)
        samples.insert(0, 'chain', np.repeat(np.arange(n_chains), n_draws))
        samples.insert(1, 'draw', np.tile(np.arange(n_draws), n_chains))
        out.divergences = divergences = int(out.divergent.sum()) if 'divergent' in out else None
        out.table = table = self.par_table(out.best)
        table['median'] = np.median(samples[names].values, axis=0)
        table['lower'], table['upper'] = np.percentile(samples[names].values, [2.5, 97.5], axis=0)
        table['rhat'], table['ess'] = rhat, n_eff
        if 'ess' in out: # One population, so no R-hat; the method says whether it converged
            out.table = table.drop(columns='rhat')
            out.table['ess'] = out.ess
        else:
            out.converged = bool(np.all(rhat <= 1.01)) and not divergences # NaN (chains too short to tell) is not converged
            out.why = f'R-hat at most {np.max(rhat):.3f} and effective sample size at least {np.min(n_eff):.0f}'
            out.why += f', with {divergences} divergent draws' if divergences is not None else f', with {sc.strjoin([f"{a:.0%}" for a in out.acceptance])} of proposals accepted'
        picks = samples[names].values[np.linspace(0, len(samples) - 1, min(100, len(samples))).astype(int)] # Spread evenly over the draws
        values = sc.ddict(list)
        for k,x in enumerate(picks):
            sim = self.run(x, seed=k + 1 if method == 'pmcmc' else None) # A stochastic run takes a seed of its own, so the band includes chance
            for col in sim.data.columns:
                summed = flow_at_data(sim, col) # A flow with data at a longer period: its totals over the data's periods, as compared
                values[col].append(summed[1] if summed is not None else np.array(match_column(sim, col).values, dtype=float))
        out.band = {col:np.percentile(vals, [2.5, 97.5], axis=0) for col,vals in values.items()}
        return out

    def unidentifiable(self, best, samples=None, by_slopes=False):
        """ Notes on the parameters that the data cannot tell apart: from the model, from the posterior, or from the slopes of the results at the best fit """
        names = self.names
        out = []
        for a,b,how in self.products:
            va, vb = best[names.index(a)], best[names.index(b)]
            out.append(f'"{a}" and "{b}" appear in the model only {"multiplied together" if how == "product" else "one divided by the other"}, so the data can determine only their {how} '
                       f'({va*vb if how == "product" else va/vb:.3g}), not each of them; fit one and fix the other')
        known = [{a, b} for a,b,how in self.products]
        if samples is not None: # From the fit: parameters almost perfectly correlated over the posterior (of their logs, for those searched on a log scale)
            draws = np.where(self.logs[:, None], np.log(np.maximum(samples[names].values.T, 1e-300)), samples[names].values.T)
            for a,b,r in correlated_pairs(names, draws, 0.95):
                if {a, b} not in known:
                    out.append(f'"{a}" and "{b}" are almost perfectly correlated in the posterior (r = {r:.3f}), so the data determine a combination of them rather than each')
        elif by_slopes and len(names) > 1: # Or parameters that change the fitted results in almost the same way at the best fit
            def read(sim):
                parts = [model/max(np.abs(expected).mean(), 1e-12) for expected,model in compare(sim).values()]
                return np.concatenate(parts)
            value, derivs = ss.gradients.gradient(self.runner, self.restart, best, read)
            slopes = np.array([d*x for d,x in zip(derivs, best)]) # Relative to each value, so the scale of a parameter does not matter
            norms = np.linalg.norm(slopes, axis=1)
            for i in range(len(names)):
                for k in range(i+1, len(names)):
                    cos = slopes[i] @ slopes[k]/(norms[i]*norms[k]) if norms[i]*norms[k] > 0 else 0.0
                    if abs(cos) > 0.999 and {names[i], names[k]} not in known:
                        out.append(f'"{names[i]}" and "{names[k]}" change the fitted results in almost the same way at the best fit (r = {cos:.4f}), so the data can hardly tell them apart')
        return out

    def result(self, method, out):
        """ The `ss.fit.Fit` from what a method found """
        names = self.names
        if 'zs' in out: # Draws from a posterior
            self.posterior(method, out)
        else:
            out.table = self.par_table(out.best)
            if out.get('spread') is not None: # Not a posterior, but some indication of the uncertainty
                out.table['spread_lower'], out.table['spread_upper'] = out.spread
        best = out.best
        notes = self.unidentifiable(best, samples=out.get('samples'), by_slopes=method in grad_methods and not self.no_grad)
        seeds = self.seeds
        before = self.run(self.x0, seed=seeds[0], fresh=True)
        after = self.run(best, seed=seeds[0], fresh=True) # A run at the best values, for plotting; also the last, so a likelihood's own parameters keep their best values
        scores = [self.measure(after)] + [self.measure(self.run(best, seed=seed)) for seed in seeds[1:]] # The score of the best values, as in the search
        for msg in notes:
            ss.note(msg, sim=after, verbose=self.verbose)
        zs, zlo, zhi = self.encode(best), self.encode(self.lows), self.encode(self.highs) # On the scale searched on, so a value near the low end of a log scale is not at its bound
        for name,z,lo,hi,val,low,high in zip(names, zs, zlo, zhi, best, self.lows, self.highs):
            if min(z - lo, hi - z) < 0.01*(hi - lo):
                ss.warn(f'the best-fit value of "{name}" ({val:n}) is at one of its bounds [{low:n}, {high:n}]; consider widening them', ss.errors.AtBound)
        mismatch = float(np.mean(scores)) + self.penalty(best)
        if out.converged and self.stochastic and 'zs' not in out and mismatch > self.mismatch_before: # A best fit of a stochastic sim, whose scores are noisy, that did not find better values
            out.converged = False
            out.why += f'; but rerun with the same seeds, the best values score {mismatch:.4g}, worse than the initial values ({self.mismatch_before:.4g}), so the best trial may have been chance; try more trials (n_trials=) or more runs a point (n_runs=)'
        if self.verbose:
            print(f'{"Converged" if out.converged else "Not converged"}: {out.why}')
            print(f'The {"negative log-posterior" if self.use_map else "negative log-likelihood" if self.use_lik else "goodness of fit"} is {mismatch:.4g} at the best values and {self.mismatch_before:.4g} at the initial values')
        best_pars = sc.obj()
        for name,val,cls,p in zip(names, best, self.types, self.fitted.values()):
            val = cls(val) if cls is not None else float(val)
            if name in self.levels: # One value per level, gathered as the parameter was given: ss.by(patch=[...])
                key = name.rpartition('[')[0]
                vals = best_pars[key].values if key in best_pars else []
                best_pars[key] = ss.by(**{p.raw.dim:list(vals) + [val]})
            elif name in self.elements: # An array or a time series, whole
                key, i = self.elements[name][:2]
                best_pars[key] = with_element(p.raw, i, val, best_pars.get(key))
            else:
                best_pars[name] = val
        diagnostics = sc.obj(divergences=out.get('divergences'), acceptance=out.get('acceptance'), band=out.get('band'))
        return Fit(sim=after, before=before, pars=best_pars, table=out.table, trials=pd.DataFrame(self.trials), method=method,
                   mismatch=mismatch, mismatch_before=self.mismatch_before,
                   measure='negative log-posterior' if self.use_map else 'negative log-likelihood' if self.use_lik else 'goodness of fit',
                   samples=out.get('samples'), diagnostics=diagnostics, converged=out.converged, why=out.why, scores=scores, study=out.get('study'))


#%% The fitting methods: each a function of a Problem, returning the best values (or draws from a posterior), whether it converged, and why

def fit_scipy(problem, method='nelder-mead', scan=None, restarts=None, **options):
    """
    A method of `scipy.optimize.minimize` (one of `scipy_methods`), from each of the best few points of a scan that are apart (`restarts`), with gradients
    from JAX for the methods that use them; it has converged if the best two searches end at the same value. By default 3 searches,
    or 1 for a single parameter of a deterministic sim, where the scan already finds the right valley and SciPy's own stopping rule is enough.

    Args:
        problem (Problem): the fit, as `sim.fit()` sets it up
        method (str): the method of `scipy.optimize.minimize`
        scan (int): the number of points in the scan (default: see `Problem.scan_size()`)
        restarts (int): the number of searches
        options (dict): passed to `scipy.optimize.minimize` as `options=`, e.g. `maxiter=500`; see `scipy.optimize.show_options()` (a setting SciPy does not know is an error)

    Examples:
        ```python
        import starsim as ss

        beta = ss.par(0.3, bounds=[0.1, 1])
        S, I, R = ss.states(3)
        infection = S >> I @ beta*I
        sir = ss.Disease(infection, I >> R @ ss.days(8), ss.outcome(0.5*infection, name='cases'), init=dict(I=0.01), name='sir')
        pars = dict(total_pop=10_000, unit='day', stop=60)
        data = ss.Sim(sir, pars=dict(beta=0.5), rand_seed=2, **pars).run().to_data('cases', resample='week')
        fit = ss.Sim(sir, data=data, **pars).fit(method='nelder-mead', scan=5, restarts=2, maxiter=200)
        print(fit.pars, fit.converged)
        ```
    """
    import scipy.optimize as spo
    p = problem
    restarts = sc.ifelse(restarts, 1 if len(p.names) == 1 and not p.stochastic else 3)
    bounds = list(zip(p.encode(p.lows), p.encode(p.highs)))
    options = sc.mergedicts(dict(xatol=1e-4, fatol=1e-6) if method == 'nelder-mead' else None, options)
    if method in grad_methods and not p.no_grad: # The gradient with respect to the scale searched on: d/dz = x d/dx where x = exp(z)
        def fun(z):
            x = p.decode(z)
            value, grad = p.with_gradient(x)
            return value, grad*np.where(p.logs, x, 1.0)
        search = lambda start: spo.minimize(fun, p.encode(start), method=method, jac=True, bounds=bounds, options=options)
    else:
        search = lambda start: spo.minimize(lambda z: p.objective(p.decode(z)), p.encode(start), method=method, bounds=bounds, options=options)
    starts = p.scan(scan, top=restarts)
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='Unknown solver options') # SciPy would only warn about a setting it does not know
        try:
            results = sorted([search(start) for start in starts], key=lambda res: res.fun)
        except spo.OptimizeWarning as E:
            raise TypeError(f'fit(method="{method}") takes scan=, restarts=, and the options of this SciPy method (see `scipy.optimize.show_options()`); SciPy says: {E}') from E
    result = results[0]
    best = p.decode(result.x)
    if not p.stochastic: # A search can end short of the best point it tried (e.g. if a line search fails); a deterministic sim's scores are exact, so take that point
        scores = [trial['objective'] for trial in p.trials]
        tried = int(np.nanargmin(scores))
        if scores[tried] < result.fun:
            best = np.array([p.trials[tried][name] for name in p.names])
    evals = sum(res.nfev for res in results)
    if len(results) == 1: # Nothing to compare it with, so SciPy's own stopping rule
        return sc.obj(best=best, converged=bool(result.success), why=f'{result.message} ({evals} evaluations)')
    tol = 0.1 if p.use_lik else max(1e-3*abs(result.fun), 1e-6) # A tenth of a unit of log-likelihood, or 0.1% of a goodness of fit (which can be 0)
    at = lambda res: f' at {sc.strjoin([f"{name}={val:.4g}" for name,val in zip(p.names, p.decode(res.x))])}' if len(p.names) <= 3 else '' # Where each search ended, if that is short
    ends = sc.strjoin([f'{res.fun:.4g}{at(res)}' for res in results], sep='; ')
    converged = bool(results[1].fun - result.fun <= tol)
    if converged:
        why = f'the best two of {len(results)} searches, from points of the scan that are apart, ended within {tol:.2g} of each other (objective values {ends}; {evals} evaluations)'
    else:
        why = f'the {len(results)} searches, from points of the scan that are apart, ended at different objective values ({ends}; {evals} evaluations), so the best may be a local optimum; try a larger scan (scan=) or more searches (restarts=)'
        if not p.use_lik and 0 <= result.fun < results[1].fun/10:
            why += '. The best is far below the others, which may have stalled where the fit is flat (e.g. where there is no epidemic), so it is likely the best there is'
    return sc.obj(best=best, converged=converged, why=why)


def smoothed_best(zs, values):
    """
    The index of the point whose neighbors score lowest, among the best tenth of the
    points (at least 5): each of their scores is replaced by the mean of all scores
    weighted by a Gaussian kernel of the distance between points, on the scale searched
    on, with a width for each parameter by Scott's rule from the spread of those best
    points (the spread of all of them, including the random first trials, would let a
    dense region of worse points pull the choice away from the best). For scores from
    fresh random seeds, whose lowest value picks out chance as much as the values.
    """
    n, d = zs.shape
    best = np.argsort(values, kind='stable')[:max(5, n//10)]
    width = np.maximum(zs[best].std(axis=0), 1e-6)*len(best)**(-1/(d + 4)) # Scott's rule, for each parameter
    dists = (((zs[best,None,:] - zs[None,:,:])/width)**2).sum(axis=-1)
    weights = np.exp(-0.5*dists)
    smoothed = (weights*values[None,:]).sum(axis=1)/weights.sum(axis=1)
    return int(best[np.argmin(smoothed)])


def optuna_study(problem, storage=None, worker=0):
    """ The Optuna study of a fit: new, or loaded from `storage` (see `fit_optuna()`), with a sampler seeded for this worker so that workers propose different points """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if isinstance(storage, (str, os.PathLike)) and '://' not in str(storage): # A file name: an Optuna journal, which processes can share without a database
        storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(storage)))
    sampler = optuna.samplers.TPESampler(seed=sc.ifelse(problem.rand_seed, 0) + worker)
    return optuna.create_study(sampler=sampler, storage=storage, study_name='calibrate', load_if_exists=True)


def optuna_trial(problem, trial):
    """ The objective at the values a trial proposes, with the trial's own seeds if each point takes fresh ones """
    p = problem
    x = [trial.suggest_float(name, lo, hi, log=bool(log)) for name,lo,hi,log in zip(p.names, p.lows, p.highs, p.logs)]
    return p.objective(np.array(x), number=trial.number)


def optuna_worker(worker, n, problem, storage, kwargs):
    """ One worker process of a fit by Optuna: `n` trials, shared with the others through the storage """
    study = optuna_study(problem, storage, worker=worker)
    study.optimize(functools.partial(optuna_trial, problem), n_trials=n, **kwargs)
    return


def fit_optuna(problem, n_trials=100, n_workers=None, storage=None, **kwargs): # The search starts from the initial values, not from a scan
    """
    Optuna's tree-structured Parzen estimator, from the initial values: a best fit, with the spread of the best trials, and Optuna's study (`fit.study`)

    Args:
        n_trials (int): the number of trials (each one point, run once for agents), including any already in `storage`
        n_workers (int): the number of processes that propose and run trials at once, sharing what they have learned (default 1)
        storage (str): where the trials are kept, so that a fit that stopped can be resumed by calling it again with the same storage:
            a file name (kept as an Optuna journal), a database URL (e.g. `'sqlite:///fit.db'`), or an Optuna storage
        kwargs (dict): Optuna's own keywords for `study.optimize()`, e.g. `timeout=600`
        problem (Problem): the fit, as `sim.fit()` sets it up

    It has not converged if the best trial is the initial values or one of the first 10 (which are random), or if the best improved
    by 1% or more over the last quarter of the trials.

    Examples:
        ```python
        import starsim as ss

        beta = ss.par(0.3, bounds=[0.1, 1])
        S, I, R = ss.states(3)
        infection = S >> I @ beta*I
        sir = ss.Disease(infection, I >> R @ ss.days(8), ss.outcome(0.5*infection, name='cases'), init=dict(I=0.01), name='sir')
        pars = dict(total_pop=10_000, unit='day', stop=60, solver='binomial')
        data = ss.Sim(sir, pars=dict(beta=0.5), rand_seed=2, **pars).run().to_data('cases', resample='week')
        fit = ss.Sim(sir, data=data, **pars).fit(method='optuna', n_trials=50, n_runs=4)
        print(fit.pars, fit.converged)
        ```
    """
    import optuna
    p = problem
    n_workers = n_workers or 1
    temp = None
    if n_workers > 1 and storage is None: # The workers share their trials through a file, deleted afterwards
        storage = temp = os.path.join(tempfile.gettempdir(), f'starsim_fit_{sc.uuid().hex}.log')
    study = optuna_study(p, storage)
    if not study.trials: # The initial values, already scored by Problem.start()
        dists = {name:optuna.distributions.FloatDistribution(lo, hi, log=bool(log)) for name,lo,hi,log in zip(p.names, p.lows, p.highs, p.logs)}
        study.add_trial(optuna.trial.create_trial(params=dict(zip(p.names, p.x0)), distributions=dists, value=p.mismatch_before))
    remaining = n_trials - len(study.trials)
    if remaining > 0 and n_workers > 1:
        shares = [len(part) for part in np.array_split(np.arange(remaining), n_workers)]
        method = ss.kernels.start_method()
        with ss.kernels.starting_workers(method): # The workers share the cores, so each takes one thread, as in ss.parallel()
            sc.parallelize(ss.kernels.worker(optuna_worker), iterarg=list(enumerate(shares)), kwargs=dict(problem=p, storage=storage, kwargs=kwargs), ncpus=n_workers, start_method=method)
        study = optuna_study(p, storage)
    elif remaining > 0:
        study.optimize(functools.partial(optuna_trial, p), n_trials=remaining, **kwargs)
    done = [t for t in study.trials if t.value is not None] # In the order they were started, from every worker and every earlier fit with this storage
    if temp is not None: # Kept in memory, since the file is deleted
        kept = optuna.create_study(sampler=study.sampler, study_name=study.study_name)
        kept.add_trials(study.trials)
        study = kept
        for path in [temp, temp + '.lock']:
            if os.path.exists(path):
                os.remove(path)
    p.trials = [dict(t.params, objective=t.value) for t in done]
    tried = pd.DataFrame(p.trials)
    values = tried.objective.values
    i = int(np.argmin(values))
    if p.fresh and len(values) >= 10: # With fresh seeds the lowest trial is the luckiest, which favors values whose runs vary most, so take the trial whose neighbors score lowest
        i = smoothed_best(p.encode(tried[p.names].values), values)
    best = tried[p.names].values[i]
    early = values[:max(1, int(np.ceil(0.75*len(values))))].min() # The best before the last quarter of the trials
    gain = (early - values.min())/max(abs(early), 1e-12)
    n_random = 10 # Optuna's first trials, which are random (TPESampler's n_startup_trials)
    if i == 0:
        converged, why = False, f'the best of the {len(values)} trials was the initial values, so the search found nothing better; try more trials (n_trials=)'
    elif i < n_random:
        converged, why = False, f'the best of the {len(values)} trials was trial {i}, one of the first {n_random}, which are random, so the search did not improve on them; try more trials (n_trials=)'
    else:
        converged = gain < 0.01
        why = f'the best score improved by {gain:.1%} over the last quarter of the {len(values)} trials' + ('' if converged else '; try more trials (n_trials=)')
    spread = None
    if len(p.trials) >= 10: # Not a posterior, but some indication of the uncertainty: the spread of the best trials
        top = tried.nsmallest(max(5, len(tried)//10), 'objective')
        spread = top[p.names].min().values, top[p.names].max().values
    return sc.obj(best=best, converged=converged, why=why, spread=spread, study=study)


def fit_if2(problem, particles=1000, iterations=50, cooling=0.5, sd=0.05, chains=4):
    """
    Maximum likelihood by iterated filtering (IF2; Ionides et al. 2015): passes of a particle filter in which each particle's values are perturbed by a shrinking random walk, and selected with it (see `Swarm`)

    For a stochastic compartmental sim that the kernels run whole, with a likelihood for every data series, and parameters
    that are single values of the model's rates. It gives the best values, not a posterior, with the range of the chains' estimates
    in `fit.table`; for a posterior, use `method='pmcmc'`. It has converged if the chains end within 2 of each other in log-likelihood.

    Args:
        problem (Problem): the fit, as `sim.fit()` sets it up
        particles (int): the number of particles in each pass; more gives a less noisy likelihood
        iterations (int): the number of passes of the filter in each chain
        cooling (float): the share of its first size that a step has after 50 passes, as `cooling.fraction.50` in pomp
        sd (float): the first size of a step of the random walk, on the scale searched on: the logit of each value's place between its bounds, or of its log for wide bounds
        chains (int): the number of chains, each from its own start: the best of the initial values and a spread of points over the bounds

    Examples:
        ```python
        import starsim as ss

        beta = ss.par(0.3, bounds=[0.1, 1])
        S, I, R = ss.states(3)
        infection = S >> I @ beta*I
        sir = ss.Disease(infection, I >> R @ ss.days(8), ss.outcome(0.5*infection, name='cases'), init=dict(I=0.01), name='sir')
        pars = dict(total_pop=10_000, unit='day', stop=60, solver='binomial')
        data = ss.Sim(sir, pars=dict(beta=0.5), rand_seed=2, **pars).run().to_data('cases', resample='week')
        sim = ss.Sim(sir, data=data, **pars)
        fit = sim.fit(method='if2', likelihood=dict(cases=ss.poisson()), particles=300, iterations=50)
        print(fit.pars, fit.converged)
        ```
    """
    p = problem
    filt = ParticleFilter(p.runner.restart(), likelihood=p.likelihood, weights=p.weights)
    swarm = Swarm(p.runner, p.fitted, p.restart, p.to_x, p.to_z(p.x0), sd=sd, cooling=cooling, n_points=len({row for s in filt.series for row in s.rows}))
    rng = np.random.default_rng(p.rand_seed)
    T = sc.timer()
    points = [p.x0] + list(p.spread(10*len(p.names))) # The chains start from the best of the initial values and a spread of points, as particle MCMC's do, since IF2 from far off can follow the late data away from the best values
    scores = []
    for x in points:
        p.restart(x)
        scores.append(filt(min(particles, 200), seed=int(rng.integers(2**31))))
    starts = [p.to_z(points[k]) for k in np.argsort(scores)[::-1][:chains]]
    ests = []
    for c,z in enumerate(starts):
        swarm.start(z, particles)
        p.restart(p.to_x(z)[0]) # The values that are the same for every particle
        for m in range(iterations):
            ll = filt(particles, seed=int(rng.integers(2**31)), swarm=swarm)
            p.trials.append(dict(zip(p.names, swarm.estimate()), objective=-ll, chain=c, iteration=m + 1)) # The log-likelihood of the pass, with the values perturbed
        ests.append(swarm.estimate())
    lls = [] # Each chain's estimate, by the filter without perturbations
    for x in ests:
        p.restart(x)
        lls.append(np.mean([filt(particles, seed=int(rng.integers(2**31))) for r in range(5)]))
    gap = max(lls) - min(lls)
    converged = bool(chains > 1 and gap < 2)
    if chains == 1:
        why = f'one chain ({iterations} passes with {particles} particles), so nothing to compare it with; use chains=4'
    elif converged:
        why = f'the {chains} chains (from different starts, each of {iterations} passes with {particles} particles) end within {gap:.2f} of the best log-likelihood ({max(lls):.2f})'
    else:
        why = f'the {chains} chains end {gap:.2f} apart in log-likelihood (best {max(lls):.2f}); try more passes (iterations=) or particles (particles=)'
    if p.verbose:
        print(f'IF2: {chains} chain{"s" if chains != 1 else ""} of {iterations} passes with {particles} particles in {T.tt(output=True):.1f} s; log-likelihood at each estimate: {sc.strjoin([f"{ll:.2f}" for ll in lls])}')
    return sc.obj(best=ests[int(np.argmax(lls))], converged=converged, why=why, spread=(np.min(ests, axis=0), np.max(ests, axis=0)))


def fit_pmcmc(problem, scan=None, particles=None, draws=1000, warmup=1000, chains=4, parallel=True):
    """
    Particle MCMC: a random walk, on a likelihood estimated by a particle filter (see `pmmh()`), from the best of the initial values and a spread of points

    For a stochastic compartmental sim that the kernels run whole, with a likelihood for every data series. It gives a posterior (`fit.samples`).

    Args:
        problem (Problem): the fit, as `sim.fit()` sets it up
        scan (int): the number of points tried for the start (default: see `Problem.scan_size()`)
        particles (int): the number of particles (default: chosen by each chain in its warm-up)
        draws (int): the number of draws in each chain, after the warmup
        warmup (int): the number of draws to discard at the start of each chain
        chains (int): the number of chains
        parallel (bool): whether to run the chains in parallel

    Examples:
        ```python
        import starsim as ss

        beta = ss.par(0.3, bounds=[0.1, 1])
        S, I, R = ss.states(3)
        infection = S >> I @ beta*I
        sir = ss.Disease(infection, I >> R @ ss.days(8), ss.outcome(0.5*infection, name='cases'), init=dict(I=0.01), name='sir')
        pars = dict(total_pop=10_000, unit='day', stop=60, solver='binomial')
        data = ss.Sim(sir, pars=dict(beta=0.5), rand_seed=2, **pars).run().to_data('cases', resample='week')
        sim = ss.Sim(sir, data=data, **pars)
        fit = sim.fit(method='pmcmc', likelihood=dict(cases=ss.poisson()), draws=200, warmup=200, chains=2, particles=100)
        print(fit.table, fit.samples.head())
        ```
    """
    p = problem
    filt = ParticleFilter(p.runner.restart(), likelihood=p.likelihood, weights=p.weights)
    rng = np.random.default_rng(p.rand_seed)

    def logpost_pf(z, rng, n):
        x, share = p.to_x(z)
        p.runner.restart(pars=p.to_pars(x), rand_seed=int(rng.integers(2**31))) # Each with its own start, if the start is drawn
        return filt(n_particles=n, seed=int(rng.integers(2**31))) + p.logprior(x) + p.logjac(x, share)

    starts = [p.x0] + list(p.spread(p.scan_size(scan or None)))
    start = starts[int(np.argmax([logpost_pf(p.to_z(x), rng, 100) for x in starts]))]
    zs, logdens, acceptance, n_particles = pmmh(logpost_pf, p.to_z(start), draws=draws, warmup=warmup, chains=chains, particles=particles, seed=p.rand_seed, parallel=parallel, verbose=p.verbose)
    return sc.obj(zs=zs, logdens=logdens, acceptance=acceptance)


def fit_abc_smc(problem, particles=None, generations=8, quantile=0.5, tolerance=None):
    """
    A posterior without a likelihood, by approximate Bayesian computation (see `abc_smc()`): the runs within a shrinking distance of the data

    For any sim, including agents. The distance is the goodness of fit (`gof=`), and each generation takes many runs of the sim. It has converged
    once the tolerance has stopped shrinking.

    Args:
        problem (Problem): the fit, as `sim.fit()` sets it up
        particles (int): the number of particles, which are the draws of the posterior (default 200)
        generations (int): the most generations
        quantile (float): the quantile of the distances that sets the next generation's tolerance
        tolerance (float): the distance to stop at (default: run every generation)

    Examples:
        ```python
        import starsim as ss

        beta = ss.par(0.3, bounds=[0.1, 1])
        S, I, R = ss.states(3)
        infection = S >> I @ beta*I
        sir = ss.Disease(infection, I >> R @ ss.days(8), ss.outcome(0.5*infection, name='cases'), init=dict(I=0.01), name='sir')
        pars = dict(total_pop=10_000, unit='day', stop=60, solver='binomial')
        data = ss.Sim(sir, pars=dict(beta=0.5), rand_seed=2, **pars).run().to_data('cases', resample='week')
        fit = ss.Sim(sir, data=data, **pars).fit(method='abc-smc', particles=100, generations=6)
        print(fit.table, fit.samples.head())
        ```
    """
    p = problem
    def distance(z, seed):
        return p.measure(p.run(p.to_x(z)[0], seed=seed if p.stochastic else None)) # A stochastic run takes a seed of its own, so the posterior includes chance

    def logprior_z(z):
        x, share = p.to_x(z)
        return p.logprior(x) + p.logjac(x, share)

    pop, weights, dists, tolerances, acceptance, n_abc = abc_smc(distance, logprior_z, p.to_z(p.x0), particles=particles or 200, generations=generations, quantile=quantile, tolerance=tolerance, seed=p.rand_seed, verbose=p.verbose)
    picks = np.random.default_rng(p.rand_seed).choice(len(pop), size=len(pop), p=weights) # Equal weights, as draws
    zs = pop[picks][None] # One chain
    logdens = (np.array([p.logjac(*p.to_x(z)) for z in zs[0]]) - dists[picks])[None] # So the best draw is the closest run
    ess = 1/np.sum(weights**2)
    settled = len(tolerances) >= 2 and tolerances[-1] > 0.9*tolerances[-2] # The tolerance has stopped shrinking
    why = (f'the tolerance {"settled" if settled else "was still shrinking"} at {tolerances[-1]:.4g} after {len(tolerances)} generations ({n_abc} runs), with {acceptance[-1]:.0%} of the last proposals accepted '
           f'and an effective sample size of {ess:.0f}' + ('' if settled else '; try more generations (generations=)'))
    return sc.obj(zs=zs, logdens=logdens, ess=ess, converged=bool(settled), why=why)


def fit_nuts(problem, draws=500, warmup=500, chains=4, scan=None):
    """
    The No-U-Turn sampler, each chain from an optimum found from one of the best points of a scan that are apart: the best, and others nearly as good (so that chains caught in different modes disagree, and none starts in a mode far below the best), on the log-density as one function in JAX if the model allows (see `ss.gradients.Density`), else with Python in the loop

    For a deterministic sim with gradients, with a likelihood for every data series. It gives a posterior (`fit.samples`), and needs JAX and BlackJAX.

    Args:
        problem (Problem): the fit, as `sim.fit()` sets it up
        draws (int): the number of draws in each chain, after the warmup
        warmup (int): the number of warmup steps of each chain
        chains (int): the number of chains
        scan (int): the number of points in the scan (default: see `Problem.scan_size()`)

    Examples:
        ```python
        import starsim as ss

        beta = ss.par(0.3, bounds=[0.1, 1])
        S, I, R = ss.states(3)
        infection = S >> I @ beta*I
        sir = ss.Disease(infection, I >> R @ ss.days(8), ss.outcome(0.5*infection, name='cases'), init=dict(I=0.01), name='sir')
        pars = dict(total_pop=10_000, unit='day', stop=60)
        data = ss.Sim(sir, pars=dict(beta=0.5), rand_seed=2, **pars).run().to_data('cases', resample='week')
        sim = ss.Sim(sir, data=data, **pars)
        fit = sim.fit(method='nuts', likelihood=dict(cases=ss.poisson()), draws=200, warmup=200, chains=2)
        print(fit.table, fit.samples.head())
        ```
    """
    p = problem
    sampling = dict(draws=draws, warmup=warmup, chains=chains)
    logs, span, base_lows = p.logs, p.span, p.base_lows
    import scipy.optimize as spo
    bounds = list(zip(p.encode(p.lows), p.encode(p.highs)))
    def fun(z): # Minus the log-likelihood and its gradient, on the scale searched on (as in fit_scipy())
        x = p.decode(z)
        value, grad = p.with_gradient(x)
        return value, grad*np.where(p.logs, x, 1.0)
    optima = [p.decode(spo.minimize(fun, p.encode(x), method='l-bfgs-b', jac=True, bounds=bounds).x) for x in p.scan(scan, top=max(chains, 3))] # The optimum near each of the best points of the scan that are apart
    scores = np.array([p.objective(x) - p.logprior(x) for x in optima]) # Minus the log of the posterior density at each
    starts = []
    for i in np.argsort(scores, kind='stable'): # The chains start at the best optimum, and at others only if they are nearly as good, so they cannot all sit in a mode far below the best (A4), while chains in modes that compete disagree
        if scores[i] - scores.min() <= 5 and all(np.abs(p.share_of(optima[i]) - p.share_of(x)).max() >= 0.01 for x in starts):
            starts.append(optima[i])
    z0 = np.array([p.to_z(starts[c % len(starts)]) for c in range(chains)])

    def logpost(z):
        x, share = p.to_x(z)
        dxdz = span*share*(1 - share)*np.where(logs, x, 1.0)
        value, derivs = ss.gradients.gradient(p.runner, p.restart, x, p.measure)
        hs = 1e-6*np.maximum(np.abs(x), 1e-6)
        dprior = np.array([(logprior(q, val + h) - logprior(q, val - h))/(2*h) if q.prior is not None else 0.0 for q,val,h in zip(p.fitted.values(), x, hs)])
        dlogjac = 1 - 2*share + np.where(logs, span*share*(1 - share), 0.0) # The derivative of log(dx/dz)
        return -float(value) + p.logprior(x) + np.log(dxdz).sum(), (dprior - np.array(derivs, dtype=float))*dxdz + dlogjac

    def exact(z): # The log of the posterior density at a point, from a run of the sim
        x, share = p.to_x(z)
        return -p.measure(p.run(x)) + p.logprior(x) + p.logjac(x, share)

    try: # The log-density as one function in JAX, if the model allows; otherwise each gradient is worked out with Python in the loop
        density = ss.gradients.Density(p.runner, p.fitted, p.x0, p.restart, likelihood=p.likelihood, weights=p.weights, external=p.external, bounds=(p.lows, p.highs))
    except ss.gradients.Unsupported as E:
        density = None
        p.python_in_loop(f'the log-density in JAX does not cover {E} yet', 'the posterior')
    if density is not None:
        jnp = ss.gradients.jax().numpy
        forward = len(p.names)*sum(X['y'].size for X in density.X0) < reverse_min # As for a best fit (see Problem.jax_density())
        if not forward:
            density.reverse()
        def logpost_jax(z):
            share = jnp.clip(1/(1 + jnp.exp(-z)), 1e-12, 1 - 1e-12)
            pos = base_lows + span*share
            x = jnp.where(logs, jnp.exp(pos), pos)
            return density(x) + jnp.log(span*share*(1 - share)*jnp.where(logs, x, 1.0)).sum()
        zs, logdens, divergent = nuts(logpost_jax, z0, traced=True, forward=forward, seed=p.rand_seed, verbose=p.verbose, **sampling)
        flat_z, flat_lp = zs.reshape(-1, len(p.names)), logdens.reshape(-1)
        for i in np.linspace(0, len(flat_lp) - 1, 3).astype(int): # The JAX log-density is the one the sim gives, at a few of the draws
            expected = exact(flat_z[i])
            if not np.isclose(flat_lp[i], expected, rtol=1e-6, atol=1e-6):
                p.python_in_loop(f'the log-density in JAX ({flat_lp[i]:.10g}) differs from the one from the sim ({expected:.10g}) at {dict(zip(p.names, p.to_x(flat_z[i])[0]))}', 'the posterior again')
                density = None
                break
    if density is None:
        zs, logdens, divergent = nuts(logpost, z0, seed=p.rand_seed, verbose=p.verbose, **sampling)
    return sc.obj(zs=zs, logdens=logdens, divergent=divergent)


reverse_min = 5000 # The values times the compartments at which a gradient of one number is taken in reverse mode (one pass) rather than forward mode (one direction per value): forward is faster for 40 values of an SIR (3.1 ms against 14 ms), and for 20 places (5 against 20 ms); reverse for 100 places (26 against 117 ms)
grad_methods = ['l-bfgs-b', 'slsqp'] # The SciPy methods that use gradients, when the sim has them
sampling_methods = ['nuts', 'pmcmc', 'abc-smc'] # The methods that give a posterior
methods = {'nuts':fit_nuts, 'pmcmc':fit_pmcmc, 'abc-smc':fit_abc_smc, 'if2':fit_if2, 'optuna':fit_optuna} | {name:functools.partial(fit_scipy, method=name) for name in scipy_methods} # The methods that sim.fit() knows, by name


def method_options(method):
    """ The settings that a fitting method takes, as keywords of `sim.fit()`: its own, and Optuna's for Optuna; a SciPy method also takes SciPy's, which SciPy checks (see `fit_scipy()`) """
    options = [key for key,par in inspect.signature(methods[method]).parameters.items() if key not in ['problem', 'method'] and par.kind != par.VAR_KEYWORD]
    if method == 'optuna':
        import optuna
        options += [key for key in inspect.signature(optuna.Study.optimize).parameters if key not in ['self', 'func', 'n_trials']]
    return options


def check_options(method, kwargs):
    """ Raise an error for a keyword of `sim.fit()` that the fitting method does not take, which would otherwise be a mistake that went unnoticed """
    options = method_options(method)
    unknown = [key for key in kwargs if key not in options]
    if unknown and method not in scipy_methods:
        raise TypeError(f'fit(method="{method}") takes {sc.strjoin([f"{key}=" for key in options])}, not {sc.strjoin(unknown)}')
    return


def split_data(df, holdout=0.25, folds=None):
    """
    Split data into the part to fit to and the part to predict, for each fold

    Each series is split on its own times. With a fraction, the last `holdout` of
    each series' points is held out; with `folds=k`, that part is cut into k blocks,
    and fold j fits to everything before block j and predicts block j. With a date
    or time, everything from it on is held out.

    Returns a list of (fitting data, held-out data) pairs, one per fold.

    Args:
        df (dataframe): the data
        holdout (float/date): the share of each series to hold out, or the date or time from which to hold out everything
        folds (int): the number of folds
    """
    k = folds or 1
    out = []
    for j in range(k):
        train = df.copy().astype(float)
        test = df.copy().astype(float)
        for col in df.columns:
            ok = np.flatnonzero(np.isfinite(df[col].values.astype(float)))
            if sc.isnumber(holdout) and 0 < holdout < 1:
                n_out = max(k, int(np.ceil(holdout*len(ok))))
                blocks = np.array_split(ok[len(ok)-n_out:], k)
                predict = blocks[j]
            else:
                if k > 1:
                    errormsg = 'folds= needs holdout= to be a fraction'
                    raise ValueError(errormsg)
                index = df.index.values
                after = (np.array([ss.date(t).years for t in index]) >= ss.date(holdout).years) if not sc.isnumber(index[0]) else (index.astype(float) >= float(holdout))
                predict = ok[after[ok]]
            if not len(predict) or predict[0] == ok[0]:
                errormsg = f'Holding out {holdout} leaves no data to fit to, or none to predict, for "{col}"'
                raise ValueError(errormsg)
            fit_part = ok[ok < predict[0]]
            train.iloc[np.setdiff1d(np.arange(len(df)), fit_part), train.columns.get_loc(col)] = np.nan
            test.iloc[np.setdiff1d(np.arange(len(df)), predict), test.columns.get_loc(col)] = np.nan
        out.append((train, test))
    return out


class ModelComparison(sc.prettyobj):
    """
    The result of comparing models by how well they predict held-out data; see `ss.fit.compare_models()`

    Attributes:
        fits (dict): each model's `ss.fit.Fit` (a list of them, one per fold, if there are several folds)
        table (df): the fitted parameters, the error on the data fitted to, the prediction error, and the rank of each model
        holdout (str): a description of what was held out
        measure (str): how the fit error and the prediction error are measured (the same for every model)
        splits (list): the data fitted to and the data held out, for each fold
        clear (bool): whether the best model is clearly better, given the spread across folds
    """
    def __init__(self, fits, table, holdout, measure, splits, clear=True):
        self.fits = fits
        self.table = table
        self.holdout = holdout
        self.measure = measure
        self.splits = splits
        self.clear = clear
        return

    def summarize(self):
        """ Print what was held out, and return the table, which is printed too in a script (see `ss.utils.return_table()`) """
        print(f'held out: {self.holdout}')
        ss.utils.return_table(self.table)
        print(f'fit error and prediction error are {self.measure}, on the data fitted to and on the held-out data (lower is better)')
        if not self.clear:
            print('the spread of the prediction error across folds is larger than the difference between the best two models, so there is no clear difference between them')
        return self.table

    def to_df(self):
        """ The table, as a dataframe """
        return self.table

    def plot(self, **kwargs):
        """ Plot each model's fit and prediction against the data, with the held-out data hollow; kwargs are as for `ss.plot_args()`

        Args:
            kwargs (dict): passed to `ss.plot_args()`
        """
        train, test = self.splits[-1]
        cols = list(train.columns)
        kw = ss.plot_args(kwargs, alpha=0.8)
        with ss.style(**kw.style):
            fig, axs = sc.getrowscols(len(cols), make=True, squeeze=False, **kw.fig)
            for ax,col in zip(axs.flatten(), cols):
                for label,fits in self.fits.items():
                    fit = sc.tolist(fits)[-1]
                    res = match_column(fit.sim, col)
                    ax.plot(res.timevec, res.values, label=label, **kw.plot)
                ax.scatter(plot_times(train), train[col], c='k', s=15, label='Fitted to', zorder=10)
                ax.scatter(plot_times(test), test[col], facecolors='none', edgecolors='k', s=15, label='Held out', zorder=10)
                ax.set_title(col)
                ax.legend(frameon=False)
            sc.figlayout(fig=fig)
        return ss.return_fig(fig, **kw.return_fig)


def compare_models(sims, data=None, holdout=0.25, folds=None, **kwargs):
    """
    Compare models by how well they predict data they were not fitted to

    Each model is fitted (with `sim.fit()`) to all but the last part of each
    data series, run forward over that part, and scored on it: by the default
    goodness of fit, or by the log predictive density if every series of every model
    has a likelihood; the same measure is used for every model. A model with more
    parameters usually fits better; the prediction error needs no separate penalty
    for them.

    Args:
        sims (list/dict): the sims to compare; labeled by their `label`, else by position (or by the keys of a dict)
        data (df/str): the data, attached to every sim (default: each sim's own data, which must be the same for all)
        holdout (float/date): the fraction of each series to hold out (default: the last 25%), or the date (or time) from which to hold it out
        folds (int): if given, repeat the comparison with this many cut points over the held-out part (fit to the first part, predict the next block), and report the mean and spread of the prediction error
        kwargs (dict): passed to `sim.fit()`, e.g. `weights=`

    Returns:
        An `ss.fit.ModelComparison`; `comp.fits` holds each model's fit

    Examples:
        ```python
        comp = ss.fit.compare_models([sim_sir, sim_seir], data=df)
        comp.summarize()
        comp.plot()
        ```
    """
    if isinstance(sims, dict):
        labels, sims = list(sims.keys()), list(sims.values())
    else:
        sims = sc.tolist(sims)
        labels = [sim.label or f'model {i+1}' for i,sim in enumerate(sims)]
    if len(set(labels)) < len(labels):
        labels = [f'{label} ({i+1})' for i,label in enumerate(labels)]

    # Find the data
    probes = []
    for sim in sims:
        probe = sim.new(verbose=0)
        if data is not None:
            probe.data = data
        probes.append(probe.init())
    dfs = [probe.data for probe in probes]
    if any(df is None for df in dfs):
        raise ValueError('compare_models() needs data: pass data=, or attach the same data to every sim.')
    if data is None and not all(df.equals(dfs[0]) for df in dfs[1:]):
        errormsg = 'The sims have different data; pass data= to compare them on the same data'
        raise ValueError(errormsg)
    df = dfs[0]
    splits = split_data(df, holdout=holdout, folds=folds)

    # Describe what was held out
    train, test = splits[-1] if folds is None else (splits[0][0], pd.concat([s[1] for s in splits]).groupby(level=0).first())
    spans = []
    for col in df.columns:
        times = test.index[np.isfinite(test[col].values.astype(float))]
        fmt = lambda t: f'{round(t, 6):g}' if sc.isnumber(t) else str(t)
        spans.append(f'{col}: {fmt(times[0])} to {fmt(times[-1])}')
    what = f'the last {holdout:.0%} of each series' if sc.isnumber(holdout) and 0 < holdout < 1 else f'from {holdout}'
    desc = f'{what}{f", in {folds} folds" if folds else ""} ({"; ".join(spans)})'

    # Fit and predict
    liks = [likelihoods(probe, likelihood=kwargs.get('likelihood')) for probe in probes]
    use_lik = all(lik is not None for lk in liks for lik in lk.values()) and kwargs.get('gof') is None
    if not use_lik: # One measure for every model, even one whose series all have a likelihood
        kwargs['gof'] = kwargs.get('gof') or 'nae'
    fits = sc.obj()
    rows = []
    for label,sim,probe in zip(labels, sims, probes):
        fold_fits = []
        fit_errors = []
        errors = []
        for train,test in splits:
            fit = sim.fit(data=train, **sc.mergedicts(dict(verbose=False), kwargs)) # A point estimate, for each fold, by the method sim.fit() chooses (e.g. Nelder-Mead for one or two parameters)
            fold_fits.append(fit)
            for df_part,out in [(train, fit_errors), (test, errors)]: # Each scored the same way
                if use_lik:
                    out.append(-loglik(fit.sim, df=df_part, likelihood=kwargs.get('likelihood'), weights=kwargs.get('weights')))
                else:
                    out.append(mismatch(fit.sim, df=df_part, weights=kwargs.get('weights'), gof=kwargs['gof']))
        fits[label] = fold_fits if folds else fold_fits[0]
        row = dict(model=label, fitted_pars=len(fold_fits[0].pars), fit_error=np.mean(fit_errors), prediction_error=np.mean(errors))
        if folds:
            row['spread'] = np.std(errors)
        rows.append(row)
    table = pd.DataFrame(rows).set_index('model')
    table['rank'] = table['prediction_error'].rank(method='min').astype(int)
    table = table.sort_values('rank')

    # Check whether the difference is larger than the spread
    clear = True
    if folds and len(table) > 1:
        diff = table['prediction_error'].iloc[1] - table['prediction_error'].iloc[0]
        clear = diff > table['spread'].iloc[:2].max()
    measure = 'the negative log predictive density' if use_lik else 'the normalized absolute error' if kwargs['gof'] == 'nae' else f'the goodness of fit ({kwargs["gof"]})'
    comp = ModelComparison(fits=fits, table=table, holdout=desc, measure=measure, splits=splits, clear=clear)
    return comp
