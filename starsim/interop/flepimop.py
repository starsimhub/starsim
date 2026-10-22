"""
Import a flepiMoP model (a configuration YAML file)

flepiMoP (https://github.com/HopkinsIDD/flepiMoP) describes compartments as a product of
named dimensions, transitions as rates with `proportional_to` terms, places with their
populations and commuting, initial numbers, modifiers that change parameters over date
ranges (grouped into scenarios), and outcomes chained from the flows.

The compartments become states (with several dimensions, one state for each combination,
named as flepiMoP names them, e.g. `S_unvaccinated`); the places become a stratification
`subpop` with commuting (`ss.geo.Mobility(kind='visit')`, as flepiMoP's force of infection
mixes in the places its people visit for half the day); each transition becomes a flow,
classified by the shared rules; the modifiers of a scenario become `ss.modify` interventions;
and the outcomes become `ss.outcome`s. Parameters drawn from a distribution are imported at
their mean, and listed as not translated.
"""
import ast
import itertools
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
from .core import Source, build
from .emulsion import number

__all__ = ['load_flepimop']

solvers = dict(rk4='ode', rk4jit='ode', bestcurrent='ode', euler='ode', stochastic='binomial')
total = '__N__' # The population of the place, which flepiMoP divides each term of the force of infection by


def strip(df):
    """ A CSV as flepiMoP writes them, with spaces around the names and values """
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()
    return df


def prior_of(val):
    """ The distribution a parameter is drawn from for each run, as the prior of its ss.par (so ss.parallel(sim, priors=True) draws it), or None for a fixed value """
    if not isinstance(val, dict):
        return None
    dist = str(val.get('distribution', 'fixed')).lower()
    num = lambda key, default=None: float(val[key]) if key in val else default
    if dist == 'truncnorm':
        return ss.truncnorm(loc=num('mean'), scale=num('sd'), low=num('a', -np.inf), high=num('b', np.inf))
    elif dist in ['norm', 'normal']:
        return ss.normal(loc=num('mean'), scale=num('sd'))
    elif dist == 'uniform':
        return ss.uniform(low=num('low'), high=num('high'))
    elif dist == 'lognorm':
        return ss.lognorm_im(mean=num('meanlog'), std=num('sdlog'))
    return None


def mean_of(val, name, src):
    """ The value of a parameter: a number, an expression of numbers, or a distribution (at its mean, listed as not translated) """
    if isinstance(val, dict):
        dist = str(val.get('distribution', 'fixed')).lower()
        if dist == 'fixed':
            return mean_of(val.get('value'), name, src)
        means = dict(truncnorm='mean', norm='mean', normal='mean', lognorm='meanlog', poisson='lam', binomial=None)
        if dist == 'uniform':
            out = (float(val['low']) + float(val['high']))/2
        elif dist in means and means[dist] in val:
            out = float(val[means[dist]])
            if dist == 'lognorm':
                out = float(np.exp(out + float(val.get('sdlog', 0))**2/2))
        else:
            errormsg = f'Cannot work out the mean of the {dist} distribution of "{name}" ({val})'
            raise ValueError(errormsg)
        src.skip(name, f'drawn from a {dist} distribution ({sc.strjoin(f"{k}={v}" for k,v in val.items() if k != "distribution")}) for each run; imported at its mean, {out:n}. A prior on the ss.par (or a value drawn for each run) would carry it')
        return out
    num = number(val, {})
    if num is None:
        errormsg = f'The value of "{name}" ({val}) is not a number'
        raise ValueError(errormsg)
    return num


def per_dim(spec, ndim):
    """ A source, destination, rate, or group of a transition as one entry for each dimension """
    if isinstance(spec, str) or not isinstance(spec, list):
        return [[str(spec)]] + [['1']]*(ndim - 1) if ndim > 1 else [[str(spec)]]
    if ndim == 1 and all(isinstance(s, str) for s in spec) and len(spec) == 1:
        return [spec]
    if len(spec) != ndim:
        if ndim == 1:
            return [spec]
        errormsg = f'Expecting an entry for each of the {ndim} dimensions of the compartments, not {spec}'
        raise ValueError(errormsg)
    return [sc.tolist(s) for s in spec]


def flatten(items):
    """ The names in a list that may hold lists (which flepiMoP sums) """
    out = []
    for item in sc.tolist(items):
        out += flatten(item) if isinstance(item, list) else [str(item)]
    return out


def compartment(levels):
    """ flepiMoP's name for the compartment with these levels, e.g. S_unvaccinated """
    return '_'.join(levels)


def expand(t, dims):
    """ Each combination of levels a transition applies to: (source, destination, rate factors, groups) """
    ndim = len(dims)
    srcs, dsts = per_dim(t['source'], ndim), per_dim(t['destination'], ndim)
    rates = per_dim(t.get('rate', ['1']*ndim), ndim)
    props = [[['source']]*ndim if g == 'source' else per_dim(g, ndim) for g in t.get('proportional_to', [])] # "source" is the source compartment
    exps = sc.tolist(t.get('proportion_exponent', ['1']*len(props)))
    out = []
    for combo in itertools.product(*[range(len(s)) for s in srcs]):
        pick = lambda lst, d: lst[d][combo[d]] if len(lst[d]) > 1 else lst[d][0]
        src = [srcs[d][combo[d]] for d in range(ndim)]
        dst = [pick(dsts, d) for d in range(ndim)]
        rate = [str(pick(rates, d)) for d in range(ndim)]
        groups = []
        for g,group in enumerate(props): # Each group is a sum of compartments: its levels in each dimension (paired with the source's, or all of them)
            levels = []
            for d in range(ndim):
                lev = group[d]
                if 'source' in flatten(lev):
                    lev = [src[d]]
                elif len(lev) == len(srcs[d]) and len(lev) > 1:
                    lev = flatten(lev[combo[d]])
                else:
                    lev = flatten(lev)
                levels.append(lev)
            exp = exps[g] if g < len(exps) else '1'
            exp = '*'.join(f'({e})' for e in flatten(exp) if str(e) != '1') or '1'
            groups.append(([compartment(c) for c in itertools.product(*levels)], exp))
        out.append((compartment(src), compartment(dst), rate, groups))
    return out


def load_flepimop(filename, scenario=None, verbose=None, **kwargs):
    """
    Read a flepiMoP configuration (a YAML file) into a sim

    Args:
        filename (str): the configuration file; the files it names are read from its folder (or its `data_path`)
        scenario (str): the scenario of `seir_modifiers` to apply (default: the first listed). The modifiers of
            every scenario are kept in `sim.imported.scenarios`, e.g. for `sim.new(interventions=sim.imported.scenarios['opt_vax'])`
        verbose (bool): whether to print how each transition was read
        kwargs (dict): passed to `ss.Sim()`

    Returns:
        An `ss.Sim`, with `sim.imported` saying how it was read and what was not translated
    """
    filename = sc.path(filename)
    cfg = sc.loadyaml(filename)
    folder = filename.parent / cfg.get('data_path', '.')
    name = cfg.get('name', filename.stem)
    src = Source(name, format='flepiMoP', file=str(filename))
    notes = [] # How the parts that are not flows were read
    seir = cfg.get('seir') or {}

    # Compartments: one state for each combination of levels
    dims = {k:[str(v) for v in sc.tolist(vals)] for k,vals in (cfg.get('compartments') or {}).items()}
    if not dims:
        errormsg = f'"{filename}" has no compartments'
        raise ValueError(errormsg)
    states = [compartment(c) for c in itertools.product(*dims.values())]
    if len(dims) > 1:
        notes.append(f'compartments: the product of {sc.strjoin(dims)}, as {len(states)} states')

    # Places
    places, pop, away = None, None, None
    setup = cfg.get('subpop_setup') or {}
    if setup.get('geodata'):
        geo = strip(pd.read_csv(folder / setup['geodata']))
        allplaces = list(geo['subpop'].astype(str))
        keep = np.array([p in [str(x) for x in sc.tolist(setup['selected'])] for p in allplaces]) if setup.get('selected') else np.ones(len(allplaces), dtype=bool) # Only the places selected, as flepiMoP does
        places = list(np.array(allplaces)[keep])
        pop = geo['population'].values.astype(float)[keep]
        if not keep.all():
            notes.append(f'subpop_setup.selected: {len(places)} of the {len(allplaces)} places in the geodata')
        if setup.get('mobility'):
            path = folder / setup['mobility']
            if path.suffix == '.txt':
                moves = np.loadtxt(path)
            elif path.suffix == '.csv':
                mob = strip(pd.read_csv(path))
                moves = np.zeros((len(allplaces), len(allplaces)))
                for ori,dest,amount in zip(mob['ori'].astype(str), mob['dest'].astype(str), mob['amount'].astype(float)):
                    moves[allplaces.index(ori), allplaces.index(dest)] += amount
            else:
                moves = None
                src.skip('subpop_setup.mobility', f'the mobility file {path.name} is not a .csv or .txt file')
            if moves is not None:
                away = 0.5*moves[keep][:, keep]/pop[:, None] # flepiMoP's travelers spend half the day away (percent_day_away = 0.5)

    # Initial numbers, by place and compartment
    ic = cfg.get('initial_conditions') or {}
    init = np.zeros((len(states), len(places) if places else 1))
    method = ic.get('method', 'Default')
    if method == 'SetInitialConditions' and ic.get('initial_conditions_file'):
        df = strip(pd.read_csv(folder / ic['initial_conditions_file']))
        for _,row in df.iterrows():
            comp = str(row['mc_name']) if 'mc_name' in df.columns else compartment([str(row[f'mc_{d}']) for d in dims])
            if comp not in states:
                src.skip(f'initial_conditions: {comp}', 'names a compartment the model does not have')
                continue
            if places and str(row['subpop']) not in places: # A place that was not selected
                continue
            col = places.index(str(row['subpop'])) if places else 0
            init[states.index(comp), col] += float(row['amount'])
        if ic.get('proportional'):
            init *= (pop if pop is not None else 1)
    elif method == 'Default' or not ic:
        init[0] = pop if pop is not None else seir.get('population', 1) # Everyone starts in the first compartment
    else:
        src.skip('initial_conditions', f'the method {method} is not translated; everyone starts in {states[0]}')
        init[0] = pop if pop is not None else 1
    sizes = init.sum(axis=0)
    if pop is not None and not np.allclose(sizes, pop):
        notes.append(f'initial conditions: {sc.strjoin(f"{p}={s:n}" for p,s in zip(places, sizes))} people, where the geodata has {sc.strjoin(f"{p:n}" for p in pop)}; the force of infection is divided by the number in each place, not by the geodata')
    for i,state in enumerate(states):
        if places:
            src.add_state(state, init=ss.by(subpop=list(init[i])) if i and init[i].any() else None) # The first starts with the rest
        else:
            src.add_state(state, init=init[i, 0])

    # Parameters
    pars = {}
    priors = {} # The distributions that parameters are drawn from for each run
    for key,p in (seir.get('parameters') or {}).items():
        p = p if isinstance(p, dict) else dict(value=p)
        if 'timeseries' in p:
            src.skip(key, f'a time series read from {p["timeseries"]}; imported as 1. An ss.timeseries would carry it')
            pars[key] = 1.0
        else:
            priors[key] = prior_of(p.get('value'))
            pars[key] = mean_of(p.get('value'), key, src if priors[key] is None else Source('', '')) # A distribution that becomes a prior is translated
        for extra in ['rolling_mean_windows']:
            if extra in p:
                src.skip(f'{key}.{extra}', f'"{extra}: {p[extra]}" was not translated')
        if p.get('stacked_modifier_method') not in [None, 'product', 'reduction_product', 'sum']:
            src.skip(f'{key}.stacked_modifier_method', f'"{p["stacked_modifier_method"]}" was not translated')
    methods = {k:(p.get('stacked_modifier_method', 'product') if isinstance(p, dict) else 'product') for k,p in (seir.get('parameters') or {}).items()}

    # Transitions: one flow for each combination of levels, written as the total flow that flepiMoP works out
    src.totals = [total]
    used = set()
    for i,t in enumerate(seir.get('transitions') or []):
        for frm,to,rates,groups in expand(t, dims):
            factors = [f'({r})' for r in rates if r != '1']
            for g,(comps,exp) in enumerate(groups):
                summed = ' + '.join(comps)
                summed = f'({summed})' if len(comps) > 1 else summed
                term = summed if exp == '1' else f'{summed} ** ({exp})'
                factors.append(term if g == 0 else f'{term} / {total}') # After the source, each term is a share of the place's people
            if len(groups) > 2:
                notes.append(f'{frm} -> {to}: {len(groups) - 1} terms after the source, each divided by the population (flepiMoP also multiplies by the rate once for each)')
            rate = ' * '.join(factors) or '1'
            used |= {n.id for n in ast.walk(ast.parse(rate, mode='eval')) if isinstance(n, ast.Name)}
            src.add_flow(frm, to, rate, per='total' if groups else 'person', name=f'{frm}_to_{to}')
    for key in sorted(used & set(pars)):
        src.add_par(key, pars[key], prior=priors.get(key))

    # Places, as a stratification with commuting; and their initial numbers
    if places:
        mob = None if away is None or not away.any() else ss.geo.Mobility(matrix=away, kind='visit')
        src.stratify(subpop=places, split=list(sizes/sizes.sum()), kind='space', mobility=mob)
        if mob is not None:
            notes.append('subpop: commuting as visits, each traveler away for half the day (flepiMoP\'s percent_day_away)')

    # Time and solver
    integ = seir.get('integration') or {}
    method = str(integ.get('method', 'rk4')).replace('.', '').replace('_', '')
    solver = solvers.get(method)
    if solver is None:
        src.skip('seir.integration.method', f'"{integ.get("method")}" is not a method flepiMoP has; run as an ODE')
        solver = 'ode'
    if method == 'euler':
        notes.append('integration: flepiMoP\'s "euler" moves 1 - exp(-rate*dt) of each compartment a step; run as an ODE')
    cfg_sim = dict(solver=solver, unit='day', total_pop=float(sizes.sum()))
    if method in ['rk4', 'rk4jit', 'bestcurrent']: # flepiMoP takes one RK4 step a step, so the numbers are its own
        cfg_sim['substeps'] = 1
    if cfg.get('start_date'): # flepiMoP's output includes what happens during its end date, so the sim stops a day later
        cfg_sim.update(start=str(cfg['start_date']), stop=str(sc.date(str(cfg['end_date'])) + pd.Timedelta(days=1))[:10])
    if float(integ.get('dt', 1)) != 1:
        cfg_sim['dt'] = float(integ['dt'])
    src.config.update(cfg_sim)

    # What is not translated
    for key,why in dict(seeding='seeding (scheduled importations): an ss.geo.Mobility(data=...) register would carry it',
                        inference='inference (fitting to data): use sim.calibrate() with ss.data()',
                        outcome_modifiers='outcome modifiers (changes to the outcomes\' probabilities over date ranges): ss.modify on the weights would carry them').items():
        if cfg.get(key):
            src.skip(key, why)

    # Modifiers and scenarios (built before the sim, since they are interventions)
    mods = (cfg.get('seir_modifiers') or {}).get('modifiers') or {}
    names = sc.tolist((cfg.get('seir_modifiers') or {}).get('scenarios') or [])
    scen_mods = {}
    for scen in names:
        scen_mods[scen] = modifiers(scen, mods, name, places, methods, src)
    src.incomplete = list(dict.fromkeys(src.incomplete))
    if scenario is None and names:
        scenario = names[0]
        notes.append(f'scenario: {scenario} (the first of {sc.strjoin(names)}; the others are in sim.imported.scenarios)')
    elif scenario is not None and scenario not in scen_mods:
        errormsg = f'There is no scenario "{scenario}"; the scenarios are {sc.strjoin(names) or "none"}'
        raise ValueError(errormsg)
    if scenario is not None:
        kwargs.setdefault('interventions', modifiers(scenario, mods, name, places, methods, src, quiet=True))

    # Outcomes, which may refer to each other
    for oname,o in ((cfg.get('outcomes') or {}).get('outcomes') or {}).items():
        outcome(oname, o, dims, src, notes)
    src.incomplete = list(dict.fromkeys(src.incomplete)) # A modifier in several scenarios is listed once
    src.notes += notes
    sim = build(src, verbose=verbose, **kwargs)
    sim.imported.scenarios = scen_mods
    return sim


def modifiers(scenario, mods, disease, places, methods, src, quiet=False):
    """ The modifiers of a scenario (a modifier, or a stack of them), as `ss.modify` interventions """
    out = []
    def add(key, seen=()):
        mod = mods.get(key)
        if mod is None:
            if not quiet:
                src.skip(f'seir_modifiers: {key}', 'not defined')
            return
        kind = mod.get('method')
        if kind == 'StackedModifier':
            for sub in mod.get('modifiers', []):
                if sub not in seen:
                    add(sub, seen + (key,))
            return
        elif kind != 'SinglePeriodModifier':
            if not quiet:
                src.skip(f'seir_modifiers: {key}', f'the method {kind} is not translated (only SinglePeriodModifier and StackedModifier are)')
            return
        par = str(mod['parameter'])
        if par not in src.pars and not quiet:
            src.skip(f'seir_modifiers: {key}', f'changes "{par}", which no transition uses')
        value = mean_of(mod['value'], f'seir_modifiers: {key}', src if not quiet else Source('', ''))
        method = methods.get(par, 'product')
        amount = dict(product=dict(multiply=value), reduction_product=dict(multiply=1 - value), sum=dict(add=value))[method]
        stop = str(sc.date(str(mod['period_end_date'])) + pd.Timedelta(days=1))[:10] # Starsim's stop is the first day it no longer applies
        subpops = mod.get('subpop', 'all')
        if subpops != 'all' and places and not quiet:
            src.skip(f'seir_modifiers: {key}', f'applies only in {sc.strjoin(sc.tolist(subpops))}; imported for every place')
        if mod.get('subpop_groups') and not quiet:
            src.skip(f'seir_modifiers: {key}', 'subpop_groups (one value drawn for a group of places) were not translated')
        out.append(ss.modify(f'{disease}.pars.{par}', start=str(mod['period_start_date']), stop=stop, name=key, **amount))
    add(scenario)
    return out


def outcome(name, o, dims, src, notes):
    """ An outcome: a share of a flow (or of another outcome), with a delay and a duration """
    source = o.get('source')
    if isinstance(source, dict) and 'incidence' in source:
        want = {d:[str(v) for v in sc.tolist(lev)] for d,lev in source['incidence'].items()}
        dimnames = list(dims)
        def matches(state):
            levels = state.split('_') if len(dimnames) > 1 else [state]
            return all(levels[dimnames.index(d)] in lev for d,lev in want.items() if d in dimnames)
        flows = [f.name for f in src.flows if f.dst and matches(f.dst) and not (f.src and matches(f.src))]
        if not flows:
            src.skip(f'outcomes: {name}', f'no flow enters {source["incidence"]}')
            return
    elif isinstance(source, str) and source in src.outcomes:
        flows = source
    else:
        src.skip(f'outcomes: {name}', f'its source {source} (a prevalence, or an outcome not yet defined) is not translated')
        return
    p = mean_of(o['probability']['value'], f'outcomes: {name}.probability', src) if 'probability' in o else None
    kw = {}
    if 'delay' in o:
        kw['delay'] = ss.days(mean_of(o['delay']['value'], f'outcomes: {name}.delay', src))
    if 'duration' in o:
        kw['duration'] = ss.constant(mean_of(o['duration']['value'], f'outcomes: {name}.duration', src), unit='days') # A fixed stay, as in flepiMoP
        notes.append(f'outcomes: {name} has a duration of {kw["duration"]}; the number in it is n_{name} (flepiMoP\'s {o["duration"].get("name", name + "_curr")})')
    src.add_outcome(name, flows, p=p, doc=f'the probability of {name}', **kw)
    return
