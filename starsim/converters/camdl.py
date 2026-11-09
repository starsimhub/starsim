"""
Import a camdl model (a .camdl file, or camdl's JSON IR), and export a sim as one

camdl (https://github.com/vsbuffalo/camdl) declares compartments, parameters (with
bounds and priors), named expressions, transitions whose rates are total flows
(`S --> I @ beta * S * I / N`), observations with their likelihoods, initial numbers,
and the time to simulate. Its compiler, `camdlc`, expands the model into a JSON
intermediate representation (IR), which is what is read here: a `.camdl` file is
compiled first, with `camdlc` from the PATH (or `CAMDLC`).

The IR has every stratification expanded, so a stratified model is imported with one
state for each combination of levels (`S_child_north`), which runs the same as the
original but is not written as `stratify()`. Each transition becomes a transition, its
rate classified by the import rules (`beta * S * I / N` is exposure to `I`); each
observation of a flow becomes an outcome with its likelihood; an event that adds people
at fixed times becomes an inflow from outside the population at those times. What is
listed as not translated (other scheduled effects, ODE compartments, forcing functions of
time, transitions that are not one state to another, extra noise in a rate, a balance,
and any other part of the IR not read) is left for a person, or the convert-model skill,
to add.

`to_camdl()` (`ss.export_to(sim, 'camdl')`) is the reverse: it writes the compartmental
form of a sim (`ss.converters.export.flatten()`) as a .camdl file, with its compartments,
parameters (kinds from their Starsim types, and bounds, priors, and docs), named
expressions as `let`, transitions with their total rates, initial numbers, the time to
simulate, and the parameters' values in a scenario, `baseline`, as camdl keeps them out
of the model itself. A rate that varies in time is carried (in camdl's `t`); what is
stratified is summed into one group, and results and outcomes are not carried.

Used with: compartments; declared modules
"""
import os
import ast
import json
import keyword
import shutil
import subprocess
import numpy as np
import sciris as sc
import starsim as ss
from .core import Source, build

__all__ = ['load_camdl', 'to_camdl']

ops = dict(add='+', sub='-', mul='*', div='/', pow='**', gt='>', lt='<', ge='>=', le='<=')
funcs = dict(exp='exp', log='log', sqrt='sqrt', abs='abs', sin='sin', cos='cos')
units = dict(days='day', day='day', weeks='week', week='week', months='month', month='month', years='year', year='year')


def ident(name):
    """ A transition's name as a Python name (camdl allows e.g. `import`) """
    return f'{name}_' if keyword.iskeyword(name) else name


class Untranslated(Exception):
    """ Part of an expression that has no translation """


def compile_camdl(filename, camdlc=None):
    """ The IR of a .camdl file, from camdl's compiler """
    camdlc = camdlc or os.environ.get('CAMDLC') or shutil.which('camdlc')
    if not camdlc:
        errormsg = f'Reading "{filename}" needs camdl\'s compiler, camdlc, which was not found: put it on the PATH, give camdlc=<path> or set CAMDLC, or write the IR with "camdlc {os.path.basename(filename)} --no-state-grad > model.json" and load that'
        raise FileNotFoundError(errormsg)
    out = subprocess.run([camdlc, str(filename), '--no-state-grad'], capture_output=True, text=True)
    if out.returncode:
        errormsg = f'camdlc could not compile "{filename}":\n{out.stderr.strip()}'
        raise ValueError(errormsg)
    return json.loads(out.stdout)


def read(source, camdlc=None):
    """ The IR as a dict, from a .camdl file, a JSON file, or a dict """
    if isinstance(source, dict):
        return source
    elif str(source).endswith('.camdl'):
        return compile_camdl(source, camdlc=camdlc)
    with open(source) as f:
        return json.load(f)


def expr(node, ir, used):
    """ An expression of the IR in Python syntax; `used` collects the named expressions and time functions it refers to """
    def ex(n):
        return expr(n, ir, used)
    (kind, val), = node.items()
    if kind == 'const':
        return repr(float(val))
    elif kind in ['pop', 'param']:
        return val
    elif kind == 'const_ref':
        return repr(float({c['name']:c['value'] for c in ir.get('constants', [])}[val]))
    elif kind in ['binding_ref', 'per_eval_ref']:
        used.add(val)
        return val
    elif kind in ['pop_sum', 'reduce']:
        parts = [p if isinstance(p, str) else ex(p) for p in val]
        return '(' + ' + '.join(parts) + ')' if parts else '0.0'
    elif kind == 'bin_op':
        if val['op'] not in ops:
            raise Untranslated(f'the operator {val["op"]}')
        return f'({ex(val["left"])} {ops[val["op"]]} {ex(val["right"])})'
    elif kind == 'un_op':
        if val['op'] == 'neg':
            return f'(-{ex(val["arg"])})'
        elif val['op'] in funcs:
            return f'{funcs[val["op"]]}({ex(val["arg"])})'
        raise Untranslated(f'the function {val["op"]}')
    elif kind == 'unchecked_dim':
        return ex(val['inner'])
    elif kind == 'cond': # Usually a guard against dividing by zero, e.g. N > 0 ? I/N : 0
        other = val.get('else', val.get('otherwise'))
        pred = val['pred'].get('bin_op', {})
        if other == dict(const=0.0) and pred.get('op') == 'gt' and pred.get('right') == dict(const=0.0):
            return f'({ex(val["pred"])} * {ex(guarded(val["then"], pred["left"]))})' # A division by what is 0 when the condition is false is kept from giving 0/0
        raise Untranslated('a condition other than a guard (x > 0 ? ... : 0)')
    elif kind == 'safe':
        return f'max({ex(val)}, 1e-300)'
    elif kind == 'table_lookup':
        table = {t['name']:t for t in ir.get('tables', [])}[val['table']]
        idx = val['indices']
        if len(idx) == 1 and 'const' in idx[0]:
            return ex(table['values'][int(idx[0]['const'])])
        raise Untranslated(f'a lookup in the table {val["table"]}')
    elif kind == 'time_func':
        used.add(('time', val['name']))
        return val['name']
    elif kind == 'untranslated':
        raise Untranslated(val)
    raise Untranslated(f'an expression of kind "{kind}"')


def guarded(node, x):
    """ The expression with each division by x made a division by max(x, 1e-300) """
    if isinstance(node, list):
        return [guarded(n, x) for n in node]
    elif not isinstance(node, dict):
        return node
    if node.get('bin_op', {}).get('op') == 'div' and node['bin_op']['right'] == x:
        return {'bin_op': dict(op='div', left=guarded(node['bin_op']['left'], x), right={'safe': x})}
    return {k:guarded(v, x) for k,v in node.items()}


def size(spec):
    """ What kind a part of the IR is, and how big, without its data (e.g. "interpolated, 312 values") """
    if not isinstance(spec, dict) or len(spec) != 1:
        return str(spec)
    (kind, val), = spec.items()
    n = max([len(v) for v in val.values() if isinstance(v, list)] if isinstance(val, dict) else [], default=0)
    return f'{kind}, {n} values' if n else kind


def fire_times(fire, t_end):
    """ The times at which camdl fires a scheduled event or intervention, as its runtime works them out; None if they are not fixed """
    (kind, spec), = fire.get('scheduled', {None: None}).items()
    if kind == 'at_times':
        return list(spec)
    elif kind != 'recurring':
        return None # At times given by expressions, or when a condition holds
    end = spec['end'] if spec.get('end') is not None else t_end
    if end is None:
        return None
    period, at_day = spec['period'], spec.get('at_day')
    t = spec['start'] if at_day is None else at_day + max(0, np.ceil((spec['start'] - at_day)/period))*period
    times = []
    while t <= end + period*1e-9:
        times.append(float(t))
        t += period
    return times


def when(fire, unit):
    """ When a scheduled effect fires, in words """
    (kind, spec), = (fire.get('scheduled') or fire).items()
    if kind == 'at_times':
        return f'at {sc.strjoin(f"{t:g}" for t in spec)} {unit}s' if len(spec) <= 5 else f'at {len(spec)} times'
    elif kind == 'recurring':
        start = spec['start'] if spec.get('at_day') is None else spec['at_day']
        return f'every {spec["period"]:g} {unit}s from {start:g}'
    return 'when a condition holds' if kind == 'reactive' else 'at times given by expressions'


def prior(spec):
    """ A camdl prior as an `ss.Dist`, or None (a flat prior is the bounds) """
    (name, kw), = spec['dist'].items()
    if name == 'log_normal':
        return ss.lognormal(log_mean=kw['mu'], log_std=kw['sigma'])
    elif name == 'normal':
        return ss.normal(mean=kw.get('mu', kw.get('mean')), std=kw.get('sigma', kw.get('sd')))
    elif name == 'beta':
        return ss.beta_dist(a=kw['alpha'], b=kw['beta'])
    elif name == 'gamma':
        return ss.gamma(shape=kw['shape'], scale=1/kw['rate'])
    elif name == 'truncated_normal':
        return ss.truncnorm(mean=kw['mean'], std=kw['sd'], low=kw.get('lower', -np.inf), high=kw.get('upper', np.inf))
    elif name == 'half_normal':
        return ss.truncnorm(mean=0, std=kw['sigma'], low=0)
    elif name == 'exponential':
        return ss.expon(mean=1/kw['rate'])
    elif name == 'uniform':
        return ss.uniform(low=kw.get('lo', kw.get('low')), high=kw.get('hi', kw.get('high')))
    return None


def load_camdl(filename, pars=None, scenario=None, solver='ode', camdlc=None, verbose=None, **kwargs):
    """
    Import a camdl model as a sim

    Args:
        filename (str/dict): a `.camdl` file (compiled with `camdlc`), a JSON file of camdl's IR, or the IR as a dict
        pars (dict): parameter values, which take the place of the model's (camdl is often given them on the command line, with `--param`)
        scenario (str): a scenario of the model whose parameter values to use (default: the first, if any)
        solver (str): `'ode'` for camdl's `--backend ode`, `'binomial'` for its chain-binomial backend (camdl's default)
        camdlc (str): the path to camdl's compiler, if it is not on the PATH
        verbose (bool): whether to print how each flow was read
        kwargs (dict): passed to `ss.Sim()`

    Example:
        ```python
        sim = ss.import_from('sir.camdl', pars=dict(beta=0.4, gamma=0.15, rho=0.6, k=10))
        ```
    """
    ir = read(filename, camdlc=camdlc)
    m = ir['model']
    file = filename if isinstance(filename, str) else None
    name = m.get('name') if m.get('name') is not None else (os.path.splitext(os.path.basename(file))[0] if file else 'camdl')
    src = Source(name, 'camdl', file)
    pars = dict(pars or {})
    docs = ir.get('docs', {}).get('parameters', {})
    unit = units.get(m.get('time_unit', 'days'), 'day')
    used = set()

    # Scenarios: the values of the one chosen fill in what pars does not
    scens = {s['name']:s for s in m.get('scenarios', [])}
    if scens:
        chosen = scenario or list(scens)[0]
        if chosen not in scens:
            errormsg = f'The scenario "{chosen}" is not in the model; choices are {list(scens)}'
            raise ValueError(errormsg)
        pars = {**scens[chosen].get('params', {}), **pars}
        src.notes.append(f'the parameter values of scenario "{chosen}" (of {sc.strjoin(scens)})')
        for key in ['scale', 'compose', 't_end_anchor']:
            if scens[chosen].get(key):
                src.skip(f'{chosen} ({key})', f'the scenario\'s {key} ({scens[chosen][key]}) has no translation')
    elif scenario is not None:
        errormsg = f'The model has no scenarios, so scenario="{scenario}" cannot be chosen'
        raise ValueError(errormsg)

    # Compartments, all of people (a stratified model spelled out may have groups that no transition connects)
    src.declared = True
    for c in m['compartments']:
        src.add_state(c['name'])
    dims = m.get('model_structure', {}).get('dimensions', [])
    if dims:
        desc = sc.strjoin(f'{d["name"]} {d["values"]}' for d in dims)
        src.notes.append(f'stratified by {desc} in camdl, which the IR spells out: imported with one state for each combination of levels ({m["compartments"][0]["name"]}, ...)')

    # Parameters
    missing = []
    for p in m.get('parameters', []):
        pname = p['name']
        v = p['value']
        doc = p.get('doc') or docs.get(pname, {}).get('text')
        bounds = v.get('bounds')
        value = pars.pop(pname, v.get('value'))
        if value is None and bounds is None:
            missing.append(pname)
            continue
        if value is None:
            src.notes.append(f'{pname} has no value, so it starts from the middle of its bounds {bounds}')
        flat = v.get('prior') in [None, 'flat']
        pr = None if flat else prior(v['prior'])
        if not flat and pr is None:
            src.skip(f'{pname} (prior)', f'the prior {v["prior"]} has no translation')
        src.add_par(pname, value, doc=doc, bounds=bounds, prior=pr)
    if missing:
        errormsg = f'The parameter(s) {sc.strjoin(missing)} have no value in the model; give them with pars=dict(...), as camdl is given them with --param'
        raise ValueError(errormsg)
    if pars:
        errormsg = f'Unknown parameter(s) {sc.strjoin(pars)}; choices are {sc.strjoin(src.pars)}'
        raise ValueError(errormsg)

    # Named expressions (let), which the IR also makes when it takes a common part out of several rates
    for b in m.get('bindings', []) + m.get('per_eval_bindings', []):
        try:
            src.exprs[b['name']] = expr(b['expr'], m, used)
        except Untranslated as E:
            src.skip(b['name'], f'the expression has {E}')

    # Transitions
    for tr in m['transitions']:
        stoich = tr['stoichiometry']
        out = [c for c,d in stoich if d < 0]
        into = [c for c,d in stoich if d > 0]
        if len(stoich) > 2 or len(out) > 1 or len(into) > 1 or any(abs(d) != 1 for c,d in stoich):
            src.skip(tr['name'], f'it is not a move from one state to another ({stoich}); split it into transitions, or write a rate with counts')
            continue
        try:
            rate = expr(tr['rate'], m, used)
        except Untranslated as E:
            src.skip(tr['name'], f'its rate has {E}')
            continue
        draw = tr.get('draw_method') or 'poisson' # How camdl's stochastic backends draw the flow
        noise = None
        if isinstance(draw, dict) and 'overdispersed' in draw: # Gamma white noise whose argument is sigma squared per unit of time (shape dt/sigma²), where ss.Transition(noise=) takes sigma, as a number or a parameter
            var = draw['overdispersed']
            if 'const' in var:
                noise = float(var['const'])**0.5
            elif 'param' in var and var['param'] in src.pars:
                p = src.pars[var['param']]
                noise = f'sqrt_{var["param"]}'
                if noise not in src.pars:
                    root = lambda v: None if v is None else float(v)**0.5
                    src.add_par(noise, root(p.value), doc=f'the square root of {var["param"]}: sigma of the noise on a rate, where camdl\'s overdispersed() takes sigma squared', bounds=None if p.bounds is None else [root(b) for b in p.bounds])
                    src.notes.append(f'{var["param"]} is sigma squared of the noise on {tr["name"]} in camdl, and noise= takes sigma, so it is fitted as {noise}, its square root' + (' (its prior is not carried over)' if p.prior is not None else ''))
            else:
                src.skip(f'{tr["name"]} (noise)', 'camdl draws it with extra noise in its rate (overdispersed), whose variance is an expression; noise= takes sigma as a number or a parameter, so the rate is used without it')
        src.add_flow(out[0] if out else None, into[0] if into else None, rate=rate, per='total', name=ident(tr['name']), noise=noise)
        if draw == 'deterministic' and solver != 'ode':
            src.skip(f'{tr["name"]} (deterministic)', 'camdl adds it as it is (deterministic), but a stochastic solver draws it at random like any other flow; under ss.solvers.ode() it is the same')
        elif draw not in ['poisson', 'deterministic'] and not (isinstance(draw, dict) and 'overdispersed' in draw):
            src.skip(f'{tr["name"]} (draw)', f'camdl draws it as {size(draw) if isinstance(draw, dict) else draw}, which has no translation')
        for key in sorted(set(tr) - {'name', 'stoichiometry', 'rate', 'metadata', 'draw_method', 'rate_grad', 'rate_state_grad'}): # E.g. lineage
            if tr[key]:
                src.skip(f'{tr["name"]} ({key})', f'camdl\'s {key} of the transition has no translation')

    # Scheduled effects: adding people is an inflow from outside the population at each time it fires; the rest are listed
    simulation = m.get('simulation', {})
    dt = simulation.get('dt') or 1.0
    enabled = scens[chosen].get('enable', []) if scens else []
    disabled = scens[chosen].get('disable', []) if scens else []
    for iv in m.get('interventions', []):
        names = {iv['name'], iv.get('base_name')}
        acts = sc.strjoin(list(a)[0] for a in iv['actions'])
        fires = when(iv['fire'], unit)
        if names & set(disabled):
            src.notes.append(f'{iv["name"]} is off in scenario "{chosen}"')
            continue
        if iv.get('kind') != 'event' and not names & set(enabled): # An intervention, unlike an event, is off unless enabled: listed all the same, since a scenario that enables it is the model's question
            where = [k for k,scen in scens.items() if names & set(scen.get('enable', []))]
            src.skip(iv['name'], f'an intervention ({acts}, {fires}) that is off unless enabled{", as scenario " + sc.strjoin(where) + " does" if where else ""}')
            continue
        times = fire_times(iv['fire'], simulation.get('t_end'))
        adds = [a['add'] for a in iv['actions'] if 'add' in a]
        for a in iv['actions']:
            (akind, act), = a.items()
            name = ident(iv['name']) if len(adds) == 1 else f'{ident(iv["name"])}_{act.get("compartment", akind)}'
            if akind == 'add' and times is not None and np.all(np.diff(times) >= dt):
                try:
                    count = expr(act['count'], m, used)
                except Untranslated as E:
                    src.skip(name, f'adds people to {act["compartment"]} {fires}, a number with {E}; write it as an inflow from outside the population, ss.unborn >> {act["compartment"]} @ ss.per{unit}(ss.timeseries(times, numbers, interpolation=\'step\')), with each number divided by the step at its time and 0 a step later')
                    continue
                t0 = simulation.get('t_start') or 0.0
                pulse = {getattr(ss, f'{unit}s')(0): 0.0} # Before the first time, since a series holds its first value from the start; keyed by the time after the start
                pulse.update({getattr(ss, f'{unit}s')(k - t0):v for t in times for k,v in [(t, 1/dt), (t + dt, 0.0)]})
                src.add_par(f'{name}_pulse', ss.timeseries(pulse), doc=f'1/dt in the step of each time camdl fires {iv["name"]} ({fires}), and 0 otherwise')
                src.add_flow(None, act['compartment'], rate=f'({count}) * {name}_pulse', per='total', name=name)
                if solver != 'ode':
                    src.skip(f'{name} (deterministic)', f'camdl adds the number to {act["compartment"]} as it is ({fires}); imported as an inflow, which a stochastic solver draws at random')
                continue
            elif akind == 'add':
                advice = f'an inflow from outside the population, ss.unborn >> {act["compartment"]} @ ss.per{unit}(ss.timeseries(times, numbers, interpolation=\'step\')), with each number divided by the step at its time and 0 a step later'
            elif akind in ['fraction_transfer', 'absolute_transfer']:
                advice = f'a move between states, ss.Event({act["src"]} >> {act["dst"]}, when=[...], {"p" if akind == "fraction_transfer" else "n"}=...)'
            else:
                advice = 'an intervention (ss.Event, or custom code)'
            src.skip(name if akind == 'add' else iv['name'], f'a scheduled effect ({akind}, {fires}); write it as {advice}')

    for item in sorted(used, key=str):
        if isinstance(item, tuple) and item[0] == 'time':
            fn = {f['name']:f for f in m.get('time_functions', [])}.get(item[1], {})
            src.add_par(item[1], 1.0, doc='a function of time in camdl, set to 1')
            src.skip(item[1], f'a function of time ({size(fn.get("kind"))}), set to 1; write it as an ss.seasonal() or ss.timeseries() value of the parameter')
    for eq in m.get('ode_equations', []):
        src.skip(eq['compartment'], 'a compartment given by an ODE; write it as a state that is not people (ss.State(people=False)) with flows against ss.dead')

    # Initial numbers, which may refer to the parameters, to each other, and to named expressions
    values = {k:(p.value if p.value is not None else sum(p.bounds)/2) for k,p in src.pars.items()}
    texts = {}
    for cname,spec in m.get('initial_conditions', {}).items():
        (kind, val), = spec.items()
        if kind == 'deterministic':
            texts[cname] = expr(val, m, set())
            continue
        (dist, kw), = val.items() # A number drawn from a distribution: its mean
        mean = kw.get('rate', kw.get('mean'))
        if mean is not None:
            texts[cname] = expr(mean.get('expr', mean), m, set())
        elif 'n' in kw and 'p' in kw:
            texts[cname] = f'({expr(kw["n"], m, set())}) * ({expr(kw["p"].get("expr", kw["p"]), m, set())})'
        else:
            src.skip(f'{cname} (initial number)', f'drawn from a {dist} distribution in camdl, with no mean to use')
            continue
        src.skip(f'{cname} (initial number)', f'drawn from a {dist} distribution in camdl; the mean is used')
    ns = {'__builtins__': {}}
    known = {**values, **{k:0.0 for k in src.states if k not in texts}} # A state with no initial number has none
    for i in range(len(texts) + 1): # Until nothing more can be worked out
        for cname,text in list(texts.items()):
            scope = dict(known)
            for k,e in src.exprs.items():
                try:
                    scope[k] = eval(e, ns, scope) # Only numbers and the names of parameters, states, and expressions
                except Exception:
                    pass
            try:
                known[cname] = float(eval(text, ns, scope))
                texts.pop(cname)
            except Exception:
                pass
    for cname,text in texts.items():
        src.skip(f'{cname} (initial number)', f'"{text}" could not be worked out')
    for cname in src.states:
        src.states[cname].init = known.get(cname) or None

    # Observations
    for ob in m.get('observations', []):
        oname = ob['name']
        for key in sorted(set(ob) - {'name', 'source', 'columns', 'scored', 'stratum', 'emit_schedule', 'covers', 'projection', 'projection_state_grad', 'likelihood', 'untranslated'}):
            if ob[key]:
                src.skip(f'{oname} ({key})', f'camdl\'s {key} of the observation has no translation')
        if 'untranslated' in ob:
            src.skip(oname, ob['untranslated'])
            continue
        proj = ob.get('projection', {})
        (pkind, pval), = proj.items() if proj else [(None, None)]
        if pkind in ['cumulative_flow', 'cumulative_flow_sum']:
            pval = [ident(f) for f in pval] if isinstance(pval, list) else ident(pval)
            lost = [f for f in sc.tolist(pval) if f not in [flow.name for flow in src.flows]]
            if lost:
                src.skip(oname, f'an observation of {sc.strjoin(lost)}, which was not translated')
                continue
        elif pkind in ['current_pop', 'derived_expr']:
            src.results[oname] = pval if pkind == 'current_pop' else expr(pval, m, set())
            src.skip(f'{oname} (likelihood)', 'an observation of a number in states, imported as a result; give its likelihood when calibrating (sim.fit(likelihood=...))')
            continue
        else:
            src.skip(oname, f'an observation of {pkind}, which has no translation')
            continue

        # The mean as a share of the projected flow, and the likelihood
        lik = ob.get('likelihood') or {}
        (lkind, lkw), = lik.items() if lik else [(None, {})]
        p = None
        mean = lkw.get('mean', lkw.get('rate')) # A Poisson's mean is its rate
        if mean:
            mean = mean['expr']
            if mean == {'projected': None}:
                p = None
            elif mean.get('bin_op', {}).get('op') == 'mul':
                sides = [mean['bin_op']['left'], mean['bin_op']['right']]
                rest = [s for s in sides if s != {'projected': None}]
                if len(rest) == 1 and 'param' in rest[0]:
                    p = rest[0]['param']
                elif len(rest) == 1 and 'const' in rest[0]:
                    p = rest[0]['const']
            if p is None and mean != {'projected': None}:
                src.skip(oname, 'its mean is not a share of the flow')
                continue
        def arg(spec):
            e = spec['expr']
            return (lambda ps: ps[e['param']]) if 'param' in e else (lambda ps: e['const']) if 'const' in e else None
        likelihood = None
        if lkind == 'neg_binomial' and arg(lkw['dispersion']):
            n = arg(lkw['dispersion'])
            likelihood = lambda ps, n=n: ss.nbinom(n=n(ps))
        elif lkind == 'poisson':
            likelihood = lambda ps: ss.poisson()
        elif lkind == 'normal' and arg(lkw.get('sd', lkw.get('sigma', {'expr': {}}))):
            sd = arg(lkw.get('sd', lkw.get('sigma')))
            likelihood = lambda ps, sd=sd: ss.normal(std=sd(ps))
        elif lkind is not None:
            src.skip(f'{oname} (likelihood)', f'the {lkind} likelihood has no translation')

        sched = ob.get('emit_schedule', {}).get('regular')
        covers = ob.get('covers', {})
        span = covers.get('span')
        total = sched and covers.get('kind') in ['from', 'until'] and span == sched['step'] # A total over each period, which data by period is compared with automatically
        if (sched or covers) and not total:
            src.skip(f'{oname} (period)', f'reported {covers or sched}, which is not a total over each period')
        shift = (covers.get('offset') or 0) - (span if covers.get('kind') == 'until' else 0) # From the time a row of data is labeled by to the start of its period
        if total and shift:
            src.skip(f'{oname} (data times)', f'camdl labels each row of its data {-shift:g} {unit}s after the start of the period it covers ({covers["kind"]}), and Starsim by the start: '
                                              f'{"subtract" if shift < 0 else "add"} {abs(shift):g} {unit}s {"from" if shift < 0 else "to"} the times of the data')
        src.add_outcome(oname, pval, p=p, likelihood=likelihood)

    # What else the model has, which is not translated: keeping a state equal to an expression, and any part not read above
    if m.get('balance'):
        bal = m['balance']
        try:
            text = expr(bal['expr'], m, set())
        except Untranslated:
            text = 'an expression'
        src.skip(f'{bal["target"]} (balance)', f'camdl sets {bal["target"]} = {text} after each step (balance), which has no translation; {bal["target"]} follows its own flows')
    known = {'name', 'version', 'time_unit', 'description', 'origin', 'origin_rata_die', 'compartments', 'transitions', 'ode_equations', 'time_functions', 'tables', 'constants', 'interventions',
            'observations', 'parameters', 'initial_conditions', 'ic_grad', 'output', 'simulation', 'scenarios', 'model_structure', 'balance', 'bindings', 'per_eval_bindings'} # Read, or not part of the model (gradients, output)
    for key in sorted(set(m) - known): # E.g. quantities, contrasts, identity_tracked_compartments
        if m[key]:
            what = sc.strjoin(x.get('name', '?') if isinstance(x, dict) else x for x in m[key]) if isinstance(m[key], list) else size(m[key]) if isinstance(m[key], dict) else m[key]
            src.skip(key, f'camdl\'s {key} ({what}): not translated')

    # The run
    tdur = getattr(ss, f'{unit}s')
    cfg = dict(unit=unit, solver=solver)
    if solver in ['ode', 'binomial']: # camdl's ODE backend takes one RK4 step a step, and its chain binomial whole steps, so the numbers are its own
        cfg['solver'] = getattr(ss.solvers, solver)(substeps=1)
    if simulation.get('t_end') is not None:
        cfg['stop'] = tdur(simulation['t_end'])
    if simulation.get('t_start'):
        cfg['start'] = tdur(simulation['t_start'])
    if simulation.get('dt'):
        cfg['dt'] = tdur(simulation['dt'])
    if scens and scens[chosen].get('t_end') is not None:
        cfg['stop'] = tdur(scens[chosen]['t_end'])
    if simulation.get('rng_seed') is not None:
        cfg['rand_seed'] = simulation['rng_seed']
    if 'stop' not in cfg and simulation.get('t_end_anchor'):
        src.skip('simulate (to)', f'camdl ends the run at a time set by the data ({simulation["t_end_anchor"]}); give stop=')
    method = (simulation.get('integrator') or {}).get('method', 'rk4')
    if solver == 'ode' and method != 'rk4':
        src.skip('simulate (integrator)', f'camdl solves its ODE with {method}, and the import takes one RK4 step a step; give ss.solvers.ode() more substeps= to come closer')
    if simulation.get('time_semantics', 'continuous') != 'continuous':
        src.skip('simulate (time)', f'camdl\'s time is {simulation["time_semantics"]}, which has no translation')
    src.config.update(cfg)
    sim = build(src, verbose=verbose, **kwargs)
    sim.imported.scenarios = {k:s.get('params', {}) for k,s in scens.items()} # The parameter values of each; the interventions each enables are listed in sim.imported.incomplete
    return sim


#%% Export

reserved = {'t', 't_start', 't_end', 'dt', 'pi', 'e', 'rate', 'probability', 'positive', 'count', 'real', 'integer', 'instant', 'duration', 'sum', 'where', 'let', 'if', 'then', 'else',
            'and', 'or', 'not', 'in', 'by', 'from', 'to', 'every', 'until', 'at', 'when', 'action', 'origin', 'init', 'simulate', 'scenarios', 'via', 'covers'} # Names camdl keeps for itself
camdl_funcs = dict(minimum='min', maximum='max', absolute='abs', ceiling='ceil')
camdl_units = dict(day='days', week='weeks', month='months', year='years')
priors = dict( # Each Starsim prior as camdl's, the reverse of prior()
    lognormal  = lambda p: 'log_normal(mu = {!r}, sigma = {!r})'.format(*((p.log_mean, p.log_std) if 'log_mean' in p else map(float, ss.lognormal.log_moments(p.mean, p.std)))),
    normal     = lambda p: f'normal(mu = {p.mean!r}, sigma = {p.std!r})',
    beta_dist  = lambda p: f'beta(alpha = {p.a!r}, beta = {p.b!r})',
    gamma      = lambda p: f'gamma(shape = {p.shape!r}, rate = {1/p.scale!r})',
    truncnorm  = lambda p: f'truncated_normal(mean = {p.mean!r}, sd = {p.std!r})',
    expon      = lambda p: f'exponential(rate = {1/p.mean!r})',
    uniform    = lambda p: f'uniform(lower = {p.low!r}, upper = {p.high!r})',
)


def camdl_name(name):
    """ A name that camdl keeps for itself, with an underscore added """
    return f'{name}_' if name in reserved else name


def camdl_expr(text, unit):
    """ An expression in Python syntax, as `ss.converters.export.flatten()` writes it, in camdl's syntax (with the time `t` a number in the unit) """
    class Sub(ast.NodeTransformer):
        def visit_Call(self, n):
            self.generic_visit(n)
            if n.func.id == 'where': # camdl's if, written whole as a name, since Python's expressions have no such form
                cond, yes, no = [ast.unparse(a) for a in n.args]
                return ast.Name(f'(if {cond} then {yes} else {no})')
            n.func.id = camdl_funcs.get(n.func.id, n.func.id)
            return n
        def visit_Name(self, n):
            return ast.Name(f"(t * 1 'per_{unit[:-1]})") if n.id == 't' else n if n.id == 'pi' else ast.Name(camdl_name(n.id)) # camdl's t is a time, not a number
    return ast.unparse(Sub().visit(ast.parse(text, mode='eval').body)).replace('**', '^')


def par_decl(name, p, lost):
    """ The declaration of a parameter: its kind (from its Starsim type), bounds, and prior """
    raw = getattr(p, 'raw', None)
    kind = 'rate' if isinstance(raw, ss.per) else 'duration' if isinstance(raw, (ss.Duration, ss.Dist)) else 'real'
    out = f'{camdl_name(name)} : {kind}'
    prior = getattr(p, 'prior', None)
    bounds = getattr(p, 'bounds', None)
    if bounds is None and isinstance(prior, ss.truncnorm): # camdl's truncated normal takes its bounds from the parameter's
        bounds = [prior.pars.low, prior.pars.high]
    if bounds is not None and np.isfinite(bounds).all():
        out += f' in [{float(bounds[0])!r}, {float(bounds[1])!r}]'
    if prior is not None:
        try:
            out += f' ~ {priors[type(prior).__name__](sc.objdict({k:float(v) for k,v in prior.pars.items()}))}'
        except (KeyError, TypeError, ValueError):
            lost.append(f'not exported: the prior of "{name}" ({prior}), which camdl has no form for')
    return out


def to_camdl(sim):
    """
    The compartmental form of a sim as a camdl model (a .camdl file): compartments, parameters, named expressions (let), transitions, initial numbers, and the time to simulate

    The parameters' values are in a scenario, `baseline` (camdl keeps values out of its
    model, except in a scenario), so it is run with
    `camdl simulate model.camdl --scenario baseline --backend ode`; their bounds and
    priors, if any, are carried as camdl's.

    Args:
        sim (`ss.Sim`): the sim
    """
    sim = ss.converters.export.initialized(sim) # Once, for each of these
    flat = ss.converters.export.flatten(sim)
    times = ss.converters.export.timing(sim)
    lost = list(flat.lost)
    pars = ss.converters.export.declared_pars(sim)
    unit = camdl_units[flat.unit]
    e = lambda text: camdl_expr(text, unit)
    lines = [f"#' {flat.name}, exported from Starsim {ss.__version__} (tested with {ss.converters.core.tested_with("camdl")})", '#  Run with: camdl simulate <this file> --scenario baseline --backend ode', '', f"time_unit = '{unit}", '']
    n = camdl_name
    lines += ['compartments { ' + ', '.join(n(c) for c in flat.comps) + ' }', '', 'parameters {']
    for name in flat.pars:
        p = pars.get(name)
        doc = getattr(p, 'doc', None)
        if doc:
            lines.append(f"  #' {' '.join(str(doc).split())}")
        lines.append(f'  {par_decl(name, p, lost)}')
    lines += ['}', '']
    lines += [f'let {n(name)} = {e(text)}' for name,text in flat.exprs.items()]
    lines += ['', 'transitions {']
    names = lambda text: {n.id for n in ast.walk(ast.parse(text, mode='eval')) if isinstance(n, ast.Name)}
    with_pars = set(flat.pars) | {name for name,text in flat.exprs.items() if names(text) & set(flat.pars)}
    for flow in flat.flows:
        rate = e(flow.rate)
        if not names(flow.rate) & with_pars: # A rate of numbers and counts alone has no time in it for camdl's check of dimensions to find
            rate = f'unchecked_dim({rate}, dim = population_rate, reason = "a rate in numbers per {flat.unit}, from Starsim")'
        src = '' if flow.src is None else f'{n(flow.src)} '
        dst = '' if flow.dst is None else f' {n(flow.dst)}'
        lines.append(f'  {n(flow.name)} : {src}-->{dst} @ {rate}')
    lines += ['}', '', 'init {']
    lines += [f'  {n(name)} = {init:.15g}' for name,init in flat.comps.items() if init]
    lines += ['}', '', 'simulate {', f"  from = {times.start:.15g} '{unit}", f"  to   = {times.stop:.15g} '{unit}"]
    if times.dt != 1:
        lines.append(f'  dt   = {times.dt:.15g}')
    lines.append('}')
    values = [f'      {n(name)} = {value:.15g}' for name,value in flat.pars.items() if np.isfinite(value)]
    if values:
        lines += ['', 'scenarios {', '  baseline {', '    set = {', *values, '    }', '  }', '}']
    if times.date is not None:
        lost.append(f'changed: the dates (from {times.date}) are written as times from 0, in {flat.unit}s')
    summary = f'exported to camdl: {flat.n_states} states, {flat.n_transitions} transitions, as {len(flat.comps)} compartments and {len(flat.flows)} transitions (time in {flat.unit}s)'
    return '\n'.join(lines) + '\n', summary, lost
