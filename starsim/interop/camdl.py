"""
Import a camdl model (a .camdl file, camdl's JSON IR, or a pycamdl model)

camdl (https://github.com/vsbuffalo/camdl) declares compartments, parameters (with
bounds and priors), named expressions, transitions whose rates are total flows
(`S --> I @ beta * S * I / N`), observations with their likelihoods, initial numbers,
and the time to simulate. Its compiler, `camdlc`, expands the model into a JSON
intermediate representation (IR), which is what is read here: a `.camdl` file is
compiled first, with `camdlc` from the PATH (or `CAMDLC`). A pycamdl model, which has
no IR, is written in the same form by `dump_pycamdl.py`.

The IR has every stratification expanded, so a stratified model is imported with one
state for each combination of levels (`S_child_north`), which runs the same as the
original but is not written as `stratify()`. Each transition becomes a transition, its
rate classified by the import rules (`beta * S * I / N` is exposure to `I`); each
observation of a flow becomes an outcome with its likelihood. What is listed as not
translated (scheduled events, ODE compartments, forcing functions of time, and
transitions that are not one state to another) is left for a person, or the
import-model skill, to add.
"""
import os
import json
import keyword
import shutil
import subprocess
import numpy as np
import sciris as sc
import starsim as ss
from .core import Source, build

__all__ = ['load_camdl']

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
    """ The IR as a dict, from a .camdl file, a JSON file, a dict, or a pycamdl model """
    if isinstance(source, dict):
        return source
    elif type(source).__module__.startswith('pycamdl'):
        from . import dump_pycamdl
        return dump_pycamdl.model_dict(source)
    elif str(source).endswith('.camdl'):
        return compile_camdl(source, camdlc=camdlc)
    with open(source) as f:
        return json.load(f)


def expr(node, ir, used):
    """ An expression of the IR in Python syntax; `used` collects the named expressions and time functions it refers to """
    def ex(n):
        return expr(n, ir, used)
    (kind, val), = node.items()
    if kind == 'text': # From dump_pycamdl.py
        return val
    elif kind == 'const':
        return repr(float(val))
    elif kind in ['pop', 'param']:
        return val
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


def prior(spec):
    """ A camdl prior as an `ss.Dist`, or None (a flat prior is the bounds) """
    (name, kw), = spec['dist'].items()
    if name == 'log_normal':
        return ss.lognorm_im(mean=kw['mu'], sigma=kw['sigma'])
    elif name == 'normal':
        return ss.normal(loc=kw.get('mu', kw.get('mean')), scale=kw.get('sigma', kw.get('sd')))
    elif name == 'beta':
        return ss.beta_dist(a=kw['alpha'], b=kw['beta'])
    elif name == 'gamma':
        return ss.gamma(a=kw['shape'], scale=1/kw['rate'])
    elif name == 'truncated_normal':
        return ss.truncnorm(loc=kw['mean'], scale=kw['sd'], low=kw.get('lower', -np.inf), high=kw.get('upper', np.inf))
    elif name == 'half_normal':
        return ss.truncnorm(loc=0, scale=kw['sigma'], low=0)
    elif name == 'exponential':
        return ss.expon(scale=1/kw['rate'])
    elif name == 'uniform':
        return ss.uniform(low=kw.get('lo', kw.get('low')), high=kw.get('hi', kw.get('high')))
    return None


def load_camdl(filename, pars=None, scenario=None, solver='ode', camdlc=None, verbose=None, **kwargs):
    """
    Import a camdl model as a sim

    Args:
        filename (str/dict): a `.camdl` file (compiled with `camdlc`), a JSON file of camdl's IR (or one written by `dump_pycamdl.py`), the IR as a dict, or a pycamdl model
        pars (dict): parameter values, which take the place of the model's (camdl is often given them on the command line, with `--param`)
        scenario (str): a scenario of the model whose parameter values to use (default: the first, if any)
        solver (str): `'ode'` for camdl's `--backend ode`, `'binomial'` for its chain-binomial backend (camdl's default)
        camdlc (str): the path to camdl's compiler, if it is not on the PATH
        verbose (bool): whether to print how each flow was read
        kwargs (dict): passed to `ss.Sim()`

    **Example**:

        sim = ss.load('sir.camdl', pars=dict(beta=0.4, gamma=0.15, rho=0.6, k=10))
    """
    ir = read(filename, camdlc=camdlc)
    m = ir['model']
    file = filename if isinstance(filename, str) else None
    name = m.get('name') if m.get('name') not in [None, 'pycamdl'] else (os.path.splitext(os.path.basename(file))[0] if file else 'camdl')
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
        src.add_flow(out[0] if out else None, into[0] if into else None, rate=rate, per='total', name=ident(tr['name']))
    for item in sorted(used, key=str):
        if isinstance(item, tuple) and item[0] == 'time':
            fn = {f['name']:f for f in m.get('time_functions', [])}.get(item[1], {})
            src.add_par(item[1], 1.0, doc='a function of time in camdl, set to 1')
            src.skip(item[1], f'a function of time ({fn.get("kind")}), set to 1; write it as an ss.seasonal() or ss.timeseries() value of the parameter')
    for eq in m.get('ode_equations', []):
        src.skip(eq['compartment'], 'a compartment given by an ODE; write it as a state that is not people (ss.State(people=False)) with flows against ss.null')

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
            src.skip(f'{oname} (likelihood)', 'an observation of a number in states, imported as a result; give its likelihood when calibrating (sim.calibrate(likelihood=...))')
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
                src.skip(oname, f'its mean ({mean}) is not a share of the flow')
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
            likelihood = lambda ps, sd=sd: ss.normal(scale=sd(ps))
        elif lkind is not None:
            src.skip(f'{oname} (likelihood)', f'the {lkind} likelihood has no translation')

        sched = ob.get('emit_schedule', {}).get('regular')
        covers = ob.get('covers', {})
        total = sched and covers.get('kind') == 'until' and covers.get('span') == sched['step'] # A total over each period, which data by period is compared with automatically
        if (sched or covers) and not total:
            src.skip(f'{oname} (period)', f'reported {covers or sched}, which is not a total over each period')
        src.add_outcome(oname, pval, p=p, likelihood=likelihood)

    # Scheduled effects
    enabled = scens[chosen].get('enable', []) if scens else []
    for iv in m.get('interventions', []):
        acts = sc.strjoin(list(a)[0] for a in iv['actions'])
        when = iv['fire']
        names = {iv['name'], iv.get('base_name')}
        if iv.get('kind') != 'event' and not names & set(enabled): # An intervention, unlike an event, is off unless enabled: listed all the same, since a scenario that enables it is the model's question
            where = [k for k,scen in scens.items() if names & set(scen.get('enable', []))]
            src.skip(iv['name'], f'an intervention ({acts}, {when}) that is off unless enabled{", as scenario " + sc.strjoin(where) + " does" if where else ""}; '
                                 f'write it as a one-off move, ss.Delivery(move=S >> V, ...), with its times')
            continue
        src.skip(iv['name'], f'a scheduled effect ({acts}, {when}); write it as a one-off move, ss.Delivery(move=S >> V, ...), with its times')

    # The run
    simulation = m.get('simulation', {})
    tdur = getattr(ss, f'{unit}s')
    cfg = dict(unit=unit, solver=solver)
    if solver in ['ode', 'binomial']: # camdl's ODE backend takes one RK4 step a step, and its chain binomial whole steps, so the numbers are its own
        cfg['substeps'] = 1
    if simulation.get('t_end') is not None:
        cfg['stop'] = tdur(simulation['t_end'])
    if simulation.get('t_start'):
        cfg['start'] = tdur(simulation['t_start'])
    if simulation.get('dt'):
        cfg['dt'] = tdur(simulation['dt'])
    src.config.update(cfg)
    sim = build(src, verbose=verbose, **kwargs)
    sim.imported.scenarios = {k:s.get('params', {}) for k,s in scens.items()} # The parameter values of each; the interventions each enables are listed in sim.imported.incomplete
    return sim
