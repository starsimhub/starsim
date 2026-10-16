"""
Import an EMULSION model (a YAML file)

EMULSION (https://sourcesup.renater.fr/www/emulsion-public) describes a model as state
machines in YAML. Each state machine becomes a disease, whose states are the machine's
states and whose transitions are its transitions; rates (per individual) and per-step
probabilities are translated, and a state with a duration whose only way out has a
probability of 1 becomes a fixed dwell time. What is listed as not translated (minimum
stays, conditions, calendars, actions on crossing, offspring, and variables of
individuals) is left for a person, or the import-model skill, to add.
"""
import ast
import math
import sciris as sc
import starsim as ss
from .core import Source, build

__all__ = ['load_emulsion']

units = dict(days='day', day='day', weeks='week', week='week', months='month', month='month', years='year', year='year')
solvers = dict(compartment='binomial', IBM='abm', hybrid='abm') # EMULSION's compartments are stochastic and in discrete time
functions = dict(StrictLessThan='<', LessThan='<=', StrictGreaterThan='>', GreaterThan='>=') # EMULSION's comparison functions
skipped_keys = ['cond', 'when', 'escape', 'on_cross', 'amount', 'flux']


def emulsion_expr(text):
    """ An EMULSION expression in Python syntax: comparison functions as comparisons, AND/OR as products and sums """
    node = ast.parse(str(text).strip(), mode='eval').body
    def visit(node):
        for field,val in ast.iter_fields(node):
            if isinstance(val, ast.AST):
                setattr(node, field, visit(val))
            elif isinstance(val, list):
                setattr(node, field, [visit(v) if isinstance(v, ast.AST) else v for v in val])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name in functions and len(node.args) == 2:
                op = dict(zip(['<', '<=', '>', '>='], [ast.Lt(), ast.LtE(), ast.Gt(), ast.GtE()]))[functions[name]]
                return ast.Compare(node.args[0], [op], [node.args[1]])
            elif name == 'AND':
                out = node.args[0]
                for arg in node.args[1:]:
                    out = ast.BinOp(out, ast.Mult(), arg)
                return out
        return node
    return ast.unparse(visit(node))


def number(text, values):
    """ The value of an expression of numbers and parameters, or None if it is not one """
    try:
        node = ast.parse(str(text).strip(), mode='eval').body
    except SyntaxError:
        return None
    if not all(n.id in values for n in ast.walk(node) if isinstance(n, ast.Name)):
        return None
    if any(isinstance(n, (ast.Call, ast.Attribute, ast.Subscript, ast.Lambda)) for n in ast.walk(node)):
        return None
    return eval(compile(ast.Expression(node), '<emulsion>', 'eval'), {'__builtins__':{}}, dict(values)) # Numbers and arithmetic only


def listed(items):
    """ EMULSION's lists of one-key dicts ([{S: {...}}, {I: {...}}]) as (key, value) pairs """
    out = []
    for item in items or []:
        if isinstance(item, dict) and len(item) == 1:
            key, val = list(item.items())[0]
            out.append((key, val or {}))
        else:
            out.append((None, item))
    return out


def names(text, src, exprs, seen=()):
    """ The states, parameters, and totals an expression uses, with its named expressions written out; None if it uses anything else """
    try:
        found = {n.id for n in ast.walk(ast.parse(emulsion_expr(text), mode='eval')) if isinstance(n, ast.Name)}
    except SyntaxError:
        return None
    out = set()
    for name in found:
        if name in src.exprs and src.exprs[name] in src.states: # total_S
            out.add(src.exprs[name])
        elif name in exprs and name not in seen:
            sub = names(exprs[name], src, exprs, seen + (name,))
            if sub is None:
                return None
            out |= sub
        elif name in src.totals or name in src.pars or name not in exprs:
            out.add(name)
    return out


def translate(src, text, exprs, values, docs):
    """ An EMULSION expression for a source, adding the parameters and named expressions it uses """
    text = emulsion_expr(text)
    for n in ast.walk(ast.parse(text, mode='eval')):
        if isinstance(n, ast.Name):
            if n.id in values and n.id not in src.pars:
                src.add_par(n.id, values[n.id], doc=docs.get(n.id))
            elif n.id in exprs and n.id not in src.exprs:
                src.exprs[n.id] = translate(src, exprs[n.id], exprs, values, docs)
    return text


def load_emulsion(filename, machines=None, verbose=None):
    """
    Read an EMULSION model (a YAML file) into a sim

    Each state machine becomes a disease (a sim with several has several diseases). The
    numbers in the YAML are used as they are, in EMULSION's time unit, and the solver
    follows EMULSION's aggregation type: 'binomial' for compartments, 'abm' for
    individuals and hybrid models.

    Args:
        filename (str): the YAML file
        machines (list): the state machines to import (default: all)
        verbose (bool): whether to print how each transition was read

    Returns:
        An `ss.Sim`, with `sim.imported` saying how it was read and what was not translated
    """
    yml = sc.loadyaml(filename)
    name = yml.get('model_name', sc.path(filename).stem)

    # Parameters: numbers (or expressions of numbers) are parameters, the rest are expressions
    values = {}
    exprs = {}
    docs = {}
    for key,par in (yml.get('parameters') or {}).items():
        val = par.get('value') if isinstance(par, dict) else par
        docs[key] = par.get('desc') if isinstance(par, dict) else None
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            values[key] = val
        else:
            exprs[key] = str(val)
    for i in range(len(exprs)): # Expressions of numbers and parameters, in any order
        for key,val in list(exprs.items()):
            num = number(val, values)
            if num is not None:
                values[key] = num
                del exprs[key]

    # Time
    tinfo = yml.get('time_info') or {}
    unit = units.get(str(tinfo.get('time_unit', 'days')).strip(), None)
    dt = number(tinfo.get('delta_t', 1), values) or 1
    stop = number(tinfo.get('total_duration', 100), values)
    levels = yml.get('levels') or {}
    kinds = [lev.get('aggregation_type') for lev in levels.values() if isinstance(lev, dict) and lev.get('aggregation_type')]
    solver = solvers.get(kinds[0], 'abm') if kinds else 'abm'

    # Initial numbers, by prototype
    protos = {}
    for level,items in (yml.get('prototypes') or {}).items():
        for key,val in listed(items):
            protos[key] = val
    amounts = [] # (prototype or {machine: state}, number)
    for level,items in (yml.get('initial_conditions') or {}).items():
        for item in items or []:
            if 'prototype' in item:
                amounts.append((protos.get(item['prototype'], {}), number(item.get('amount', 0), values)))
            else: # Compartments: a list of vars and amounts under the level's name
                for group in item.values():
                    for entry in group or []:
                        if 'vars' in entry:
                            amounts.append(({None:entry['vars'][0]}, number(entry.get('amount', 0), values)))

    sources = []
    all_machines = yml.get('state_machines') or {}
    for mname,machine in all_machines.items():
        if machines is not None and mname not in machines:
            continue
        src = Source(mname if len(all_machines) > 1 else name, format='EMULSION', file=str(filename))
        states = listed(machine.get('states'))
        snames = [s for s,_ in states]
        default = next((s for s,info in states if info.get('default')), snames[0])

        # States and their initial numbers
        init = {s:0 for s in snames}
        for proto,amount in amounts:
            state = proto.get(mname, proto.get(None, default))
            if state in init and amount is not None:
                init[state] += amount
        for s,info in states:
            if info.get('autoremove'): # The dead are removed: an outflow
                continue
            src.add_state(s, init=init[s], label=info.get('name'))
        removed = [s for s,info in states if info.get('autoremove')]

        # Counts and totals, and the parameters and expressions
        src.totals = ['total_population']
        for s in snames:
            src.exprs[f'total_{s}'] = s
        use = lambda text: translate(src, text, exprs, values, docs)

        # Transitions
        durations = {s:info['duration'] for s,info in states if 'duration' in info}
        trans = machine.get('transitions') or []
        for t in trans:
            frm, to = t.get('from'), t.get('to')
            dst = None if to in removed else to
            label = t.get('desc')
            what = f'{mname}: {frm} -> {to}'
            extra = [k for k in skipped_keys if k in t and not (k == 'escape' and t[k] is True)] # A plain escape is translated: with no minimum stay, every way out applies throughout
            if 'rate' in t:
                src.add_flow(frm, dst, use(t['rate']), label=label)
            elif 'proba' in t or t.get('amount-all-but') == 0: # Everyone, once any minimum stay is over
                p = t.get('proba', 1)
                if isinstance(p, list):
                    src.skip(what, f'a list of probabilities ({p}), for a transition with several destinations')
                    continue
                pval = number(p, values)
                if pval == 1:
                    dur = durations.get(frm)
                    others = [o for o in trans if o.get('from') == frm and o is not t and not o.get('escape')]
                    if dur is not None and not others and number(dur, values) is not None:
                        days = number(dur, values)
                        src.add_flow(frm, dst, dur=ss.constant(getattr(ss, f'{unit}s')(days) if unit else days), label=label) # A fixed dwell time
                        durations.pop(frm)
                    else:
                        src.skip(what, 'happens with probability 1 at the next step ("as soon as"); written as a very high rate would be the closest')
                        continue
                else: # A probability per step: the rate that gives it
                    src.add_flow(frm, dst, f'-log(1 - ({use(p)})) / {dt}', label=label)
            else:
                src.skip(what, 'has neither a rate nor a probability')
                continue
            for k in extra:
                src.skip(f'{what} ({k})', f'"{k}: {t[k]}" was not translated' + (' (the transition was translated without it)' if k != 'escape' else ''))
        for s,dur in durations.items():
            src.skip(f'{mname}: duration of {s}', f'a minimum stay of {dur} {unit or "steps"} before the other ways out apply; the rates out of {s} were translated as if it had none')
        for s,info in states:
            for key in ['on_stay', 'on_enter', 'on_exit']:
                if key in info:
                    src.skip(f'{mname}: {key} of {s}', f'actions ({info[key]}) are not translated')

        sources.append(src)

    if not sources:
        errormsg = f'"{filename}" has no state machines to import'
        raise ValueError(errormsg)

    # What applies to the whole model
    first = sources[0]
    if yml.get('statevars'):
        first.skip('statevars', f'the variables of individuals {sc.strjoin(yml["statevars"])} are not translated')
    if tinfo.get('calendars'):
        first.skip('calendars', f'{sc.strjoin(tinfo["calendars"])} not translated: a period on a calendar could be an ss.timeseries')
    for key,val in (yml.get('grouping') or {}).items():
        for gname,g in (val or {}).items():
            if len(g.get('key_variables', [])) > 1:
                first.skip(f'grouping {gname}', f'grouped by {g["key_variables"]}: a transition that depends on another machine\'s state needs a where= condition')
    for level,out in (yml.get('outputs') or {}).items():
        for var in (out.get('extra_vars') or []) if isinstance(out, dict) else []:
            owner = None
            if var == 'total_population':
                owner = sources[0]
                owner.results[var] = var
            elif var in exprs:
                for src in sources: # The disease whose counts it uses
                    found = names(var, src, exprs)
                    if found is not None and found & set(src.states):
                        owner = src
                        src.results[var] = translate(src, var, exprs, values, docs)
                        break
            if owner is None:
                first.skip(f'outputs: {var}', 'an extra output that is not an expression of the numbers in states (it uses a variable of individuals or a calendar)')

    # Build: one disease for each state machine
    cfg = dict(solver=solver, dt=dt)
    if unit:
        cfg['unit'] = unit
    if stop is not None:
        cfg['stop'] = getattr(ss, f'{unit}s')(stop) if unit else stop
    return build(sources, verbose=verbose, **cfg)
