"""
Import an EMULSION model (a YAML file)

EMULSION (https://sourcesup.renater.fr/www/emulsion-public) describes a model as state
machines in YAML. Each state machine becomes a disease, whose states are the machine's
states and whose transitions are its transitions; rates (per individual) and
probabilities (per time unit, as EMULSION reads them) are translated, and a state with
a duration whose only way out has a probability of 1 becomes a fixed dwell time. What
is listed as not translated (conditions, calendars, actions on crossing, offspring and
productions, levels above the population, and variables of individuals) is left for a
person, or the import-model skill, to add; a minimum stay (a state's duration) becomes
`after=` on the ways out of it.

Used with: agents and compartments; declared modules
"""
import ast
import math
import sciris as sc
import starsim as ss
from .core import Source, build, funcs as known_funcs

__all__ = ['load_emulsion']

units = dict(days='day', day='day', weeks='week', week='week', months='month', month='month', years='year', year='year')
solvers = dict(compartment='binomial', IBM='abm', hybrid='abm') # EMULSION's compartments are stochastic and in discrete time
functions = dict(StrictLessThan='<', LessThan='<=', StrictGreaterThan='>', GreaterThan='>=') # EMULSION's comparison functions
skipped_keys = ['cond', 'when', 'escape', 'on_cross', 'amount', 'flux']
top_keys = ['model_name', 'model_info', 'time_info', 'levels', 'grouping', 'processes', 'state_machines', 'parameters', 'statevars', 'prototypes', 'initial_conditions', 'outputs'] # Read, or (for model_info, processes) not needed
machine_keys = ['desc', 'states', 'transitions', 'productions']


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
    """ EMULSION's lists of one-key dicts ([{S: {...}}, {I: {...}}]), or of plain names ([S, I, R]), as (key, value) pairs """
    out = []
    for item in items or []:
        if isinstance(item, dict) and len(item) == 1:
            key, val = list(item.items())[0]
            out.append((key, val or {}))
        elif isinstance(item, str): # A plain name, with nothing else to say about it
            out.append((item, {}))
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


def unknown(text, src, exprs, values, seen=()):
    """ The names and functions an expression uses that the import cannot translate (a variable of individuals or of a level, or a function of EMULSION's) """
    try:
        node = ast.parse(emulsion_expr(text), mode='eval')
    except SyntaxError:
        return {str(text)}
    calls = {n.func.id for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    out = {f'{c}()' for c in calls if c not in known_funcs}
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id not in calls:
            if n.id in exprs and n.id not in seen:
                out |= unknown(exprs[n.id], src, exprs, values, seen + (n.id,))
            elif not (n.id in values or n.id in src.exprs or n.id in src.states or n.id in src.totals):
                out.add(n.id)
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


def load_emulsion(filename, verbose=None):
    """
    Read an EMULSION model (a YAML file) into a sim

    Each state machine becomes a disease (a sim with several has several diseases). The
    numbers in the YAML are used as they are, in EMULSION's time unit, and the solver
    follows EMULSION's aggregation type: 'binomial' for compartments, 'abm' for
    individuals and hybrid models.

    Args:
        filename (str): the YAML file
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
    levels = {k:v for k,v in (yml.get('levels') or {}).items() if isinstance(v, dict)}
    upper = {k:v for k,v in levels.items() if v.get('aggregation_type') == 'metapopulation'} # A level of populations (herds), not of animals
    kinds = [lev.get('aggregation_type') for key,lev in levels.items() if lev.get('aggregation_type') and key not in upper]
    solver = solvers.get(kinds[0], 'abm') if kinds else 'abm'
    skipped = [] # What applies to the whole model and is not translated, as (name, why)
    for key,lev in levels.items():
        if lev.get('file') or lev.get('class_name'):
            skipped.append((f'levels: {key}', f'the level is Python code ({lev.get("class_name")} in {lev.get("file")}), which is not translated'))
    groups = {g for val in (yml.get('grouping') or {}).values() for g in (val or {})}
    for key,procs in (yml.get('processes') or {}).items():
        for proc in procs or []:
            pname = proc if isinstance(proc, str) else str(proc)
            if pname not in groups and pname not in (yml.get('state_machines') or {}):
                skipped.append((f'processes: {pname}', f'a process of the {key} level that is not a state machine (a method in the Python code add-on), which is not translated'))
    for key,lev in upper.items():
        skipped.append((f'levels: {key}', f'a metapopulation of {sc.strjoin(sc.tolist(lev.get("contains", [])))} levels: the model was imported as one population. '
                                           'Each population is a place: stratify the disease with kind=\'space\' (e.g. dis.stratify(pop=[...], kind=\'space\')), and give the initial numbers by place with ss.by(pop=...)'))

    # Initial numbers, by prototype
    protos = {}
    for level,items in (yml.get('prototypes') or {}).items():
        for key,val in listed(items):
            protos[key] = val
    amounts = [] # (prototype {machine: state}, or {None: [states]}, number)
    for level,items in (yml.get('initial_conditions') or {}).items():
        if level in upper:
            skipped.append((f'initial_conditions: {level}', f'the initial populations of the {level} level ({sc.strjoin(item.get("prototype", "?") for item in items or [])}) are not translated'))
            continue
        for item in items or []:
            if 'prototype' in item:
                amounts.append((protos.get(item['prototype'], {}), item.get('amount', 0)))
            else: # Compartments: a list of vars and amounts under the level's name
                for group in item.values():
                    for entry in group or []:
                        if 'vars' in entry:
                            amounts.append(({None:sc.tolist(entry['vars'])}, entry.get('amount', 0)))
    for proto,amount in amounts:
        if number(amount, values) is None:
            skipped.append((f'initial_conditions: {amount}', f'the initial number "{amount}" is not a number or an expression of parameters (it uses a variable of individuals or of a level), so it was left out'))

    sources = []
    removes = [] # The machines with births or deaths
    all_machines = yml.get('state_machines') or {}
    for mname,machine in all_machines.items():
        src = Source(mname if len(all_machines) > 1 else name, format='EMULSION', file=str(filename))
        states = listed(machine.get('states'))
        snames = [s for s,_ in states]
        default = next((s for s,info in states if info.get('default')), snames[0])

        # States and their initial numbers
        init = {s:0 for s in snames}
        for proto,amount in amounts:
            state = proto.get(mname)
            if state is None:
                state = next((v for v in proto.get(None, []) if v in init), default)
            amount = number(amount, values)
            if state in init and amount is not None:
                init[state] += amount
        for s,info in states:
            if info.get('autoremove'): # The dead are removed: an outflow
                continue
            src.add_state(s, init=init[s], label=info.get('name'))
        removed = [s for s,info in states if info.get('autoremove')]

        # Counts and totals, and the parameters and expressions
        src.totals = ['total_population'] + [f'total_{lev}' for lev in levels if lev not in upper] # The number in a level of animals (total_herd) is the population too
        for s in snames:
            src.exprs[f'total_{s}'] = s
        use = lambda text: translate(src, text, exprs, values, docs)

        # Transitions
        durations = {s:info['duration'] for s,info in states if 'duration' in info}
        stays = set() # The states whose minimum stay was translated, as after= on the ways out
        def stay(frm, t):
            """ The minimum stay before a way out of a state applies (EMULSION's duration), unless it is an escape """
            if frm not in durations or t.get('escape') is True or number(durations[frm], values) is None:
                return None
            stays.add(frm)
            days = number(durations[frm], values)
            return getattr(ss, f'{unit}s')(days) if unit else days
        trans = machine.get('transitions') or []
        for t in trans:
            frm, to = t.get('from'), t.get('to')
            dst = None if to in removed else to
            label = t.get('desc')
            what = f'{mname}: {frm} -> {to}'
            extra = [k for k in skipped_keys if k in t and not (k == 'escape' and t[k] is True)] # A plain escape is translated: with no minimum stay, every way out applies throughout
            missing = set().union(*[unknown(t[k], src, exprs, values) for k in ['rate', 'proba'] if k in t and not isinstance(t[k], list)])
            if missing:
                src.skip(what, f'uses {sc.strjoin(sorted(missing))}, which {"is" if len(missing) == 1 else "are"} not a state, parameter, or expression of them (e.g. a variable of individuals or of a level, or a function)')
                continue
            if 'rate' in t:
                src.add_flow(frm, dst, use(t['rate']), label=label, after=stay(frm, t))
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
                else: # A probability per time unit (EMULSION makes it 1 - (1 - p)**delta_t for a step): the rate that gives it
                    src.add_flow(frm, dst, f'-log(1 - ({use(p)}))', label=label, after=stay(frm, t))
            else:
                src.skip(what, 'has neither a rate nor a probability')
                continue
            for k in extra:
                src.skip(f'{what} ({k})', f'"{k}: {t[k]}" was not translated' + (' (the transition was translated without it)' if k != 'escape' else ''))
        for s,dur in durations.items():
            if s not in stays:
                src.skip(f'{mname}: duration of {s}', f'a minimum stay of {dur} {unit or "steps"} before the other ways out apply; the rates out of {s} were translated as if it had none')
        for s,info in states:
            for key in ['on_stay', 'on_enter', 'on_exit']:
                if key in info:
                    src.skip(f'{mname}: {key} of {s}', f'actions ({info[key]}) are not translated')
        for t in machine.get('productions') or []: # Births: newborns in one state, made by those in another
            what = f'{mname}: production {t.get("from")} -> {t.get("to")}'
            if 'prototype' in t:
                src.skip(what, f'newborns of the prototype {t["prototype"]} are not translated: an inflow into each machine\'s state of the prototype')
            else:
                src.skip(what, f'births into {t.get("to")} from those in {t.get("from")} ({t.get("rate", t.get("proba"))} each) are not translated: an inflow, ss.null >> {t.get("to")} @ <rate>*{t.get("from")}.n, entering every machine together')
        for key in machine:
            if key not in machine_keys:
                src.skip(f'{mname}: {key}', f'"{key}" in the state machine is not read by the importer')
        if removed or machine.get('productions'):
            removes.append(src.name)

        sources.append(src)

    if not sources:
        errormsg = f'"{filename}" has no state machines to import'
        raise ValueError(errormsg)

    # What applies to the whole model
    first = sources[0]
    for item in skipped:
        first.skip(*item)
    for key in yml:
        if key not in top_keys:
            first.skip(key, f'"{key}" is not read by the importer')
    if len(sources) > 1 and solver != 'abm' and removes:
        first.skip('state machines', f'as compartments, each of the {len(sources)} machines is a disease with its own counts, so the births and deaths in {sc.strjoin(removes)} do not reach the others: '
                                     'cross the machines into one disease (e.g. states SJ, IJ, ... for health and age) to keep their counts together')
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
    if solver == 'binomial': # EMULSION's compartments take whole steps
        cfg['substeps'] = 1
    if unit:
        cfg['unit'] = unit
    if stop is not None:
        cfg['stop'] = getattr(ss, f'{unit}s')(stop) if unit else stop
    return build(sources, verbose=verbose, **cfg)
