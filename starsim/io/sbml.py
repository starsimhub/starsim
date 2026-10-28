"""
Import an SBML model (an XML file)

SBML (https://sbml.org) describes a model as species and the reactions between them,
each with a kinetic law in MathML. Each species becomes a state, each parameter a
parameter, and each reaction that takes one species to another (counting a species
that is both used and made, such as I in S + I -> 2 I, as unchanged) a transition,
whose kinetic law is its rate in total. Assignment rules become named expressions,
and function definitions are written out where they are called. What is listed as
not translated (events, rate rules, algebraic rules, reversible reactions, and
reactions with other stoichiometries) is left for a person, or the import-model
skill, to add. Only the standard library is used to read the file.
"""
import ast
import math
import xml.etree.ElementTree as ET
import sciris as sc
import starsim as ss
from .core import Source, build

__all__ = ['load_sbml']

ops = dict(plus='+', minus='-', times='*', divide='/', power='**')
funcs = dict(exp='exp', ln='log', abs='abs', floor='floor', ceiling='ceiling', min='min', max='max', sin='sin', cos='cos')
compares = dict(lt='<', leq='<=', gt='>', geq='>=')


def tag(el):
    """ An element's tag without its namespace """
    return el.tag.split('}')[-1]


def children(el, name):
    """ The children of an element with a tag (in any namespace) """
    return [c for c in el if tag(c) == name] if el is not None else []


def child(el, name):
    """ The first child of an element with a tag, or None """
    found = children(el, name)
    return found[0] if found else None


def mathml(el, lambdas=None):
    """ A MathML element as an expression in Python syntax """
    lambdas = lambdas or {}
    t = tag(el)
    if t == 'math':
        return mathml(list(el)[0], lambdas)
    if t == 'ci':
        return el.text.strip()
    if t == 'cn':
        kind = el.get('type', 'real')
        parts = [p.strip() for p in el.itertext() if p.strip()]
        if kind == 'e-notation':
            return f'{parts[0]}e{parts[1]}'
        elif kind == 'rational':
            return f'({parts[0]} / {parts[1]})'
        return parts[0]
    if t == 'csymbol':
        url = el.get('definitionURL', '')
        if url.endswith('time'):
            raise ValueError('the time (csymbol time)')
        raise ValueError(f'the symbol {url}')
    if t in ['pi', 'exponentiale', 'true', 'false']:
        return dict(pi=str(math.pi), exponentiale=str(math.e), true='1', false='0')[t]
    if t == 'apply':
        head, *args = list(el)
        h = tag(head)
        vals = [mathml(a, lambdas) for a in args if tag(a) not in ['logbase', 'degree']]
        if h in ops:
            if h == 'minus' and len(vals) == 1:
                return f'(-{vals[0]})'
            return '(' + f' {ops[h]} '.join(vals) + ')'
        if h in funcs:
            return f'{funcs[h]}({", ".join(vals)})'
        if h == 'root':
            degree = child(el, 'degree')
            n = mathml(list(degree)[0]) if degree is not None else '2'
            return f'sqrt({vals[0]})' if n == '2' else f'({vals[0]} ** (1 / {n}))'
        if h == 'log':
            base = child(el, 'logbase')
            return f'(log({vals[0]}) / log({mathml(list(base)[0])}))' if base is not None else f'(log({vals[0]}) / log(10))'
        if h in compares and len(vals) == 2:
            return f'({vals[0]} {compares[h]} {vals[1]})'
        if h == 'ci' and head.text.strip() in lambdas: # A function definition, written out
            params, body = lambdas[head.text.strip()]
            node = ast.parse(body, mode='eval').body
            values = dict(zip(params, [ast.parse(v, mode='eval').body for v in vals]))
            class Sub(ast.NodeTransformer):
                def visit_Name(self, n):
                    return values.get(n.id, n)
            return '(' + ast.unparse(Sub().visit(node)) + ')'
        raise ValueError(f'the MathML operator "{h}"')
    if t == 'piecewise':
        raise ValueError('a piecewise expression')
    raise ValueError(f'the MathML element "{t}"')


def load_sbml(filename, unit=None, stop=None, solver='ode', verbose=None, **kwargs):
    """
    Read an SBML model (an XML file) into a sim

    Species become states, parameters (global, and those local to a reaction)
    parameters, and reactions transitions, with the kinetic law as the rate in
    total, classified by the rules in 16-interop.md. SBML's time units are rarely
    given, so the numbers are taken to be in `unit`: by default, the model's
    `timeUnits` if it is a day, week, month, or year (as Starsim's export writes), else a day.

    Args:
        filename (str): the SBML file
        unit (str): the time unit of the model's rates
        stop (float): when to stop, in that unit (SBML does not say)
        solver (str): the solver to use (SBML models are usually ODEs)
        verbose (bool): whether to print how each reaction was read
        kwargs (dict): passed to `ss.Sim()`

    Returns:
        An `ss.Sim`, with `sim.imported` saying how it was read and what was not translated
    """
    root = ET.parse(str(filename)).getroot()
    model = child(root, 'model')
    if model is None:
        errormsg = f'"{filename}" is not an SBML file: it has no <model>'
        raise ValueError(errormsg)
    if unit is None:
        unit = model.get('timeUnits') if model.get('timeUnits') in ['day', 'week', 'month', 'year'] else 'day'
    name = model.get('id') or model.get('name') or sc.path(filename).stem
    if not name.isidentifier():
        name = sc.path(filename).stem
    src = Source(name, format='SBML', file=str(filename))

    # Function definitions, written out where they are used
    lambdas = {}
    for fd in children(child(model, 'listOfFunctionDefinitions'), 'functionDefinition'):
        lam = child(child(fd, 'math'), 'lambda')
        try:
            params = [mathml(child(b, 'ci')) for b in children(lam, 'bvar')]
            body = [c for c in lam if tag(c) != 'bvar'][0]
            lambdas[fd.get('id')] = (params, mathml(body, lambdas))
        except (ValueError, IndexError, TypeError) as E:
            src.skip(f'function {fd.get("id")}', f'could not be read ({E})')

    # Compartments: a size is a parameter, as kinetic laws often multiply by it
    sizes = {}
    for c in children(child(model, 'listOfCompartments'), 'compartment'):
        size = c.get('size') or c.get('volume') or 1
        sizes[c.get('id')] = float(size)
        if float(size) != 1:
            src.skip(f'compartment {c.get("id")}', f'has size {size}; species given as concentrations are taken as numbers of people, so a law in concentrations may need dividing by it')

    # Species: states
    species = []
    boundary = []
    for sp in children(child(model, 'listOfSpecies'), 'species'):
        sid = sp.get('id')
        amount = sp.get('initialAmount')
        if amount is None and sp.get('initialConcentration') is not None:
            amount = float(sp.get('initialConcentration')) * sizes.get(sp.get('compartment'), 1)
        if sp.get('boundaryCondition') == 'true' or sp.get('constant') == 'true': # Held fixed: a parameter
            src.add_par(sid, float(amount or 0), doc=f'a species held fixed in SBML ({sp.get("name") or sid})')
            boundary.append(sid)
            continue
        src.add_state(sid, init=float(amount) if amount else None, label=sp.get('name'))
        species.append(sid)

    # Parameters, and initial assignments that are numbers
    ruled = {rule.get('variable') for rules in children(model, 'listOfRules') for rule in rules}
    for p in children(child(model, 'listOfParameters'), 'parameter'):
        if p.get('value') is not None:
            src.add_par(p.get('id'), float(p.get('value')), doc=p.get('name'))
        elif p.get('id') not in ruled: # A parameter given by a rule has no value of its own
            src.skip(f'parameter {p.get("id")}', 'has no value')
    for c,size in sizes.items():
        src.add_par(c, size, doc='the size of an SBML compartment')
    for ia in children(child(model, 'listOfInitialAssignments'), 'initialAssignment'):
        sym = ia.get('symbol')
        try:
            text = mathml(child(ia, 'math'), lambdas)
            vals = {k:v.value for k,v in src.pars.items()}
            val = eval(compile(ast.Expression(ast.parse(text, mode='eval').body), '<sbml>', 'eval'), {'__builtins__':{}}, {**vals, **{k:getattr(math, k) for k in ['exp', 'log', 'sqrt']}})
            if sym in src.states:
                src.states[sym].init = val
            else:
                src.add_par(sym, float(val))
        except Exception as E:
            src.skip(f'initial assignment of {sym}', f'is not a number of the parameters ({E})')

    # Rules: an assignment is a named expression; others are not translated
    for rules in children(model, 'listOfRules'):
        for rule in rules:
            var = rule.get('variable')
            kind = tag(rule)
            if kind == 'assignmentRule':
                try:
                    src.exprs[var] = mathml(child(rule, 'math'), lambdas)
                    src.pars.pop(var, None)
                except ValueError as E:
                    src.skip(f'rule for {var}', f'uses {E}, which is not translated')
            else:
                src.skip(f'{kind} {var or ""}'.strip(), f'a {kind} is not translated: a rate rule for a parameter is a state that is not people (ss.State(people=False)) with inflows and outflows')
    for ev in children(child(model, 'listOfEvents'), 'event'):
        src.skip(f'event {ev.get("id") or ev.get("name") or ""}'.strip(), 'events are not translated: one at a fixed time is an intervention or an ss.timeseries, and one on a condition is an intervention with a where= condition')

    # Reactions: a species used and one made, with the kinetic law as the flow in total
    for r in children(child(model, 'listOfReactions'), 'reaction'):
        rid = r.get('id')
        net = {}
        fixed = True # Whether every stoichiometry is a number
        for key,sign in [('listOfReactants', -1), ('listOfProducts', 1)]:
            for ref in children(child(r, key), 'speciesReference'):
                sp = ref.get('species')
                fixed = fixed and child(ref, 'stoichiometryMath') is None
                if sp not in boundary:
                    net[sp] = net.get(sp, 0) + sign*float(ref.get('stoichiometry', 1))
        used = [s for s,n in net.items() if n == -1]
        made = [s for s,n in net.items() if n == 1]
        changes = [f'{s} ({n:+g})' for s,n in net.items() if n]
        law = child(r, 'kineticLaw')
        if law is None:
            src.skip(f'reaction {rid}', 'has no kinetic law')
            continue
        if not fixed or len(used) > 1 or len(made) > 1 or len(used) + len(made) < len(changes):
            why = 'a stoichiometry is an expression' if not fixed else f'it changes the species by {sc.strjoin(changes)}'
            src.skip(f'reaction {rid}', f'{why}; only reactions that take one person from one state to another, or into or out of the model, are translated')
            continue

        # Local parameters, renamed if they clash
        local = {}
        for lp in children(child(law, 'listOfLocalParameters'), 'localParameter') + children(child(law, 'listOfParameters'), 'parameter'):
            pid = lp.get('id')
            local[pid] = pid if pid not in src.pars and pid not in src.states else f'{rid}_{pid}'
            src.add_par(local[pid], float(lp.get('value')), doc=f'local to reaction {rid}')
        try:
            node = ast.parse(mathml(child(law, 'math'), lambdas), mode='eval').body
        except ValueError as E:
            src.skip(f'reaction {rid}', f'its kinetic law uses {E}, which is not translated')
            continue
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id in local:
                n.id = local[n.id]
        if r.get('reversible') == 'true':
            src.skip(f'reaction {rid} (reversible)', 'it is reversible: the kinetic law was taken as the net flow forward, which is right only while it stays positive')
        src.add_flow(used[0] if used else None, made[0] if made else None, ast.unparse(node), per='total', name=rid if rid and rid.isidentifier() else None, label=r.get('name'))

    # Run settings
    cfg = dict(solver=solver, unit=unit)
    if stop is not None:
        cfg['stop'] = getattr(ss, f'{unit}s')(stop) if unit else stop
    cfg.update(kwargs)
    return build(src, verbose=verbose, **cfg)
