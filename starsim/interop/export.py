"""
Export a sim to other formats

`ss.save(filename, sim, format=...)` writes a sim's declared diseases in another
format; see `v4_design/design/16-interop.md`. Only the model spec (`sim.to_spec()`)
is complete: the other formats carry less, and what is left out or changed is listed
as `ss.errors.LossyExport`. An export does not fail because something cannot be carried.

- `mermaid`: a diagram of each disease's states and transitions, labelled with their rates
- `sbml`: the compartmental form as species and reactions (https://sbml.org), for any SBML tool
- `odin`: the compartmental form as `deriv()` equations (https://mrc-ide.github.io/odin)

The runs of a MultiSim are written as a submission to a forecast hub by `ss.interop.to_hub()`.

SBML and odin are written from the compartmental form of the model (what
`sir.equations()` prints), with the numbers in the sim's unit of time: a dwell time
that is not exponential is its Erlang stages, and each stratification is summed into
one group. Both are read back by `ss.load()`.
"""
import re
import ast
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

__all__ = ['exporters', 'export', 'flatten', 'to_mermaid', 'to_sbml', 'to_odin', 'to_hub']

math_funcs = dict(exp='exp', log='ln', sqrt='root', abs='abs', absolute='abs', minimum='min', maximum='max', min='min', max='max', sin='sin', cos='cos', floor='floor', ceiling='ceiling')


#%% The compartmental form

def initialized(sim):
    """ The sim initialized: itself if it is, else a new one from the same inputs """
    if sim.initialized:
        return sim
    return sim.new(verbose=0).init() if hasattr(sim, 'inputs') else sc.dcp(sim).init()


def compartmental(sim):
    """ An initialized sim whose diseases run as compartments, and what was changed to get it """
    sim = initialized(sim)
    if sim.solver != 'abm':
        return sim, None
    return sim.new(solver='ode', verbose=0).init(), 'changed: the agent-based sim was written as its compartmental form (an ODE)'


def strata(sim, mod):
    """ The stratifications of a disease, as text (e.g. 'age: 0-4, 5-17, 18+') """
    seen = {}
    for strat in list(getattr(getattr(sim, 'groups', None), 'strats', [])) + list(getattr(mod.graph, 'strats', [])):
        seen[strat.name] = f'{strat.name}: {", ".join(str(l) for l in strat.labels)}'
    return list(seen.values())


def rename(text, names, unit):
    """
    An expression with its names changed, as plain numbers and counts

    A typed value (`peryear(0.02)`) is its number in the sim's unit, a count (`(S + I).n`)
    is the sum it counts, a fraction (`I.frac`) that over `N`, and `mean(x)` (the mean
    of a dwell time) is `x`, whose value is the mean.
    """
    node = ast.parse(text, mode='eval').body
    class Sub(ast.NodeTransformer):
        def visit_Call(self, n):
            func = n.func.id if isinstance(n.func, ast.Name) else None
            if func == 'mean' and len(n.args) == 1:
                return self.visit(n.args[0])
            cls = getattr(ss, func, None) if func else None
            if isinstance(cls, type) and issubclass(cls, ss.TimePar) and all(isinstance(a, ast.Constant) for a in n.args):
                return ast.Constant(float(ss.solvers.convert(cls(*[a.value for a in n.args]), unit)))
            self.generic_visit(n)
            return n
        def visit_Attribute(self, n):
            if n.attr == 'n':
                return self.visit(n.value)
            elif n.attr == 'frac':
                return ast.BinOp(self.visit(n.value), ast.Div(), ast.Name(names.get('N', 'N')))
            self.generic_visit(n)
            return n
        def visit_Name(self, n):
            return ast.Name(names.get(n.id, n.id))
    return ast.unparse(Sub().visit(node))


def flatten(sim):
    """
    The compartmental form of a sim's declared diseases, as plain data for the exporters

    Returns an `sc.objdict` with the unit of time; the compartments and their initial
    numbers; the parameters (numbers in that unit); named expressions (a state with
    several stages is their sum, and `N` is the people in a disease); the flows, each
    with its compartments and its rate in total; and what was left out or changed.
    """
    sim, how = compartmental(sim)
    spec = sim.to_spec()
    unit = sim.unit_base
    lost = [how] if how else []
    diseases = [mod for mod in sim.diseases.values() if getattr(mod, 'graph', None) is not None]
    many = len(diseases) > 1
    out = sc.objdict(name=diseases[0].name if len(diseases) == 1 else 'model', unit=unit, comps=sc.objdict(), pars=sc.objdict(), exprs=sc.objdict(), flows=[], lost=lost, n_states=0, n_transitions=0)

    for mod in diseases:
        g, e = mod.graph, mod.engine
        pre = f'{mod.name}_' if many else ''
        names, flows = g.flow_terms(e)
        Y = np.asarray(e.y).sum(axis=1) # Summed over the groups of any stratification
        table = {}
        for i,name in enumerate(names):
            table[name] = pre + name
            out.comps[pre + name] = float(Y[i])
        for state in g.states.values():
            stages = [table[n] for i,n in enumerate(names) if e.comps[i].state is state]
            if len(stages) > 1: # The state is the sum of its stages
                table[state.name] = pre + state.name
                out.exprs[pre + state.name] = ' + '.join(stages)
        table['N'] = pre + 'N'
        out.exprs[pre + 'N'] = ' + '.join(pre + s.name for s in g.people_states)
        out.n_states += len(g.states)
        out.n_transitions += len(g.transitions)

        # Parameters, as numbers in the sim's unit
        dwells = {id(dw.par): dw for dw in e.dwells.values()}
        for name,p in g.pars.items():
            table[name] = pre + name
            raw = p.raw
            if isinstance(raw, ss.Dist):
                dw = dwells.get(id(p))
                value = float(dw.mean) if dw is not None else np.nan
                if dw is not None and dw.staged:
                    lost.append(f'changed: the {dw.family} dwell time "{name}" written as {dw.k} exponential stages (Erlang) with its mean, {value:n} {unit}s')
                elif dw is not None and dw.family not in ['expon', None]:
                    lost.append(f'changed: the {dw.family} dwell time "{name}" written as exponential with its mean, {value:n} {unit}s')
            else:
                try:
                    value = float(ss.solvers.convert(raw, unit))
                except (TypeError, ValueError):
                    value = np.nan
                    lost.append(f'not exported: the value of parameter "{name}" ({type(raw).__name__}), which is not a number; written as NaN')
            out.pars[pre + name] = value

        # Flows, each named after its transition (with a number when it is several)
        counts = sc.ddict(int)
        for flow in flows:
            counts[flow.tr.name] += max(len(flow.dst), 1)
        seen = sc.ddict(int)
        for flow in flows:
            src = None if flow.src is None else table[names[flow.src]]
            for dst,term in flow.dst or [(None, flow.term)]:
                name = pre + flow.tr.name
                if counts[flow.tr.name] > 1:
                    seen[flow.tr.name] += 1
                    name = f'{name}_{seen[flow.tr.name]}'
                out.flows.append(sc.objdict(name=name, src=src, dst=None if dst is None else table[names[dst]], rate=rename(term, table, unit), transition=flow.tr.name))

    # Anything a rate uses that is not exported
    known = set(out.comps) | set(out.pars) | set(out.exprs)
    for flow in out.flows:
        used = {n.id for n in ast.walk(ast.parse(flow.rate, mode='eval')) if isinstance(n, ast.Name)} - known - set(math_funcs)
        if used:
            lost.append(f'not exported: {sc.strjoin(sorted(used))}, used in the rate of "{flow.name}" ({flow.rate})')

    # What the spec has that these formats do not
    if not diseases:
        lost.append('not exported: the sim has no declared diseases')
    for m in spec['modules']:
        if not any(m['name'] == mod.name for mod in diseases):
            lost.append(f'not exported: module "{m["name"]}" ({m.get("type")}), which has no declared transitions')
            continue
        mod = sim.diseases[m['name']]
        for text in strata(sim, mod):
            lost.append(f'changed: "{m["name"]}" is stratified ({text}), written as one group, the sum of its groups, with the base values of its parameters')
        for r in m.get('results', []):
            lost.append(f'not exported: {r.get("type", "result")} "{r["name"]}"')
        for f in m.get('functions', []):
            lost.append(f'not exported: function "{f["name"]}" (Python code)')
    for key,what in dict(networks='network', interventions='intervention', modifiers='modifier').items():
        for item in spec.get(key, []):
            name = item.get('name') if isinstance(item, dict) else str(item)
            extra = ' (contacts written as homogeneous mixing)' if key == 'networks' else ''
            lost.append(f'not exported: {what} "{name}"{extra}')
    if getattr(sim, 'crossings', None):
        lost.append('not exported: the coupling between diseases (each is written on its own)')
    if sim.solver not in ['ode', 'auto'] and how is None:
        lost.append(f'changed: solver="{sim.solver}" is not carried; the model is written as an ODE')
    return out


#%% Mermaid

def to_mermaid(sim):
    """ A Mermaid flowchart of each disease's states and transitions, labelled with their rates; transmissions are dashed """
    sim = initialized(sim)
    diseases = [mod for mod in sim.diseases.values() if getattr(mod, 'graph', None) is not None]
    many = len(diseases) > 1
    lines = ['flowchart LR']
    n_states = n_transitions = 0
    for mod in diseases:
        g = mod.graph
        pre = f'{mod.name}_' if many else ''
        indent = '    '*(2 if many else 1)
        strats = strata(sim, mod)
        title = mod.name + (f' (stratified by {"; ".join(strats)})' if strats else '')
        if many:
            lines.append(f'    subgraph {mod.name}["{title}"]')
        elif strats: # A title above the chart
            lines = ['---', f'title: {title}', '---'] + lines
        lines += g.diagram(prefix=pre, indent=indent)
        if many:
            lines.append('    end')
        n_states += len(g.states)
        n_transitions += len(g.transitions)
    summary = f'exported to Mermaid: {n_states} states, {n_transitions} transitions'
    return '\n'.join(lines) + '\n', summary, []


#%% SBML

def mathml(node):
    """ A Python expression (an AST node) as a MathML element """
    m = 'http://www.w3.org/1998/Math/MathML'
    def el(tag, text=None):
        e = ET.Element(f'{{{m}}}{tag}')
        if text is not None:
            e.text = f' {text} '
        return e
    def apply(op, args):
        a = el('apply')
        a.append(el(op))
        for arg in args:
            a.append(arg)
        return a
    def visit(n):
        if isinstance(n, ast.Name):
            return el('ci', n.id)
        elif isinstance(n, ast.Constant):
            return el('cn', repr(float(n.value)))
        elif isinstance(n, ast.BinOp):
            ops = {ast.Add:'plus', ast.Sub:'minus', ast.Mult:'times', ast.Div:'divide', ast.Pow:'power'}
            return apply(ops[type(n.op)], [visit(n.left), visit(n.right)])
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return apply('minus', [visit(n.operand)])
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in math_funcs:
            return apply(math_funcs[n.func.id], [visit(a) for a in n.args])
        raise ValueError(f'the expression "{ast.unparse(n)}" has no MathML form here')
    out = el('math')
    out.append(visit(n) if isinstance(node, ast.AST) and not isinstance(node, ast.Expression) else visit(node.body))
    return out


def to_sbml(sim):
    """ The compartmental form of a sim as an SBML model (Level 3 Version 1): compartments as species, flows as reactions """
    flat = flatten(sim)
    sb = 'http://www.sbml.org/sbml/level3/version1/core'
    ET.register_namespace('', sb)
    ET.register_namespace('math', 'http://www.w3.org/1998/Math/MathML')
    q = lambda tag: f'{{{sb}}}{tag}'
    sub = lambda parent, tag, **attrs: ET.SubElement(parent, q(tag), {k:str(v) for k,v in attrs.items()})
    root = ET.Element(q('sbml'), dict(level='3', version='1'))
    model = sub(root, 'model', id=flat.name if flat.name.isidentifier() else 'model', name=flat.name, timeUnits=flat.unit, substanceUnits='item', extentUnits='item')
    seconds = dict(day=86400, week=7*86400, month=365.25/12*86400, year=365.25*86400)
    units = sub(model, 'listOfUnitDefinitions')
    ud = sub(units, 'unitDefinition', id=flat.unit)
    sub(sub(ud, 'listOfUnits'), 'unit', kind='second', exponent=1, scale=0, multiplier=seconds.get(flat.unit, 1))
    sub(sub(model, 'listOfCompartments'), 'compartment', id='population', size=1, constant='true', spatialDimensions=3, units='dimensionless')
    species = sub(model, 'listOfSpecies')
    for name,init in flat.comps.items():
        sub(species, 'species', id=name, compartment='population', initialAmount=repr(init), hasOnlySubstanceUnits='true', boundaryCondition='false', constant='false')
    params = sub(model, 'listOfParameters')
    for name,value in flat.pars.items():
        sub(params, 'parameter', id=name, value=repr(value), constant='true')
    for name in flat.exprs:
        sub(params, 'parameter', id=name, constant='false')
    lost = list(flat.lost)
    rules = sub(model, 'listOfRules')
    for name,text in flat.exprs.items():
        sub(rules, 'assignmentRule', variable=name).append(mathml(ast.parse(text, mode='eval')))
    reactions = sub(model, 'listOfReactions')
    for flow in flat.flows:
        try:
            law = mathml(ast.parse(flow.rate, mode='eval'))
        except ValueError as E:
            lost.append(f'not exported: the flow "{flow.name}": {E}')
            continue
        r = sub(reactions, 'reaction', id=flow.name, reversible='false', fast='false')
        if flow.src is not None:
            sub(sub(r, 'listOfReactants'), 'speciesReference', species=flow.src, stoichiometry=1, constant='true')
        if flow.dst is not None:
            sub(sub(r, 'listOfProducts'), 'speciesReference', species=flow.dst, stoichiometry=1, constant='true')
        used = {n.id for n in ast.walk(ast.parse(flow.rate, mode='eval')) if isinstance(n, ast.Name)}
        modifiers = [name for name in flat.comps if name in used and name not in [flow.src, flow.dst]] # Other species the law uses, as SBML requires
        if modifiers:
            mods = sub(r, 'listOfModifiers')
            for name in modifiers:
                sub(mods, 'modifierSpeciesReference', species=name)
        sub(r, 'kineticLaw').append(law)
    ET.indent(root, space='  ')
    text = ET.tostring(root, encoding='unicode')
    text = text.replace(' xmlns:math="http://www.w3.org/1998/Math/MathML"', '').replace('<math:math>', '<math xmlns="http://www.w3.org/1998/Math/MathML">') # MathML in its own default namespace, as SBML tools write it
    text = re.sub(r'<(/?)math:', r'<\1', text)
    text = '<?xml version="1.0" encoding="UTF-8"?>\n' + text + '\n'
    summary = f'exported to SBML: {flat.n_states} states, {flat.n_transitions} transitions, as {len(flat.comps)} species and {len(flat.flows)} reactions (rates per {flat.unit})'
    return text, summary, lost


#%% odin

def to_odin(sim):
    """ The compartmental form of a sim as an odin model: a `deriv()` equation for each compartment, with each flow a named variable """
    flat = flatten(sim)
    r = lambda text: text.replace('**', '^')
    lines = [f'## {flat.name}, exported from Starsim {ss.__version__}; time is in {flat.unit}s', '']
    taken = set(flat.comps) | set(flat.pars) | set(flat.exprs)
    flow_names = []
    for flow in flat.flows: # Each flow is its own variable, so the equations pair up when read back
        name = flow.name if flow.name not in taken else f'{flow.name}_flow'
        taken.add(name)
        flow_names.append(name)
    for comp in flat.comps:
        terms = [f'+ {n}' for f,n in zip(flat.flows, flow_names) if f.dst == comp] + [f'- {n}' for f,n in zip(flat.flows, flow_names) if f.src == comp]
        rhs = ' '.join(terms).lstrip('+ ') if terms else '0'
        if rhs.startswith('- '):
            rhs = '-' + rhs[2:]
        lines.append(f'deriv({comp}) <- {rhs}')
    lines.append('')
    for comp,init in flat.comps.items():
        lines.append(f'initial({comp}) <- {init:.15g}')
    lines.append('')
    for flow,name in zip(flat.flows, flow_names):
        lines.append(f'{name} <- {r(flow.rate)}')
    for name,text in flat.exprs.items():
        lines.append(f'{name} <- {r(text)}')
    lines.append('')
    for name,value in flat.pars.items():
        lines.append(f'{name} <- user({"NA" if np.isnan(value) else f"{value:.15g}"})')
    summary = f'exported to odin: {flat.n_states} states, {flat.n_transitions} transitions, as {len(flat.comps)} compartments and {len(flat.flows)} flows (time in {flat.unit}s)'
    return '\n'.join(lines) + '\n', summary, list(flat.lost)


#%% Exporting

exporters = dict(
    mermaid = to_mermaid,
    sbml    = to_sbml,
    odin    = to_odin,
)


def export(sim, format, filename=None, verbose=None):
    """
    Write a sim in another format; usually called as `ss.save(filename, sim, format=...)`

    What the format cannot carry is listed as `ss.errors.LossyExport`; the export still
    writes everything it can.

    Args:
        sim (`ss.Sim`): the sim, with declared diseases
        format (str): one of `ss.interop.exporters`: 'mermaid', 'sbml', or 'odin'
        filename (str): the file to write, if any (for Mermaid, a `.md` file has the diagram in a fenced block)
        verbose (bool): whether to print what was exported (by default, unless something was lost)

    Returns:
        The text of the export

    **Examples**:

        ss.save('sir.xml', sim, format='sbml')
        print(ss.interop.export(sim, 'mermaid'))
    """
    key = str(format).lower()
    if key not in exporters:
        errormsg = f'Cannot export to format "{format}"; the formats are {sc.strjoin(exporters)}'
        raise ValueError(errormsg)
    text, summary, lost = exporters[key](sim)
    if lost:
        ss.warn(f'{summary}\n' + '\n'.join(f'  {line}' for line in lost), ss.errors.LossyExport)
    elif verbose is not False:
        print(summary)
    if filename is not None:
        out = f'```mermaid\n{text}```\n' if key == 'mermaid' and str(filename).endswith('.md') else text
        sc.savetext(filename, out)
    return text


#%% Exporting runs

hub_quantiles = [0.01, 0.025] + [round(q, 2) for q in np.arange(0.05, 0.951, 0.05)] + [0.975, 0.99] # The 23 quantiles of the US forecast hubs

def to_hub(msim, targets, by=None, every='week', quantiles=hub_quantiles, location=None):
    """
    The runs of a MultiSim as a submission to a forecast hub: quantiles of each target, by location and period

    Args:
        msim (`ss.MultiSim`): the runs
        targets (dict): the hub's name for each result, e.g. `dict(incidHosp='inc hosp')`
        by (str): the dimension whose levels are the locations (default: one location, the total)
        every (str): the period: a flow is totaled over it, and a count is taken at its end: 'day' or 'week' (of the sim's steps, which must then be days)
        quantiles (list): the quantiles (default: the 23 that the US forecast hubs ask for)
        location (str): the name of the single location, if `by` is not given

    Returns a long dataframe with columns `target`, `location`, `target_end_date`, `output_type`, `quantile`, and `value`.

    **Example**:

        msim = ss.parallel(sim, n_runs=20)
        hub = ss.interop.to_hub(msim, targets=dict(infection='inc inf'), by='patch', every='week')
    """
    msim.check_run('to_hub')
    steps = dict(day=1, week=7)[every]
    found = {key:res for modname,key,res in msim.results.walk() if modname is not None}
    rows = []
    for key,target in targets.items():
        if key not in found:
            errormsg = f'No result called "{key}"; the results are {sc.strjoin(found)}'
            raise KeyError(errormsg)
        res = found[key]
        places = res.by(by) if by is not None else {location or 'total':res}
        for place,sub in places.items():
            vals = sub.values[:, 1:] # The steps, after the start
            n = vals.shape[1]//steps
            if sub.kind == 'flow':
                periods = vals[:, :n*steps].reshape(len(vals), n, steps).sum(axis=2)
            else:
                periods = vals[:, steps-1:n*steps:steps]
            ends = np.array(sub.timevec)[1:][steps-1:n*steps:steps]
            for q,qvals in zip(quantiles, np.quantile(periods, quantiles, axis=0)):
                rows.append(pd.DataFrame(dict(target=target, location=str(place), target_end_date=ends, output_type='quantile', quantile=q, value=qvals)))
    return pd.concat(rows, ignore_index=True)
