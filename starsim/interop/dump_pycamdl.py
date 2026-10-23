"""
Write a pycamdl model to JSON, for `ss.interop.load(..., format='camdl')`

pycamdl, camdl's Python front end, builds a model as Python objects and runs it in
Python, without camdl's compiler, so there is no camdl IR to read. This writes the
same subset of the IR that the camdl importer reads: the model as pycamdl expands
it (one compartment per combination of levels, one transition per member of each
family), with each rate as an expression in Python syntax. It needs only pycamdl
(and NumPy), not Starsim or Sciris. Run it on a module that makes a model called
`m` or `model`, or on a function that returns one:

    python dump_pycamdl.py stratified.py model.json
    python dump_pycamdl.py my_model.py:build model.json

or call `dump(model, 'model.json')` from Python; `ss.interop.load(model)` also works on a
live pycamdl model, if pycamdl can be imported where Starsim is.
"""
import re
import sys
import json
import runpy
import numbers

__all__ = ['dump', 'model_dict']

funcs = dict(exp='exp', log='log', sqrt='sqrt', abs='abs', minimum='min', maximum='max', sin='sin', cos='cos', floor='floor', ceil='ceil', tanh='tanh', mod='mod')


def compiled(model):
    """ The compiled model, with the code of each rate (pycamdl keeps only the functions made from it) """
    from pycamdl import compiler
    calls = []
    fn = compiler.Compiled.fn
    def record(self, codes, args='y, p, t, dt'):
        calls.append(list(codes))
        return fn(self, codes, args)
    compiler.Compiled.fn = record
    try:
        comp = model.compile()
    finally:
        compiler.Compiled.fn = fn
    return comp, calls[0] # The first function made is the rates (see Compiled.setup_transitions())


def python(code, comp):
    """ A rate's code, with the cells and parameters by name """
    def const(match):
        val = comp.consts[int(match.group(1))]
        if not isinstance(val, numbers.Number) and getattr(val, 'size', 2) != 1:
            raise ValueError(f'the rate uses an array, {code}')
        return repr(float(val))
    code = re.sub(r'\by\[(\d+)\]', lambda m: comp.cells[int(m.group(1))], code)
    code = re.sub(r'\bp\[(\d+)\]', lambda m: comp.pars[int(m.group(1))], code)
    code = re.sub(r'\bK\[(\d+)\]', const, code)
    code = re.sub(r'\bnp\.(\w+)', lambda m: funcs.get(m.group(1), f'np.{m.group(1)}'), code)
    return code


def model_dict(model, pars=None):
    """ The model as a dict, in the form of camdl's IR (see the module docstring) """
    comp, rates = compiled(model)
    pars = pars or {}
    out = dict(name=getattr(model, 'name', None) or 'pycamdl', time_unit=comp.time_unit, compartments=[dict(name=c) for c in comp.cells])

    # Parameters, in the model's time unit
    out['parameters'] = []
    values = []
    for i,(name,par) in enumerate(zip(comp.pars, comp.parinfo)):
        try:
            value = float(comp.values({name: pars[name]} if name in pars else {})[i])
            entry = dict(mode='fixed', value=value)
        except Exception: # No value
            value = float('nan')
            entry = dict(mode='required')
        if par.bounds:
            entry['bounds'] = list(comp.bounds[i])
        values.append(value)
        out['parameters'].append(dict(name=name, value=entry, doc=par.doc))

    # Transitions
    out['transitions'] = []
    for tr,code in zip(comp.trans, rates):
        stoich = [[comp.cells[i], int(d)] for i,d in tr.stoich.items()]
        try:
            rate = dict(text=python(code, comp))
        except ValueError as E:
            rate = dict(untranslated=str(E))
        out['transitions'].append(dict(name=tr.name, stoichiometry=stoich, rate=rate))

    # Initial numbers, worked out from the parameters
    init = comp.init(values)
    out['initial_conditions'] = {c: dict(deterministic=dict(const=float(v))) for c,v in zip(comp.cells, init) if v}

    # Scheduled effects, worked out from the parameters
    cfg = model.config
    start = float(getattr(cfg['start'], 'value', cfg['start']))
    stop = float(getattr(cfg['stop'], 'value', cfg['stop']))
    out['interventions'] = []
    for act in comp.actions:
        times = [float(t) for t in act.times(start, stop)]
        amounts = act.amounts([0.0]*len(comp.cells), values, start, 1.0)
        if act.kind == 'transfer':
            actions = [{'fraction_transfer' if act.how == 'fraction' else 'transfer': dict(src=comp.cells[s], dst=comp.cells[d], fraction=dict(const=float(a)))} for (s,d),a in zip(act.cells, amounts)]
        else:
            actions = [{act.kind: dict(compartment=comp.cells[c], count=dict(const=float(a)))} for c,a in zip(act.cells, amounts)]
        out['interventions'].append(dict(name=act.name, fire=dict(scheduled=dict(at_times=times)), actions=actions, kind='event' if act.event else 'intervention'))

    out['observations'] = [dict(name=s.name, untranslated='pycamdl observations are not written by dump_pycamdl.py') for s in comp.streams]
    out['simulation'] = dict(t_start=start, t_end=stop, dt=None)
    dims = [dict(name=dim.name, values=list(dim)) for dim,_ in model.strata]
    out['model_structure'] = dict(dimensions=dims, base_compartments=list(model.compartments))
    return dict(format='pycamdl', model=out)


def dump(model, filename=None, pars=None):
    """ Write a pycamdl model to a JSON file, and return it as a dict """
    out = model_dict(model, pars=pars)
    if filename:
        with open(filename, 'w') as f:
            json.dump(out, f, indent=1)
    return out


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    path, func = (sys.argv[1].split(':') + [None])[:2]
    ns = runpy.run_path(path, run_name='dump_pycamdl')
    model = ns[func]() if func else ns.get('m', ns.get('model'))
    dump(model, sys.argv[2])
    print(f'Wrote {sys.argv[2]}')
