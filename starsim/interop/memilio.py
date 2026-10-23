"""
Import a MEmilio ODE model

MEmilio (https://github.com/SciCompMod/memilio) builds its ODE models in C++, and a
Python script only sets their parameters, so a model is read from the JSON written by
`dump_memilio.py`, which runs the script in MEmilio's own environment and records
the model it simulates. The equations of each supported model (`osir`, `oseir`,
`oseirdb`, and `osecir`) are written here as MEmilio's C++ has them
(`cpp/models/ode_*/model.h`); the parameters, the initial populations by age group,
and the contact matrices with their dampings come from the file. Age groups become
an age stratification with a contact matrix, a damping a factor on the force of
infection (smoothed over the day before it, as MEmilio does), and those who die
leave the model, as they leave MEmilio's denominator. A graph (`MobilityGraph`, or
the `ModelGraph` of a `GraphParameterStudy`) becomes a stratification by place, one
level per node, with its edges as commuting (`ss.geo.Mobility`); the distributions of a
parameter study become the parameters' priors, for `ss.parallel(sim, n_runs=,
priors=True)`.
"""
import numpy as np
import sciris as sc
import starsim as ss
from .core import Source, build

__all__ = ['load_memilio']

unlimited = 1e300 # MEmilio's default capacities are the largest double
pi = '3.141592653589793'
once = ['TimeExposed', 'TimeInfectedNoSymptoms', 'RelativeTransmissionNoSymptoms', 'RiskOfInfectionFromSymptomatic', 'MaxRiskOfInfectionFromSymptomatic'] # In a study, MEmilio's osecir draws these once for every age group, from the first group's distribution (draw_sample_infection())


def smoother(x, left, right, yleft, yright):
    """ MEmilio's smoother_cosine(): yleft below left, yright above right, and half a cosine between """
    frac = f'minimum(maximum(({x} - {left})/({right} - {left}), 0), 1)'
    return f'({yleft} + ({yright} - {yleft})*(1 - cos({pi}*{frac}))/2)'


def load_memilio(source, verbose=None, **kwargs):
    """
    Read a MEmilio ODE model into a sim

    Args:
        source (str/dict): the JSON file written by `dump_memilio.py` (or its contents)
        verbose (bool): whether to print how each flow was read
        kwargs (dict): passed to `ss.Sim()`

    Returns:
        An `ss.Sim`, with `sim.imported` saying how it was read and what was not translated

    **Example**:

        # In the MEmilio environment: python dump_memilio.py ode_secir_simple.py model.json
        import starsim.interop
        sim = ss.interop.load('model.json', format='memilio')
    """
    d = source if isinstance(source, dict) else sc.loadjson(source)
    kind = d['model']
    readers = dict(osir=sir, oseir=sir, oseirdb=sir, osecir=secir)
    if kind not in readers:
        errormsg = f'The MEmilio model "{kind}" has no importer; those that do are {sc.strjoin(readers)}. See the import-model skill\'s guide to MEmilio for writing it by hand'
        raise NotImplementedError(errormsg)
    src = Source(name=kind, format='memilio', file=source if isinstance(source, str) else None)
    src.totals = ['N']
    src.declared = True
    n = d['groups']
    run = d['run']
    t0, tmax = run['t0'], run['tmax']
    pars = d['parameters']
    study = d.get('study')
    used = set()

    def par(name, doc=None, scalar=False):
        """ Add a MEmilio parameter (by age group if it differs between them, with its distribution as the prior in a parameter study), and return its name """
        p = pars[name]
        val = p['value']
        doc = doc or f'MEmilio parameter {name}'
        dists = (p.get('distributions') or [p.get('distribution')]) if study else [None]
        if p['kind'] == 'age' and kind == 'osecir' and name in once:
            dists = dists[:1]
        drawn = any(dists)
        if p['kind'] == 'age':
            if len(set(val)) == 1 and (len(dists) == 1 or not drawn): # In a study, each group's value is drawn on its own, so it stays by group
                val = val[0]
            elif drawn and len(dists) == 1: # Drawn once for every group
                if len(set(val)) > 1:
                    src.skip(f'{name} by age group', f'it differs between age groups ({val}), but MEmilio\'s study draws one value for all of them; the first group\'s, {val[0]:g}, is used')
                val = val[0]
            elif scalar: # Where it belongs to the infectious person's group, or is summed over groups, a rate has one value
                differs = f'it differs between age groups ({val})' if len(set(val)) > 1 else 'it is drawn for each age group in the study'
                src.skip(f'{name} by age group', f'{differs} where it applies to the infectious group, or is summed over groups, which a rate cannot say; the mean, {np.mean(val):g}, is used there' + (', and not drawn' if drawn else ''))
                return repr(round(float(np.mean(val)), 12))
            else:
                val = ss.by(val, dims=['age'])
        if name in used:
            return name
        used.add(name)
        if not drawn:
            src.add_par(name, val, doc=doc)
        elif len(dists) == 1 or all(dist == dists[0] for dist in dists): # One distribution: the prior
            src.add_par(name, val, doc=doc, prior=prior(dists[0]))
        else: # A distribution for each group: the parameter is a draw of the standard distribution, shifted and scaled by group
            standard(src, name, val.values if isinstance(val, ss.by) else [val]*n, dists, doc)
        return name

    # States: as MEmilio's, except those that nothing flows into and that start empty
    graph = d.get('graph')
    nodes = graph['nodes'] if graph else [d]
    places = [str(node['id']) for node in nodes] if graph else None
    pops = {s:np.array([node['populations'][s] for node in nodes]).T for s in d['states']} # Age group by node
    keep = [s for s in d['states'] if not (s.endswith('Confirmed') and not pops[s].any())]
    dead = 'Dead' if kind == 'osecir' else None # Those who die leave the model (MEmilio leaves them out of the total)
    dims = (['age'] if n > 1 else []) + (['patch'] if graph else [])
    for s in keep:
        if s == dead:
            if pops[s].any():
                src.notes.append(f'Dead          the {pops[s].sum():n} who start dead are not people in the sim; add them to the deaths')
            continue
        vals = pops[s].reshape([n]*(n > 1) + [len(nodes)]*bool(graph))
        init = None if s == keep[0] or not vals.any() else ss.by(vals.tolist(), dims=dims) if dims else float(vals)
        src.add_state(s, init=init)
    for node in nodes:
        for s,dists in (node.get('population_distributions') or {}).items():
            src.skip(f'the initial number in {s}' + (f' in node {node["id"]}' if graph else ''), f'it is drawn in each run of the study ({sc.strjoin(set(dist["type"] for dist in dists if dist))}), which is not translated; the value given, {sum(node["populations"][s]):g}, is used')
    for s in keep:
        if s.endswith('Confirmed'):
            src.skip(f'{s} in the total', f'it starts with people in it, who count in the sim\'s total but not in MEmilio\'s')

    # Contacts: the matrix at the start, with the dampings as a factor on the force of infection
    c = d.get('contacts')
    if study and c and c.get('sampled_dampings'): # A study replaces the matrices' dampings with those it draws
        force = sampled(src, c, n, t0)
    else:
        contacts = factor(src, c, n, t0)
        if contacts is not None:
            src.add_par('contacts', contacts, doc='The contacts a day (with one age group), or their share of those at the start, with MEmilio\'s dampings')
        force = ['contacts'] if contacts is not None else []

    # Seasonality: MEmilio's 1 + k*sin(pi*((StartDay + t)/182.5 + 0.5)) is a cosine that peaks on day 0 of its year
    k = pars.get('Seasonality', {}).get('value', 0)
    if k:
        start = pars.get('StartDay', {}).get('value', 0)
        src.add_par('season', ss.seasonal(mean=1.0, amplitude=k, peak=float(-(start + t0)/365 % 1)), doc=f'MEmilio Seasonality, with StartDay {start:g}')
        force.append('season')
    readers[kind](src, d, par, force)
    if study:
        for name,p in pars.items():
            if name not in used and (p.get('distribution') or any(p.get('distributions') or [])):
                src.skip(f'the distribution of {name}', 'it is drawn in each run of the study, which is not translated; the value given is used')

    # Groups and contacts
    totals = np.sum([pops[s] for s in keep if s != dead], axis=0) # Age group by node
    if n > 1:
        src.stratify(age=[str(i) for i in range(n)], split=(totals.sum(1)/totals.sum()).tolist())
        matrix = np.array(d['contacts']['samples'][0]['matrix'])
        src.config.networks = ss.contacts(age=matrix.tolist())
        src.code['networks'] = f'ss.contacts(age={matrix.tolist()!r})'
        src.notes.append(f'contacts      the {n} age groups as labels ("0" to "{n-1}"), with MEmilio\'s contact matrix at the start (summed over its {len(d["contacts"]["matrices"])} matrices)')
    if graph:
        nodes_of(src, d, places, totals)
    for name,what in (d.get('other') or {}).items():
        src.skip(name, f'{what}, which is not translated')
    if study:
        src.notes.append(f'study         a parameter study of {study["runs"]} runs, with each distribution as the prior of its parameter: ss.parallel(sim, n_runs={study["runs"]}, priors=True)')

    src.config.update(unit='day', stop=ss.days(tmax - t0), solver='ode', total_pop=float(totals.sum()))
    sim = build(src, verbose=verbose, **kwargs)
    if study:
        sim.imported.runs = study['runs']
    return sim


def prior(dist):
    """ A MEmilio parameter distribution as a prior """
    if dist['type'] == 'uniform':
        return ss.uniform(dist['lower'], dist['upper'])
    elif dist['type'] == 'normal': # MEmilio's normal is cut at bounds that its Python bindings do not show
        return ss.normal(dist['mean'], dist['std'])
    errormsg = f'The MEmilio distribution {dist["type"]} has no prior in Starsim'
    raise NotImplementedError(errormsg)


def standard(src, name, vals, dists, doc):
    """ A parameter drawn from a different distribution in each age group, as a draw of the standard uniform (or normal) for each group, shifted and scaled by group """
    types = {dist['type'] for dist in dists if dist}
    if len(types) > 1 or not types <= {'uniform', 'normal'}:
        src.skip(f'the distributions of {name}', f'they are of different kinds in different age groups ({sc.strjoin(types)}), which is not translated; it is not drawn in the study')
        src.add_par(name, ss.by(vals, dims=['age']), doc=doc)
        return
    uniform = types == {'uniform'}
    shift = [dist['lower'] if uniform else dist['mean'] if dist else v for v,dist in zip(vals, dists)] # A group with no distribution keeps its value
    scale = [dist['upper'] - dist['lower'] if uniform else dist['std'] if dist else 0.0 for dist in dists]
    draw = [(v - a)/b if b else 0.0 for v,a,b in zip(vals, shift, scale)] # So a run with no draws has the values given
    src.add_par(f'{name}_shift', ss.by(shift, dims=['age']), doc=f'{doc}: the {"lower bound" if uniform else "mean"} of its distribution in each age group')
    src.add_par(f'{name}_scale', ss.by(scale, dims=['age']), doc=f'{doc}: the {"width" if uniform else "standard deviation"} of its distribution in each age group')
    src.add_par(f'{name}_draw', ss.by(draw, dims=['age']), doc=f'{doc}: its draw in each age group, as a share of the width of its distribution' if uniform else f'{doc}: its draw in each age group, in standard deviations', prior=ss.uniform(0, 1) if uniform else ss.normal(0, 1))
    src.exprs[name] = f'({name}_shift + {name}_scale*{name}_draw)'
    src.notes.append(f'{name:13s} drawn from a different {"uniform" if uniform else "normal"} distribution in each age group: {name}_shift + {name}_scale*{name}_draw, with {name}_draw drawn from the standard one')
    return


def nodes_of(src, d, places, totals):
    """ A graph's nodes as places, and its edges as commuting between them """
    nodes, edges = d['graph']['nodes'], d['graph']['edges']
    n = d['groups']
    first = nodes[0]

    # What differs between nodes: the populations are by node, and the rest must be the same
    for name,p in first['parameters'].items():
        vals = [node['parameters'][name]['value'] for node in nodes]
        if any(v != vals[0] for v in vals):
            src.skip(f'{name} by node', f'it differs between nodes, which is not translated; the first node\'s, {vals[0]}, is used')
        elif name in ['ICUCapacity', 'TestAndTraceCapacity'] and vals[0] < unlimited:
            src.skip(f'{name} in each node', f'it applies to each node\'s own numbers in MEmilio, but to the whole model\'s here')
    matrices = [[s['matrix'] for s in node['contacts']['samples']] if node['contacts'] else None for node in nodes]
    if any(m != matrices[0] for m in matrices):
        src.skip('contacts by node', 'the contact matrices or their dampings differ between nodes, which is not translated; the first node\'s are used')

    # Edges: each moves a share of a node's people to another for every other step, so they are away half the time (MobilityEdge::apply_mobility())
    states = d['states']
    alive = [i for i,s in enumerate(states) if s != 'Dead']
    M = np.zeros((len(nodes), len(nodes)))
    uneven = 0
    damped = 0
    for edge in edges:
        coeffs = np.reshape(edge['coefficients'], (n, len(states)))[:, alive] # MEmilio's flat index is age group, then state
        M[edge['start'], edge['end']] += coeffs.mean()
        uneven += bool(np.ptp(coeffs) > 1e-12*max(coeffs.max(), 1e-300))
        damped += edge.get('dampings', 0)
    if uneven:
        src.skip('commuting by group', f'on {uneven} of the {len(edges)} edges, the share who commute differs between age groups or states, which is not translated; the mean share is used')
    if damped:
        src.skip('commuting dampings', f'{damped} dampings of the commuting, which are not translated')
    if not M.any():
        src.stratify(patch=places, split=ss.by(totals/totals.sum(), dims=['age', 'patch']) if n > 1 else (totals[0]/totals.sum()).tolist())
        src.notes.append(f'places        the {len(nodes)} nodes as the places "{places[0]}" to "{places[-1]}", with no commuting')
        return
    away = 0.5*M # The share of their time that people spend in each other place
    split = ss.by(totals/totals.sum(), dims=['age', 'patch']) if n > 1 else (totals[0]/totals.sum()).tolist()
    src.stratify(patch=places, split=split, mobility=ss.geo.Mobility(matrix=away.tolist(), kind='commute'))
    src.notes.append(f'places        the {len(nodes)} nodes as the places "{places[0]}" to "{places[-1]}", and the {len(edges)} edges as commuting: MEmilio moves an edge\'s share of a node\'s people to the other node for every other step and back, so they spend half their time there (ss.geo.Mobility(kind="commute") with half the share); '
                     f'MEmilio integrates the commuters in the other node, where Starsim weights each place\'s exposure by the time spent there, so counties whose epidemic starts from commuters can differ (switching/memilio.md)')
    if d['model'] == 'osecir':
        src.notes.append('commuting     MEmilio\'s commuters with symptoms commute less (by RiskOfInfectionFromSymptomatic, get_mobility_factors()); here everyone commutes alike, which carries infection between places faster '
                         '(in a test of three places with RiskOfInfectionFromSymptomatic 0.25, by 5 to 19% of each place\'s peak and 1% of the total; with 1, by 1 to 3%)')
    return


def factor(src, c, n, t0):
    """ The contacts as a number, or as an ss.timeseries through the dampings (with one age group, the contacts themselves) """
    if c is None:
        return None
    samples = c['samples']
    base = np.array(samples[0]['matrix'], dtype=float)
    points = {}
    for s in samples:
        m = np.array(s['matrix'], dtype=float)
        if n == 1:
            val = float(m[0, 0])
        else:
            on = base > 0
            ratios = m[on]/base[on]
            val = float(np.mean(ratios)) if on.any() else 1.0
            if on.any() and np.ptp(ratios) > 1e-9*max(1, abs(val)):
                src.skip(f'the damping at day {s["time"]:g}', f'it cuts contacts between some age groups more than others (by {1 - ratios.min():.3g} to {1 - ratios.max():.3g}); the mean share, {val:.3g}, is used for all of them')
        points[s['time'] - t0] = val
    if c.get('school_holidays'):
        src.skip('contacts: school_holidays', f'{c["school_holidays"]} school holidays, which are not translated')
    times = sorted(points)
    for a,b in zip(times[:-1], times[1:]):
        if b - a < 1 and points[a] != points[b] and b - a != 1:
            src.notes.append(f'contacts      two dampings less than a day apart (days {a:g} and {b:g}); the smoothing between them is not exactly MEmilio\'s')
    if len(set(points.values())) == 1:
        val = points[times[0]]
        return None if val == 1 else val
    return ss.timeseries({t:points[t] for t in times}, mode='cosine')


def sampled(src, c, n, t0):
    """ The factors on the force of infection in a parameter study: the dampings it draws, which replace the contact matrices' own (UncertainContactMatrix::make_matrix()), each from its time on (smoothed over the day before) until the next """
    if c.get('school_holidays'):
        src.skip('contacts: school_holidays', f'{c["school_holidays"]} school holidays, which are not translated')
    base = np.sum([m['baseline'] for m in c['matrices']], axis=0)
    if n == 1:
        src.add_par('contacts', float(base[0, 0]), doc='The contacts a day')
    damps = sorted(c['sampled_dampings'], key=lambda dm: dm['time'])
    every = list(range(len(c['matrices'])))
    plain = all(dm['level'] == damps[0]['level'] and dm['type'] == damps[0]['type'] and sorted(dm['matrices']) == every and set(dm['weights']) == {1.0} for dm in damps)
    if not plain or any(np.any(m['minimum']) for m in c['matrices']):
        src.skip('contacts: sampled dampings', f'{len(damps)} dampings drawn in the study, of different levels or types, on some of the matrices, weighted by group, or with a minimum, which are not translated')
        return ['contacts'] if n == 1 else []
    terms = []
    for i,dm in enumerate(damps):
        name = f'damping_{i}' if len(damps) > 1 else 'damping'
        dist = dm['value'].get('distribution')
        src.add_par(name, dm['value']['value'], doc=f'MEmilio\'s damping of contacts from day {dm["time"]:g}' + (' (drawn in the study)' if dist else ''), prior=prior(dist) if dist else None)
        on = {dm['time'] - 1 - t0: 0.0, dm['time'] - t0: 1.0}
        if i + 1 < len(damps): # Until the next damping replaces it
            nxt = damps[i+1]['time'] - t0
            on.update({nxt - 1: 1.0, nxt: 0.0})
        src.add_par(f'{name}_on', ss.timeseries(on, mode='cosine'), doc=f'When {name} applies, smoothed over the day before (as MEmilio does)')
        terms.append(f'{name}*{name}_on')
    src.exprs['damped'] = f'(1 - {" - ".join(terms)})'
    src.notes.append(f'contacts      the {len(damps)} damping(s) drawn in the study, which replace the contact matrices\' own there')
    return (['contacts'] if n == 1 else []) + ['damped']


def sir(src, d, par, force):
    """ osir, oseir, and oseirdb: infection by contact with the infected (and, in oseirdb, the dead) """
    kind = d['model']
    first = 'Exposed' if kind != 'osir' else 'Infected'
    beta = '*'.join(force + [par('TransmissionProbabilityOnContact', 'The probability of infection on contact with an infected person')])
    if kind == 'oseirdb':
        dead = par('TransmissionProbabilityFromDead', 'The probability of infection on contact with someone dead but not yet buried')
        src.add_flow('Susceptible', first, f'{"*".join(force) + "*" if force else ""}(TransmissionProbabilityOnContact*Infected + {dead}*Dead)/N', name='infection')
    else:
        src.add_flow('Susceptible', first, f'{beta}*Infected/N', name='infection')
    if kind != 'osir':
        src.add_flow('Exposed', 'Infected', f'1/{par("TimeExposed")}', name='onset')
    if kind == 'oseirdb':
        src.add_flow('Infected', ['Recovered', 'Dead'], f'1/{par("TimeInfected")}', name='recovery', p=[par('ProbabilityToRecover'), None])
        src.add_flow('Dead', 'Buried', f'1/{par("TimeToBurial")}', name='burial')
    else:
        src.add_flow('Infected', 'Recovered', f'1/{par("TimeInfected")}', name='recovery')
    return


def secir(src, d, par, force):
    """ osecir: the SECIR (SECIHURD) model, with test and trace, and ICU capacity """
    pars = d['parameters']
    INS, ISy = 'InfectedNoSymptoms', 'InfectedSymptoms'

    # Those with symptoms infect more once test and trace is overwhelmed: by the number becoming symptomatic each day, over all groups
    risk = par('RiskOfInfectionFromSymptomatic', scalar=True)
    cap = pars['TestAndTraceCapacity']['value']
    if cap < unlimited:
        need = f'(1 - {par("RecoveredPerInfectedNoSymptoms", scalar=True)})/{par("TimeInfectedNoSymptoms", scalar=True)}*{INS}'
        src.add_par('TestAndTraceCapacity', cap, doc='MEmilio parameter TestAndTraceCapacity: the number becoming symptomatic each day that can be traced')
        src.add_par('TestAndTraceCapacityMaxRisk', pars['TestAndTraceCapacityMaxRisk']['value'], doc='MEmilio parameter TestAndTraceCapacityMaxRisk')
        src.exprs['risk'] = smoother(need, 'TestAndTraceCapacity', 'TestAndTraceCapacity*TestAndTraceCapacityMaxRisk', risk, par('MaxRiskOfInfectionFromSymptomatic', scalar=True))
        risk = 'risk'
        src.notes.append('risk          from RiskOfInfectionFromSymptomatic to MaxRiskOfInfectionFromSymptomatic as the number becoming symptomatic each day (summed over groups) goes from TestAndTraceCapacity to TestAndTraceCapacityMaxRisk times it, along half a cosine')
    infectious = f'({par("RelativeTransmissionNoSymptoms", scalar=True)}*{INS} + {risk}*{ISy})'
    confirmed = 'InfectedNoSymptomsConfirmed' in src.states
    src.add_flow('Susceptible', 'Exposed', '*'.join(force + [par('TransmissionProbabilityOnContact', 'The probability of infection on contact'), infectious]) + '/N', name='infection')
    src.add_flow('Exposed', INS, f'1/{par("TimeExposed")}', name='onset')
    mu = par('RecoveredPerInfectedNoSymptoms')
    src.add_flow(INS, [ISy, 'Recovered'], f'1/{par("TimeInfectedNoSymptoms")}', name='symptoms', p=[f'1 - {mu}', None])
    if confirmed:
        src.add_flow('InfectedNoSymptomsConfirmed', ['InfectedSymptomsConfirmed', 'Recovered'], '1/TimeInfectedNoSymptoms', name='symptoms_confirmed', p=[f'1 - {mu}', None])
    severe = par('SeverePerInfectedSymptoms')
    src.add_flow(ISy, ['InfectedSevere', 'Recovered'], f'1/{par("TimeInfectedSymptoms")}', name='severe', p=[severe, None])
    if 'InfectedSymptomsConfirmed' in src.states:
        src.add_flow('InfectedSymptomsConfirmed', ['InfectedSevere', 'Recovered'], '1/TimeInfectedSymptoms', name='severe_confirmed', p=[severe, None])

    # The severely ill become critical, recover, or die; with ICU capacity, those who would be critical die once the ICU is nearly full
    cps, dps, tsev = par('CriticalPerSevere'), par('DeathsPerSevere'), par('TimeInfectedSevere')
    icu = pars['ICUCapacity']['value']
    if icu < unlimited:
        src.add_par('ICUCapacity', icu, doc='MEmilio parameter ICUCapacity')
        frac = 'minimum(maximum((InfectedCritical - 0.9*ICUCapacity)/(0.1*ICUCapacity), 0), 1)' # MEmilio's smoother_cosine(), from CriticalPerSevere to 0
        src.exprs['critical_share'] = f'{cps}*(1 + cos({pi}*{frac}))/2'
        src.add_flow('InfectedSevere', 'InfectedCritical', f'critical_share/{tsev}', name='critical')
        src.add_flow('InfectedSevere', 'Recovered', f'(1 - {cps} - {dps})/{tsev}', name='severe_recovery')
        src.add_flow('InfectedSevere', None, f'({dps} + {cps} - critical_share)/{tsev}', name='severe_death')
        src.notes.append('ICU           those who would become critical die instead as the number critical (summed over groups) goes from 90% to 100% of ICUCapacity, along half a cosine')
    elif any(np.atleast_1d(pars['DeathsPerSevere']['value'])):
        src.add_flow('InfectedSevere', ['InfectedCritical', 'Recovered'], f'(1 - {dps})/{tsev}', name='critical', p=[f'{cps}/(1 - {dps})', None])
        src.add_flow('InfectedSevere', None, f'{dps}/{tsev}', name='severe_death')
    else:
        src.add_flow('InfectedSevere', ['InfectedCritical', 'Recovered'], f'1/{tsev}', name='critical', p=[cps, None])
    dpc, tcr = par('DeathsPerCritical'), par('TimeInfectedCritical')
    src.add_flow('InfectedCritical', 'Recovered', f'(1 - {dpc})/{tcr}', name='critical_recovery')
    src.add_flow('InfectedCritical', None, f'{dpc}/{tcr}', name='death')
    return
