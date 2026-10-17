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
leave the model, as they leave MEmilio's denominator.
"""
import numpy as np
import sciris as sc
import starsim as ss
from .core import Source, build

__all__ = ['load_memilio']

unlimited = 1e300 # MEmilio's default capacities are the largest double
pi = '3.141592653589793'


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
        sim = ss.load('model.json', format='memilio')
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
    used = set()

    def par(name, doc=None, scalar=False):
        """ Add a MEmilio parameter (by age group if it differs between them), and return its name """
        p = pars[name]
        val = p['value']
        if p['kind'] == 'age':
            if len(set(val)) == 1:
                val = val[0]
            elif scalar: # Where it belongs to the infectious person's group, or is summed over groups, a rate has one value
                src.skip(f'{name} by age group', f'it differs between age groups ({val}) where it applies to the infectious group, or is summed over groups, which a rate cannot say; the mean, {np.mean(val):g}, is used there')
                return repr(round(float(np.mean(val)), 12))
            else:
                val = ss.by(val, dims=['age'])
        if name not in used:
            src.add_par(name, val, doc=doc or f'MEmilio parameter {name}')
            used.add(name)
        return name

    # States: as MEmilio's, except those that nothing flows into and that start empty
    pops = d['populations']
    keep = [s for s in d['states'] if not (s.endswith('Confirmed') and not any(pops[s]))]
    dead = 'Dead' if kind == 'osecir' else None # Those who die leave the model (MEmilio leaves them out of the total)
    for s in keep:
        if s == dead:
            if any(pops[s]):
                src.notes.append(f'Dead          the {sum(pops[s]):n} who start dead are not people in the sim; add them to the deaths')
            continue
        vals = pops[s]
        init = None if s == keep[0] or not any(vals) else ss.by(vals, dims=['age']) if n > 1 else float(vals[0])
        src.add_state(s, init=init)
    for s in keep:
        if s.endswith('Confirmed'):
            src.skip(f'{s} in the total', f'it starts with people in it, who count in the sim\'s total but not in MEmilio\'s')

    # Contacts: the matrix at the start, with the dampings as a factor on the force of infection
    contacts = factor(src, d.get('contacts'), n, t0)
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

    # Groups and contacts
    if n > 1:
        totals = np.sum([pops[s] for s in keep if s != dead], axis=0)
        src.stratify(age=[str(i) for i in range(n)], split=(totals/totals.sum()).tolist())
        matrix = np.array(d['contacts']['samples'][0]['matrix'])
        src.config.networks = ss.contacts(age=matrix.tolist())
        src.code['networks'] = f'ss.contacts(age={matrix.tolist()!r})'
        src.notes.append(f'contacts      the {n} age groups as labels ("0" to "{n-1}"), with MEmilio\'s contact matrix at the start (summed over its {len(d["contacts"]["matrices"])} matrices)')
    for name,what in (d.get('other') or {}).items():
        src.skip(name, f'{what}, which is not translated')

    total = sum(np.sum(pops[s]) for s in keep if s != dead)
    src.config.update(unit='day', stop=ss.days(tmax - t0), solver='ode', total_pop=float(total))
    return build(src, verbose=verbose, **kwargs)


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
    for extra,what in [('dynamic_dampings', 'dampings drawn for a parameter study'), ('school_holidays', 'school holidays')]:
        if c.get(extra):
            src.skip(f'contacts: {extra}', f'{c[extra]} {what}, which are not translated')
    times = sorted(points)
    for a,b in zip(times[:-1], times[1:]):
        if b - a < 1 and points[a] != points[b] and b - a != 1:
            src.notes.append(f'contacts      two dampings less than a day apart (days {a:g} and {b:g}); the smoothing between them is not exactly MEmilio\'s')
    if len(set(points.values())) == 1:
        val = points[times[0]]
        return None if val == 1 else val
    return ss.timeseries({t:points[t] for t in times}, mode='cosine')


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
