"""
Import an epidemics model

epidemics (https://epiverse-trace.github.io/epidemics) is an R package of the Epiverse-TRACE
project: a library of published compartmental models, each run by a function such as
`model_default()` on a `population()` (a contact matrix, the size of each group, and the
fraction of each group in each compartment), with interventions on contacts or rates and
vaccination. Its scripts are R, so they are read from the JSON written by `dump_epidemics.R`,
which runs the script with the model functions replaced by ones that record their
arguments; `ss.import_from()` runs it if R is installed.

`model_default()` (an SEIR model with vaccination, V) is translated, giving the same numbers
as epidemics. Its contact matrix is scaled as epidemics scales it (by its largest eigenvalue),
and becomes `ss.contacts()` with β per contact; the groups become a stratification (by age, if
they are labeled as age groups); an intervention on contacts becomes modifiers of infection
and infectiousness by group, one on a rate a modifier of that parameter; and vaccination a
transition from S to V, of a number of people a day in each group. The other models
(`model_vacamole()`, `model_diphtheria()`, `model_ebola()`) are not translated.

Used with: compartments; declared modules
"""
import os
import shutil
import subprocess
import tempfile
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
from .core import Source, Code, build, code_value

__all__ = ['load_epidemics']

here = os.path.dirname(os.path.abspath(__file__))
translated = ['model_default'] # The models of epidemics that are translated
names = dict(transmission_rate='beta', infectiousness_rate='sigma', recovery_rate='gamma') # The parameters of model_default(), as named here


def dump(filename):
    """
    Run `dump_epidemics.R` on an R script, and return the JSON it writes

    Args:
        filename (str): the R script
    """
    rscript = shutil.which('Rscript')
    if rscript is None:
        errormsg = f'To read "{filename}", put Rscript (with epidemics installed) on the path, or run dump_epidemics.R on it and load the JSON'
        raise FileNotFoundError(errormsg)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'model.json')
        proc = subprocess.run([rscript, os.path.join(here, 'dump_epidemics.R'), os.path.abspath(filename), out], capture_output=True, text=True)
        if proc.returncode or not os.path.exists(out):
            errormsg = f'dump_epidemics.R failed on "{filename}":\n{proc.stderr}'
            raise RuntimeError(errormsg)
        return sc.loadjson(out)


def read(source):
    """ The model as a dict: from a dict, the JSON written by dump_epidemics.R, or an R script (which needs R) """
    if isinstance(source, dict):
        return source
    if str(source).lower().endswith('.json'):
        return sc.loadjson(source)
    return dump(source)


def age_edges(groups):
    """ The lower edge of each group, if the groups are labeled as age groups that start at 0 and go up (e.g. "[0,20)", "20-64", "65+"); else None """
    if not ss.data.is_age_index(pd.Index([str(g) for g in groups])):
        return None
    edges = ss.data.age_edges(groups)
    return edges if edges[0] == 0 and all(b > a for a,b in zip(edges, edges[1:])) else None


def rows(x, n):
    """ A matrix from the JSON (a list of rows, or of numbers if each row has one) as an array with n rows """
    return np.array(x, dtype=float).reshape(n, -1)


def window(begin, end):
    """ The start and stop of a modifier, as numbers """
    return dict(start=float(begin), stop=float(end))


def load_epidemics(source, run=0, verbose=None, **kwargs):
    """
    Read an epidemics model into a sim

    Args:
        source (str/dict): an R script that calls `model_default()` (read by running `dump_epidemics.R`, which needs R with epidemics), or the JSON it writes (or its contents)
        run (int): which call of a model function in the script to read, if it has several (default: the first)
        verbose (bool): whether to print how each transition was read
        kwargs (dict): passed to `ss.Sim()`

    Returns:
        An `ss.Sim`, with `sim.imported` saying how it was read and what was not translated

    Example:
        ```python
        sim = ss.import_from('model.R', format='epidemics') # Or the JSON from: Rscript dump_epidemics.R model.R model.json
        sim.run()
        ```
    """
    d = read(source)
    runs = d['runs']
    r = runs[run]
    model = r['model']
    if model not in translated:
        errormsg = f'The script runs epidemics::{model}(), which is not translated; only {sc.strjoin(translated)}() is'
        raise NotImplementedError(errormsg)
    file = source if isinstance(source, str) else d.get('file')
    src = Source(name='seirv', format='epidemics', file=file)
    src.totals = ['N']
    if len(runs) > 1:
        src.notes.append(f'runs          the script runs {len(runs)} models; this is run {run} (choose with run=)')

    # The population: groups, contacts, and initial conditions
    pop = r['population']
    dem = np.array(pop['demography_vector'], dtype=float)
    n = len(dem)
    groups = pop.get('groups') or [f'group{i}' for i in range(n)]
    cm = rows(pop['contact_matrix'], n)
    edges = age_edges(groups) if n > 1 else None
    dim = 'age' if edges is not None else 'group'
    levels = edges if edges is not None else [str(g) for g in groups]
    by = lambda vals: ss.by(list(map(float, vals)), dims=[dim]) if n > 1 else float(vals[0])

    # epidemics takes the transpose of the matrix it is given, divides it by its largest eigenvalue, and by the size of each row's group:
    # lambda_i = beta * sum_j C[j,i] / rho / d_i * I_j. As contacts per person, for beta*I (beta * sum_j n_ij * I_j / N_j), n_ij = C[j,i] / rho * d_j / d_i
    M = cm.T
    rho = float(np.max(np.linalg.eigvals(M).real))
    contacts = M / rho * dem[None, :] / dem[:, None]
    src.notes.append(f'contacts      the contact matrix of "{pop.get("name") or "the population"}", transposed and divided by its largest eigenvalue ({rho:.4g}), as epidemics does, then as contacts a day per person')

    comps = pop['compartments'] or list('SEIRV')
    init = rows(pop['initial_conditions'], n) * dem[:, None] # The number in each compartment, by group
    for k,c in enumerate(['S', 'E', 'I', 'R', 'V']):
        vals = init[:, k]
        src.add_state(c, init=None if c == 'S' or not vals.sum() else by(vals), label=comps[k] if k < len(comps) else None)

    # Parameters
    pars = r['pars']
    for key,name in names.items():
        val = np.atleast_1d(pars[key])
        if len(val) > 1:
            src.skip(f'{key} (several values)', f'epidemics runs a scenario for each of the {len(val)} values; this is the first ({val[0]:n})')
        src.add_par(name, ss.perday(float(val[0])) if name != 'beta' else float(val[0]), doc=f'epidemics {key}' + (', per contact a day' if name == 'beta' else ', per day'))
    for key in r.get('time_dependence') or []:
        src.skip(f'time_dependence of {key}', 'an R function of time, which is not translated; write it as an ss.timeseries() or ss.seasonal() value of the parameter')
    if r.get('population_change'):
        src.skip('population_change', 'changes in the size of the groups are not translated')

    # Transitions
    src.add_flow('S', 'E', 'beta * I / N', name='infection', label='infection, through the contact matrix')
    src.add_flow('E', 'I', 'sigma', name='progression', label='becoming infectious')
    src.add_flow('I', 'R', 'gamma', name='recovery', label='recovery')

    # Vaccination: nu is the fraction of each group vaccinated a day, so nu * d_i people a day in group i, from S while there are any
    vax = r.get('vaccination')
    if vax:
        nu = rows(vax['nu'], n)
        begin, end = rows(vax['time_begin'], n), rows(vax['time_end'], n)
        if nu.shape[1] > 1:
            src.skip('vaccination (doses)', f'{nu.shape[1]} doses; only the first is translated')
        doses = nu[:, 0] * dem
        same = np.all(begin[:, 0] == begin[0, 0]) and np.all(end[:, 0] == end[0, 0])
        terms = []
        for i in range(n):
            if not doses[i]:
                continue
            src.add_par(f'in_{i}', ss.by(list(np.eye(n)[i]), dims=[dim]) if n > 1 else 1.0, doc=f'1 in group {groups[i]}, else 0')
            on = 'vax_on' if same else f'vax_on_{i}'
            if on not in src.pars:
                src.add_par(on, ss.timeseries({0.0: 0.0, float(begin[i, 0]): 1.0, float(end[i, 0]) + 1e-9: 0.0}), doc=f'1 while vaccinating{"" if same else f" group {groups[i]}"} (from day {begin[i, 0]:n} to day {end[i, 0]:n}), else 0')
            terms.append(f'in_{i} * {doses[i]!r} * {on} / (in_{i} * S).n')
        if terms:
            src.add_flow('S', 'V', ' + '.join(terms), name='vaccination', label='vaccination, of a number of people a day in each group')
            src.notes.append(f'vaccination   {sc.strjoin(f"{x:n}" for x in doses)} people a day by group, as nu ({sc.strjoin(f"{x:n}" for x in nu[:, 0])}) times the size of each group; (in_i * S).n is the number in S in group i')

    # Interventions
    ivs = r.get('intervention') or {}
    mods = []
    for key,iv in ivs.items():
        k = len(np.atleast_1d(iv['time_begin']))
        if key == 'contacts':
            red = rows(iv['reduction'], n)
            if k > 1:
                src.notes.append(f'contacts      {k} interventions on contacts, as a modifier each: epidemics adds up the reductions where they overlap (to at most 1), while modifiers multiply')
            for j in range(k):
                f = by(1 - np.minimum(red[:, j], 1))
                when = window(np.atleast_1d(iv['time_begin'])[j], np.atleast_1d(iv['time_end'])[j])
                for target in ['seirv.infection', 'seirv.rel_trans']:
                    mods.append(f'ss.modify({target!r}, multiply={code_value(f)}, start={when["start"]!r}, stop={when["stop"]!r})')
            src.notes.append(f'contacts      intervention "{iv.get("name")}": the contacts of each group scaled down at both ends, as modifiers of infection (whom it reaches) and infectiousness (whom it comes from)')
        elif key in names:
            red = np.atleast_1d(np.array(iv['reduction'], dtype=float)).reshape(-1, k)[0]
            for j in range(k):
                when = window(np.atleast_1d(iv['time_begin'])[j], np.atleast_1d(iv['time_end'])[j])
                mods.append(f'ss.modify(\'seirv.pars.{names[key]}\', multiply={1 - min(red[j], 1)!r}, start={when["start"]!r}, stop={when["stop"]!r})')
            if key != 'transmission_rate':
                src.notes.append(f'{key:13s} epidemics 0.5 applies the first rate intervention to transmission, whatever its parameter; this applies it to {key}')
        else:
            src.skip(f'intervention on {key}', 'a parameter model_default() does not have')
    if mods:
        src.config.interventions = Code('[\n    ' + ',\n    '.join(mods) + ',\n]')

    # Groups, contacts, and run settings
    if n > 1:
        src.stratify(**{dim:levels}, fractions=(dem/dem.sum()).tolist())
        src.config.networks = Code(f'ss.contacts({dim}={np.round(contacts, 15).tolist()!r})')
    else:
        src.config.networks = Code(f'ss.MixingPool(n_contacts={float(contacts[0, 0])!r})')
    src.config.update(unit='day', dt=float(pars.get('increment', 1)), start=0, stop=float(pars.get('time_end', 100)), solver='ode', total_pop=float(dem.sum()))
    if edges is not None:
        src.config.aging = False
    return build(src, verbose=verbose, **kwargs)

