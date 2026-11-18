"""
Epiverse-TRACE: reading its parameters and contact surveys into Starsim, and writing a sim's outbreak as the data its packages analyze

Epiverse-TRACE (https://epiverse-trace.github.io) is a set of R packages for outbreak analytics. Its models
(from the epidemics package) are imported with `ss.import_from()` (see `ss.converters.epidemics`); the rest is here,
as `ss.converters.epiverse`:

- In: `parameter()` gives an estimate from the epiparameter library (an incubation period, serial interval, delay,
  or offspring distribution) as a Starsim distribution, and `parameters()` lists them; `contacts()` reads a contact
  matrix from socialmixr.
- Out: `linelist()` gives a sim's cases as a line list (for linelist, cleanepi, and cfr), `incidence()` its counts
  over time (for incidence2 and cfr), and `chains()` and `offspring()` who infected whom (for epichains and superspreading).

Used with: agents and compartments (a line list, chains, and offspring: agents only); declared and coded modules
"""
import json
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

sps = sc.lazyimport('scipy.stats') # Imported when first used, since most runs never fit a distribution
spo = sc.lazyimport('scipy.optimize')
sspec = sc.lazyimport('scipy.special')

__all__ = ['parameters', 'parameter', 'contact_matrix', 'contacts', 'linelist', 'incidence', 'chains', 'offspring']


#%% Parameters (epiparameter)

version = 'v0.1.0' # The release of epiparameterDB read by default: the one the epiparameter R package on CRAN reads
url = 'https://raw.githubusercontent.com/epiverse-trace/epiparameterDB/{version}/inst/extdata/parameters.json'
loaded = {} # The database, by version, once read


def load(version=version):
    """ The entries of the database, downloaded on first use into starsim_data's cache """
    if version not in loaded:
        ssd = ss.data.starsim_data()
        try:
            path = ssd.cache.cached_download(url.format(version=version), filename=f'epiparameterDB_{version}.json', subdir='epiparameter', verbose=False)
        except OSError as E:
            errormsg = f'The epiparameter database ({url.format(version=version)}) could not be downloaded into the cache ({ssd.cache_dir()}): {E}'
            raise ConnectionError(errormsg) from E
        loaded[version] = json.loads(sc.path(path).read_text())
    return loaded[version]


def value(x):
    """ A number from the JSON, or None """
    return float(x) if sc.isnumber(x) and np.isfinite(x) else None


def sym_percentiles(names, values):
    """ The pair of percentiles that epiparameter fits to: one at or below 50 and one above, as nearly symmetric as possible """
    pairs = [(float(n), value(v)) for n,v in zip(names, values) if value(v) is not None]
    lower = [p for p in pairs if p[0] <= 50]
    upper = [p for p in pairs if p[0] > 50]
    if not lower or not upper:
        return None
    return min(((a, b) for b in upper for a in lower), key=lambda ab: abs(ab[0][0] - (100 - ab[1][0]))) # Ties go to the first in R's (column-major) order


def from_percentiles(family, lo, hi):
    """ The parameters of a distribution with two given percentiles, (percent, value) each: solved exactly, as epiparameter's `extract_param()` fits them """
    (p1, q1), (p2, q2) = (lo[0]/100, lo[1]), (hi[0]/100, hi[1])
    if family == 'lnorm':
        z1, z2 = sps.norm.ppf([p1, p2])
        sdlog = (np.log(q2) - np.log(q1)) / (z2 - z1)
        return dict(meanlog=np.log(q1) - sdlog*z1, sdlog=sdlog)
    elif family == 'gamma':
        f = lambda logk: np.log(sps.gamma.ppf(p2, np.exp(logk)) / sps.gamma.ppf(p1, np.exp(logk))) - np.log(q2/q1)
        with np.errstate(divide='ignore'): # A tiny shape puts the lower percentile at 0
            shape = np.exp(spo.brentq(f, np.log(1e-3), np.log(1e4)))
        return dict(shape=shape, scale=q1/sps.gamma.ppf(p1, shape))
    elif family == 'weibull':
        shape = np.log(np.log(1 - p2) / np.log(1 - p1)) / np.log(q2/q1)
        return dict(shape=shape, scale=q1 / (-np.log(1 - p1))**(1/shape))
    return None


def from_summary(family, ss_):
    """ The parameters of a distribution from its summary statistics, by the rules of epiparameter (mean and SD; median and dispersion; or two percentiles), or None """
    mean, sd, median, disp = [value(ss_.get(k)) for k in ['mean', 'sd', 'median', 'dispersion']]
    if mean is not None and sd is not None:
        if family == 'gamma':
            return dict(shape=mean**2/sd**2, scale=sd**2/mean)
        elif family == 'lnorm':
            return dict(meanlog=np.log(mean**2/np.sqrt(sd**2 + mean**2)), sdlog=np.sqrt(np.log(sd**2/mean**2 + 1)))
        elif family == 'weibull':
            f = lambda k: sd**2/mean**2 - sspec.gamma(1 + 2/k)/sspec.gamma(1 + 1/k)**2 + 1 # As epiparameter solves it
            shape = spo.brentq(f, 0.1, 1000)
            return dict(shape=shape, scale=mean/sspec.gamma(1 + 1/shape))
        elif family == 'norm':
            return dict(mean=mean, sd=sd)
    if family == 'lnorm' and median is not None and disp is not None:
        return dict(meanlog=np.log(median), sdlog=np.log(disp))
    pair = sym_percentiles(ss_.get('quantile_names') or [], ss_.get('quantile_values') or [])
    if pair is not None:
        return from_percentiles(family, *pair)
    return None


def entry_pars(entry):
    """ An entry's distribution and its parameters (given, or from its summary statistics), or (family, None) if it has none """
    pd_ = entry['probability_distribution']
    family = pd_.get('prob_distribution')
    given = {k:value(v) for k,v in (pd_.get('parameters') or {}).items() if not k.endswith(('_ci', '_ci_limits')) and k != 'precision' and value(v) is not None}
    if family is None:
        return None, None
    if given:
        return family, given
    return family, from_summary(family, entry.get('summary_statistics') or {})


def to_dist(family, pars, unit=None, offset=0):
    """ A Starsim distribution from epiparameter's name for it and its parameters """
    kw = dict(unit=unit) if unit else {}
    loc = dict(loc=float(offset)) if offset else {}
    if family == 'gamma':
        return ss.gamma(shape=pars['shape'], scale=pars['scale'], **loc, **kw)
    elif family == 'weibull':
        return ss.weibull(shape=pars['shape'], scale=pars['scale'], **loc, **kw)
    elif family == 'lnorm':
        if offset:
            errormsg = f'This entry is a lognormal shifted by {offset}, which ss.lognormal() cannot be; use its parameters ({pars}) by hand'
            raise ValueError(errormsg)
        return ss.lognormal(log_mean=pars['meanlog'], log_std=pars['sdlog'], **kw)
    elif family == 'nbinom':
        return ss.nbinom(n=pars['dispersion'], mean=pars['mean'], **kw)
    elif family == 'norm':
        return ss.normal(mean=pars['mean'] + offset, std=pars['sd'], **kw)
    errormsg = f'The distribution "{family}" has no counterpart here'
    raise ValueError(errormsg)


def citation(entry):
    """ The first author, year, and DOI of an entry, as text """
    c = entry.get('citation') or {}
    authors = c.get('author') or []
    first = (authors[0].get('family') or authors[0].get('given') or '') if authors else '' # An organization is a given name only
    etal = ' et al.' if len(authors) > 1 else ''
    doi = f', doi:{c["doi"]}' if c.get('doi') else ''
    return f'{first}{etal} ({c.get("year", "n.d.")}){doi}'


def parameters(disease=None, epi_name=None, version=version):
    """
    The entries of the Epiverse-TRACE parameter library (epiparameter) as a table: one row per entry, with its distribution

    The library (https://github.com/epiverse-trace/epiparameterDB) holds published estimates of incubation
    periods, serial intervals, delays, and offspring distributions for about 20 diseases, each with its source.
    It is downloaded on first use (about 300 kB) into starsim_data's cache.

    Args:
        disease (str): only entries whose disease or pathogen contains this, ignoring case (e.g. 'ebola')
        epi_name (str): only entries whose parameter contains this, ignoring case (e.g. 'incubation')
        version (str): the release of epiparameterDB (default 'v0.1.0', as the epiparameter R package reads), or 'main' for the latest

    Returns:
        A dataframe with the disease, the parameter (`epi_name`), the distribution, its parameters, the unit,
        the sample size, the region, the citation, and the DOI

    Example:
        ```python
                ss.converters.epiverse.parameters('covid', 'incubation')
        ```
    """
    rows = []
    for i,e in enumerate(load(version)):
        if not matches(e, disease, epi_name):
            continue
        family, pars = entry_pars(e)
        meta = e.get('metadata') or {}
        rows.append(dict(index=i, disease=e['disease'], epi_name=e['epi_name'], distribution=family, parameters=pars, units=meta.get('units'),
                         sample_size=meta.get('sample_size'), region=meta.get('region'), citation=citation(e), doi=(e.get('citation') or {}).get('doi')))
    return pd.DataFrame(rows).set_index('index') if rows else pd.DataFrame()


def matches(entry, disease, epi_name):
    """ Whether an entry's disease (or pathogen) and parameter contain the given text, ignoring case """
    ok = lambda text, s: s is None or s.lower() in str(text).lower()
    return (ok(entry['disease'], disease) or ok(entry.get('pathogen'), disease)) and ok(entry['epi_name'], epi_name)


def parameter(disease, epi_name, author=None, index=None, version=version):
    """
    One estimate from the Epiverse-TRACE parameter library (epiparameter), as a Starsim distribution

    Chooses as `epiparameter::epiparameter_db(single_epiparameter = TRUE)` does: among the entries that match,
    those with a distribution, then those adjusted for right truncation, then the one with the largest sample size.
    An entry given only as summary statistics is converted as epiparameter converts it: from its mean and standard
    deviation, its median and dispersion (a lognormal), or two of its percentiles. A duration in days is a
    distribution of durations (`unit='day'`), so it can go straight into a transition (`dur=`).

    Args:
        disease (str): the disease or pathogen, or part of its name, ignoring case (e.g. 'ebola', 'COVID-19')
        epi_name (str): the parameter, or part of its name (e.g. 'incubation period', 'serial interval', 'onset to death', 'offspring')
        author (str): only entries whose first author contains this
        index (int): instead, the entry at this row of `parameters()`
        version (str): the release of epiparameterDB (see `parameters()`)

    Returns:
        An `ss.Dist`, with `dist.ref`, a dict of where it came from (the citation, DOI, region, sample size, and the entry's index)

    Example:
        ```python
        import starsim as ss
        
        incubation = ss.converters.epiverse.parameter('ebola', 'incubation period') # ss.gamma(shape=1.58, scale=6.53, unit='day')
        S, E, I, R = ss.states('S E I R')
        ebola = ss.Disease(S >> E @ 0.3*I, ss.Transition(E, I, dur=incubation), I >> R @ ss.days(10))
        ```
    """
    db = load(version)
    if index is not None:
        chosen = [(index, db[index])]
    else:
        found = [(i, e) for i,e in enumerate(db) if matches(e, disease, epi_name)]
        if author is not None:
            found = [(i, e) for i,e in found if author.lower() in citation(e).lower()]
        if not found:
            options = sorted({e['disease'] for e in db}) if not any(matches(e, disease, None) for e in db) else sorted({e['epi_name'] for e in db if matches(e, disease, None)})
            what = 'diseases' if not any(matches(e, disease, None) for e in db) else f'parameters for "{disease}"'
            errormsg = f'No entry of the epiparameter library matches disease="{disease}", epi_name="{epi_name}"' + (f', author="{author}"' if author else '') + f'; the {what} are: {sc.strjoin(options)}'
            raise sc.KeyNotFoundError(errormsg)
        chosen = [(i, e) for i,e in found if entry_pars(e)[1] is not None] or found
        adjusted = [(i, e) for i,e in chosen if (e.get('method_assessment') or {}).get('right_truncated')]
        chosen = adjusted or chosen
    size = lambda ie: (ie[1].get('metadata') or {}).get('sample_size') or -1
    i, e = max(chosen, key=size) # The first of the largest, as R's which.max()
    family, pars = entry_pars(e)
    if pars is None:
        errormsg = f'The entry ({e["disease"]}, {e["epi_name"]}, {citation(e)}) has no distribution, or no summary statistics it can be worked out from'
        raise ValueError(errormsg)
    meta = e.get('metadata') or {}
    unit = 'day' if meta.get('units') == 'days' else None
    dist = to_dist(family, pars, unit=unit, offset=e['probability_distribution'].get('offset') or 0)
    dist.ref = dict(provider='epiparameter', version=version, index=i, disease=e['disease'], epi_name=e['epi_name'], citation=citation(e),
                    doi=(e.get('citation') or {}).get('doi'), region=meta.get('region'), sample_size=meta.get('sample_size'))
    return dist


#%% Contact matrices (socialmixr)

def contact_matrix(source):
    """
    A contact matrix written by socialmixr, as a table by age group

    socialmixr's `contact_matrix()` gives the mean number of contacts a day that a participant in each age group
    (the rows) has with people in each age group (the columns), which is what `ss.contacts()` takes. Save it in R with
    `write.csv(m$matrix, "contacts.csv")`, or the whole result with `jsonlite::write_json(m, "contacts.json", digits = NA)`.

    Args:
        source (str/dict/DataFrame): the CSV or JSON file, or what `sc.loadjson()` gives for the JSON, or a table

    Returns:
        A dataframe of contacts a day, with the age groups as labels ('[0,5)', ..., '[65,Inf)'), which `ss.contacts()` reads
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, dict) or str(source).lower().endswith('.json'):
        d = source if isinstance(source, dict) else sc.loadjson(source)
        matrix = np.array(d['matrix'], dtype=float)
        groups = None
        for key in ['participants', 'demography']: # Either has the age groups, in order
            rows = d.get(key)
            if rows and 'age.group' in rows[0] and len(rows) == len(matrix):
                groups = [r['age.group'] for r in rows]
                break
        if groups is None:
            errormsg = 'The JSON has no age groups for the rows of the matrix ("participants" or "demography", each with "age.group"); save it with jsonlite::write_json(m) from the whole result of contact_matrix()'
            raise ValueError(errormsg)
        df = pd.DataFrame(matrix, index=groups, columns=groups)
    else:
        df = pd.read_csv(source, index_col=0)
    df.index = [str(i) for i in df.index]
    df.columns = [str(c) for c in df.columns]
    if not ss.data.is_age_index(df.index):
        errormsg = f'The rows of the matrix are not labeled by age group (e.g. "[0,5)"), but {list(df.index)}'
        raise ValueError(errormsg)
    df.attrs['ref'] = dict(provider='socialmixr')
    return df


def contacts(source, **kwargs):
    """
    Mixing between age groups from a contact matrix written by socialmixr, as `ss.contacts()`

    The matrix is regrouped to the model's age groups, as for any matrix by age (see `ss.contacts()`). The same as
    `ss.contacts(ss.converters.epiverse.contact_matrix(source))`; a CSV written by socialmixr can also go straight into `ss.contacts()`.

    Args:
        source (str/dict/DataFrame): the CSV or JSON written from socialmixr's `contact_matrix()` (see `contact_matrix()`)
        kwargs (dict): passed to `ss.contacts()`, e.g. `name='polymod'`

    Example:
        ```python
        # In R: m <- socialmixr::contact_matrix(polymod, countries = "United Kingdom", age_limits = c(0, 5, 18, 65), symmetric = TRUE)
        #       write.csv(m$matrix, "uk_contacts.csv")
        import starsim as ss
        
        S, I, R = ss.states('S I R')
        flu = ss.Disease(S >> I @ 0.05*I, I >> R @ ss.days(5), init_prev=0.001).stratify(age=[0, 5, 18, 65])
        sim = ss.Sim(flu, networks=ss.converters.epiverse.contacts('uk_contacts.csv'), unit='day', stop=200)
        ```
    """
    kwargs.setdefault('unit', 'day')
    return ss.contacts(contact_matrix(source), **kwargs)


#%% Outputs (linelist, incidence2, cfr, epichains, superspreading)

outcome_names = dict(D='died', dead='died', died='died', R='recovered', recovered='recovered') # The states that are outcomes, by default, and what each is called


def get_disease(sim, disease):
    """ The disease, by name or index (default: the only one) """
    if disease is None:
        if len(sim.diseases) != 1:
            errormsg = f'The sim has {len(sim.diseases)} diseases ({sc.strjoin(sim.diseases.keys())}); choose one with disease='
            raise ValueError(errormsg)
        disease = 0
    return sim.diseases[disease]


def transmissions(dis):
    """ The transitions of a disease that are transmission, and the states they infect people into """
    trans = [t for t in dis.transitions.values() if t.acquisition]
    if not trans:
        errormsg = f'The disease "{dis.name}" has no transmission'
        raise ValueError(errormsg)
    dests = [d.name for t in trans for d in t.dest]
    return trans, dests


def times(sim):
    """ The time of each step: dates (as pandas timestamps) if the sim has a calendar, else numbers in its unit """
    tv = sim.timevec
    if isinstance(tv, ss.DateArray) and tv.is_date:
        return pd.DatetimeIndex([pd.Timestamp(str(d)) for d in tv])
    return pd.Index(np.asarray(tv, dtype=float))


def at(tvec, ti):
    """ The times of these step indices (NaN where an index is NaN) """
    ti = np.asarray(ti, dtype=float)
    out = pd.Series(pd.NaT if isinstance(tvec, pd.DatetimeIndex) else np.nan, index=range(len(ti)), dtype=tvec.dtype)
    ok = ~np.isnan(ti)
    out[ok] = tvec[ti[ok].astype(int)]
    return out.values


def entered(dis, state, uids):
    """ The step index at which each person last entered a state (NaN if never), which needs the times kept """
    try:
        return np.asarray(getattr(dis, f'ti_{state}').raw[uids], dtype=float)
    except AttributeError as E:
        errormsg = f'The time each person entered "{state}" is not kept; run the sim with solver=ss.solvers.abm(keep_times=True), or declare the state with ss.State(keep_time=True)'
        raise ValueError(errormsg) from E


def log_df(dis):
    """ The disease's infection log as a dataframe, or None if it was not kept """
    log = getattr(dis, 'infection_log', None)
    if log is None:
        return None
    df = log.to_df()
    return df if len(df) else None


def linelist(sim, disease=None, onset=None, outcomes=None):
    """
    A line list of a sim's cases, one row per infection, as Epiverse-TRACE's packages read it (e.g. linelist, cleanepi, cfr)

    The columns follow simulist's line lists: `id`, `sex` ("f" or "m"), `age` (in years, at infection), `date_infection`,
    `date_onset`, `outcome`, and `date_outcome`, with who infected them (`source`, NaN if not known) and the route
    (`network`) if the infection log was kept (`analyzers='infection_log'`). A date is a date if the sim has a
    calendar, else a time in its unit since the start. A person infected more than once has a row for each infection,
    and onset and outcome only for the last, since the time of entering a state is kept only for the last time.

    Agents only, and the times of entering the onset and outcome states must be kept: run the sim with
    `solver=ss.solvers.abm(keep_times=True)`. The column for each linelist tag is in `df.attrs['tags']`, for R's
    `linelist::make_linelist(df, id = "id", date_onset = "date_onset", ...)`.

    Args:
        sim (Sim): a sim that has run
        disease (str/int): the disease (default: the only one)
        onset (str): the state whose entry is the onset of symptoms (default: `I`, if the disease has it)
        outcomes (dict): the states that are outcomes, and what each is called in `outcome`, e.g. `dict(D='died', R='recovered')`
            (the default, for those of these states the disease has, and `dead`, `died`, and `recovered`)

    Returns:
        A dataframe, one row per infection, in the order they happened

    Example:
        ```python
        import starsim as ss
        
        S, E, I, R, D = ss.states('S E I R D')
        ebola = ss.Disease(S >> E @ 0.4*I, E >> I @ ss.days(9), ss.Transition(I, {D: 0.5, R: None}, dur=ss.days(8)), init_prev=0.005, name='ebola')
        sim = ss.Sim(ebola, networks='random', n_agents=5000, start='2025-01-01', stop='2025-07-01', unit='day',
                     analyzers='infection_log', solver=ss.solvers.abm(keep_times=True)).run()
        df = ss.converters.epiverse.linelist(sim)
        df.to_csv('linelist.csv', index=False) # For R
        ```
    """
    dis = get_disease(sim, disease)
    if not sim.solver.agents:
        errormsg = 'A line list is of people, so needs a sim run as agents'
        raise ValueError(errormsg)
    tvec = times(sim)
    _, dests = transmissions(dis)
    states = list(dis.states.keys()) if hasattr(dis, 'states') else []
    onset = sc.ifelse(onset, 'I' if 'I' in states else None)
    if outcomes is None:
        outcomes = {k:v for k,v in outcome_names.items() if k in states}

    # One row per infection: from the log if it was kept, else from when each person entered the state they are infected into
    log = log_df(dis)
    if log is not None:
        df = pd.DataFrame(dict(id=log.target.values.astype(int), date_infection=log.t.values))
        src = log.source.values.astype(float)
        df['source'] = np.where(src < 0, np.nan, src) # Seeds have a source of -1
        if 'network' in log:
            df['network'] = log.network.values
        ti_inf = np.searchsorted(np.asarray(tvec), np.asarray(log.t.values)) # The step of each infection
    else:
        ti = np.full(len(sim.people.uid.raw), np.nan)
        for d in dests:
            e = entered(dis, d, slice(None))
            ti = np.fmin(ti, e) if not np.all(np.isnan(ti)) else e
        uids = np.flatnonzero(~np.isnan(ti))
        order = np.argsort(ti[uids], kind='stable')
        uids, ti_inf = uids[order], ti[uids][order]
        df = pd.DataFrame(dict(id=uids, date_infection=at(tvec, ti_inf)))
    uids = df.id.values
    last = ~pd.Series(uids).duplicated(keep='last').values # The last infection of each person, which the kept times are for

    # Who they are
    people = sim.people
    df.insert(1, 'sex', np.where(np.asarray(people.female.raw[uids], dtype=bool), 'f', 'm'))
    age = np.asarray(people.age.raw[uids], dtype=float)
    if sim.pars.aging: # People have aged since: back to their age at infection, from their age at the end, or at death
        end = np.full(len(uids), float(sim.t.ti))
        if hasattr(people, 'ti_dead'):
            dead = np.asarray(people.ti_dead.raw[uids], dtype=float)
            end = np.where(np.isnan(dead) | (dead > end), end, dead)
        age = age - (end - np.asarray(ti_inf, dtype=float)) * float(sim.t.dt_year)
    df.insert(2, 'age', age)

    # Onset and outcome, for the last infection of each person
    if onset is not None:
        ti_on = entered(dis, onset, uids)
        ti_on = np.where(last & (ti_on >= ti_inf), ti_on, np.nan)
        df.insert(df.columns.get_loc('date_infection') + 1, 'date_onset', at(tvec, ti_on))
    if outcomes:
        out = np.full(len(uids), None, dtype=object)
        ti_out = np.full(len(uids), np.nan)
        for state,label in outcomes.items():
            ti = entered(dis, state, uids)
            hit = last & (ti >= ti_inf) & (np.isnan(ti_out) | (ti < ti_out))
            out[hit] = label
            ti_out[hit] = ti[hit]
        pos = df.columns.get_loc('date_onset') + 1 if 'date_onset' in df else df.columns.get_loc('date_infection') + 1
        df.insert(pos, 'outcome', out)
        df.insert(pos + 1, 'date_outcome', at(tvec, ti_out))
    df.attrs['tags'] = {tag:tag for tag in ['id', 'date_onset', 'date_outcome', 'outcome', 'age'] if tag in df} | dict(gender='sex')
    return df


def incidence(sim, counts=None, disease=None, wide=False):
    """
    New cases (or other counts) on each step, as incidence2 and cfr read them

    In the long form (the default), the columns are incidence2's: `date_index`, `count_variable`, and `count`. In the
    wide form, they are `date` and a column for each count, e.g. `date`, `cases`, `deaths`, as `cfr::cfr_static()` takes.
    Any solver: in a compartmental run, the counts are numbers of people, not whole.

    Args:
        sim (Sim): a sim that has run
        counts (dict): the counts, by name, each a result of the disease (e.g. `dict(cases='infection', deaths='outcome.D')`); by default,
            `cases` (its transmissions) and `deaths` (its flows into a state named `D`, `dead`, or `died`, if any)
        disease (str/int): the disease (default: the only one)
        wide (bool): whether to give one column per count, rather than the long form

    Returns:
        A dataframe, one row per step (and count, in the long form)

    Example:
        ```python
        df = ss.converters.epiverse.incidence(sim, wide=True) # date, cases, deaths
        df.to_csv('incidence.csv', index=False)     # In R: cfr::cfr_static(read.csv("incidence.csv"))
        ```
    """
    dis = get_disease(sim, disease)
    res = sim.results[dis.name]
    if counts is None:
        trans, _ = transmissions(dis)
        counts = dict(cases=[t.name for t in trans])
        deaths = [key for key in res.keys() if '.' in key and key.split('.')[-1] in ['D', 'dead', 'died']]
        deaths += [t.name for t in dis.transitions.values() if len(t.dest) == 1 and t.dest[0].name in ['D', 'dead', 'died']]
        if deaths:
            counts['deaths'] = deaths
    tvec = times(sim)
    data = {}
    for name,keys in counts.items():
        data[name] = np.sum([np.asarray(res[k].values, dtype=float) for k in sc.tolist(keys)], axis=0)
    wide_df = pd.DataFrame(dict(date=tvec, **data))
    if wide:
        return wide_df
    df = wide_df.melt(id_vars='date', var_name='count_variable', value_name='count').rename(columns=dict(date='date_index'))
    return df


def chains(sim, disease=None):
    """
    Who infected whom, as a tree of chains of transmission, as epichains gives them: `chain`, `infector`, `infectee`, `generation`, and `time`

    Each seed (a person infected at the start, or by no one in the model) starts a chain, numbered from 1, in which
    it is generation 1, with an infector of NaN. Needs the infection log (`analyzers='infection_log'`), with
    transmission over networks, where each infection has a source; infections by a mixing pool or a contact matrix
    have no particular source (unless run event by event, `solver=ss.solvers.abm(step='events')`), so each starts a chain.

    Args:
        sim (Sim): a sim that has run, with `analyzers='infection_log'`
        disease (str/int): the disease (default: the only one)

    Returns:
        A dataframe, one row per infection, in the order they happened

    Example:
        ```python
        tree = ss.converters.epiverse.chains(sim)
        tree.groupby('chain').size() # The size of each chain
        ```
    """
    dis = get_disease(sim, disease)
    log = log_df(dis)
    if log is None:
        errormsg = 'Chains of transmission need the infection log: run the sim with analyzers=\'infection_log\''
        raise ValueError(errormsg)
    src = log.source.values.astype(float)
    tgt = log.target.values.astype(int)
    current = {} # The row of each person's latest infection so far, so a reinfected infector is in the chain of their latest infection
    chain = np.zeros(len(log), dtype=int)
    gen = np.zeros(len(log), dtype=int)
    n_chains = 0
    for i in range(len(log)):
        s = src[i]
        j = current.get(int(s)) if np.isfinite(s) and s >= 0 else None
        if j is None:
            n_chains += 1
            chain[i], gen[i] = n_chains, 1
        else:
            chain[i], gen[i] = chain[j], gen[j] + 1
        current[tgt[i]] = i
    known = np.isfinite(src) & (src >= 0) & np.array([int(s) in current for s in np.nan_to_num(src, nan=-1)])
    if not np.any(known) and len(log) > 1:
        ss.note('No infection in the log has a source, so every one is its own chain: transmission by a mixing pool or contact matrix has no particular source; use a network, or solver=ss.solvers.abm(step=\'events\')')
    return pd.DataFrame(dict(chain=chain, infector=np.where(np.isfinite(src) & (src >= 0), src, np.nan), infectee=tgt, generation=gen, time=log.t.values))


def offspring(sim, disease=None, until=None):
    """
    The number of people each case infected: the offspring distribution, as superspreading fits it (its R and dispersion, k)

    Counts each infection's onward infections, from the infection log (`analyzers='infection_log'`). Cases infected near the end
    of the sim have not finished infecting others, so give `until=` to count only those infected by then.

    Args:
        sim (Sim): a sim that has run, with `analyzers='infection_log'`
        disease (str/int): the disease (default: the only one)
        until (date/float): count only cases infected up to this time (a date, or a time in the sim's unit, as in the log)

    Returns:
        A dataframe with each case's `id`, `time` of infection, and `n_offspring`

    Example:
        ```python
        counts = ss.converters.epiverse.offspring(sim, until='2025-05-01').n_offspring
        counts.to_csv('offspring.csv', index=False) # In R: superspreading's fitdistrplus::fitdist(x, "nbinom")
        ```
    """
    tree = chains(sim, disease)
    tree['row'] = np.arange(len(tree))
    current = {}
    parent = np.full(len(tree), -1)
    for i,(s, t) in enumerate(zip(tree.infector.values, tree.infectee.values)):
        if np.isfinite(s) and int(s) in current:
            parent[i] = current[int(s)]
        current[int(t)] = i
    n = np.bincount(parent[parent >= 0], minlength=len(tree))
    df = pd.DataFrame(dict(id=tree.infectee.values, time=tree.time.values, n_offspring=n))
    if until is not None:
        cut = pd.Timestamp(str(until)) if isinstance(tree.time.dtype, np.dtype) and np.issubdtype(tree.time.dtype, np.datetime64) else float(until)
        df = df[df.time <= cut].reset_index(drop=True)
    return df
