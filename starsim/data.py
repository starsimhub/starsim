"""
Data by place: ages, populations, birth and death rates, contact matrices, and administrative units

    ss.data.age('Kenya')                    # The age_data of ss.People: counts by 5-year age group ('world' needs no download)
    ss.data.population('Kenya', year=2020)  # By age and sex (also for a US state, e.g. 'United States:Massachusetts')
    ss.data.rates('Kenya', 'deaths')        # Birth, death, or fertility rates, by year
    ss.data.contacts('Kenya', layer='home') # A contact matrix by age
    ss.data.geography('Kenya', level=1)     # Administrative units, with populations and positions

These are tables. The model objects that take a place by name read them from here:
`ss.people('Kenya')`, `ss.geo.geography('Kenya', level=1)`, `ss.contacts('Kenya')`,
`ss.Births(birth_rate='Kenya')`, `ss.Deaths(death_rate='Kenya')`, and `ss.Pregnancy(fertility_rate='Kenya')`. The data comes from
`starsim_data` (https://github.com/starsimhub/starsim_data), which bundles what is
small and common and fetches the rest on demand. Each lookup returns the data with a
record of where it came from (`ref`; for a table, in `.attrs['ref']`), which `sim.describe()` shows.
"""
import re
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

__all__ = ['age', 'population', 'rates', 'contacts', 'geography', 'world', 'age_split', 'age_labels', 'age_edges', 'provider']

layers = ['all', 'home', 'school', 'work', 'community'] # The settings of the contact matrices


# The default ages of the people of a declared model, as a guess: the world's population in 2024 by 5-year age group
# (the last is 100 and over), in millions; the sum of the 187 countries of the UN World Population Prospects 2024 in starsim_data
world = sc.obj(
    ages   = np.arange(0, 101, 5, dtype=float),
    counts = np.array([664.2, 666.0, 663.6, 647.4, 625.3, 609.7, 601.1, 566.7, 530.8, 498.0, 452.7,
                       410.7, 353.9, 296.2, 219.8, 154.3, 98.6, 49.7, 19.8, 4.7, 0.6]),
    guess  = "ages as the world's population in 2024",
)


def age_labels(edges):
    """ The label of each age group from its lower edge, as a stratification by age labels them, e.g. '0-4', '5-9', and '75+' (a single year is '3') """
    ends = list(edges[1:]) + [None]
    return [f'{lo:n}+' if hi is None else f'{lo:n}' if hi - lo <= 1 else f'{lo:n}-{hi-1:n}' for lo,hi in zip(edges, ends)]


def age_edges(labels):
    """ The lower edge of each age group from its label ('0-4', '75+', or '3'), as floats: the inverse of `age_labels()` """
    return [float(str(label).split('-')[0].rstrip('+')) for label in labels]


def is_age_index(index):
    """ Whether a table's index is by age group: named "age", or labeled as `age_labels()` labels them ('0-4', '75+', or '3'), and not just the positions of the rows (a RangeIndex) """
    if index.name == 'age':
        return True
    return not isinstance(index, pd.RangeIndex) and len(index) > 0 and all(re.fullmatch(r'\d+(\.\d+)?(-\d+(\.\d+)?|\+)?', str(label).strip()) for label in index)


def age_pyramid(sim, strat=None):
    """
    The people by 5-year age group (their lower edges, and the number in each) that a sim's ages come from:
    a named population's, or, as a guess, the world's; None if there is no named population and the split of
    the age groups was given

    Args:
        sim (Sim): the sim
        strat (Strat): the age stratification, if any
    """
    named = getattr(sim, 'named_people', None)
    if named is not None:
        return named.ages, named.counts
    elif getattr(sim, 'given_pyramid', None) is not None:
        return sim.given_pyramid
    elif strat is None or strat.guessed.get('split') == world.guess:
        return world.ages, world.counts
    return None


def to_age_data(ages, counts):
    """ The `age_data` of `ss.People` for counts by 5-year age group (the last is 100 and over, taken to end at 105); not for the user """
    return np.column_stack([np.append(ages, ages[-1] + 5), np.append(counts, 0)])


def age(place='world', year=None):
    """
    The ages of a place's people, as the `age_data` of `ss.People`: counts by 5-year age group

    Args:
        place (str): a country, by name or code, or 'world' (the world's population in 2024, which needs no download)
        year (int): the year (default: the year of the data); not for 'world'

    Examples:
        ```python
        ppl = ss.People(10_000, age_data=ss.data.age('Kenya', year=2020))
        ```
    """
    if place == 'world':
        if year is not None:
            errormsg = f"year={year} is not available for the world's ages, which are those of 2024; give a country for another year"
            raise ValueError(errormsg)
        return to_age_data(world.ages, world.counts)
    df = population(place, year=year)
    return to_age_data(np.array(age_edges(df.index)), (df.female + df.male).values)


def age_split(edges, ages, counts):
    """
    The share of people in each age group, from counts by 5-year age group, with ages uniform within each

    Args:
        edges (array): the lower edge of each age group; the last is open-ended
        ages (array): the lower edge of each 5-year group of the counts
        counts (array): the number of people in each
    """
    ends = np.append(ages, ages[-1] + 5)
    below = np.interp(np.append(edges, np.inf), ends, np.append(0, np.cumsum(counts))) # People younger than each edge
    out = np.diff(below)
    return out/out.sum()


def provider():
    """ The data provider, `starsim_data`, imported when it is first used """
    try:
        import starsim_data as ssd
    except ImportError as E:
        errormsg = 'Data by place (ss.data, ss.people("Kenya"), ss.contacts("Kenya")) comes from the starsim_data package: pip install starsim_data'
        raise ImportError(errormsg) from E
    return ssd


def unknown(place, E, what='place'):
    """ Raise a `KeyError` for a place that the provider cannot resolve, with its suggestions """
    msg = str(E).strip('"\'')
    raise sc.KeyNotFoundError(f'the {what} "{place}" was not found in starsim_data: {msg}.')


def subnational(ssd, place):
    """ Whether a place is a unit within a country, such as a US state ('United States:Massachusetts'), whose data are its own datasets """
    try:
        return ssd.locations.SEP in ssd.resolve(place)
    except KeyError as E:
        unknown(place, E)


def fetch(ssd, place, which, **kwargs):
    """ Data that starsim_data does not bundle, downloaded on first use into its cache; a KeyError (an unknown place) is left to the caller """
    try:
        return ssd.fetch(place, which=which, **kwargs)
    except OSError as E: # No connection, or the file is not where the registry says
        errormsg = f'The data "{which}" for "{place}" is not bundled with starsim_data, and could not be downloaded into its cache ({ssd.cache_dir()}): {E}'
        raise ConnectionError(errormsg) from E


def ref(ssd, df, **kwargs):
    """ Where the data came from: the provider, its version, and the dataset's own provenance """
    prov = df.attrs.get('provenance', {})
    out = dict(provider='starsim_data', version=ssd.__version__, **kwargs)
    for key,name in dict(source='study', license='license', population_source='population', population_year='population_year').items():
        if prov.get(key) is not None:
            out[name] = str(prov[key])
    return out


def nodes(ssd, place, level, year):
    """ The node table of one place at one level: bundled where it can be, else fetched """
    bundled = level in ssd.registry.dataset('nodes').levels
    try:
        if year is None and bundled:
            df = ssd.load_nodes(place, level=level)
            missing = df.population.isna().values
            if missing.any(): # WorldPop has no population for these (e.g. Kiribati): the rest of the national total, shared by area
                national = float(ssd.load_nodes(place, level=0).population.iloc[0])
                rest = max(national - df.population[~missing].sum(), 0)
                df = df.copy()
                df.loc[missing, 'population'] = rest*df.area_km2[missing]/df.area_km2[missing].sum()
                ss.warn(f'starsim_data has no population for {missing.sum()} of the {len(df)} units of {place} at level {level}; they share the rest of the national total by area, which is a guess: give pop= by hand if it matters')
        else: # Another year, or a finer level: fetched (this needs starsim_data's geospatial extra)
            df = ssd.fetch_nodes(place, level=level, year=year)
    except KeyError as E:
        unknown(place, E)
    if not len(df):
        unknown(place, f'it has no administrative units at level {level}')
    return df


def geography(place, level=1, year=None, places=None):
    """
    The administrative units of a place, for `ss.geo.geography('Kenya', level=1)`

    Args:
        place (str/list): a country, by name or code ('Kenya', 'KE', 'KEN'), or several
        level (int/str): the administrative level: 0 for each country, 1 for its first-level units (counties, provinces, states), 2 for the next
        year (int): the year of the populations (default: those bundled, without a download)
        places (list): only these units, by name, `dot_name` ('Kenya:Nairobi'), or p-code

    Returns:
        A dict of `names`, `pop`, `lat`, `lon`, `area` (km²), `codes` (p-codes), `adjacency` (or None), and `ref`
    """
    ssd = provider()
    level = ssd.parse_level(level)
    many = isinstance(place, (list, tuple))
    frames = [nodes(ssd, p, level, year) for p in sc.tolist(place)]
    df = sc.dataframe.cat(*frames) if many else frames[0]
    if places is not None:
        wanted = set(sc.tolist(places))
        keep = df.name.isin(wanted) | df.dot_name.isin(wanted) | df.pcode.isin(wanted)
        missing = wanted - set(df.name) - set(df.dot_name) - set(df.pcode)
        if missing:
            unknown(sc.strjoin(sorted(missing)), f'not among the {len(df)} units of {sc.strjoin(sc.tolist(place))} at level {level}', what='unit')
        df = df[keep.values]
    if many: # Each with its country, e.g. 'Kenya:Busia' and 'Uganda:Busia'
        names = list(df.dot_name)
    elif df.name.is_unique:
        names = list(df.name)
    else: # The path below the country, e.g. 'Nairobi:Westlands'
        names = [n.split(':', 1)[-1] for n in df.dot_name]
    adjacency = None
    if level > 0 and hasattr(ssd, 'adjacency'): # Which units share a border, where the provider has it (within each country)
        pos = {key:i for i,key in enumerate(df.dot_name)}
        adjacency = np.zeros((len(df), len(df)), dtype=bool)
        try:
            for p,frame in zip(sc.tolist(place), frames):
                a = ssd.adjacency(p, level=level, nodes=frame) # A fetched table carries its own borders
                for src,dst in zip(*np.nonzero(a.values)):
                    i, j = pos.get(a.index[src]), pos.get(a.columns[dst])
                    if i is not None and j is not None:
                        adjacency[i, j] = True
        except ValueError: # Not available at this level
            adjacency = None
    info = ref(ssd, frames[0], place=place, level=level, year=year, places=None if places is None else list(names), boundaries=sc.strjoin(sorted(set(df.source))))
    out = dict(names=names, pop=df.population.values.astype(float), lat=df.lat.values.astype(float), lon=df.lon.values.astype(float),
               area=df.area_km2.values.astype(float), codes=list(df.pcode), adjacency=adjacency, ref=info)
    return out


def rates(place, which, revision=None):
    """
    The UN World Population Prospects birth, death, or fertility rates of a named place, for `ss.Births(birth_rate='Kenya')`, `ss.Deaths(death_rate='Kenya')`, and `ss.Pregnancy(fertility_rate='Kenya')`

    Args:
        place (str): a country, by name or code
        which (str): 'births' (crude birth rates per 1,000, by year), 'deaths' (death rates per person, by year, sex, and age), or 'fertility' (age-specific fertility rates per 1,000 women, by year and 5-year age group)
        revision (str): the WPP revision (default: the latest bundled)
    """
    ssd = provider()
    if which not in ssd.list_datasets(): # A series added in a later version, e.g. fertility (2.1.0)
        errormsg = f'The data "{which}" are not in starsim_data {ssd.__version__}; a newer version has them: pip install -U starsim_data'
        raise ImportError(errormsg)
    try:
        return ssd.load(place, which=which, revision=revision)
    except KeyError as E:
        unknown(place, E)


def population(place, year=None, revision=None):
    """
    The population of a named place by age and sex, for `ss.people('Kenya', year=2024)`

    For a country, the UN World Population Prospects give it every ten years (1950 to 2100); a year in
    between is interpolated linearly, group by group. For a US state, the US Census gives it for 2023
    only, by single year of age to 84+, and not by sex: it is grouped by 5 years (the last group, 80+,
    is taken to end at 85, as `ss.People` reads it), and split evenly between women and men.

    Args:
        place (str): a country, by name or code; or a US state, e.g. 'United States:Massachusetts' (downloaded on first use)
        year (int): the year (default: the year of the WPP revision, e.g. 2024); a US state has only 2023, which is used with a note for any other
        revision (str): the WPP revision (default: the latest bundled); not for a US state

    Returns:
        A dataframe of the number of women and of men (columns `female` and `male`) in each 5-year age group (the index, labeled
        '0-4', '5-9', ..., '100+', or '80+' for a US state), with where it came from in `.attrs['ref']`

    Examples:
        ```python
        pop = ss.data.population('Kenya', year=2024)
        pop.loc['0-4'].sum() # Children under 5
        pop.attrs['ref']['study'] # The source
        ```
    """
    ssd = provider()
    if subnational(ssd, place):
        if revision is not None:
            errormsg = f'revision={revision} is a revision of the UN World Population Prospects, which are national; the population of {place} is from the US Census'
            raise ValueError(errormsg)
        try:
            df = fetch(ssd, place, 'population_us')
        except KeyError as E:
            unknown(place, E)
        data_year = int(df.Time.iloc[0])
        if year is not None and year != data_year:
            ss.note(f'the population of {place} is that of {data_year}, the only year starsim_data has for it, not {year}')
        single = df.sort_values('AgeGrp').Value.values.astype(float)
        ages = np.arange(0, len(single), 5, dtype=float)
        both = np.add.reduceat(single, ages.astype(int)) # By 5-year age group
        info = ref(ssd, df, place=place, year=data_year, location=df.attrs.get('location'))
        return by_age(ages, dict(female=both/2, male=both/2), info)
    try:
        df = ssd.load(place, which='population', revision=revision)
    except KeyError as E:
        unknown(place, E)
    rev = df.attrs.get('provenance', {}).get('revision')
    year = sc.ifelse(year, int(rev) if rev else 2024)
    years = np.array(sorted(set(df.Time)))
    if not years[0] <= year <= years[-1]:
        errormsg = f'year={year} is outside the years of the UN World Population Prospects ({years[0]} to {years[-1]})'
        raise ValueError(errormsg)
    ages = np.array(sorted(set(df.AgeGrp)), dtype=float)
    counts = {}
    for sex in ['Female', 'Male']:
        table = df[df.Sex == sex].pivot(index='Time', columns='AgeGrp', values='Value').loc[years, ages.astype(int)].values
        counts[sex] = np.array([np.interp(year, years, table[:,i]) for i in range(len(ages))])
    info = ref(ssd, df, place=place, year=int(year), location=df.attrs.get('location'), revision=rev)
    return by_age(ages, dict(female=counts['Female'], male=counts['Male']), info)


def by_age(edges, cols, info):
    """ A table by age group, labeled from the lower edges, with where it came from in `.attrs['ref']`; not for the user """
    df = pd.DataFrame(cols, index=pd.Index(age_labels(edges), name='age'))
    df.attrs['ref'] = info
    return df


def contacts(place, source=None, layer='all'):
    """
    A contact matrix by age for a named place, for `ss.contacts('Kenya')`

    Args:
        place (str): a country, by name or code; or a US state, e.g. 'United States:Massachusetts'
        source (str): the study: 'prem2021' (the default for a country; bundled; 177 places), 'prem2017' (152), 'mistry2021' (36, by single year of age), or 'mistry2021_us' (the default for a US state; the 50 states and DC, by single year of age); all but the first are fetched on first use
        layer (str): the setting: 'all', 'home', 'school', 'work', or 'community'

    Returns:
        A dataframe of the contacts a day that a person in each age group (a row) has with people in each (a column), with the groups labeled
        '0-4', '5-9', ..., '75+' (or by single year of age), and where it came from in `.attrs['ref']`; `ss.contacts()` takes it

    Examples:
        ```python
        school = ss.data.contacts('Kenya', layer='school')
        school.loc['10-14', '10-14'] # Contacts a day of a child aged 10-14 at school with children aged 10-14
        mixing = ss.contacts(school) # The same as ss.contacts('Kenya', layer='school')
        ```
    """
    ssd = provider()
    source = sc.ifelse(source, 'mistry2021_us' if subnational(ssd, place) else 'prem2021')
    if layer not in layers:
        errormsg = f'layer="{layer}" is not one of the settings of the contact matrices: {sc.strjoin(layers)}'
        raise ValueError(errormsg)
    try:
        ds = ssd.registry.dataset(source)
    except ValueError as E:
        errormsg = f'source="{source}" is not a contact matrix in starsim_data: {E}'
        raise ValueError(errormsg) from E
    try:
        df = ssd.load(place, which=source, kind=layer) if ds.tier == 0 else fetch(ssd, place, source, kind=layer)
    except KeyError as E:
        try:
            key = ssd.resolve(place)
        except KeyError:
            unknown(place, E)
        others = [s for s in ['prem2021', 'prem2017', 'mistry2021', 'mistry2021_us'] if s != source and key in ssd.list_locations(s)]
        hint = f'; it is in {sc.strjoin(others)}, e.g. ss.contacts("{place}", source="{others[0]}")' if others else ''
        raise sc.KeyNotFoundError(f'"{source}" has no contact matrix for {key}{hint}.')
    labels = age_labels([float(a) for a in df.index])
    out = pd.DataFrame(df.values.astype(float), index=pd.Index(labels, name='age'), columns=pd.Index(labels, name='contact age'))
    out.attrs['ref'] = ref(ssd, df, place=place, source=source, layer=layer, location=df.attrs.get('location'))
    return out
