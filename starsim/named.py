"""
Named data: places and contact matrices looked up by name

`ss.people('Kenya')`, `ss.geo('Kenya', level=1)`, and `ss.contacts('Kenya')` read their data from
`starsim_data` (https://github.com/starsimhub/starsim_data), which bundles what is
small and common and fetches the rest on demand. Each lookup returns the data with a
record of where it came from (`ref`), which is written into the model spec, so the
sim's hash changes if the data does.
"""
import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['provider', 'geography', 'population', 'contact_matrix', 'world', 'age_data', 'age_split']

layers = ['all', 'home', 'school', 'work', 'community'] # The settings of the contact matrices


# The default ages of the people of a declared model, as a guess: the world's population in 2024 by 5-year age group
# (the last is 100 and over), in millions; the sum of the 187 countries of the UN World Population Prospects 2024 in starsim_data
world = sc.objdict(
    ages   = np.arange(0, 101, 5, dtype=float),
    counts = np.array([664.2, 666.0, 663.6, 647.4, 625.3, 609.7, 601.1, 566.7, 530.8, 498.0, 452.7,
                       410.7, 353.9, 296.2, 219.8, 154.3, 98.6, 49.7, 19.8, 4.7, 0.6]),
    guess  = "ages as the world's population in 2024",
)


def age_data(ages, counts):
    """ The `age_data` of `ss.People` for counts by 5-year age group (the last is 100 and over, taken to end at 105) """
    return np.column_stack([np.append(ages, ages[-1] + 5), np.append(counts, 0)])


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
    """ The data provider, `starsim_data`, imported when named data is first used """
    try:
        import starsim_data as ssd
    except ImportError as E:
        errormsg = 'Named data (ss.geo("Kenya"), ss.contacts("Kenya")) comes from the starsim_data package: pip install starsim_data'
        raise ImportError(errormsg) from E
    return ssd


def unknown(place, E, what='place'):
    """ Raise `ss.UnknownLocation` for a place that the provider cannot resolve, with its suggestions """
    msg = str(E).strip('"\'')
    raise ss.UnknownLocation(f'the {what} "{place}" was not found in starsim_data: {msg}.')


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
    The places of a named geography, for `ss.geo('Kenya', level=1)`

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
    The UN World Population Prospects birth or death rates of a named place, for `ss.Births(birth_rate='Kenya')` and `ss.Deaths(death_rate='Kenya')`

    Args:
        place (str): a country, by name or code
        which (str): 'births' (crude birth rates per 1,000, by year) or 'deaths' (death rates per person, by year, sex, and age)
        revision (str): the WPP revision (default: the latest bundled)
    """
    ssd = provider()
    try:
        return ssd.load(place, which=which, revision=revision)
    except KeyError as E:
        unknown(place, E)


def population(place, year=None, revision=None):
    """
    The population of a named place by age and sex, for `ss.people('Kenya', year=2024)`

    The UN World Population Prospects give it every ten years (1950 to 2100); a year in
    between is interpolated linearly, group by group.

    Args:
        place (str): a country, by name or code
        year (int): the year (default: the year of the WPP revision, e.g. 2024)
        revision (str): the WPP revision (default: the latest bundled)

    Returns:
        The lower edge of each 5-year age group (the last is 100+), the number of women and of men in each, and `ref`
    """
    ssd = provider()
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
    return ages, counts['Female'], counts['Male'], info


def contact_matrix(place, source='prem2021', layer='all'):
    """
    A contact matrix by age for a named place, for `ss.contacts('Kenya')`

    Args:
        place (str): a country, by name or code
        source (str): the study: 'prem2021' (bundled; 177 places), 'prem2017' (152), or 'mistry2021' (36, by single year of age); the last two are fetched on first use
        layer (str): the setting: 'all', 'home', 'school', 'work', or 'community'

    Returns:
        The matrix (contacts a day that a person in each age group has with people in each), the lower edge of each age group, and `ref`
    """
    ssd = provider()
    if layer not in layers:
        errormsg = f'layer="{layer}" is not one of the settings of the contact matrices: {sc.strjoin(layers)}'
        raise ValueError(errormsg)
    try:
        ds = ssd.registry.dataset(source)
    except ValueError as E:
        errormsg = f'source="{source}" is not a contact matrix in starsim_data: {E}'
        raise ValueError(errormsg) from E
    try:
        df = ssd.load(place, which=source, kind=layer) if ds.tier == 0 else ssd.fetch(place, which=source, kind=layer)
    except KeyError as E:
        try:
            key = ssd.resolve(place)
        except KeyError:
            unknown(place, E)
        others = [s for s in ['prem2021', 'prem2017', 'mistry2021'] if s != source and key in ssd.list_locations(s)]
        hint = f'; it is in {sc.strjoin(others)}, e.g. ss.contacts("{place}", source="{others[0]}")' if others else ''
        raise ss.UnknownLocation(f'"{source}" has no contact matrix for {key}{hint}.')
    ages = [float(a) for a in df.index]
    return df.values.astype(float), ages, ref(ssd, df, place=place, source=source, layer=layer, location=df.attrs.get('location'))
