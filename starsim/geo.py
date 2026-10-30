"""
Places and the movement between them: a geography (`ss.geo.geography`), and movement between its places (`ss.geo.Mobility`, from a matrix, a data file of moves, or a movement model: `ss.geo.gravity` or `ss.geo.radiation`)
"""
import numpy as np
import pandas as pd
import scipy.sparse as spsp
import sciris as sc
import starsim as ss

__all__ = ['geography', 'gravity', 'radiation', 'Mobility']

shapefiles = ('.shp', '.geojson', '.gpkg') # Files of shapes, read with geopandas


class geography:
    """
    A geography: named places with populations and positions, used as the levels of a spatial stratification

    A country's administrative units come from `starsim_data` by name (`ss.geo.geography('Kenya', level=1)`):
    their names, populations, centroids, areas, p-codes, and which share a border. The first
    two levels of every country are bundled, so this needs no download.

    Args:
        data (dataframe/str/list): a country (or a list of them) by name or code, e.g. 'Kenya' or 'KE'; or a table (or the name of a file holding one) with a column `name`, and optionally `pop` (or `population`), and `x` and `y` or `lat` and `lon`;
            or your own shapes, as a GeoDataFrame or a .shp, .geojson, or .gpkg file (read with geopandas), whose places are at the centroids of their shapes (in degrees, if it has a coordinate reference system)
        level (int/str): for a country, the administrative level: 1 (the default) for its first-level units (counties, provinces, states), 2 for the next (fetched), 0 for the country itself
        year (int): for a country, the year of the populations (default: those bundled, for 2020; another year is fetched)
        places (list): for a country, only these units, by name, `dot_name` ('Kenya:Nairobi'), or p-code
        names (list): the name of each place (instead of `data`)
        pop (list): the population of each place; sets the share of people who start in each (for a country, in place of the bundled ones)
        x (list): a position for each place, in any unit of distance
        y (list): the other coordinate
        lat (list): latitudes in degrees (instead of `x` and `y`); distances are then great-circle distances in km
        lon (list): longitudes in degrees
        distances (array): the distance between each pair of places (instead of positions)
        adjacency (array): whether each pair of places shares a border

    Examples:
        ```python
        kenya = ss.geo.geography('Kenya', level=1) # Its 47 counties
        towns = ss.geo.geography(names=['north', 'south', 'east'], pop=[50_000, 30_000, 20_000], x=[0, 0, 40], y=[0, 30, 0])
        districts = ss.geo.geography('districts.shp') # Your own shapes, with columns name and population
        S, I, R = ss.states(3)
        sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=10), init_prev=0.01)
        sir.stratify(patch=towns, mobility=ss.geo.gravity(k=0.01))
        ```
    """
    def __init__(self, data=None, names=None, pop=None, x=None, y=None, lat=None, lon=None, distances=None, adjacency=None, level=1, year=None, places=None):
        self.ref = None # Where a named geography came from, for describe()
        self.area = None
        self.codes = None
        named = isinstance(data, str) and not data.endswith(('.csv', '.xlsx') + shapefiles) or (isinstance(data, (list, tuple)) and all(isinstance(d, str) for d in data))
        if named: # A country's administrative units, from starsim_data
            found = ss.data.geography(data, level=level, year=year, places=places)
            names, lat, lon, adjacency = [found[key] for key in ['names', 'lat', 'lon', 'adjacency']]
            pop = found['pop'] if pop is None else pop # Populations given by hand win
            self.area, self.codes, self.ref = found['area'], found['codes'], found['ref']
            data = None
        elif isinstance(data, str) and data.endswith(shapefiles):
            try:
                import geopandas as gpd # Optional, since only files of shapes need it
            except ImportError as E:
                errormsg = f'Reading the shapes in "{data}" needs geopandas: pip install geopandas'
                raise ImportError(errormsg) from E
            data = gpd.read_file(data)
        elif isinstance(data, str):
            data = sc.dataframe.read_csv(data) if data.endswith('.csv') else sc.dataframe.read_excel(data)
        if data is not None:
            cols = {key:data[key].tolist() for key in ['name', 'pop', 'population', 'x', 'y', 'lat', 'lon'] if key in data.columns}
            names, pop, x, y, lat, lon = [cols.get(key) for key in ['name', 'pop', 'x', 'y', 'lat', 'lon']]
            pop = sc.ifelse(pop, cols.get('population'))
            if 'geometry' in data.columns: # Shapes: each place is at the centroid of its shape
                if getattr(data, 'crs', None) is not None: # Found on an equal-area projection, in degrees
                    centroids = data.geometry.to_crs('EPSG:6933').centroid.to_crs('EPSG:4326')
                    lat, lon = centroids.y.tolist(), centroids.x.tolist()
                else: # Positions in the shapes' own units
                    centroids = data.geometry.centroid
                    x, y = centroids.x.tolist(), centroids.y.tolist()
        if names is None:
            errormsg = 'ss.geo.geography() needs the names of the places: names=[...], or a table with a column called "name"'
            raise ValueError(errormsg)
        self.names = list(names)
        self.pop = None if pop is None else np.array(pop, dtype=float)
        self.lat = None if lat is None else np.array(lat, dtype=float)
        self.lon = None if lon is None else np.array(lon, dtype=float)
        self.adjacency = None if adjacency is None else np.array(adjacency, dtype=bool)
        self.distances = None
        if distances is not None:
            self.distances = np.array(distances, dtype=float)
        elif x is not None and y is not None:
            x, y = np.array(x, dtype=float), np.array(y, dtype=float)
            self.distances = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
        elif lat is not None and lon is not None: # Great-circle distances, in km
            lat, lon = np.radians(np.array(lat, dtype=float)), np.radians(np.array(lon, dtype=float))
            a = np.sin((lat[:, None] - lat[None, :])/2)**2 + np.cos(lat[:, None])*np.cos(lat[None, :])*np.sin((lon[:, None] - lon[None, :])/2)**2
            self.distances = 2*6371*np.arcsin(np.sqrt(a))
        for key,val in dict(pop=self.pop, distances=self.distances, adjacency=self.adjacency).items():
            if val is not None and len(val) != len(self.names):
                errormsg = f'ss.geo.geography() has {len(self.names)} places but {len(val)} entries in {key}'
                raise ValueError(errormsg)
        return

    def __repr__(self):
        if self.ref is not None:
            return f"geography({self.ref['place']!r}, level={self.ref['level']}: {len(self)} places)"
        return f'geography({sc.strjoin(self.names)})'

    def __len__(self):
        return len(self.names)


class gravity:
    """
    The gravity model of movement: the share of their time that a person from place `i` spends in place `j` (or, for migration,
    the rate at which they move there) is `k × pop[i]^(a−1) × pop[j]^b / distance[i,j]^c`

    Populations and distances come from the geography of the dimension it is used on (`ss.geo.geography`).
    The model gives the shape of the movement, and `away=` its amount: the share of their time
    that people spend away from home, split between the other places by gravity. `k` has units
    of distance^c / pop^(a+b−1), so no default fits both three towns and a country; give it only
    if it has been calibrated. With neither, people spend 10% of their time away.

    Args:
        away (float): for commuting and visiting, the share of their time people spend away from home (default 0.1, unless `k` is given)
        k (float): the overall amount of movement, instead of `away`; for migration, the scale of the rates (required)
        a (float): the exponent of the population of the place people come from
        b (float): the exponent of the population of the place they go to
        c (float): the exponent of the distance between the two

    Examples:
        ```python
        sir.stratify(patch=towns, mobility=ss.geo.gravity(away=0.1, a=1.0, b=1.0, c=2.0))
        sir.stratify(patch=towns, mobility=ss.geo.gravity(k=2e-4)) # A calibrated amount
        ```
    """
    def __init__(self, away=None, k=None, a=1.0, b=1.0, c=2.0):
        if away is not None and k is not None:
            errormsg = 'ss.geo.gravity() takes away= (the share of time spent away) or k= (a calibrated amount of movement), not both'
            raise ValueError(errormsg)
        self.away = away
        self.k = k
        self.a = a
        self.b = b
        self.c = c
        return

    def __repr__(self):
        amount = f'away={self.away}, ' if self.away is not None else f'k={self.k}, ' if self.k is not None else ''
        return f'gravity({amount}a={self.a}, b={self.b}, c={self.c})'

    def matrix(self, strat):
        """ The movement between each pair of levels of a stratification, from its geography """
        place = strat.geo
        if place is None or place.distances is None or place.pop is None:
            raise ValueError(f'ss.geo.gravity() is used on "{strat.name}", which has no geography to give the populations and distances it needs.\n'
                                 f'     Give the places with ss.geo.geography(names=[...], pop=[...], x=[...], y=[...]), or give the movement itself with ss.geo.Mobility(matrix=...).')
        pop, dist = place.pop, place.distances
        k, a, b, c = [float(v) for v in [sc.ifelse(self.k, 1.0), self.a, self.b, self.c]] # Any can be a fitted ss.par
        with np.errstate(divide='ignore'):
            out = k*pop[:, None]**(a - 1)*pop[None, :]**b/dist**c # Without k, only the shape matters: see Mobility.time_share()
        np.fill_diagonal(out, 0)
        return out


class radiation:
    """
    The radiation model of movement (Simini et al. 2012), which has no parameters to fit: of the people from place `i` who travel,
    the share who go to place `j` is `pop[i] × pop[j] / ((pop[i] + s) × (pop[i] + pop[j] + s))`, where `s` is the population of the
    other places nearer to `i` than `j` is

    It is used as `ss.geo.gravity` is, with the populations and distances of the geography of the dimension it is used on
    (`ss.geo.geography`): the model gives where people go, and `away=` how much they travel. With neither `away` nor `k`,
    people spend 10% of their time away. The shares are normalized to sum to 1 for each place, so for migration `k` is the
    rate at which each person moves.

    Args:
        away (float): for commuting and visiting, the share of their time people spend away from home (default 0.1, unless `k` is given)
        k (float): the overall amount of movement, instead of `away`: the share of their time away, or for migration the rate of moving (required)

    Examples:
        ```python
        sir.stratify(patch=towns, mobility=ss.geo.radiation(away=0.1))
        sir.stratify(patch=towns, mobility=ss.geo.Mobility(ss.geo.radiation(k=2e-5), kind='migrate')) # 2e-5 moves per person per unit of time
        ```
    """
    def __init__(self, away=None, k=None):
        if away is not None and k is not None:
            errormsg = 'ss.geo.radiation() takes away= (the share of time spent away) or k= (the amount of movement), not both'
            raise ValueError(errormsg)
        self.away = away
        self.k = k
        return

    def __repr__(self):
        amount = f'away={self.away}' if self.away is not None else f'k={self.k}' if self.k is not None else ''
        return f'radiation({amount})'

    def matrix(self, strat):
        """ The movement between each pair of levels of a stratification, from its geography """
        place = strat.geo
        if place is None or place.distances is None or place.pop is None:
            raise ValueError(f'ss.geo.radiation() is used on "{strat.name}", which has no geography to give the populations and distances it needs.\n'
                                 f'     Give the places with ss.geo.geography(names=[...], pop=[...], x=[...], y=[...]), or give the movement itself with ss.geo.Mobility(matrix=...).')
        pop, dist = place.pop, place.distances
        order = np.argsort(dist, axis=1, kind='stable')
        nearer = np.zeros((len(pop), len(pop) + 1))
        nearer[:, 1:] = np.cumsum(pop[order], axis=1) # The population of the n nearest places to each, itself included
        s = np.empty_like(dist)
        for i in range(len(pop)): # The population of the places strictly nearer to i than each j, without i itself
            s[i] = nearer[i, np.searchsorted(dist[i, order[i]], dist[i], side='left')] - pop[i]
        m, n = pop[:, None], pop[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            out = m*n/((m + s)*(m + n + s))/(1 - pop[:, None]/pop.sum()) # Normalized for the finite population, so each row sums to 1
        np.fill_diagonal(out, 0)
        out = np.nan_to_num(out)
        return float(sc.ifelse(self.k, 1.0))*out


class Mobility:
    """
    Movement between the levels of a spatial stratification

    People either commute (they keep their home, and spend a share of their time in
    other places, where they are exposed and expose others), visit (the same, but
    they are exposed to the residents of the places they go to and do not expose
    anyone there, as flepiMoP has it), or migrate (they relocate). Or, with `kind='force'`,
    nobody moves and the force of infection does, as in LASER: with `M[i,k]` the movement
    from place i to place k, each place keeps `1 − Σ_k M[i,k]` of its own force of infection
    and receives `M[k,i]` of each other place's.

    Other forms are written with the general pieces. Commuters who come back at a finite
    rate (Keeling and Rohani's model 7.3) are a second spatial dimension of where people
    are now, with transfers between its levels at rates by home.

    A movement register (`data=`) is a table of events on given dates, with columns
    `date`, `from`, `to`, `n`, and optionally `states`. Each row moves `n` people from
    one place to another; a row with no `from` adds them to `to` (births, purchases),
    and one with no `to` removes them from `from` (deaths, sales out of the model).
    The people who move or leave are chosen at random from those in `states` (by
    default, everyone); people who are added go into `states` (by default, the first
    state). Each event happens at the start of the first step that begins on or after
    its date, before that step's transitions, and the events of a step happen in the
    order of the table.

    Note: CRN safe as agents.

    Args:
        model (gravity/radiation): a movement model (instead of `matrix`)
        matrix (array): the movement from each place (rows) to each place: for commuting the share of time spent there, and for migration the rate of moving there (a SciPy sparse array for many places, most pairs of which share no one)
        data (str/df): a movement register: a file name or a dataframe (instead of `model` or `matrix`)
        kind (str): 'commute', 'visit', 'migrate', or 'force'; by default 'migrate' for a register and 'commute' otherwise
        away (float): for commuting and visiting, the share of their time people spend away from home (for 'force', the share of each place's force of infection that goes to the others); the movement is rescaled to give every place that share (to keep the differences between places, scale the matrix by hand and give no `away=`)

    Examples:
        ```python
        sir.stratify(patch=towns, mobility=ss.geo.Mobility(ss.geo.gravity(), kind='commute', away=0.4))
        sir.stratify(patch=['a', 'b'], mobility=ss.geo.Mobility(matrix=[[0, 0.1], [0.2, 0]]))
        sir.stratify(herd=herds, mobility=ss.geo.Mobility(data='movements.csv'))

        # LASER's coupling, with the force of infection moving: M[i,k] is the share of i's force that goes to k
        sir.stratify(patch=places, mobility=ss.geo.Mobility(matrix=M, kind='force'))
        ```
    """
    def __init__(self, model=None, matrix=None, kind=None, away=None, data=None):
        if sum(arg is not None for arg in [model, matrix, data]) != 1:
            errormsg = 'ss.geo.Mobility() takes one of a movement model (e.g. ss.geo.gravity()), matrix=, or data='
            raise ValueError(errormsg)
        if kind is None:
            kind = 'migrate' if data is not None else 'commute'
        if kind not in ['commute', 'visit', 'migrate', 'force']:
            errormsg = f'The kind of ss.geo.Mobility() is "commute", "visit", "migrate", or "force", not "{kind}"'
            raise ValueError(errormsg)
        if data is not None and kind != 'migrate':
            errormsg = f'A movement register moves people from one place to another, so its kind is "migrate", not "{kind}"'
            raise ValueError(errormsg)
        self.guessed = None
        if isinstance(model, (gravity, radiation)): # The amount of movement: away= here, or on the movement model, or its k
            away = sc.ifelse(away, model.away)
            if kind == 'migrate' and model.k is None:
                errormsg = f'ss.geo.{type(model).__name__}() for migration needs k=, the scale of the rates of moving; the share of time away (away=) is for commuting and visiting'
                raise ValueError(errormsg)
            if away is None and model.k is None:
                away = 0.1
                self.guessed = '10% of time away, since neither away= nor k= was given'
        self.model = model
        self.given = None if matrix is None else spsp.csr_array(matrix, dtype=float) if spsp.issparse(matrix) else np.array(matrix, dtype=float)
        self.data = None if data is None else self.load(data)
        self.kind = kind
        self.away = away
        return

    def fitted(self):
        """ The parameters (`ss.par`) in the movement, by attribute (`away`, and the movement model's `k`, and gravity's `a`, `b`, `c`), which set the mixing when a sim is built """
        objs = dict(away=self) | {key:self.model for key in ['k', 'a', 'b', 'c'] if hasattr(self.model, key)}
        return {key:getattr(obj, key) for key,obj in objs.items() if isinstance(getattr(obj, key), ss.par)}

    def __repr__(self):
        if self.model is not None:
            what = self.model
        elif self.data is not None:
            what = f'register of {len(self.data):n} events'
        elif self.given.shape[0] <= 4 and not spsp.issparse(self.given): # A small matrix is shown, as the textbooks write it
            what = 'matrix=[' + ', '.join('[' + ', '.join(f'{v:n}' for v in row) + ']' for row in self.given) + ']'
        else:
            what = f'a {self.given.shape[0]}×{self.given.shape[1]} matrix'
        away = f', away={self.away}' if self.away is not None else ''
        return f'Mobility({what}, kind={self.kind!r}{away})'

    @staticmethod
    def blank(v):
        """ Whether an entry of a register is empty """
        return v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip())

    @staticmethod
    def by_value(vals, func, missing):
        """
        A function of each entry of a column of a register, worked out once for each distinct value, since a register can have millions of rows

        Returns the results for the distinct values, then `missing` (for None and NaN), and the index of each entry's result.
        """
        codes, uniq = pd.factorize(np.asarray(vals, dtype=object)) # None and NaN are -1, which is the last result
        return [func(v) for v in uniq] + [missing], codes

    @staticmethod
    def load(data):
        """ Read a movement register, with a blank `from` or `to` (or a blank `states`) as None """
        df = sc.dataframe(pd.read_csv(data) if isinstance(data, str) else data)
        missing = [col for col in ['date', 'from', 'to', 'n'] if col not in df.columns]
        if missing:
            errormsg = f'A movement register needs the columns date, from, to, and n (and optionally states), but {sc.strjoin(missing)} {"is" if len(missing) == 1 else "are"} missing; its columns are {list(df.columns)}'
            raise KeyError(errormsg)
        out = sc.dataframe(date=df['date'].values, n=np.asarray(df['n'], dtype=float))
        empty = sc.obj() # Whether each entry of each column is blank
        for col in ['from', 'to', 'states', 'to_states']:
            vals = np.array(df[col].values, dtype=object) if col in df.columns else np.full(len(df), None, dtype=object)
            blanks, codes = Mobility.by_value(vals, Mobility.blank, True)
            empty[col] = np.array(blanks)[codes]
            vals[empty[col]] = None
            out[col] = vals
        if (~empty.to_states & (empty['from'] | empty.to | empty.states)).any():
            errormsg = 'A row of a movement register with to_states moves people from its states to those states (in order, e.g. states "S1, I1" and to_states "S2, I2"), so it needs a from, a to (which can be the same place), and states'
            raise ValueError(errormsg)
        if (empty['from'] & empty.to).any():
            errormsg = 'Every row of a movement register needs a from, a to, or both'
            raise ValueError(errormsg)
        if (out['n'] < 0).any():
            errormsg = 'The numbers of people in a movement register cannot be negative'
            raise ValueError(errormsg)
        return out

    def events(self, strat, sim):
        """
        The register, resolved against the levels and the timeline of a sim

        Returns the step at whose start each event happens, the position of the place each leaves
        (-1 for none) and goes to (-1 for none), the number of people, and the states it names (None
        for the default), in the order of the table within each step. Events before the start or
        after the last step are dropped, with a note. Also the distinct pairs of `states` and
        `to_states` (`sets`, in the order they first appear), and which of them each event names (`set_of`).
        """
        df = self.data
        times = sc.dataframe(index=df['date'].values)
        sim_t, data_t = ss.results.data_times(sim, times)
        tol = 1e-9*max(np.abs(sim_t).max(initial=1), 1)
        ti = np.searchsorted(sim_t, data_t - tol, side='left') # The first step that starts on or after each date
        inside = (data_t >= sim_t[0] - tol) & (ti < len(sim_t))
        if not inside.all():
            ss.note(f'{(~inside).sum():n} of the {len(df):n} events in the movement register for "{strat.name}" fall outside the run, and are left out', sim=sim)
        labels = {str(label):i for i,label in enumerate(strat.labels)}
        def level(v):
            """ Places are given by label (a number read from a file with blanks in it is read as a float, so 5.0 is "5") """
            if Mobility.blank(v):
                return -1
            key = str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
            if key not in labels:
                errormsg = f'The movement register names a place "{v}", which is not a level of "{strat.name}"; places are given by label, e.g. {sc.strjoin(list(labels)[:3])}'
                raise KeyError(errormsg)
            return labels[key]
        order = np.argsort(ti[inside], kind='stable')
        rows = np.flatnonzero(inside)[order]
        col = lambda name: np.asarray(df[name].values, dtype=object)[rows] # As arrays, since reading a dataframe one cell at a time is slow
        def places(name):
            """ The position of the place in each entry of a column """
            vals, codes = Mobility.by_value(col(name), level, -1)
            return np.array(vals, dtype=np.int64)[codes]
        names = lambda v: None if Mobility.blank(v) else tuple(s.strip() for s in str(v).replace(';', ',').split(','))
        st_vals, st_codes = Mobility.by_value(col('states'), names, None)
        ts_vals, ts_codes = Mobility.by_value(col('to_states'), names, None)
        n_ts = len(ts_vals)
        set_of, pairs = pd.factorize((st_codes % len(st_vals))*n_ts + ts_codes % n_ts) # The distinct pairs of states and to_states, by their positions in st_vals and ts_vals (a code of -1 is the last, None)
        sets = [(st_vals[pair//n_ts], ts_vals[pair%n_ts]) for pair in pairs]
        for a,b in sets:
            if b is not None and len(a) != len(b):
                errormsg = f'A row of a movement register moves people from the states {a} to {b}, which need to be the same number'
                raise ValueError(errormsg)
        states, to_states = np.empty(len(sets), dtype=object), np.empty(len(sets), dtype=object) # Filled one at a time, since NumPy would read a list of tuples as rows
        for i,(a,b) in enumerate(sets):
            states[i], to_states[i] = a, b
        return sc.obj(ti=ti[rows], src=places('from'), dst=places('to'), n=np.asarray(df['n'].values[rows], dtype=float),
                          states=states[set_of], to_states=to_states[set_of], sets=sets, set_of=set_of)

    def matrix(self, strat):
        """ The movement between each pair of levels, with nothing on the diagonal """
        if self.data is not None:
            errormsg = 'A movement register has no matrix of rates; its events are applied on their dates'
            raise ValueError(errormsg)
        out = self.model.matrix(strat) if self.model is not None else self.given.copy()
        if out.shape != (strat.n, strat.n):
            raise ValueError(f'the mobility matrix for "{strat.name}" has shape {out.shape}, but there are {strat.n} levels ({sc.strjoin(strat.labels)}).')
        if spsp.issparse(out):
            out = spsp.csr_array(out - spsp.diags_array(out.diagonal()))
            out.eliminate_zeros()
        else:
            np.fill_diagonal(out, 0)
        return out

    def coupling(self, strat):
        """ For a moving force of infection: the share of the force of infection of each level (columns) that each level (rows) receives, as a sparse array """
        out = spsp.csr_array(self.time_share(strat).T) # Each place keeps its own share, and receives what each other place's force sends it
        out.sort_indices()
        return out

    def time_share(self, strat):
        """ For commuting: the share of their time that people from each level (rows) spend in each level; rows sum to 1. A sparse array, since most pairs of places share no one """
        out = self.matrix(strat)
        total = np.asarray(out.sum(axis=1)).ravel()
        away = None if self.away is None else float(self.away) # A fitted ss.par, as its value
        if away is not None:
            scale = np.divide(away, total, out=np.zeros(strat.n), where=total>0)
        else: # Nobody can spend more than all their time away
            capped = (total > 1).sum()
            if capped:
                ss.warn(f'The movement for "{strat.name}" would have the people of {capped:n} of its {strat.n:n} places spend more than all their time away, so it was capped at all of it (nobody is ever at home there). Give the share of time away with away= instead.')
            scale = 1/np.maximum(total, 1)
        if spsp.issparse(out):
            out = spsp.csr_array(out.multiply(scale[:, None]))
            out = spsp.csr_array(out + spsp.diags_array(np.maximum(1 - np.asarray(out.sum(axis=1)).ravel(), 0)))
        else:
            out = out*scale[:, None]
            np.fill_diagonal(out, np.maximum(1 - out.sum(axis=1), 0))
            out = spsp.csr_array(out)
        out.eliminate_zeros()
        out.sort_indices()
        return out
