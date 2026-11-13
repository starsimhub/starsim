"""
Places and the movement between them: a geography (`ss.geo.geography`), and movement between its places: a movement model (`ss.geo.gravity`, `ss.geo.radiation`), a matrix (`ss.geo.Mobility`), or a register of moves on given dates (`ss.geo.register`)

These exist so that a model of more than one place can be built from named places and a standard movement model. Movement applies to a model stratified by place (`sir.stratify(patch=...)`), whose networks then pair people within each place; a geography can also be used by a network (`ssl.GeoNet`), for a disease of either kind.

Used with: agents and compartments; declared and coded modules
"""
import numpy as np
import pandas as pd
import scipy.sparse as spsp
import sciris as sc
import starsim as ss

__all__ = ['geography', 'gravity', 'radiation', 'Mobility', 'register']

shapefiles = ('.shp', '.geojson', '.gpkg') # Files of shapes, read with geopandas


class geography:
    """
    A geography: named places with populations and positions, used as the levels of a spatial stratification

    A country's administrative units come from `starsim_data` by name (`ss.geo.geography('Kenya', level=1)`):
    their names, populations, centroids, areas, p-codes, and which share a border. Levels 0 and 1
    are bundled (for 184 countries and territories), so they need no download; level 2 is downloaded on first use.

    Args:
        data (dataframe/str/list): a country (or a list of them) by name or code, e.g. 'Kenya' or 'KE'; or a table (or the name of a file holding one) with a column `name`, and optionally `population` (or `pop`), and `x` and `y` or `lat` and `lon`;
            or your own shapes, as a GeoDataFrame or a .shp, .geojson, or .gpkg file (read with geopandas), whose places are at the centroids of their shapes (in degrees, if it has a coordinate reference system)
        level (int/str): for a country, the administrative level: 1 (the default) for its first-level units (counties, provinces, states), 2 for the next (fetched), 0 for the country itself
        year (int): for a country, the year of the populations (default: those bundled, for 2020; another year is fetched)
        places (list): for a country, only these units, by name, `dot_name` ('Kenya:Nairobi'), or p-code
        names (list): the name of each place (instead of `data`)
        populations (list): the number of people in each place; sets the fraction of people who start in each (for a country, in place of the bundled ones), and the sizes of the places for movement between them (`ss.geo.gravity`, `ss.geo.radiation`)
        x (list): a position for each place, in any unit of distance
        y (list): the other coordinate
        lat (list): latitudes in degrees (instead of `x` and `y`); distances are then great-circle distances in km
        lon (list): longitudes in degrees
        distances (array): the distance between each pair of places (instead of positions)
        adjacency (array): whether each pair of places shares a border

    Examples:
        ```python
        kenya = ss.geo.geography('Kenya', level=1) # Its 47 counties
        towns = ss.geo.geography(names=['north', 'south', 'east'], populations=[50_000, 30_000, 20_000], x=[0, 0, 40], y=[0, 30, 0])
        districts = ss.geo.geography('districts.shp') # Your own shapes, with columns name and population
        S, I, R = ss.states(3)
        sir = ss.Disease(infection=S >> I @ 0.3*I, recovery=ss.Transition(I, R, dur=10), init_prev=0.01)
        sir.stratify(patch=ss.levels(towns, mobility=ss.geo.gravity(scale=0.01)))
        ```
    """
    def __init__(self, data=None, names=None, populations=None, x=None, y=None, lat=None, lon=None, distances=None, adjacency=None, level=1, year=None, places=None):
        self.ref = None # Where a named geography came from, for describe()
        self.area = None
        self.codes = None
        named = isinstance(data, str) and not data.endswith(('.csv', '.xlsx') + shapefiles) or (isinstance(data, (list, tuple)) and all(isinstance(d, str) for d in data))
        if named: # A country's administrative units, from starsim_data
            found = ss.data.geography(data, level=level, year=year, places=places)
            names, lat, lon, adjacency = [found[key] for key in ['names', 'lat', 'lon', 'adjacency']]
            populations = found['pop'] if populations is None else populations # Populations given by hand win
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
            cols = {key:data[key].tolist() for key in ['name', 'population', 'pop', 'x', 'y', 'lat', 'lon'] if key in data.columns}
            names, populations, x, y, lat, lon = [cols.get(key) for key in ['name', 'population', 'x', 'y', 'lat', 'lon']]
            populations = sc.ifelse(populations, cols.get('pop'))
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
        self.shapes = data.geometry if data is not None and 'geometry' in data.columns else None # Kept for drawing maps
        self.names = list(names)
        self.populations = None if populations is None else np.array(populations, dtype=float)
        self.x = None if x is None else np.array(x, dtype=float)
        self.y = None if y is None else np.array(y, dtype=float)
        self.lat = None if lat is None else np.array(lat, dtype=float)
        self.lon = None if lon is None else np.array(lon, dtype=float)
        self.adjacency = None if adjacency is None else np.array(adjacency, dtype=bool)
        self.distances = None
        if distances is not None:
            self.distances = np.array(distances, dtype=float)
        elif x is not None and y is not None:
            x, y = self.x, self.y
            self.distances = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
        elif lat is not None and lon is not None: # Great-circle distances, in km
            lat, lon = np.radians(self.lat), np.radians(self.lon)
            a = np.sin((lat[:, None] - lat[None, :])/2)**2 + np.cos(lat[:, None])*np.cos(lat[None, :])*np.sin((lon[:, None] - lon[None, :])/2)**2
            self.distances = 2*6371*np.arcsin(np.sqrt(a))
        for key,val in dict(populations=self.populations, distances=self.distances, adjacency=self.adjacency).items():
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

    def by_place(self, values):
        """ Values in the order of the places, from an array, or a dict or series by name """
        if isinstance(values, (dict, pd.Series)):
            missing = [name for name in self.names if name not in values]
            if missing:
                errormsg = f'The values have none for {sc.strjoin(missing[:5])}{" ..." if len(missing) > 5 else ""}; give one for each place'
                raise ValueError(errormsg)
            values = [values[name] for name in self.names]
        values = np.asarray(values, dtype=float)
        if len(values) != len(self):
            errormsg = f'{len(values)} values were given for {len(self)} places'
            raise ValueError(errormsg)
        return values

    def get_shapes(self, shapes=None):
        """ The shapes of the places to draw, if any: those the geography was made from, or for a country, its boundaries (`shapes=True`, which downloads them, and needs geopandas) """
        if shapes is False or (shapes is None and self.shapes is None):
            return None
        if self.shapes is None:
            if self.ref is None:
                errormsg = 'shapes=True needs a geography of a country (e.g. ss.geo.geography("Kenya")) or one made from shapes (a .shp, .geojson, or .gpkg file, or a GeoDataFrame)'
                raise ValueError(errormsg)
            ssd = ss.data.starsim_data()
            found = pd.concat([ssd.download_shapes(place, level=self.ref['level']) for place in sc.tolist(self.ref['place'])]) # Keeps them a GeoDataFrame
            found = found.drop_duplicates('pcode').set_index('pcode')
            missing = [name for name,code in zip(self.names, self.codes) if code not in found.index]
            if missing:
                errormsg = f'The boundaries found have no shapes for {len(missing)} of the {len(self)} places (e.g. {missing[0]}), so they are drawn as points instead'
                ss.warn(errormsg)
                return None
            self.shapes = found.geometry.loc[self.codes].reset_index(drop=True)
        return self.shapes

    def positions(self):
        """ Where each place is, as x and y: the positions given, or longitudes and latitudes """
        if self.lon is not None:
            return self.lon, self.lat
        if self.x is not None:
            return self.x, self.y
        errormsg = 'This geography has no positions to draw it with: give x and y, or lat and lon'
        raise ValueError(errormsg)

    def draw(self, ax, colors=None, shapes=None, names=None):
        """ Draw the places into axes, as shapes or as points sized by population, for `plot()` and maps of results; returns what was drawn, whose colors `map()` updates """
        shapes = self.get_shapes(shapes)
        if shapes is not None:
            shapes.plot(ax=ax, color='0.85' if colors is None else colors, edgecolor='white', linewidth=0.4)
            drawn = ax.collections[-1]
            centers = shapes.representative_point()
            x, y = centers.x.values, centers.y.values
        else:
            x, y = self.positions()
            biggest = 1500/max(1, len(self)/5)**0.5 # Smaller points for more places, so they overlap less
            sizes = biggest/4 if self.populations is None else 10 + biggest*self.populations/self.populations.max() # Area in proportion to population
            drawn = ax.scatter(x, y, s=sizes, c='0.6' if colors is None else colors, edgecolor='white', linewidth=0.8, zorder=2)
        if self.lon is not None or shapes is not None: # Degrees of longitude are shorter away from the equator
            ax.set_aspect(1/np.cos(np.radians(np.nanmean(y))))
        else:
            ax.set_aspect('equal')
        if sc.ifelse(names, len(self) <= 15):
            for name,xi,yi in zip(self.names, x, y):
                ax.annotate(name, (xi, yi), xytext=(6, 4), textcoords='offset points', fontsize=8, color='0.25')
        ax.margins(0.15)
        ax.set_axis_off()
        return drawn

    def plot(self, values=None, label=None, shapes=None, names=None, cmap='Reds', vmin=None, vmax=None, ax=None, **kwargs):
        """
        Draw a map of the places, colored by a value for each place if given

        Places are drawn as shapes if the geography has them (it was made from shapes, or `shapes=True` for a country),
        and otherwise as points at their positions, with areas in proportion to their populations.
        For a map of a result over time, use the result's `map()`, e.g. `sim.results.sir.prevalence.map()`.

        Args:
            values (array/dict/series): a value for each place, in order or by name, e.g. a summary of a result by place
            label (str): what the values are, for the color bar
            shapes (bool): whether to draw shapes (by default, if the geography has them); `True` for a country downloads its boundaries, which needs geopandas
            names (bool): whether to label the places by name (by default, if there are 15 or fewer)
            cmap (str): the colormap of the values
            vmin (float): the value at the bottom of the color scale (by default, 0, or the smallest value if it is negative)
            vmax (float): the value at the top of the color scale (by default, the largest value)
            ax (Axes): if given, draw into these axes
            kwargs (dict): passed to `ss.plot_args()`, e.g. `figsize` or `filename`

        Examples:
            ```python
            towns = ss.geo.geography(names=['a', 'b', 'c'], populations=[50_000, 30_000, 20_000], x=[0, 0, 40], y=[0, 30, 0])
            towns.plot() # Where the places are, and their populations
            towns.plot(values=sim.results.sir.infection.by('patch').summarize(), label='Infections') # A summary of a result
            ss.geo.geography('Kenya').plot(shapes=True) # Kenya's counties, with their boundaries
            ```
        """
        import matplotlib.pyplot as plt
        kw = ss.plot_args(kwargs, figsize=(6, 5))
        given = ax is not None
        with ss.style(**kw.style):
            if ax is None:
                fig, ax = plt.subplots(**kw.fig)
            colors = None
            if values is not None:
                values = self.by_place(values)
                norm = plt.Normalize(sc.ifelse(vmin, min(0, np.nanmin(values))), sc.ifelse(vmax, np.nanmax(values))) # From 0, so that small differences don't look large
                colors = plt.get_cmap(cmap)(norm(values))
                ax.figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.7, label=label)
            self.draw(ax, colors=colors, shapes=shapes, names=names)
        return ss.return_fig(ax.figure, given=given, **kw.return_fig)


def draw_map(res, t=None, animate=False, shapes=None, names=None, cmap='Reds', vmin=0, vmax=None, n=None, fps=8, ax=None, **kwargs):
    """
    Draw a result by place on a map, at several times or as an animation: the function behind `Result.map()`, which documents the arguments

    The result needs a spatial stratification with a geography (`stratify(patch=ss.geo.geography(...))`).
    """
    import matplotlib.pyplot as plt
    if res.runs is not None:
        errormsg = f'"{res.name}" is a result across runs; map one of them, e.g. res.select(run=0)'
        raise ValueError(errormsg)
    if res.split is not None: # Already split by a dimension: map the result it came from
        res = res.split[0]
    strat = next((s for s in (getattr(res, 'strats', None) or []) if s.geo is not None), None)
    if strat is None:
        errormsg = 'A map needs a result stratified by place, with a geography: e.g. sir.stratify(patch=ss.geo.geography(...))'
        raise ValueError(errormsg)
    geo = strat.geo
    byplace = res.by(strat.name)
    values = np.asarray(byplace.values, dtype=float) # Time points by places
    times = np.array([float(t) for t in byplace.timevec])

    # What to draw: the summary over the run (by default), the times given, or several across the time when something is happening
    dated = isinstance(byplace.timevec[0], ss.date)
    if t is None and n is None and not animate: # The summary of each place, as summarize() gives it
        values = np.asarray(byplace.summarize(), dtype=float)[None, :]
        inds = [0]
        labels = [dict(sum='Total over the run', last='At the end of the run', mean='Mean over the run').get(byplace.summarize_by, 'Over the run')]
    elif t is not None:
        if not dated and any(isinstance(ti, (str, ss.date)) for ti in sc.tolist(t)):
            errormsg = f'The times of "{res.name}" are not dates, so give t= as times like those of the result, e.g. t={times[len(times)//2]:g}'
            raise ValueError(errormsg)
        targets = [float(ss.date(ti)) if isinstance(ti, str) else float(ti) for ti in sc.tolist(t)]
        inds = [int(np.argmin(np.abs(times - target))) for target in targets]
    else:
        active = np.nonzero(np.nanmax(np.abs(values), axis=1) > 0.01*np.nanmax(np.abs(values)))[0]
        first, last = (active[0], active[-1]) if len(active) else (0, len(times)-1)
        n = sc.ifelse(n, 60 if animate else 6)
        inds = np.unique(np.linspace(first, last, min(n, last - first + 1)).round().astype(int))
    if t is not None or n is not None or animate:
        labels = [str(byplace.timevec[i]) if dated else f'{times[i]:g}' for i in inds]

    single = animate or len(inds) == 1
    kw = ss.plot_args(kwargs, **(dict(figsize=(6, 5)) if single else {}))
    norm = plt.Normalize(vmin, sc.ifelse(vmax, np.nanmax(values[inds])))
    colormap = plt.get_cmap(cmap)
    title = res.full_label
    with ss.style(**kw.style):
        if single:
            given = ax is not None
            if ax is None:
                fig, ax = plt.subplots(**kw.fig)
            axes = [ax]
        else:
            given = False
            ncols = 2 if len(inds) == 4 else min(len(inds), 3)
            nrows = int(np.ceil(len(inds)/ncols))
            fig_kw = sc.mergedicts(dict(figsize=(3*ncols + 1.5, 2.8*nrows), layout='constrained'), kw.fig)
            fig, axes = plt.subplots(nrows, ncols, **fig_kw)
            axes = np.atleast_1d(axes).ravel()
            for extra in axes[len(inds):]:
                extra.remove()
            axes = axes[:len(inds)]
        fig = axes[0].figure
        drawn = [geo.draw(a, colors=colormap(norm(values[i])), shapes=shapes, names=names if len(axes) == 1 else sc.ifelse(names, False)) for a,i in zip(axes, inds)]
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap), ax=list(axes), shrink=0.7, label=title)
        if not animate:
            for a,label in zip(axes, labels):
                a.set_title(label)
            return ss.return_fig(fig, given=given, **kw.return_fig)

        # Animate: recolor the places at each time point
        from matplotlib import animation
        def frame(j):
            drawn[0].set_facecolor(colormap(norm(values[inds[j]])))
            axes[0].set_title(labels[j])
            return drawn
        frame(0)
        anim = animation.FuncAnimation(fig, frame, frames=len(inds), interval=1000/fps)
        filename = kw.return_fig.get('filename')
        if filename is not None: # E.g. spread.gif (with Pillow, which comes with Matplotlib) or spread.mp4 (with ffmpeg)
            anim.save(filename, fps=fps)
        if ss.options.is_jupyter and kw.return_fig.get('do_show') is not False: # Shown in the notebook, with controls to play it
            from IPython.display import HTML, display
            display(HTML(anim.to_jshtml()))
            plt.close(fig)
            return None
        return anim


class gravity:
    """
    The gravity model of movement: the fraction of their time that a person from place `i` spends in place `j` (or, for migration,
    the rate at which they move there) is `scale × pop[i]^(source_power−1) × pop[j]^dest_power / distance[i,j]^distance_power`

    Populations and distances come from the geography of the dimension it is used on (`ss.geo.geography`).
    The model gives the shape of the movement, and `away=` its amount: the fraction of their time
    that people spend away from home, split between the other places by gravity. `scale` has units
    of distance^distance_power / pop^(source_power+dest_power−1), so no default fits both three towns
    and a country; give it only if it has been calibrated. With neither, people spend 10% of their time away.
    `kind=` says how people move (see `ss.geo.Mobility`).

    Args:
        away (float): for commuting and visiting, the fraction of their time people spend away from home (for 'force', the fraction of each place's force of infection that goes to the others; default 0.1, unless `scale` is given)
        scale (float): the overall amount of movement, instead of `away`; for migration, the scale of the rates (required)
        kind (str): 'commute' (the default), 'visit', 'migrate', or 'force' (see `ss.geo.Mobility`)
        source_power (float): the exponent of the population of the place people come from
        dest_power (float): the exponent of the population of the place they go to
        distance_power (float): the exponent of the distance between the two

    Examples:
        ```python
        sir.stratify(patch=ss.levels(towns, mobility=ss.geo.gravity(away=0.1, source_power=1.0, dest_power=1.0, distance_power=2.0)))
        sir.stratify(patch=ss.levels(towns, mobility=ss.geo.gravity(scale=2e-4))) # A calibrated amount
        sir.stratify(patch=ss.levels(towns, mobility=ss.geo.gravity(away=0.1, kind='visit'))) # Travelers do not infect the people they visit
        ```
    """
    def __init__(self, away=None, scale=None, kind=None, source_power=1.0, dest_power=1.0, distance_power=2.0):
        if away is not None and scale is not None:
            errormsg = 'ss.geo.gravity() takes away= (the fraction of time spent away) or scale= (a calibrated amount of movement), not both'
            raise ValueError(errormsg)
        self.away = away
        self.scale = scale
        self.kind = kind
        self.source_power = source_power
        self.dest_power = dest_power
        self.distance_power = distance_power
        return

    def __repr__(self):
        amount = f'away={self.away}, ' if self.away is not None else f'scale={self.scale}, ' if self.scale is not None else ''
        kind = f'kind={self.kind!r}, ' if self.kind is not None else ''
        return f'gravity({amount}{kind}source_power={self.source_power}, dest_power={self.dest_power}, distance_power={self.distance_power})'

    def matrix(self, strat):
        """ The movement between each pair of levels of a stratification, from its geography

        Args:
            strat (Strat): the stratification, whose geography gives the populations and distances
        """
        place = strat.geo
        if place is None or place.distances is None or place.populations is None:
            raise ValueError(f'ss.geo.gravity() is used on "{strat.name}", which has no geography to give the populations and distances it needs.\n'
                                 f'     Give the places with ss.geo.geography(names=[...], populations=[...], x=[...], y=[...]), or give the movement itself with ss.geo.Mobility(matrix=...).')
        pop, dist = place.populations, place.distances
        scale, a, b, c = [float(v) for v in [sc.ifelse(self.scale, 1.0), self.source_power, self.dest_power, self.distance_power]] # Any can be a fitted ss.par
        with np.errstate(divide='ignore'):
            out = scale*pop[:, None]**(a - 1)*pop[None, :]**b/dist**c # Without a scale, only the shape matters: see Mobility.time_frac()
        np.fill_diagonal(out, 0)
        return out


class radiation:
    """
    The radiation model of movement (Simini et al. 2012), which has no parameters to fit: of the people from place `i` who travel,
    the fraction who go to place `j` is `pop[i] × pop[j] / ((pop[i] + s) × (pop[i] + pop[j] + s))`, where `s` is the population of the
    other places nearer to `i` than `j` is

    It is used as `ss.geo.gravity` is, with the populations and distances of the geography of the dimension it is used on
    (`ss.geo.geography`): the model gives where people go, and `away=` how much they travel. With neither `away` nor `scale`,
    people spend 10% of their time away. The fractions are normalized to sum to 1 for each place, so for migration `scale` is the
    rate at which each person moves. `kind=` says how people move (see `ss.geo.Mobility`).

    Args:
        away (float): for commuting and visiting, the fraction of their time people spend away from home (for 'force', the fraction of each place's force of infection that goes to the others; default 0.1, unless `scale` is given)
        scale (float): the overall amount of movement, instead of `away`: the fraction of their time away, or for migration the rate of moving (required)
        kind (str): 'commute' (the default), 'visit', 'migrate', or 'force' (see `ss.geo.Mobility`)

    Examples:
        ```python
        sir.stratify(patch=ss.levels(towns, mobility=ss.geo.radiation(away=0.1)))
        sir.stratify(patch=ss.levels(towns, mobility=ss.geo.radiation(scale=2e-5, kind='migrate'))) # 2e-5 moves per person per unit of time
        ```
    """
    def __init__(self, away=None, scale=None, kind=None):
        if away is not None and scale is not None:
            errormsg = 'ss.geo.radiation() takes away= (the fraction of time spent away) or scale= (the amount of movement), not both'
            raise ValueError(errormsg)
        self.away = away
        self.scale = scale
        self.kind = kind
        return

    def __repr__(self):
        args = [f'away={self.away}' if self.away is not None else f'scale={self.scale}' if self.scale is not None else '', f'kind={self.kind!r}' if self.kind is not None else '']
        return f'radiation({", ".join(a for a in args if a)})'

    def matrix(self, strat):
        """ The movement between each pair of levels of a stratification, from its geography

        Args:
            strat (Strat): the stratification, whose geography gives the populations and distances
        """
        place = strat.geo
        if place is None or place.distances is None or place.populations is None:
            raise ValueError(f'ss.geo.radiation() is used on "{strat.name}", which has no geography to give the populations and distances it needs.\n'
                                 f'     Give the places with ss.geo.geography(names=[...], populations=[...], x=[...], y=[...]), or give the movement itself with ss.geo.Mobility(matrix=...).')
        pop, dist = place.populations, place.distances
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
        return float(sc.ifelse(self.scale, 1.0))*out


class Mobility:
    """
    Movement between the levels of a spatial stratification, given as a matrix

    People either commute (they keep their home, and spend a fraction of their time in
    other places, where they are exposed and expose others), visit (the same, but
    they are exposed to the residents of the places they go to and do not expose
    anyone there, as flepiMoP has it), or migrate (they relocate). Or, with `kind='force'`,
    nobody moves and the force of infection does, as in LASER: with `M[i,k]` the movement
    from place i to place k, each place keeps `1 − Σ_k M[i,k]` of its own force of infection
    and receives `M[k,i]` of each other place's.

    The movement models (`ss.geo.gravity()`, `ss.geo.radiation()`) take the same `kind=` and `away=`
    themselves, and a register of moves on given dates is `ss.geo.register()`.

    Other forms are written with the general pieces. Commuters who come back at a finite
    rate (Keeling and Rohani's model 7.3) are a second spatial dimension of where people
    are now, with transfers between its levels at rates by home.

    Note: CRN safe as agents.

    Args:
        matrix (array): the movement from each place (rows) to each place: for commuting the fraction of time spent there, and for migration the rate of moving there (a SciPy sparse array for many places, most pairs of which share no one)
        kind (str): 'commute', 'visit', 'migrate', or 'force' (default 'commute')
        away (float): for commuting and visiting, the fraction of their time people spend away from home (for 'force', the fraction of each place's force of infection that goes to the others); the movement is rescaled to give every place that fraction (to keep the differences between places, scale the matrix by hand and give no `away=`)

    Examples:
        ```python
        sir.stratify(patch=ss.levels(['a', 'b'], mobility=ss.geo.Mobility([[0, 0.1], [0.2, 0]])))

        # LASER's coupling, with the force of infection moving: M[i,k] is the fraction of i's force that goes to k
        sir.stratify(patch=ss.levels(places, mobility=ss.geo.Mobility(M, kind='force')))
        ```
    """
    def __init__(self, matrix=None, kind=None, away=None, **kwargs):
        if isinstance(matrix, (gravity, radiation)):
            errormsg = f'ss.geo.Mobility() takes a matrix; a movement model is given on its own, with its kind and amount, e.g. mobility=ss.geo.{type(matrix).__name__}(kind={kind or "commute"!r}{f", away={away}" if away is not None else ""})'
            raise TypeError(errormsg)
        if kwargs:
            errormsg = f'ss.geo.Mobility() takes matrix=, kind=, and away=, not {sc.strjoin(kwargs)} (a register of moves on given dates is ss.geo.register())'
            raise TypeError(errormsg)
        if matrix is None:
            errormsg = 'ss.geo.Mobility() takes a matrix of the movement between places; for a movement model, give it on its own (e.g. mobility=ss.geo.gravity()), and for a register of moves, ss.geo.register()'
            raise ValueError(errormsg)
        self.setup(kind=kind, away=away, given=matrix)
        return

    @classmethod
    def from_model(cls, model):
        """ The movement of a movement model (`ss.geo.gravity()`, `ss.geo.radiation()`), with its kind and amount """
        out = cls.__new__(cls)
        out.setup(kind=model.kind, away=model.away, model=model)
        return out

    def setup(self, kind=None, away=None, model=None, given=None, data=None):
        """ Check the kind and amount, and set them """
        kind = kind or ('migrate' if data is not None else 'commute')
        if kind not in ['commute', 'visit', 'migrate', 'force']:
            errormsg = f'The kind of movement is "commute", "visit", "migrate", or "force", not "{kind}"'
            raise ValueError(errormsg)
        self.guessed = None
        if model is not None: # The amount of movement: away=, or the model's scale
            if kind == 'migrate' and model.scale is None:
                errormsg = f'ss.geo.{type(model).__name__}() for migration needs scale=, the scale of the rates of moving; the fraction of time away (away=) is for commuting and visiting'
                raise ValueError(errormsg)
            if away is None and model.scale is None:
                away = 0.1
                self.guessed = '10% of time away, since neither away= nor scale= was given'
        self.model = model
        self.given = None if given is None else spsp.csr_array(given, dtype=float) if spsp.issparse(given) else np.array(given, dtype=float)
        self.data = None if data is None else self.load(data)
        self.kind = kind
        self.away = away
        return

    def fitted(self):
        """ The parameters (`ss.par`) in the movement, by attribute (`away`, and the movement model's `scale`, and gravity's `source_power`, `dest_power`, `distance_power`), which set the mixing when a sim is built """
        objs = dict(away=self) | {key:self.model for key in ['scale', 'source_power', 'dest_power', 'distance_power'] if hasattr(self.model, key)}
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
        """ The movement between each pair of levels, with nothing on the diagonal

        Args:
            strat (Strat): the stratification
        """
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
        """ For a moving force of infection: the fraction of the force of infection of each level (columns) that each level (rows) receives, as a sparse array

        Args:
            strat (Strat): the stratification
        """
        out = spsp.csr_array(self.time_frac(strat).T) # Each place keeps its own fraction, and receives what each other place's force sends it
        out.sort_indices()
        return out

    def time_frac(self, strat):
        """ For commuting: the fraction of their time that people from each level (rows) spend in each level; rows sum to 1. A sparse array, since most pairs of places share no one

        Args:
            strat (Strat): the stratification
        """
        out = self.matrix(strat)
        total = np.asarray(out.sum(axis=1)).ravel()
        away = None if self.away is None else float(self.away) # A fitted ss.par, as its value
        if away is not None:
            scale = np.divide(away, total, out=np.zeros(strat.n), where=total>0)
        else: # Nobody can spend more than all their time away
            capped = (total > 1).sum()
            if capped:
                ss.warn(f'The movement for "{strat.name}" would have the people of {capped:n} of its {strat.n:n} places spend more than all their time away, so it was capped at all of it (nobody is ever at home there). Give the fraction of time away with away= instead.')
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


class register(Mobility):
    """
    A movement register: a table of moves between places on given dates, e.g. livestock sales between farms

    It has columns `date`, `from`, `to`, `n`, and optionally `states` (and `to_states`). Each row moves `n` people from
    one place to another; a row with no `from` adds them to `to` (births, purchases),
    and one with no `to` removes them from `from` (deaths, sales out of the model).
    The people who move or leave are chosen at random from those in `states` (by
    default, everyone); people who are added go into `states` (by default, the first
    state). With `to_states`, they also move from each of `states` to the matching one of `to_states`.
    Each event happens at the start of the first step that begins on or after
    its date, before that step's transitions, and the events of a step happen in the
    order of the table.

    Note: CRN safe as agents.

    Args:
        data (str/df): the register: a file name or a dataframe

    Examples:
        ```python
        sir.stratify(herd=ss.levels(herds, mobility=ss.geo.register('movements.csv')))
        ```
    """
    def __init__(self, data):
        self.setup(kind='migrate', data=data)
        return
