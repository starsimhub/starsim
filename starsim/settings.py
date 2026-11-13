"""
Options for Starsim, and the plotting style

These exist so that behavior that isn't part of a model (how much to print, how to treat warnings, which
kernels to use) is set in one place, e.g.:

    ss.options(verbose=0)

Used with: agents and compartments; declared and coded modules
"""
import os
import tempfile
import numpy as np
import numba as nb
import numba.core.caching as nb_caching
import sciris as sc
import matplotlib.font_manager as fm
import starsim as ss # Only used to set references

thread_limit = None # The fewest threads asked for in the environment when Starsim was imported (STARSIM_BLAS_THREADS, SCIRIS_NUM_THREADS, or OMP_NUM_THREADS), before it set its own (see ss.gradients.jax()); None if none was


def cache_numba():
    """ Keep Numba's cache in a temporary folder if it has nowhere else to go (a read-only package and home folder, as in some containers), since `cache=True` would otherwise fail on import; Numba finds it by `NUMBA_CACHE_DIR`, as do the workers of `ss.parallel()` """
    if any(locator.from_function(cache_numba, __file__) for locator in nb_caching.CacheImpl._locator_classes):
        return
    uid = getattr(os, 'getuid', int)() # Not on Windows, whose temporary folder is the user's own
    path = os.path.join(tempfile.gettempdir(), f'starsim_numba_{uid}') # The same folder each time, so only the first process compiles
    os.makedirs(path, mode=0o700, exist_ok=True)
    if os.stat(path).st_uid != uid: # Someone else's, so not to be loaded from
        path = tempfile.mkdtemp(prefix='starsim_numba_')
    os.environ['NUMBA_CACHE_DIR'] = nb.config.CACHE_DIR = path
    return

cache_numba()


# Define Starsim-default data types
dtypes = sc.dictobj(
    bool = bool,
    int = np.int64, # np.int32 is actually much slower for indexing arrays
    uid = np.int64, # The active agents (people.auids): 64-bit, since NumPy converts a 32-bit index array at every use (2-4 times slower), at 4 bytes more per agent; ss.uids keeps either width
    nbint = nb.int64,
    rand_int = np.int32,
    rand_uint = np.uint32,
    float = np.float32,
    nbfloat = nb.float32,
    result_float = np.float64,
    int_nan = -999999, # Arbitrary placeholder value
)

# Define simple plotting options -- similar to Matplotlib default
rc_starsim = {
    'font.family':       'sans-serif', # Replaced with Mulish in load_fonts() if import succeeds
    'axes.axisbelow':    True, # So grids show up behind
    'axes.spines.right': False,
    'axes.spines.top':   False,
    'legend.frameon':    False,
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'axes.grid':         True,
    'grid.color':        '#f2f2f2',
    'grid.linewidth':    1,
    'lines.linewidth':   2.5,      # Slightly thicker lines
}


class Options(sc.obj): # Not public (only ss.options is), to avoid confusion with it
    """
    The options for Starsim, as `ss.options`: settings for everything in a session that aren't part of a model, such as how much to print and how to treat warnings

    Set one with `ss.options(verbose=0)` (or `ss.options.verbose = 0`), or for a `with` block only, with
    `with ss.options(verbose=0):`; each can also be set before Python starts, by its environment variable
    (e.g. `STARSIM_VERBOSE=0`). `ss.options.set('defaults')` resets them all, and `ss.options.set(verbose='default')`
    one. Use `ss.style()` to set the plotting style for a block.

    The options most often set are:

        - verbose:  how much a sim prints (0 for nothing)
        - warnings: what to do with warnings ("warn", "print", "error", or "ignore")

    See `ss.options.help(detailed=True)` for all of them, with what each is for.

    Examples:
        ```python
        ss.options(verbose=2) # Print everything
        ss.options(warnings='error') # Be more strict about warnings
        with ss.options(jax='cpu'): # Just for this block
            sim.gradient()
        ```
    """

    __slots__ = ('optdesc', 'orig_options', '_locked', '_style')

    def __init__(self):
        super().__init__()
        optdesc, options = self.get_orig_options()
        super().update(options)
        self.setattribute('optdesc', optdesc) # As attributes, not options
        self.setattribute('orig_options', sc.dcp(options)) # The defaults
        self.setattribute('_locked', True)
        return

    @staticmethod
    def get_orig_options():
        """ The description and the default of each option, the default read from its environment variable if set; to reset the options, use `ss.options.set('defaults')` """
        optdesc = sc.obj()
        options = sc.obj()

        optdesc.verbose = "How much a sim prints by default: 0 for nothing (e.g. in a batch of runs or a fit), 1 for the solver chosen, notes, and progress (default), 2 for everything, such as each step (for debugging). A sim's own verbose= overrides it."
        options.verbose = sc.parse_env('STARSIM_VERBOSE', 1, int)

        optdesc.warnings = 'What to do with a warning: "warn" (default) to show it as a Python warning, "print" to print it as plain text, "error" to raise it as an error (useful in tests, or to find where a warning comes from), or "ignore" to drop it.'
        options.warnings = sc.parse_env('STARSIM_WARNINGS', 'warn', str)

        optdesc.max_steps = 'Warn that a sim may be slow if it has more timesteps than this (0 to never warn); the default is more than a century at a daily step.'
        options.max_steps = sc.parse_env('STARSIM_MAX_STEPS', 50_000, int) # Enough for the textbooks' models, some of which run a century at a daily step
        optdesc.max_years = 'Warn that a run is probably a mistake if it lasts longer than this many years (0 to never warn); a run this long is almost always a stop meant as a year, or a time in the wrong unit.'
        options.max_years = sc.parse_env('STARSIM_MAX_YEARS', 1000, int)

        optdesc.jupyter = 'Whether to use the settings for Jupyter notebooks (sharper "retina" figures): -1 to detect a notebook (default), 0 for never, 1 for always.'
        options.jupyter = sc.parse_env('STARSIM_JUPYTER', -1, int)

        optdesc.style = 'The plotting style: "starsim" (default), "fancy", "simple", or any of Matplotlib\'s styles. To change it for one plot only, use ss.style() in a with block.'
        options.style = sc.parse_env('STARSIM_STYLE', 'starsim', str)

        optdesc.show = 'Whether plots are shown as soon as they are made (with plt.show()); set False to make figures without showing them, e.g. in a script that saves them to files.'
        options.show = sc.parse_env('STARSIM_SHOW', True, bool)

        optdesc.reticulate = 'Whether a NumPy integer array indexes a state by UIDs, as needed for a module written in R (through reticulate), which cannot keep ss.uids; leave False in Python.'
        options.reticulate = sc.parse_env('STARSIM_RETICULATE', False, bool)

        optdesc.precision = 'The precision of the numbers kept for each agent: 32 (default) for float32, which takes half the memory, or 64 for float64, for when small differences matter (e.g. comparing against another model to many decimal places).'
        options.precision = sc.parse_env('STARSIM_PRECISION', 32, int)

        optdesc.threads = 'The number of threads that the passes over the agents of a large sim, and the particles of a particle filter, use: 0 (default) for as many as Numba has (by default, one per core), or 1 for no extra threads, e.g. on a shared machine. Each worker of ss.parallel() and of a calibration uses one, since the workers already share the cores.'
        options.threads = sc.parse_env('STARSIM_THREADS', 0, int)

        optdesc.blas_threads = 'The number of threads for BLAS, the linear algebra in NumPy and SciPy: 1 (default), since more threads rarely speed up a sim but do slow down many sims run in parallel; or 0 to leave BLAS as it is. For the threads of Starsim itself, see "threads"; see also the "Deployment" page of the user guide.'
        options.blas_threads = sc.parse_env('STARSIM_BLAS_THREADS', int(os.getenv('SCIRIS_NUM_THREADS') or 1), int) # 1 by default, set before Sciris is imported (see starsim/__init__.py)

        optdesc.kernels = 'Which compiled code runs the solvers (all give the same results):\n\n- "auto" (default): Rust if it is there (in the wheels, or built: see starsim/solvers/rust/README.md); otherwise Numba, running compartmental sims in plain Python until Numba has compiled (Numba compiles first if a run, or the runs of a session, would take longer in plain Python than compiling takes, about two minutes, once per install; a note says which)\n- "rust": Rust, compiled ahead of time, so a first run takes no compiling\n- "numba": always compile with Numba first (see ss.compile())\n- "python": the compartmental kernels in plain Python (the passes over agents are still Numba), e.g. for debugging\n\nJAX is chosen by the run (see sim.jax), and ss.jit always uses Numba.'
        options.kernels = sc.parse_env('STARSIM_KERNELS', 'auto', str)

        optdesc.jax = 'Where JAX runs when Starsim uses it (sim.gradient(), fits, and ODE runs with sim.jax = True): "cpu" (default), "auto" for JAX\'s own choice (a GPU if it finds one), or another platform, e.g. "cuda". The CPU is the default since Starsim\'s runs in JAX are long loops of small steps, which a GPU runs several times slower. It sets JAX_PLATFORMS, so it applies to everything in the session that uses JAX; if JAX has already started on other platforms, it is restarted on these the next time Starsim uses it (and "auto" goes back to the platforms it had before).'
        options.jax = sc.parse_env('STARSIM_JAX', 'cpu', str)

        optdesc.uncompiled_warning = 'Whether to say on import that Starsim has not been compiled yet (see ss.compile()); True by default. Since it is read on import, it only has an effect as the environment variable STARSIM_UNCOMPILED_WARNING.'
        options.uncompiled_warning = sc.parse_env('STARSIM_UNCOMPILED_WARNING', True, bool)

        optdesc.crn = 'Whether to use common random numbers (default True): each agent gets the same random numbers in every scenario, so the difference between two scenarios is due to what changed rather than to chance. Set False for slightly faster draws (one plain random number per agent or edge), which are statistically valid but differ between scenarios.'
        options.crn = sc.parse_env('STARSIM_CRN', True, bool)

        optdesc.keep_times = 'Whether a declared disease run as agents keeps the time each agent entered every state (e.g. ti_S) and the history of every transition (e.g. sir.infection.n_times, .since, and .ever). Off by default to save memory: only those that the engine or the model needs are kept (with, for each, the time each agent last left the state, e.g. for I.stays), and reading another raises an error. ss.Sim(keep_times=...) sets it for one sim.'
        options.keep_times = sc.parse_env('STARSIM_KEEP_TIMES', False, bool)

        optdesc.max_joint = 'The most compartments that diseases tracked jointly in a compartmental run may use (their joint states times the groups of the other dimensions; see ss.solvers.JointStates). Above it, solver="auto" runs as agents instead, and a compartmental solver that was asked for raises an error before anything is made, since so many compartments would take too much memory. 0 for no limit.'
        options.max_joint = sc.parse_env('STARSIM_MAX_JOINT', 10_000, int) # 6,912 compartments (three diseases with Erlang stages, by four age groups) take about 2 GB over 400 steps

        return optdesc, options

    def __call__(self, key=None, value=None, **kwargs):
        """ Allow `ss.options(verbose=0)` instead of `ss.options.set(verbose=0)`, and `with ss.options(verbose=0):` for a block (as `context()`) """
        return self.context(key, value, **kwargs)

    def update(self, *args, **kwargs):
        """ Set several options, as `set()` does (so e.g. the kernels are swapped)

        Args:
            args (dict): the options to set, as a dict
            kwargs (dict): the options to set, by name
        """
        return self.set(**dict(*args, **kwargs))

    def __setitem__(self, key, value):
        """ Set an option through `set()`, so that it is checked and takes effect (e.g. `ss.options.verbose = 0`); a name that is not an option raises, rather than adding one """
        try:
            locked = self.getattribute('_locked')
        except AttributeError: # While being made
            locked = False
        if not locked:
            return super().__setitem__(key, value)
        if key not in self:
            errormsg = f'Cannot set option "{key}"; valid options are:\n{sc.newlinejoin(sorted(self.keys()))}\n\nSee ss.options.disp() for details.'
            raise sc.KeyNotFoundError(errormsg)
        self.set(key=key, value=value)

    def to_dict(self):
        """ The options as a plain dict """
        return dict(self.items())

    def __repr__(self):
        return sc.objectid(self) + 'Starsim options (see also ss.options.disp()):\n' + sc.pp(self.to_dict(), output=True)

    def disp(self):
        """ Print each option and its value """
        output = 'Starsim options (see also ss.options.help()):\n'
        keylen = 10  # Maximum key length
        for k, v in self.items():
            keystr = sc.colorize(f'  {k:>{keylen}s}: ', fg='cyan', output=True)
            reprstr = sc.pp(v, output=True)
            reprstr = sc.indent(n=keylen + 4, text=reprstr, width=None)
            output += f'{keystr}{reprstr}'
        print(output)
        return

    def help(self, *args, detailed=False, output=False):
        """
        Print what each option is for, its value, its default, and its environment variable

        Args:
            *args (list): the options to print help on (default: all, if `detailed`)
            detailed (bool): whether to print the help on every option, rather than only the overview
            output (bool): whether to return the help on each option, as a dict

        Examples:
            ```python
            ss.options.help(detailed=True)
            ```
        """
        if not detailed and not len(args): # The overview, the docstring of ss.options
            print(self.__doc__)
            return

        n = 15 # Size of indent
        optdict = sc.obj()
        orig_keys = self.orig_options.keys()
        for key in orig_keys:
            entry = sc.obj()
            entry.key = key
            entry.current = sc.indent(n=n, width=None, text=sc.pp(self[key], output=True)).rstrip()
            entry.default = sc.indent(n=n, width=None, text=sc.pp(self.orig_options[key], output=True)).rstrip()
            entry.variable = f'STARSIM_{key.upper()}' # As read in get_orig_options()
            entry.desc = sc.indent(n=n, text=self.optdesc[key])
            optdict[key] = entry

        if len(args):
            try:
                optdict = sc.obj({k:optdict[k] for k in args})
            except Exception as e:
                errormsg = f'Cannot provide help on invalid setting(s) "{sc.strjoin(args)}"; valid keys are:\n{sc.newlinejoin(sorted(orig_keys))}'
                raise sc.KeyNotFoundError(errormsg) from e

        print('Starsim global options ("Environment" = name of corresponding environment variable):')
        for k, key, entry in optdict.enumitems():
            sc.heading(f'{k}. {key}', spaces=0, spacesafter=0)
            changestr = '' if entry.current == entry.default else ' (modified)'
            print(f'          Key: {key}')
            print(f'      Current: {entry.current}{changestr}')
            print(f'      Default: {entry.default}')
            print(f'  Environment: {entry.variable}')
            print(f'  Description: {entry.desc}')

        if detailed:
            sc.heading('Methods:', spacesafter=0)
            print("""
    ss.options(key=value) -- set key to value
    ss.options[key] -- get or set key
    ss.options.set() -- set option(s)
    ss.options.get_default() -- get default setting(s)
    ss.options.set_style() -- change the style for plotting (can also use ss.style() in a context block)
""")

        return optdict if output else None

    def set(self, key=None, value=None, **kwargs):
        """
        Set one or more options (see `ss.options.help()` for what each is for)

        Args:
            key (str): the option to set, or 'defaults' to reset every option to its default
            value (varies): its value; None or 'default' for its default
            kwargs (dict): the options to set, by name

        Examples:
            ```python
            ss.options.set(verbose=0) # Equivalent to ss.options(verbose=0)
            ```
        """
        if key in ['default', 'defaults']:
            kwargs = self.orig_options
        elif key is not None:
            kwargs = sc.mergedicts(kwargs, {key: value})

        for key, value in kwargs.items():
            if key not in self:
                keys = '\n'.join(self.orig_options.keys())
                errormsg = f'Option "{key}" not recognized; options are "defaults" or:\n{keys}\n\nSee help(ss.options.set) for more information.'
                raise sc.KeyNotFoundError(errormsg)
            if value in [None, 'default']:
                value = self.orig_options[key]
            if key == 'kernels': # Checked here rather than at the first run, and swapped before the option is set, so it is not set if the Rust kernels have not been built
                if value not in ['auto', 'rust', 'numba', 'python']:
                    errormsg = f'ss.options.kernels must be "auto", "rust", "numba", or "python", not {value!r}'
                    raise ValueError(errormsg)
                import starsim.solvers.backends.rust as rust # Imported only now, since it imports the modules whose kernels it swaps
                rust.auto() if value == 'auto' else rust.use(value == 'rust')
            elif key == 'jax' and not (isinstance(value, str) and value):
                errormsg = f'ss.options.jax must be the name of a JAX platform, e.g. "cpu", or "auto", not {value!r}'
                raise ValueError(errormsg)
            super().__setitem__(key, value) # Not self[key], which calls this

            # The options that take effect when set
            setter = dict(jupyter=self.set_jupyter, precision=self.set_precision, blas_threads=self.set_blas_threads, style=self.set_style).get(key)
            if setter is not None:
                setter()
        return

    def context(self, key=None, value=None, **kwargs):
        """
        Set options, as `set()` does, until the end of a "with" block (the same as `ss.options(...)`)

        Args:
            key (str): the option to set, or 'defaults' to set every option to its default
            value (varies): its value
            kwargs (dict): the options to set, by name

        Returns:
            An object that sets them back to what they were at the end of the "with" block (and outside one, does nothing)

        Examples:
            ```python
            # Silence all output
            with ss.options(verbose=0):
                ss.Sim().run()

            # Convert warnings to errors
            with ss.options.context(warnings='error'):
                ss.Sim().init()

            # Use ss.style(), not context(), for plotting options
            with ss.style(dpi=50):
                ss.Sim().run().plot()
            ```
        """
        keys = self.keys() if key in ['default', 'defaults'] else [key, *kwargs] if key is not None else kwargs
        before = {k:self[k] for k in keys if k in self} # Unknown ones are raised by set()
        self.set(key, value, **kwargs)
        return OptionsContext(self, before)

    def get_default(self, key):
        """ The default of an option

        Args:
            key (str): the name of the option
        """
        return self.orig_options[key]

    @property
    def is_jupyter(self):
        """ Whether this is a Jupyter notebook (or `jupyter=1`); jupyter=0 doesn't change this, since a notebook still shows what is returned """
        return self.jupyter == 1 or sc.isjupyter()

    def set_jupyter(self):
        """ Set Sciris' Jupyter settings as well """
        sc.options(jupyter=bool(self.jupyter) and self.is_jupyter) # 0 = False, 1 = True, -1 = sc.isjupyter()
        return

    def set_blas_threads(self):
        """ Set the number of BLAS threads, for the libraries already loaded and any loaded later, as in new workers; unlike `sc.limit_threads()`, this can also raise it """
        n = self.blas_threads
        if n:
            os.environ.update({key:str(n) for key in sc._threadvars})
            for setter,getter in sc.limit_threads.find_libs():
                setter(n)
        return

    def set_precision(self, refresh=True):
        """ Set the data types of the arrays of agents and kernels (`ss.dtypes`) to the precision of `ss.options.precision`

        Args:
            refresh (bool): whether to pass the new data types on to the modules that keep their own copies
        """
        if self.precision == 32:
            dtypes.update(rand_int=np.int32, rand_uint=np.uint32, float=np.float32, nbfloat=nb.float32)
        elif self.precision == 64:
            dtypes.update(rand_int=np.int64, rand_uint=np.uint64, float=np.float64, nbfloat=nb.float64)
        else:
            errormsg = f'Precision {self.precision} not recognized; must be 32 or 64'
            raise ValueError(errormsg)
        # The integers and the results stay 64-bit at either precision, since 32-bit integers broke the indexing of arrays
        if refresh:
            self.refresh_references()
        return

    def refresh_references(self):
        """ Pass the data types on to the modules that keep their own copies, for speed (e.g. `ss.arrays.ss_int`) """
        for file in ['arrays', 'demographics', 'diseases', 'dists', 'distributions', 'networks']:
            ss_file = getattr(ss, file)
            for key, value in dtypes.items():
                if hasattr(ss_file, f'ss_{key}'):
                    setattr(ss_file, f'ss_{key}', value)
        return

    def set_style(self):
        """ Set the plotting style from `ss.options.style` (used by `ss.style()`) """
        self.setattribute('_style', sc.dcp(rc_starsim) if self.style == 'starsim' else self.style) # An attribute, not an option
        return



def with_fallback(font):
    """ A named font, followed by Matplotlib's sans-serif fonts (DejaVu Sans by default) for the glyphs it lacks, e.g. the Greek letters that Mulish, Raleway, and Rosario don't have; a generic family or a list is kept as it is """
    generic = ['serif', 'sans-serif', 'monospace', 'cursive', 'fantasy']
    return [font, 'sans-serif'] if isinstance(font, str) and font not in generic else font


def load_fonts(folder=None, name='Mulish', rebuild=False, verbose=False, **kwargs):
    """
    Load the fonts bundled with Starsim (Mulish, Raleway, and Rosario) for plotting; done on import, so usually not for the user

    Note: if fonts don't load, try running `ss.settings.load_fonts(rebuild=True)`,
    and/or rebooting the system.

    Args:
        folder (str): the folder to add fonts from
        name (str): the name of the font to load
        rebuild (bool): whether to rebuild the font cache
        verbose (bool): whether to print out progress/errors
        kwargs (dict): passed to `sc.fonts()`
    """
    if folder is None:
        folder = str(sc.thispath(__file__) / 'internals' / 'assets')
    sc.fonts(add=folder, rebuild=rebuild, verbose=verbose, **kwargs)

    # Try to find the font, and if it succeeds, update the styles
    try:
        fm.findfont(name, fallback_to_default=False) # Raise an exception if the font isn't found
        rc_starsim['font.family'] = with_fallback(name)
        if verbose: print(f'Default Starsim font reset to "{name}"')
    except Exception as E:
        if verbose: print(f'Could not find font {name}: {str(E)}')
    return


class OptionsContext:
    """ What `ss.options(...)` returns: at the end of a "with" block, it sets the options it changed back to what they were """
    def __init__(self, options, before):
        self.options = options
        self.before = before
        return

    def __enter__(self):
        return self.options

    def __exit__(self, *args):
        self.options.set(**{k:v for k,v in self.before.items() if self.options[k] != v}) # Only those that have changed
        return

    def __repr__(self):
        return 'Starsim options set (see ss.options.disp())'

    def _ipython_display_(self):
        """ Show nothing when it ends a Jupyter cell (e.g. ss.options(verbose=0)) """
        return


# Create the options on module load
options = Options()
options.set_precision(refresh=False) # Other modules are not loaded yet, so nothing to refresh
options.set_blas_threads()

# Load the fonts, rebuilding Matplotlib's font cache the first time (marked by a file in internals/assets/; if it can't be written, the cache isn't rebuilt)
marker = sc.thispath(__file__) / 'internals' / 'assets' / 'fonts_installed_successfully'
first = not marker.exists()
try:
    if first:
        marker.touch()
    load_fonts(rebuild=first)
    if first and options.verbose:
        print('\nNote: rebuilding the font cache only happens once on first import.')
except Exception: # Not writable, or something else went wrong: plot without them
    pass

# Set the style
options.set_style()

# In a notebook, use Jupyter's settings (retina figures) as if ss.options(jupyter=True) had been called, unless another backend was chosen (e.g. %matplotlib widget)
if options.jupyter and options.is_jupyter:
    import matplotlib as mpl
    if 'inline' in mpl.get_backend():
        options.set_jupyter()


def style(style=None, **kwargs):
    """
    Set the plotting style for a `with` block

    Note: Starsim comes bundled with three fonts: Mulish (default), Raleway, and Rosario.
    Use `font='sans-serif'` to use the Matplotlib default font.

    Args:
        style (str): the style to use; if None, use current; otherwise, 'starsim', 'simple', 'fancy', plus all of the Matplotlib styles are options
        **kwargs (dict): passed to `sc.options.with_style()`

    Examples:
        ```python
        # Create a plot using default Starsim styling
        with ss.style():
            plt.plot()

        # Create a plot using a built-in Matplotlib style
        with ss.style('seaborn-v0_8-whitegrid'):
            plt.plot()

        # Customize the current style
        with ss.style(font='Rosario'):
            plt.plot()
        ```
    """
    if style is None:
        style = options._style
    elif style in ['starsim', 'default']:
        style = sc.dcp(rc_starsim)

    # Short names for Matplotlib's settings, e.g. font for font.family
    mapping = {'font':'font.family', 'facecolor':'axes.facecolor', 'grid':'axes.grid'}
    for key,val in mapping.items():
        if key in kwargs:
            kwargs[val] = kwargs.pop(key)
    if 'font.family' in kwargs:
        kwargs['font.family'] = with_fallback(kwargs['font.family'])
    return sc.options.with_style(style, **kwargs)
