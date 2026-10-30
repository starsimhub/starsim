"""
Define options for Starsim.
All options should be set using set() or directly, e.g.:

    ss.options(verbose=False)
"""
import numpy as np
import numba as nb
import sciris as sc
import matplotlib.font_manager as fm
import starsim as ss # Only used to set references


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


# Not public to avoid confusion with ss.options
class Options(sc.objdict):
    """
    Set options for Starsim.

    Use `ss.options.set('defaults')` to reset all values to default, or `ss.options.set(verbose='default')`
    to reset one parameter to default. See `ss.options.help(detailed=True)` for
    more information.

    See `ss.options.context()` to set options temporarily, and `ss.style()` to
    set the plotting style temporarily.

    Common options are (see also `ss.options.help(detailed=True)`):

        - verbose:        default verbosity for simulations to use
        - warnings:       how to handle warnings (e.g. print, raise as errors, ignore)

    Examples:
        ```python
        ss.options(verbose=True) # Set more verbosity
        ss.options(warnings='error') # Be more strict about warnings
        ```
    """

    def __init__(self):
        super().__init__()
        optdesc, options = self.get_orig_options()  # Get the options
        super().update(options)  # Update this object with them
        self.setattribute('optdesc', optdesc)  # Set the description as an attribute, not a dict entry
        self.setattribute('orig_options', sc.dcp(options))  # Copy the default options
        self.setattribute('on_entry', [])  # Stack of settings to restore, to allow nested context() blocks
        self.setattribute('_locked', True)
        return

    @staticmethod
    def get_orig_options():
        """
        Set the default options for Starsim -- not to be called by the user, use
        `ss.options.set('defaults')` instead.
        """

        # Options acts like a class, but is actually an objdict for simplicity
        optdesc = sc.objdict()  # Help for the options
        options = sc.objdict()  # The options

        optdesc.verbose = 'Set default level of verbosity: 0 prints nothing, 1 the solver, notes, and progress, and 2 everything (e.g. each step).'
        options.verbose = sc.parse_env('STARSIM_VERBOSE', 1, int)

        optdesc.warnings = 'How warnings are handled: options are "warn" (default), "print", and "error"'
        options.warnings = sc.parse_env('STARSIM_WARNINGS', 'warn', str)

        optdesc.warn_convert = 'Whether to warn about automatic time conversions'
        options.warn_convert = sc.parse_env('STARSIM_WARN_CONVERT', True, bool)

        optdesc.max_steps = 'The number of timesteps above which a sim warns that it may be slow (0 to never warn)'
        options.max_steps = sc.parse_env('STARSIM_MAX_STEPS', 50_000, int) # More than a century at a daily step, which the textbooks' models run

        optdesc.jupyter = 'Set whether to use Jupyter settings (retina figures): -1=auto, 0=False, 1=True'
        options.jupyter = sc.parse_env('STARSIM_JUPYTER', -1, int)

        optdesc.style = 'Set the plotting style: choices are "starsim", "fancy", "simple", or any of Matplotlib\'s'
        options.style = sc.parse_env('STARSIM_STYLE', 'starsim', str)

        optdesc.install_fonts = 'Choose whether or not to install Starsim-specific fonts on first import -- NOTE, only has effect via the environment variable since executed on load'
        options.install_fonts = sc.parse_env('STARSIM_INSTALL_FONTS', True, bool)

        optdesc.show = 'Whether to show the plot immediately (i.e. call plt.show())'
        options.show = sc.parse_env('STARSIM_SHOW', True, bool)

        optdesc.reticulate = 'Whether a NumPy integer array indexes a state by UIDs, as for a module written in R (through reticulate), which cannot keep ss.uids'
        options.reticulate = sc.parse_env('STARSIM_RETICULATE', False, bool)

        optdesc.precision = 'Set arithmetic precision'
        options.precision = sc.parse_env('STARSIM_PRECISION', 32, int)

        optdesc.threads = 'The number of threads that the passes over the agents of a large sim, and the particles of a particle filter, take: 0 (default) for as many as Numba has (by default, one per core), 1 for none. Each worker of ss.parallel() and of a calibration takes one, since the workers share the cores already'
        options.threads = sc.parse_env('STARSIM_THREADS', 0, int)

        optdesc.kernels = 'Which kernels run the solvers (all give the same results, except as said for "python"): "auto" (default) for Rust if it has been built (see starsim/solvers/rust/README.md), and otherwise Numba, with compartmental runs taken as plain Python until Numba has compiled (if a run, or the runs of a session, would take longer as plain Python than compiling takes, about two minutes once per install, Numba compiles first; a note says which); "rust" for Rust, compiled ahead of time, so a first run takes no compiling; "numba" to always compile with Numba first (see ss.compile()); "python" for the compartmental kernels as plain Python (and the kernels, not JAX, for the chain binomial, whose draws in JAX are compiled, so a long seeded chain-binomial run gives other numbers than with the others; the passes over agents are Numba). JAX is chosen by the run (see sim.jax), and ss.jit is always Numba'
        options.kernels = sc.parse_env('STARSIM_KERNELS', 'auto', str)

        optdesc.jax = 'The platform that JAX runs on where Starsim uses it (sim.gradient(), posteriors, and long compartmental runs; see sim.jax): "auto" (default) for JAX\'s own choice (a GPU if it finds one), or a platform, e.g. "cpu" (for a GPU that JAX cannot use) or "cuda". It sets JAX_PLATFORMS, so it applies to everything in the session that uses JAX, and must be set before JAX is first used'
        options.jax = sc.parse_env('STARSIM_JAX', 'auto', str)

        optdesc.uncompiled_warning = 'If True (default), say on import when Starsim has not been compiled yet (see ss.compile()) -- NOTE, only has effect via the environment variable since executed on load'
        options.uncompiled_warning = sc.parse_env('STARSIM_UNCOMPILED_WARNING', True, bool)

        optdesc.crn = 'If True (default), use common random numbers (CRN) for reproducibility across scenarios. Set False for faster draws (one plain uniform per agent/edge, skipping the slot machinery); statistically valid but not CRN-safe across scenarios.'
        options.crn = sc.parse_env('STARSIM_CRN', True, bool)

        optdesc.keep_times = 'If True, a declared disease run as agents keeps the time each agent entered every state (e.g. ti_S) and the history of every transition (e.g. sir.infection.count, .since, and .ever); by default (False), only those the engine or the model needs (with, for each, the time each agent last left the state, for its stays, e.g. I.stays), to save memory (reading another raises an error)'
        options.keep_times = sc.parse_env('STARSIM_KEEP_TIMES', False, bool)

        optdesc.max_joint = 'The most compartments that the diseases tracked jointly in a compartmental run may take (their joint states times the groups of the other dimensions; see ss.solvers.Crossing): above it, solver="auto" runs agents, and a compartmental solver that was asked for stops before anything is made. 0 for no limit'
        options.max_joint = sc.parse_env('STARSIM_MAX_JOINT', 10_000, int) # 6,912 compartments (three diseases with Erlang stages, by four age groups) take about 2 GB over 400 steps

        optdesc.keep_memory = 'If True (default), large arrays that a sim frees are kept by the process for reuse, rather than given back to the system, which makes sims with networks 15-30% faster, but the process keeps its peak memory until it ends (Linux only). Set False (before importing Starsim, via the environment variable) to give memory back as it is freed; ss.solvers.backends.release_memory() gives back what is free now'
        options.keep_memory = sc.parse_env('STARSIM_KEEP_MEMORY', True, bool)

        return optdesc, options

    def __call__(self, *args, **kwargs):
        """Allow `ss.options(verbose=0)` instead of `ss.options.set(verbose=0)` """
        return self.set(*args, **kwargs)

    def update(self, *args, **kwargs):
        """ Set several options, as `set()` does (so e.g. the kernels are swapped) """
        return self.set(**dict(*args, **kwargs))

    def __setitem__(self, key, value):
        """ Do not allow accidental modifications """

        try:
            assert self.getattribute('_locked') # This handles False, not present, etc.
            locked = True
        except (AssertionError, AttributeError):
            locked = False

        if locked:
            if key in self:
                self.set(key=key, value=value)
            else:
                errormsg = f'Cannot set option "{key}"; valid options are:\n{sc.newlinejoin(sorted(self.keys()))}\n\nSee ss.options.disp() for details.'
                raise sc.KeyNotFoundError(errormsg)
        else:
            return super().__setitem__(key, value)

    def to_dict(self):
        ''' Pull out only the settings from the options object '''
        return {k:v for k,v in self.items()}

    def __repr__(self):
        """ Brief representation """
        output = sc.objectid(self)
        output += 'Starsim options (see also ss.options.disp()):\n'
        output += sc.pp(self.to_dict(), output=True)
        return output

    def __enter__(self):
        """ Allow to be used in a with block """
        return self

    def __exit__(self, *args, **kwargs):
        """ Allow to be used in a with block """
        try:
            reset = {}
            for k, v in self.on_entry.pop().items():
                if self[k] != v:  # Only reset settings that have changed
                    reset[k] = v
            self.set(**reset)
        except IndexError as E:
            errormsg = 'Please use ss.options.context() if using a with block'
            raise AttributeError(errormsg) from E
        return

    def disp(self):
        """ Detailed representation """
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
        Print information about options.

        Args:
            *args (list): if provided, print out detailed help on one or more specific options
            detailed (bool): whether to print out full help
            output (bool): whether to return a list of the options

        Examples:
            ```python
            ss.options.help(detailed=True)
            ```
        """
        # If not detailed, just print the docstring for sc.options
        if not detailed and not len(args):
            print(self.__doc__)
            return

        n = 15 # Size of indent
        optdict = sc.objdict()
        orig_keys = self.orig_options.keys()
        for key in orig_keys:
            entry = sc.objdict()
            entry.key = key
            entry.current = sc.indent(n=n, width=None, text=sc.pp(self[key], output=True)).rstrip()
            entry.default = sc.indent(n=n, width=None, text=sc.pp(self.orig_options[key], output=True)).rstrip()
            entry.variable = f'STARSIM_{key.upper()}' # NB, hard-coded above!
            entry.desc = sc.indent(n=n, text=self.optdesc[key])
            optdict[key] = entry

        if len(args):
            try:
                optdict = sc.objdict({k:optdict[k] for k in args})
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

        if output:
            return optdict
        else:
            return

    def set(self, key=None, value=None, use=False, **kwargs):
        """
        Actually change the style. See `ss.options.help()` for more information.

        Args:
            key    (str):    the parameter to modify, or 'defaults' to reset everything to default
            value  (varies): the value to specify; use None or 'default' to reset to default
            use (bool): whether to use the chosen style
            kwargs (dict):   if supplied, set multiple key-value pairs

        Examples:
            ```python
            ss.options.set(verbose=0) # Equivalent to ss.options(verbose=0)
            ```
        """

        # Reset to defaults
        if key in ['default', 'defaults']:
            kwargs = self.orig_options  # Reset everything to default

        # Handle other keys
        elif key is not None:
            kwargs = sc.mergedicts(kwargs, {key: value})

        # Reset options
        for key, value in kwargs.items():
            if key not in self:
                keylist = self.orig_options.keys()
                keys = '\n'.join(keylist)
                errormsg = f'Option "{key}" not recognized; options are "defaults" or:\n{keys}\n\nSee help(ss.options.set) for more information.'
                raise sc.KeyNotFoundError(errormsg)
            else:
                if value in [None, 'default']:
                    value = self.orig_options[key]
                if key == 'kernels': # Checked here rather than at the first run (CB-9), and swapped before the option is set, so it is not set if the Rust kernels have not been built
                    if value not in ['auto', 'rust', 'numba', 'python']:
                        errormsg = f'ss.options.kernels must be "auto", "rust", "numba", or "python", not {value!r}'
                        raise ValueError(errormsg)
                    import starsim.solvers.backends.rust as rust # Imported only now, since it imports the modules whose kernels it swaps
                    rust.auto() if value == 'auto' else rust.use(value == 'rust')
                elif key == 'jax' and not (isinstance(value, str) and value):
                    errormsg = f'ss.options.jax must be "auto" or the name of a JAX platform, e.g. "cpu", not {value!r}'
                    raise ValueError(errormsg)
                super().__setitem__(key, value) # Needed since we overwrite __setitem__ to call this

                # Handle special cases
                if key == 'jupyter':
                    self.set_jupyter()
                elif key == 'precision':
                    self.set_precision()
                elif key == 'style':
                    self.set_style()

        return

    def context(self, **kwargs):
        """
        Alias to set(), for use in a "with" block.

        Examples:
            ```python
            # Silence all output
            with ss.options.context(verbose=0):
                ss.Sim().run()

            # Convert warnings to errors
            with ss.options.context(warnings='error'):
                ss.Sim(location='not a location').init()

            # Use ss.style(), not context(), for plotting options
            with ss.style(dpi=50):
                ss.Sim().run().plot()
            ```
        """

        # Store current settings
        on_entry = {k: self[k] for k in kwargs.keys()}
        self.on_entry.append(on_entry)

        # Make changes
        self.set(**kwargs)
        return self

    def get_default(self, key):
        """ Helper function to get the original default options """
        return self.orig_options[key]

    @property
    def is_jupyter(self):
        """ Check if we're in an Jupyter environment; jupyter=False doesn't change this, since a notebook still shows what is returned """
        return self.jupyter == 1 or sc.isjupyter()

    def set_jupyter(self):
        """ Set Sciris' Jupyter settings as well """
        sc.options(jupyter=bool(self.jupyter) and self.is_jupyter) # 0 = False, 1 = True, -1 = sc.isjupyter()
        return

    def set_precision(self, refresh=True):
        """ Change the arithmetic precision used by Starsim/NumPy """
        if self.precision == 32:
            dtypes.update(dict(
                int       = np.int64, # Never set this or result_float to 32-bit since messed up array indexing
                rand_int  = np.int32,
                rand_uint = np.uint32,
                float     = np.float32,
                nbfloat   = nb.float32,
            ))
        elif self.precision == 64:
            dtypes.update(dict(
                int       = np.int64,
                rand_int  = np.int64,
                rand_uint = np.uint64,
                float     = np.float64,
                nbfloat   = nb.float64,
            ))
        else:
            errormsg = f'Precision {self.precision} not recognized; must be 32 or 64'
            raise ValueError(errormsg)
        if refresh:
            self.refresh_references()
        return

    def refresh_references(self):
        """ Many modules copy settings into them for speed; here we refresh those """
        files = ['arrays', 'demographics', 'diseases', 'distributions', 'networks']

        # Update precision
        for file in files:
            ss_file = getattr(ss, file) # e.g. ss.arrays
            for key, value in dtypes.items():
                ss_key = f'ss_{key}' # e.g. int -> ss_int
                if hasattr(ss_file, ss_key): # e.g. ss.arrays.ss_int
                    setattr(ss_file, ss_key, value)
        return

    def set_style(self):
        """ Change the plotting style """
        if self.style == 'starsim':
            style = sc.dcp(rc_starsim)
        else:
            style = self.style

        # Since Options is locked
        self.setattribute('_style', style)
        return



def with_fallback(font):
    """ A named font, followed by Matplotlib's sans-serif fonts (DejaVu Sans by default) for the glyphs it lacks, e.g. the Greek letters that Mulish, Raleway, and Rosario don't have; a generic family or a list is kept as it is """
    generic = ['serif', 'sans-serif', 'monospace', 'cursive', 'fantasy']
    return [font, 'sans-serif'] if isinstance(font, str) and font not in generic else font


def load_fonts(folder=None, name='Mulish', rebuild=False, verbose=False, **kwargs):
    """
    Helper function to load custom fonts for plotting -- (usually) not for the user.

    Note: if fonts don't load, try running `ss.settings.load_fonts(rebuild=True)`,
    and/or rebooting the system.

    Args:
        folder (str): the folder to add fonts from
        name (str): the name of the font to load
        rebuild (bool): whether to rebuild the font cache
        verbose (bool): whether to print out progress/errors
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


# Create the options on module load
options = Options()
options.set_precision(refresh=False) # Other modules are not loaded yet, so nothing to refresh

# Load the fonts, rebuilding Matplotlib's font cache the first time (marked by a file in internals/assets/; if it can't be written, the cache isn't rebuilt)
if options.install_fonts:
    marker = sc.thispath(__file__) / 'internals' / 'assets' / 'fonts_installed_successfully'
    first = not marker.exists()
    try:
        if first:
            marker.touch()
        load_fonts(rebuild=first)
        if first and options.verbose:
            print('\nNote: rebuilding the font cache only happens once on first import, or set the environment variable STARSIM_INSTALL_FONTS=0 to disable.')
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
    Set the style in a with block.

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

    # Rename to avoid name collisions (e.g. font vs font.family)
    mapping = {'font':'font.family', 'facecolor':'axes.facecolor', 'grid':'axes.grid'}
    for key,val in mapping.items():
        if key in kwargs:
            kwargs[val] = kwargs.pop(key)
    if 'font.family' in kwargs:
        kwargs['font.family'] = with_fallback(kwargs['font.family'])

    # Reset
    out = sc.options.with_style(style, **kwargs)
    return out
