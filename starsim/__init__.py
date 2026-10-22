"""
Import all Starsim modules
"""

# Time how long each import takes -- disabled by default, only used for developer debugging
debug = False
if debug:
    print('Note: importing Starsim in debug mode')
    from time import perf_counter as pytime
    start = pytime()
    timings = {}
    def t(label):
        timings[label] = pytime() - start
else:
    def t(label):
        pass

# Assign the root folder
t('sciris') # Slow since importing NumPy, Matplotlib, etc.
import sciris as sc
root = sc.thispath(__file__).parent

# Start imports: version and settings
t('settings') # Slow since importing Numba
from .version import __version__, __versiondate__, __license__
from .settings import dtypes, options, style # Also ss.settings.load_fonts

# Optionally print the license
t('license')
if options.license:
    print(__license__)

# Double-check key requirements -- should match pyproject.toml
t('reqs')
sc.require(reqs=['sciris>=3.2.8', 'pandas>=2.0.0'], die=False,
    message = f'\n\nThe following dependencies for Starsim {__version__} were not met:\n  <MISSING>.\nYou can update with: pip install <MISSING> --upgrade\n'
)

# Finish imports
t('utils')
from .utils import (
    ndict, warn, parse_age_range,
    load, save, shrink, plot_args, show,
    return_fig,
)

t('errors')
from .errors import *

t('debugtools')
from .debugtools import check_requires # Also ss.debugtools.Profile, the mock_* objects for testing, and Diagnostics

t('arrays')
from .arrays import (
    BaseArr, Arr, FloatArr, ModifiableArr, IntArr, BoolArr, BoolState, IndexArr, uids,
)

t('distributions') # Slow import due to scipy.stats
from .distributions import ( # Also ss.distributions.Dists, link_dists, make_dist, dist_list, rand_raw
    Dist, random, uniform, normal, truncnorm, lognorm_ex, lognorm_im, expon,
    poisson, nbinom, binom, beta_dist, beta_mean, weibull, gamma, erlang, constant,
    randint, bernoulli, choose_n, choice, histogram,
    multi_random,
)

t('time')
from .time import (
    TimePar, dur, Rate, prob, per, freq, peract,
    years, months, weeks, days, year, month, week, day,
    perday, perweek, permonth, peryear,
    probperday, probperweek, probpermonth, probperyear,
    freqperday, freqperweek, freqpermonth, freqperyear,
)

t('dates')
from .dates import DateArray, date, datedur
from . import timeplot # Registers Starsim's dates and durations with Matplotlib

t('timeline')
from .timeline import Timeline

t('parameters')
from .parameters import Pars, SimPars, by, timeseries, seasonal

t('people')
from .people import People, Person, people, ppl

t('modules')
from .modules import (
    register_modules, required,
    Base, Module,
)

t('networks')
from .networks import (
    Network, DynamicNetwork,
    RandomNet, SexualNet,
    PrenatalNet, MaternalNet, PostnatalNet, BreastfeedingNet,
    AgeGroup, MixingPools, MixingPool, contacts,
)
from . import geo # Places and the movement between them: ss.geo.geography, ss.geo.Mobility, ss.geo.gravity

t('results')
from .results import Result, Results, simdata

t('demographics')
from .demographics import (
    Demographics, Births, Deaths, PregnancyPars,
)

t('products')
from .products import Product, Vaccine, Treatment, Diagnostic

t('interventions')
from .interventions import (
    Intervention, Event, Delivery, vaccinate, treat, screen,
)

t('analyzers')
from .analyzers import Analyzer, infection_log, dynamics_by_age

t('connectors')
from .connectors import Connector, modify, change

t('conditions')
from . import conditions
from .conditions import state, child # Conditions on a state by path, and on a mother when her child goes through an event
from . import perflib # Hidden machinery for speed (exact draws in Numba), which the kernels use
from . import kernels
from .kernels import compile # Compile the solver kernels ahead of time

t('transitions')
from .expressions import par, mother, attr
from .transitions import (
    State, states, null, stay, on,
    Transition, birth, death, lifetable, immigration, emigration, outcome,
)

t('dimensions')
from .dimensions import transfer # Also ss.dimensions.Strat, ss.dimensions.Groups

t('solvers')
from . import solvers # The engines are internal: ss.solvers.Compartmental, ss.solvers.Agents

t('diseases')
from .diseases import Disease, Infection, SIR, SEIR, SIS
from .strains import immunity, Strains

t('pregnancy')
from . import pregnancy
from .pregnancy import Pregnancy, Lactation

t('loop')
from .loop import Loop

t('spec')
from .spec import to_code, diff

t('data')
from . import data # Data by place, from starsim_data: ss.data.age('Kenya'), ss.data.population(), ss.data.contacts(), ss.data.geography(); starsim_data is imported only when they are used

t('gradients')
from . import gradients # Used as ss.gradients; JAX is imported only when it is used

t('jit')
from .perflib import jit # Compile users' own code; the machinery for speed is hidden in ss.perflib

t('sim')
from .sim import Sim, AlreadyRunError, demo

t('run')
from .run import MultiSim, parallel

# ss.fit (fitting to data) is imported when it is first used, so that import starsim stays fast
def __getattr__(name):
    if name == 'fit':
        import importlib
        return importlib.import_module('.fit', __name__)
    hint = '; import it first: import starsim.library as ssl' if name == 'library' else ''
    raise AttributeError(f'module "starsim" has no attribute "{name}"{hint}')

# Load fonts
def _load_fonts(debug=debug):
    """ Try installing and loading fonts, but fail silently if folder is not writable """
    if options.install_fonts:
        test_file = root / 'starsim' / 'assets' / 'fonts_installed_successfully'

        # Check if fonts have already been installed
        if not test_file.exists():
            if debug: print(f'File {test_file} does not exist, continuing...')
            try:
                if debug: print('Checking write permissions...')
                test_file.touch()  # Mark that fonts were installed
                if debug: print('Rebuilding fonts...')
                settings.load_fonts(rebuild=True)
                if options.verbose:
                    print('\nNote: rebuilding the font cache only happens once on first import, or set the environment variable STARSIM_INSTALL_FONTS=0 to disable.')
            except Exception as E: # Not writable or something else went wrong; ignore
                if debug: print(f'File not writable or another error: {E}')
        else:
            if debug: print(f'File {test_file} exists, loading fonts...')
            settings.load_fonts(rebuild=False)
    return

# Try loading fonts
t('fonts')
_load_fonts()

# If we're in debug mode, show the timings
if debug:
    t('') # Final timing, label is discarded
    timing_keys = list(timings.keys())[:-1] # The labels happen *before* the code, so correct the offset here
    timing_vals = list(timings.values())[1:]
    lastval = 0
    print('\nImport timings:')
    for key,val in zip(timing_keys, timing_vals):
        print(f'{key:15s} | Σ = {val:0.3f} s | Δ = {(val-lastval)*1000:0.1f} ms')
        lastval = val

# Don't keep these in the module
del t, sc, debug
