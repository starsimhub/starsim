"""
Import all Starsim modules

The order matters: each group builds on the ones before it, and within a group,
later modules use earlier ones as they are imported.

The public names are imported by name, so that this file is the list of what `ss.X` is. Each module's
`__all__` says the same, and is what the API reference shows; the rest of a module is its helpers.
The three packages differ, each for a reason: `internals/__init__.py` imports nothing, since its modules
are attached here (`ss.time`, `ss.arrays`, ...); `solvers/__init__.py` imports everything, since the
solvers' modules call each other's helpers as `ss.solvers.X`; and `library/__init__.py` re-exports its
subpackages' classes, so that `ssl.Cholera` works as `ss.SIR` does.
"""
# Authorship: written by humans, edited by AI

def _thread_env():
    """
    Ask for one BLAS thread (see ss.options.blas_threads) before Sciris imports NumPy, which starts them (if something else already has, the options limit them);
    returns the fewest threads asked for in the environment before Starsim sets its own, which JAX keeps to too (see ss.gradients.jax()), or None
    """
    import os
    asked = {key:os.getenv(key, '') for key in ['STARSIM_BLAS_THREADS', 'SCIRIS_NUM_THREADS', 'OMP_NUM_THREADS']}
    threads = asked['STARSIM_BLAS_THREADS'] or asked['SCIRIS_NUM_THREADS'] or '1'
    if threads != '0':
        os.environ['SCIRIS_NUM_THREADS'] = threads
    return min([int(n) for n in asked.values() if n.isdigit() and int(n) > 0], default=None)

_limit = _thread_env() # Before Sciris is imported

# Assign the root folder
import sciris as sc
root = sc.thispath(__file__).parent

#%% Setup: version, settings, utilities, and errors
from .version import __version__, __versiondate__, __license__
from .settings import dtypes, options, style
settings.thread_limit = _limit
del _limit, _thread_env
from .utils import ndict, warn, load, save, plot_args, show, return_fig
from .internals import errors, debugtools, v3_deprecations
from .internals.errors import *

#%% Building blocks: arrays, distributions, time, and parameters
from .internals import arrays
from .internals.arrays import Arr, FloatArr, TimeArr, IntArr, CatArr, BoolArr, BoolState, uids

from .internals import dists
from .internals.dists import Dist
from .distributions import (
    random, uniform, normal, truncnorm, lognormal, expon,
    poisson, nbinom, binom, beta_dist, weibull, gamma, erlang, constant,
    randint, bernoulli, choose_n, choice, histogram, multi_random,
)

from .internals import time, dates, timeline, parameters
from .internals.time import (
    TimePar, Duration, Rate, prob, per, freq, peract,
    years, months, weeks, days, year, month, week, day,
    perday, perweek, permonth, peryear,
    probperday, probperweek, probpermonth, probperyear,
    freqperday, freqperweek, freqpermonth, freqperyear,
)
from .internals.dates import DateArray, date
from .internals.timeline import Timeline, schedule
from .internals.parameters import Pars, by, timeseries, seasonal

#%% People and modules
from .people import People, population, person
from .internals import modules
from .internals.modules import register_modules, Module
from .networks import (
    Network, DynamicNetwork,
    RandomNet, DegreeNet, SexualNet,
    PrenatalNet, PostnatalNet, BreastfeedingNet,
    MixingPool, contacts,
)
from . import geo
from .results import Result, Results, SimData
from .products import Product, Vaccine, Treatment, Diagnostic, boxcar
from .interventions import Intervention, Event, Delivery, at_age, capacity, vaccinate, treat, screen
from .analyzers import Analyzer, infection_log
from .connectors import Connector, modify

#%% Declared models: conditions, expressions, transitions, and the solvers that run them
from .internals import conditions
from .internals.conditions import ref
from .solvers.backends.compiling import compile # The module is ss.solvers.backends.compiling
from .internals import expressions
from .internals.expressions import par, mother, attr
from .internals import transitions
from .internals.transitions import (
    State, states, pars, unborn, dead, outside, stay, on,
    Transition, move, birth, death, lifetable, immigration, emigration, migration, outcome, lifelong,
)
from .internals import graph, display, progress, dimensions
from .internals.dimensions import transfer, levels
from . import solvers

#%% Diseases and demographics
from .diseases import Disease, AgentDisease, SIR, SEIR, SIS
from .strains import immunity, Strains
from .demographics import Demographics, Births, Deaths, Pregnancy, preterm # After the diseases, since a pregnancy is one

#%% Running: the loop, the sim, and many sims
from .internals import loop
from . import data # starsim_data is imported only when this is used
from .internals import gradients # JAX is imported only when this is used
from .solvers.backends import jit
from .solvers.numba import draws # Draws, for ss.draws.uniform() and ss.draws.next_u() in compiled code (see ss.Dist.key() and ss.Dist.streams())
from . import sim
from .simpars import SimPars
from .sim import Sim, demo
from .run import MultiSim, parallel

#%% Importing from and exporting to other frameworks
from . import converters
from .converters import import_from, export_to

fitting = sc.lazyimport('starsim.internals.fitting') # Imported when first used, so that import starsim stays fast
library = sc.lazyimport('starsim.library')

def __getattr__(name):
    """ A name of Starsim 3's says what replaces it """
    hint = '' if name.startswith('__') else v3_deprecations.hint(names=name)
    raise AttributeError(f'module "starsim" has no attribute "{name}"{hint}')

# The backend: Rust if it has been built (or asked for by STARSIM_BACKEND), else Numba (see ss.options.backend)
options.set(backend=options.backend)

# Say if Starsim has not been compiled yet
solvers.backends.compiling.note_uncompiled()

# Don't keep these in the module
del sc
