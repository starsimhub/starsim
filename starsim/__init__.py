"""
Import all Starsim modules

The order matters: each group builds on the ones before it, and within a group,
later modules use earlier ones as they are imported.
"""
# Authorship: written by humans, edited by AI

# Ask for one BLAS thread (see ss.options.blas_threads) before Sciris imports NumPy, which starts them; if something else already has, the options limit them
import os as _os
_limit = [_os.getenv(key, '') for key in ['STARSIM_BLAS_THREADS', 'SCIRIS_NUM_THREADS', 'OMP_NUM_THREADS']] # A limit on threads set before Starsim sets its own, which JAX keeps to too (see ss.gradients.jax())
_threads = _limit[0] or _limit[1] or '1'
if _threads != '0':
    _os.environ['SCIRIS_NUM_THREADS'] = _threads
_limit = min([int(n) for n in _limit if n.isdigit() and int(n) > 0], default=None)
del _os, _threads

# Assign the root folder
import sciris as sc
root = sc.thispath(__file__).parent

#%% Setup: version, settings, utilities, and errors
from .version import __version__, __versiondate__, __license__
from .settings import dtypes, options, style
settings.thread_limit = _limit
del _limit
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
from .internals.modules import register_modules, required, Module
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
from .solvers.backends import kernels
from .solvers.backends.kernels import compile
from .internals import expressions
from .internals.expressions import par, mother, attr
from .transitions import (
    State, states, pars, unborn, dead, outside, stay, on,
    Transition, move, birth, death, lifetable, immigration, emigration, migration, outcome, lifelong,
)
from .internals import graph, display, dimensions
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
from .solvers.backends import jit, draws # Draws, for ss.draws.next_u() in compiled code (see ss.Dist.streams())
from . import sim
from .simpars import SimPars
from .sim import Sim, demo
from .run import MultiSim, parallel

#%% Importing from and exporting to other frameworks
from . import converters
from .converters import import_from, export_to

def __getattr__(name):
    """ Import ss.fitting when it is first used, so that import starsim stays fast """
    if name == 'fitting':
        import importlib
        return importlib.import_module('.internals.fitting', __name__)
    if name == 'library':
        hint = '; import it first: import starsim.library as ssl'
    else:
        hint = '' if name.startswith('__') else v3_deprecations.hint(names=name) # A name of Starsim 3's says what replaces it
    raise AttributeError(f'module "starsim" has no attribute "{name}"{hint}')

# The kernels: Rust if it has been built (or asked for by STARSIM_KERNELS), else Numba (see ss.options.kernels)
options.set(kernels=options.kernels)

# Say if Starsim has not been compiled yet
kernels.note_uncompiled()

# Don't keep these in the module
del sc
