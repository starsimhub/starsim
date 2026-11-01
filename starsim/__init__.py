"""
Import all Starsim modules

The order matters: each group builds on the ones before it, and within a group,
later modules use earlier ones as they are imported.
"""

# Ask for one BLAS thread (see ss.options.num_threads) before Sciris imports NumPy, which starts them; if something else already has, the options limit them
import os as _os
_threads = _os.getenv('STARSIM_NUM_THREADS') or _os.getenv('SCIRIS_NUM_THREADS') or '1'
if _threads != '0':
    _os.environ['SCIRIS_NUM_THREADS'] = _threads
del _os, _threads

# Assign the root folder
import sciris as sc
root = sc.thispath(__file__).parent

#%% Setup: version, settings, utilities, and errors
from .version import __version__, __versiondate__, __license__
from .settings import dtypes, options, style
from .utils import ndict, warn, load, save, plot_args, show, return_fig
from .internals import errors, debugtools
from .internals.errors import *

#%% Building blocks: arrays, distributions, time, and parameters
from .internals import arrays
from .internals.arrays import Arr, FloatArr, TimeArr, IntArr, BoolArr, BoolState, IndexArr, uids

from .internals import dists
from .internals.dists import Dist
from .distributions import (
    random, uniform, normal, truncnorm, lognorm_ex, lognorm_im, expon,
    poisson, nbinom, binom, beta_dist, beta_mean, weibull, gamma, erlang, constant,
    randint, bernoulli, choose_n, choice, histogram, multi_random,
)

from .internals import time, dates, timeline, parameters
from .internals.time import (
    TimePar, dur, Rate, prob, per, freq, peract,
    years, months, weeks, days, year, month, week, day,
    perday, perweek, permonth, peryear,
    probperday, probperweek, probpermonth, probperyear,
    freqperday, freqperweek, freqpermonth, freqperyear,
)
from .internals.dates import DateArray, date, datedur
from .internals.timeline import Timeline
from .internals.parameters import Pars, by, timeseries, seasonal

#%% People and modules
from .people import People, people, ppl
from .internals import modules
from .internals.modules import register_modules, required, Base, Module
from .networks import (
    Network, DynamicNetwork,
    RandomNet, DegreeNet, SexualNet,
    PrenatalNet, MaternalNet, PostnatalNet, BreastfeedingNet,
    AgeGroup, MixingPools, MixingPool, contacts,
)
from . import geo
from .results import Result, Results, simdata
from .products import Product, Vaccine, Treatment, Diagnostic, box
from .interventions import Intervention, Delivery, vaccinate, treat, screen
from .analyzers import Analyzer, infection_log
from .connectors import Connector, modify

#%% Declared models: conditions, expressions, transitions, and the solvers that run them
from .internals import conditions
from .internals.conditions import state
from .solvers.backends import kernels
from .solvers.backends.kernels import compile, compile_status
from .internals import expressions
from .internals.expressions import par, mother, attr
from .transitions import (
    State, states, pars, null, stay, on,
    Transition, move, birth, death, lifetable, immigration, emigration, outcome,
)
from .internals import graph, display, dimensions
from .internals.dimensions import transfer
from . import solvers

#%% Diseases and demographics
from .diseases import Disease, Infection, SIR, SEIR, SIS
from .strains import immunity, Strains
from .demographics import Demographics, Births, Deaths, Pregnancy # After the diseases, since a pregnancy is one

#%% Running: the loop, the sim, and many sims
from .internals import loop
from .internals.loop import Loop
from . import data # starsim_data is imported only when this is used
from .internals import gradients # JAX is imported only when this is used
from .solvers.backends import jit
from .sim import SimPars, Sim, demo
from .run import MultiSim, parallel

#%% Importing from and exporting to other frameworks
from . import converters
from .converters import import_from, export_to

def __getattr__(name):
    """ Import ss.fit when it is first used, so that import starsim stays fast """
    if name == 'fit':
        import importlib
        return importlib.import_module(f'.{name}', __name__)
    hint = '; import it first: import starsim.library as ssl' if name == 'library' else ''
    raise AttributeError(f'module "starsim" has no attribute "{name}"{hint}')

# The kernels: Rust if it has been built (or asked for by STARSIM_KERNELS), else Numba (see ss.options.kernels)
options.set(kernels=options.kernels)

# Say if Starsim has not been compiled yet
kernels.note_uncompiled()

# Don't keep these in the module
del sc
