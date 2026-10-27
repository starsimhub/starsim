"""
Import all Starsim modules
"""

# Assign the root folder
import sciris as sc
root = sc.thispath(__file__).parent

# Start imports: version and settings
from .version import __version__, __versiondate__, __license__
from .settings import dtypes, options, style # Also ss.settings.load_fonts

# Double-check key requirements -- should match pyproject.toml
sc.require(reqs=['sciris>=3.2.8', 'pandas>=2.0.0'], die=False,
    message = f'\n\nThe following dependencies for Starsim {__version__} were not met:\n  <MISSING>.\nYou can update with: pip install <MISSING> --upgrade\n'
)

# Finish imports
from .utils import (
    ndict, warn,
    load, save, plot_args, show,
    return_fig,
)

from .internals import errors # Errors, warnings, and notes, as classes (e.g. ss.errors.UnknownState); the base classes and ss.CapabilityError also here
from .internals.errors import *

from .internals import debugtools

from .arrays import (
    Arr, FloatArr, IntArr, BoolArr, BoolState, IndexArr, uids, # Also ss.arrays.BaseArr and ss.arrays.ModifiableArr
)

from .distributions import ( # Also ss.distributions.Dists, link_dists, make_dist, dist_list
    Dist, random, uniform, normal, truncnorm, lognorm_ex, lognorm_im, expon,
    poisson, nbinom, binom, beta_dist, beta_mean, weibull, gamma, erlang, constant,
    randint, bernoulli, choose_n, choice, histogram,
    multi_random,
)

from .time import (
    TimePar, dur, Rate, prob, per, freq, peract,
    years, months, weeks, days, year, month, week, day,
    perday, perweek, permonth, peryear,
    probperday, probperweek, probpermonth, probperyear,
    freqperday, freqperweek, freqpermonth, freqperyear,
)

from .dates import DateArray, date, datedur

from .timeline import Timeline

from .parameters import Pars, SimPars, by, timeseries, seasonal

from .people import People, people, ppl

from .modules import (
    register_modules, required,
    Base, Module,
)

from .networks import (
    Network, DynamicNetwork,
    RandomNet, SexualNet,
    PrenatalNet, MaternalNet, PostnatalNet, BreastfeedingNet,
    AgeGroup, MixingPools, MixingPool, contacts,
)
from . import geo # Places and the movement between them: ss.geo.geography, ss.geo.Mobility, ss.geo.gravity, ss.geo.radiation

from .results import Result, Results, simdata

from .products import Product, Vaccine, Treatment, Diagnostic, box

from .interventions import (
    Intervention, Delivery, vaccinate, treat, screen,
)

from .analyzers import Analyzer, infection_log

from .connectors import Connector, modify, change

from . import conditions
from .conditions import state # Conditions on a state by path
from . import perflib # Hidden machinery for speed (exact draws in Numba), which the kernels use
from .perflib import kernels
from .perflib.kernels import compile # Compile the solver kernels ahead of time

from .internals import expressions
from .internals.expressions import par, mother, attr
from .transitions import (
    State, states, pars, null, stay, on,
    Transition, move, birth, death, lifetable, immigration, emigration, outcome,
)

from .internals import graph, presentation # The graph of a declared module (ss.transitions.Graph), and how it is shown

from .dimensions import transfer # Also ss.dimensions.Strat, ss.dimensions.Groups

from . import solvers # The engines are internal: ss.solvers.Compartmental, ss.solvers.Agents

from .diseases import Disease, Infection, SIR, SEIR, SIS
from .strains import immunity, Strains

from .demographics import Demographics, Births, Deaths, Pregnancy # After the diseases, since a pregnancy is one

from .loop import Loop

from .internals import spec
from .internals.spec import diff # Also ss.spec.to_code, which sim.to_code() uses

from . import data # Data by place, from starsim_data: ss.data.age('Kenya'), ss.data.population(), ss.data.contacts(), ss.data.geography(); starsim_data is imported only when they are used

from .internals import gradients # Used as ss.gradients; JAX is imported only when it is used

from .perflib import jit # Compile users' own code; the machinery for speed is hidden in ss.perflib

from .sim import Sim, demo

from .run import MultiSim, parallel

# ss.fit (fitting to data) is imported when it is first used, so that import starsim stays fast
def __getattr__(name):
    if name in ['fit', 'interop']: # Imported when first used
        import importlib
        return importlib.import_module(f'.{name}', __name__)
    hint = '; import it first: import starsim.library as ssl' if name == 'library' else ''
    if not hint and not name.startswith('_'):
        try:
            from .interop.migrate import removed # A name of Starsim 3's: its replacement, from the migration script's tables (loaded only now)
            note = removed(name)
        except Exception: # E.g. asked for while starsim is still being imported
            note = None
        if note:
            errormsg = f'module "starsim" has no attribute "{name}" (removed in Starsim 4): {note}. The migration script rewrites Starsim 3 code: python -m starsim.interop.migrate mypackage/'
            raise AttributeError(errormsg, name=name, obj=None) # No object, so Python suggests no closest name (e.g. "emigration" for "Calibration")
    raise AttributeError(f'module "starsim" has no attribute "{name}"{hint}')

# Say if Starsim has not been compiled yet
kernels.note_uncompiled()

# Don't keep these in the module
del sc
