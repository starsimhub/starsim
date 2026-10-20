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
    ndict, warn, standardize_netkey, parse_age_range,
    standardize_data, load, save, shrink, plot_args, show,
    return_fig,
)

t('errors')
from .errors import *

t('debugtools')
from .debugtools import (
    Profile, Debugger, Diagnostics, check_requires, check_version,
    mock_time, mock_sim, mock_people, mock_module,
)

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
from .parameters import Pars, SimPars, Requires, by, timeseries, seasonal

t('people')
from .people import People, Person, people, ppl

t('modules')
from .modules import (
    register_modules, required,
    Base, Module,
)

t('networks')
from .networks import (
    Network, DynamicNetwork, SexualNetwork,
    StaticNet, RandomNet, PartnerNet, MFNet,
    PrenatalNet, MaternalNet, PostnatalNet, BreastfeedingNet,
    AgeGroup, MixingPools, MixingPool, contacts, geo, gravity, Mobility,
)

t('results')
from .results import Result, Results, data, compare

t('demographics')
from .demographics import (
    Demographics, Births, Deaths, PregnancyPars,
)

t('products')
from .products import Product, Dx, Tx, Vx, Vaccine, Treatment, Diagnostic

t('interventions')
from .interventions import (
    Intervention, RoutineDelivery, CampaignDelivery,
    BaseTest, BaseScreening, BaseTriage, BaseTreatment, treat_num, BaseVaccination,
    Event, Delivery, Resource, vaccinate, treat, screen, visit, cascade, share,
)

t('analyzers')
from .analyzers import Analyzer, infection_log, dynamics_by_age

t('connectors')
from .connectors import Connector, modify, change

t('conditions')
from . import conditions
from .conditions import clock, state, any_ as any, child # Conditions on a state by path, on any of several, and on a mother when her child goes through an event
from . import kernels

t('transitions')
from .expressions import par, mother, attr, func, spread
from .transitions import (
    State, states, null, stay, on,
    Transition, transmission, progression, birth, death, lifetable, immigration, emigration, outcome, update,
)

t('dimensions')
from .dimensions import transfer # Also ss.dimensions.Strat, ss.dimensions.Groups

t('solvers')
from . import solvers # The engines are internal: ss.solvers.Compartmental, ss.solvers.Agents

t('diseases')
from .diseases import Disease, Infection, NCD, SIR, SEIR, SIS
from .strains import immunity, Strains

t('pregnancy')
from . import pregnancy
from .pregnancy import Pregnancy, Lactation

t('loop')
from .loop import Loop

t('spec')
from .spec import to_code, diff

t('named')
from . import named # Places and contact matrices by name (ss.geo('Kenya'), ss.contacts('Kenya')); starsim_data is imported only when they are used

t('interop')
from . import interop # Used by ss.load(..., format=...)

t('gradients')
from . import gradients # Used as ss.gradients; JAX is imported only when it is used

t('sim')
from .sim import Sim, AlreadyRunError, demo, diff_sims, check_sims_match

t('run')
from .run import MultiSim, single_run, parallel

t('calibration') # Lazy import to save 500 ms
from .calibration import (
    Calibration, CalibComponent,
    BetaBinomial, Binomial, DirichletMultinomial, GammaPoisson, Normal,
    mismatch, loglik, ParticleFilter, Fit, fit, ModelComparison, compare_models, Optimization, optimize,
)

t('samples')
from .samples import Dataset, Samples

# The library of example/reference modules; not exported at the top level, access via e.g. ss.library.mnch.FetalHealth
t('library')
from . import library

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

# Names that Starsim no longer exports, with what to write instead (see the migrate-starsim3 skill)
_removed = {
    'LegacyPregnancy':    "ss.Pregnancy (the declared pregnancy, with Starsim 3's arguments, attributes, results, and callbacks); Starsim 3's pregnancy written as code was removed",
    'rate':               "ss.peryear(x) (a rate), or ss.freqperyear(x) (a count per unit time); ss.rate was deprecated in Starsim 3",
    'time_prob':          "ss.probperyear(x); ss.time_prob was deprecated in Starsim 3",
    'rate_prob':          "ss.peryear(x); ss.rate_prob was deprecated in Starsim 3",
    'simple_vx':          "ss.Vaccine(efficacy=e) (leaky, as by default in Starsim 3; ss.Vaccine(efficacy=e, leaky=False) for all-or-nothing)",
    'routine_vx':         "ss.vaccinate(ss.Vaccine(efficacy=e), coverage=p, start=y, interval=ss.years(1)) (leaky, as by default in Starsim 3; ss.Vaccine(efficacy=e, leaky=False) for all-or-nothing)",
    'campaign_vx':        "ss.vaccinate(ss.Vaccine(efficacy=e), coverage=p, at=[y1, y2]) (leaky, as by default in Starsim 3; ss.Vaccine(efficacy=e, leaky=False) for all-or-nothing)",
    'routine_screening':  "ss.screen(ss.Diagnostic(sensitivity=..., specificity=...), coverage=p, start=y, interval=ss.years(1))",
    'campaign_screening': "ss.screen(ss.Diagnostic(sensitivity=..., specificity=...), coverage=p, at=[...])",
    'routine_triage':     "a follow-up test, as a step of its own: screening = ss.screen(test1, ...); triage = ss.screen(test2, who=screening.positive, ...)",
    'campaign_triage':    "a follow-up test, as a step of its own: screening = ss.screen(test1, ...); triage = ss.screen(test2, who=screening.positive, ...)",
    'Test':               "ss.Diagnostic(sensitivity=..., specificity=...) (renamed, so that no product and delivery differ only by case)",
    'seasonality':        "ss.modify(flu.rel_trans, multiply=ss.seasonal(amplitude=a, peak=p)), passed in connectors=",
    'multi_run':          "ss.parallel(sim, n_runs=n) (or sim.run(n_runs=n)); its .sims are the list multi_run returned",
    'RandomExactNet':     "ss.RandomNet(method='exact')",
    'RandomSafeNet':      "ssl.RandomSafeNet (import starsim.library as ssl)",
    'ClusterNet':         "ssl.ClusterNet (import starsim.library as ssl)",
    'HybridNet':          "ssl.HybridNet (import starsim.library as ssl)",
    'MSMNet':             "ssl.MSMNet (import starsim.library as ssl)",
    'Route':              "ss.networks.Route (most code subclasses ss.Network)",
    'InfectionLog':       "ss.diseases.InfectionLog",
    'stratify':           "model.stratify(...)",
    'Expr':               "ss.transitions.Expr",
    'Graph':              "ss.transitions.Graph",
    'Strat':              "ss.transitions.Strat",
    'Transfer':           "ss.transitions.Transfer",
    'Cond':               "ss.conditions.Cond",
    'MultiResults':       "ss.run.MultiResults",
    'Engine':             "ss.solvers.Engine",
    'Compartmental':      "ss.solvers.Compartmental",
    'Agents':             "ss.solvers.Agents",
    'Crossing':           "ss.solvers.Crossing",
    'Mixing':             "ss.solvers.Mixing",
    'CountState':         "ss.solvers.CountState",
    'make_engine':        "ss.solvers.make_engine",
    'solver_names':       "ss.solvers.solver_names",
    'CodedDisease':       "ss.Disease (a subclass that declares nothing is written as code, as in Starsim 3)",
    # Internal helpers, now only in their submodules
    'link_dists':         "ss.distributions.link_dists",
    'make_dist':          "ss.distributions.make_dist",
    'dist_list':          "ss.distributions.dist_list",
    'scale_types':        "ss.distributions.scale_types",
    'rand_raw':           "ss.distributions.rand_raw",
    'Dists':              "ss.distributions.Dists",
    'load_fonts':         "ss.settings.load_fonts",
    'module_map':         "ss.modules.module_map",
    'module_types':       "ss.modules.module_types",
    'find_modules':       "ss.modules.find_modules",
    'linear_interp':      "ss.calibration.linear_interp",
    'linear_accum':       "ss.calibration.linear_accum",
    'step_containing':    "ss.calibration.step_containing",
    'load_pars':          "ss.parameters.load_pars",
    'validate_sim_data':  "ss.results.validate_sim_data",
    'metadata':           "ss.debugtools.metadata",
    'apply_age_range':    "ss.utils.apply_age_range",
    'find_contacts':      "ss.utils.find_contacts",
}

def __getattr__(name):
    """ Say what a name that Starsim no longer exports has become """
    if name in _removed:
        errormsg = f'ss.{name} has been removed from Starsim. Instead, write: {_removed[name]}. (The migrate-starsim3 skill lists every change.)'
        raise AttributeError(errormsg)
    raise AttributeError(f"module 'starsim' has no attribute '{name}'")

# Don't keep these in the module
del t, sc, debug
