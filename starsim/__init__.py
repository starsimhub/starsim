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
t('settings') # SLow since import Numba
from .version import __version__, __versiondate__, __license__
from .settings import dtypes, options, style, load_fonts

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
    ndict, warn, find_contacts, standardize_netkey, parse_age_range, apply_age_range,
    standardize_data, validate_sim_data, load, save, plot_args, show,
    return_fig,
)

t('debugtools')
from .debugtools import (
    Profile, Debugger, Diagnostics, check_requires, check_version,
    metadata, mock_time, mock_sim, mock_people, mock_module,
)

t('arrays')
from .arrays import (
    BaseArr, Arr, FloatArr, IntArr, BoolArr, BoolState, IndexArr, uids,
)

t('distributions') # Slow import due to scipy.stats
from .distributions import (
    link_dists, make_dist, dist_list, scale_types, Dists, Dist,
    random, uniform, normal, lognorm_ex, lognorm_im, expon,
    poisson, nbinom, beta_dist, beta_mean, weibull, gamma, constant,
    randint, rand_raw, bernoulli, choice, histogram,
    multi_random,
)

t('time')
from .time import (
    DateArray, date, TimePar, dur, datedur, Rate, prob, per, freq,
    years, months, weeks, days, year, month, week, day,
    perday, perweek, permonth, peryear,
    probperday, probperweek, probpermonth, probperyear,
    freqperday, freqperweek, freqpermonth, freqperyear,
    rate, time_prob, rate_prob,
)

t('timeline')
from .timeline import Timeline

t('parameters')
from .parameters import Pars, SimPars

t('people')
from .people import People, Person

t('modules')
from .modules import (
    module_map, module_types, register_modules, find_modules, required,
    Base, Module,
)

t('networks') # Slow import due to networkx
from .networks import (
    Route, Network, DynamicNetwork, SexualNetwork,
    StaticNet, RandomNet, RandomSafeNet, MFNet, MSMNet,
    MaternalNet, PrenatalNet, PostnatalNet, BreastfeedingNet,
    HouseholdNet,
    AgeGroup, MixingPools, MixingPool,
)

t('results')
from .results import Result, Results

t('demographics')
from .demographics import (
    Demographics, Births, Deaths, PregnancyPars, Pregnancy, FetalHealth,
)

t('products')
from .products import Product, Dx, Tx, Vx, simple_vx

t('interventions')
from .interventions import (
    Intervention, RoutineDelivery, CampaignDelivery,
    BaseTest, BaseScreening, routine_screening, campaign_screening,
    BaseTriage, routine_triage, campaign_triage,
    BaseTreatment, treat_num,
    BaseVaccination, routine_vx, campaign_vx,
)

t('analyzers')
from .analyzers import Analyzer, infection_log, dynamics_by_age

t('connectors')
from .connectors import Connector, seasonality

t('diseases')
from .diseases import Disease, Infection, InfectionLog, NCD, SIR, SIS

t('loop')
from .loop import Loop

t('sim')
from .sim import Sim, AlreadyRunError, demo, diff_sims, check_sims_match

t('run')
from .run import MultiSim, single_run, multi_run, parallel

t('calibration') # Lazy import to save 500 ms
from .calibration import (
    Calibration, CalibComponent,
    linear_interp, linear_accum, step_containing,
    BetaBinomial, Binomial, DirichletMultinomial, GammaPoisson, Normal,
)

t('samples')
from .samples import Dataset, Samples

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
                load_fonts(rebuild=True)
                if options.verbose:
                    print('\nNote: rebuilding the font cache only happens once on first import, or set the environment variable STARSIM_INSTALL_FONTS=0 to disable.')
            except Exception as E: # Not writable or something else went wrong; ignore
                if debug: print(f'File not writable or another error: {E}')
        else:
            if debug: print(f'File {test_file} exists, loading fonts...')
            load_fonts(rebuild=False)
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
