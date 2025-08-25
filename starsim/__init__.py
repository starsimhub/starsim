"""
Import all Starsim modules
"""

# Time how long each import takes -- disabled by default, only used for developer debugging
debug = False
if debug:
    print('Note: importing Starsim in debug mode')
    from time import time as pytime
    start = pytime()
    timings = {}
    def t(label):
        timings[label] = pytime() - start
else:
    def t(label):
        pass

# Assign the root folder
t('sciris') # SLow since importing NumPy, Matplotlib, etc.
import sciris as sc
root = sc.thispath(__file__).parent

# Start imports: version and settings
t('settings') # SLow since import Numba
from .version import __version__, __versiondate__, __license__
from .settings import *

# Optionally print the license
t('license')
if options.license:
    print(__license__)

# Double-check key requirements -- should match pyproject.toml
t('reqs')
sc.require(
    reqs = ['sciris>=3.2.4', 'pandas>=2.0.0'], 
    message = f'\nThe following dependencies for Starsim {__version__} were not met:\n  <MISSING>.\n\nYou can update with:\n  pip install <MISSING> --upgrade'
)

# Finish imports
t('utils        '); from .utils         import *
t('debugtools   '); from .debugtools    import *
t('arrays       '); from .arrays        import *
t('distributions'); from .distributions import * # Slow import due to scipy.stats
t('time         '); from .time          import *
t('timeline     '); from .timeline      import *
t('parameters   '); from .parameters    import *
t('people       '); from .people        import *
t('modules      '); from .modules       import *
t('networks     '); from .networks      import * # Slow import due to networkx
t('results      '); from .results       import *
t('demographics '); from .demographics  import *
t('products     '); from .products      import *
t('interventions'); from .interventions import *
t('analyzers    '); from .analyzers     import *
t('connectors   '); from .connectors    import *
t('diseases     '); from .diseases      import *
t('loop         '); from .loop          import *
t('sim          '); from .sim           import *
t('run          '); from .run           import *
t('calibration  '); from .calibration   import * # Lazy import to save 500 ms
t('samples      '); from .samples       import *

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
            except Exception as E:
                if debug: print(f'File not writable or another error: {E}')
                pass  # Not writable or something else went wrong; ignore
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
