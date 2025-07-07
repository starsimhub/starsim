"""
Import all Starsim modules
"""

# Double-check key requirements -- should match pyproject.toml
import sciris as sc
reqs = ['sciris>=3.2.2', 'pandas>=2.0.0']
msg = f'The following dependencies for Starsim {__version__} were not met: <MISSING>.'
sc.require(reqs, message=msg)
del sc, reqs, msg # Don't keep this in the module

# Assign the root folder
root = sc.thispath(__file__).parent

# Start imports: version and settings
from .version import __version__, __versiondate__, __license__
from .settings import *

# Optionally print the license
if options.license:
    print(__license__)

# Finish imports
from .utils            import *
from .arrays           import *
from .time             import *
from .parameters       import *
from .distributions    import *
from .people           import *
from .modules          import *
from .networks         import *
from .results          import *
from .demographics     import *
from .products         import *
from .interventions    import *
from .demographics     import *
from .diseases         import *
from .diseases         import *
from .loop             import *
from .sim              import *
from .run              import *
from .calibration      import *
from .calib_components import *
from .samples          import *

# Load fonts
def _load_fonts(verbose=True):
    """ Try installing and loading fonts, but fail silently if folder is not writable """
    test_file = root / 'starsim' / 'assets' / 'fonts_installed'

    # Check if fonts have already been installed
    if not test_file.exists():
        if verbose: print(f'File {test_file} does not exist, continuing...')
        try:
            if verbose: print('Checking write permissions...')
            test_file.touch()  # Mark that fonts were installed
            if verbose: print('Rebuilding fonts...')
            settings.load_fonts(rebuild=True)
            if options.verbose:
                print('Note: rebuilding the font cache only happens once, or set STARSIM_FONTS=0 to disable.')
        except Exception as E:
            if verbose: print(f'File not writable or another error: {E}')
            pass  # Not writable or something else went wrong; ignore
    else:
        if verbose: print(f'File {test_file} exists, loading fonts...')
        settings.load_fonts(rebuild=False)
    return

# Try loading fonts
_load_fonts()
