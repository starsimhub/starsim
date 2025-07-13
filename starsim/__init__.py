"""
Import all Starsim modules
"""

# Start imports: version and settings
from .version import __version__, __versiondate__, __license__
from .settings import *

# Optionally print the license
if options.license:
    print(__license__)

# Assign the root folder
import sciris as sc
root = sc.thispath(__file__).parent

# Double-check key requirements -- should match pyproject.toml
reqs = ['sciris>=3.2.3', 'pandas>=2.0.0']
msg = f'\nThe following dependencies for Starsim {__version__} were not met:\n  <MISSING>.\n\n'
msg += 'You can update with:\n  pip install <MISSING> --upgrade'
sc.require(reqs, message=msg)
del sc, reqs, msg # Don't keep this in the module

# Finish imports
from .utils         import *
from .debugtools    import *
from .arrays        import *
from .time          import *
from .parameters    import *
from .distributions import *
from .people        import *
from .modules       import *
from .networks      import *
from .results       import *
from .demographics  import *
from .products      import *
from .interventions import *
from .analyzers     import *
from .connectors    import *
from .demographics  import *
from .diseases      import *
from .diseases      import *
from .loop          import *
from .sim           import *
from .run           import *
from .calibration   import *
from .samples       import *

# Load fonts
def _load_fonts(debug=False):
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
_load_fonts()
