"""
Import all Starsim modules
"""

# Start imports: version and settings
from .version import __version__, __versiondate__, __license__
from .settings import dtypes, options

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
from .disease          import *
from .diseases         import *
from .loop             import *
from .sim              import *
from .run              import *
from .calibration      import *
from .calib_components import *
from .samples          import *

# Assign the root folder
import sciris as sc
root = sc.thispath(__file__).parent

# Double-check key requirements -- should match pyproject.toml
reqs = ['sciris>=3.2.0', 'pandas>=2.0.0']
msg = f'The following dependencies for Starsim {__version__} were not met: <MISSING>.'
sc.require(reqs, message=msg)
del sc, reqs, msg # Don't keep this in the module
