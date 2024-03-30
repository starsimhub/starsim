from .version import __version__, __versiondate__, __license__
from .settings      import *
from .parameters    import *
from .utils         import *
from .distributions import *
from .states        import *
from .people        import *
from .modules       import *
from .network       import *
from .results       import *
from .demographics  import *
from .products      import *
from .interventions import *
from .demographics  import *
from .connectors    import *
from .disease       import *
from .diseases      import *
from .sim           import *
from .run           import *
from .samples       import *

# Assign the root folder
import sciris as sc
root = sc.thispath(__file__).parent

# Import the version and print the license
if options.verbose:
    print(__license__)

# Double-check key requirements -- should match setup.py
sc.require(['sciris>=3.1.6', 'pandas>=2.0.0', 'scipy', 'numba', 'networkx'], message=f'The following dependencies for Starsim {__version__} were not met: <MISSING>.')
del sc # Don't keep this in the module