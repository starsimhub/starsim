from .version import __version__, __versiondate__, __license__
from .settings      import *
from .parameters    import *
from .utils         import *
from .distributions import *
from .states        import *
from .random        import *
from .people        import *
from .modules       import *
from .networks      import *
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

import sciris as sc
root = sc.thispath(__file__).parent

# Import the version and print the license
if options.verbose:
    print(__license__)