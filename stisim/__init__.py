from .version import __version__, __versiondate__, __license__
from .settings      import *
from .parameters    import *
from .utils         import *
from .distributions import *
from .states        import *
from .people        import *
from .modules       import *
from .networks      import *
from .results       import *
from .demographics  import *
from .interventions import *
from .analyzers     import *
from .demographics  import *
from .hiv           import *
from .gonorrhea     import *
from .sim           import *

import sciris as sc
root = sc.thispath().parent

# Import the version and print the license
if options.verbose:
    print(__license__)