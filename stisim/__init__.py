from .version import __version__, __versiondate__, __license__
from .settings      import *
from .parameters    import *
from .utils         import *
from .people        import *
from .networks      import *
from .modules       import *
from .demographics  import *
from .hiv           import *
from .gonorrhea     import *
from .connectors    import *
from .sim           import *


# Import the version and print the license
if options.verbose:
    print(__license__)