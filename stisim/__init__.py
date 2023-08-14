from .version import __version__, __versiondate__, __license__
from .settings      import *
from .misc          import *
from .parameters    import *
from .utils         import *
from .people        import *
from .networks      import *
from .sim           import *
from .modules       import *


# Import the version and print the license
if options.verbose:
    print(__license__)


# Set the root directory for the codebase
import pathlib
rootdir = pathlib.Path(__file__).parent