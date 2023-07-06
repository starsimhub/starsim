from .version import __version__, __versiondate__, __license__
from .settings      import *
from .misc          import *
from .parameters    import *
from .utils         import *
from .base          import *
from .people        import *
from .networks      import *
from .sim           import *
from .conditions    import *
# from .hpv           import *


# Import the version and print the license
if settings.options.verbose:
    print(__license__)

# # Import data and check
# from . import data
# if not data.check_downloaded():
#     try:
#         data.quick_download(init=True)
#     except:
#         import sciris as sc
#         errormsg = f"Warning: couldn't download data:\n\n{sc.traceback()}\nProceeding anyway..."
#         print(errormsg)

# Set the root directory for the codebase
import pathlib
rootdir = pathlib.Path(__file__).parent