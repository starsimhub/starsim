"""
Import all Starsim modules
"""

import time as pytime

_init = pytime.time()

class quicktime:
    def __init__(self, label):
        self.init = _init
        self.label = label

    def __enter__(self):
        """ Reset start time when entering with-as block """
        self.start = pytime.time()
        return self

    def __exit__(self, *args):
        """ Print elapsed time when leaving a with-as block """
        self.elapsed = pytime.time() - self.start
        self.total = pytime.time() - self.init
        print(f'{self.label:20s}: {self.elapsed*1000:5.0f} ms | {self.total:n} s')
        return

print('\n\ncommand : delta | total\n')
with quicktime('import numpy'): import numpy
with quicktime('import pandas'): import pandas
with quicktime('import matplotlib'): import matplotlib
with quicktime('import sciris'): import sciris
with quicktime('import numba'): import numba
with quicktime('import scipy'): import scipy.stats
with quicktime('import networkx'): import networkx
with quicktime('import optuna'): import optuna
with quicktime('import optuna-vis'): import optuna.visualization.matplotlib
with quicktime('import seaborn'): import seaborn

# Start imports: version and settings
with quicktime('version'):    from .version import __version__, __versiondate__, __license__
with quicktime('settings'):   from .settings import dtypes, options

# Optionally print the license
if options.license:
    print(__license__)

# Finish imports
with quicktime('utils'):            from .utils            import *
with quicktime('arrays'):           from .arrays           import *
with quicktime('time'):             from .time             import *
with quicktime('parameters'):       from .parameters       import *
with quicktime('distributions'):    from .distributions    import *
with quicktime('people'):           from .people           import *
with quicktime('modules'):          from .modules          import *
with quicktime('networks'):         from .networks         import *
with quicktime('results'):          from .results          import *
with quicktime('demographics'):     from .demographics     import *
with quicktime('products'):         from .products         import *
with quicktime('interventions'):    from .interventions    import *
with quicktime('demographics'):     from .demographics     import *
with quicktime('disease'):          from .disease          import *
with quicktime('diseases'):         from .diseases         import *
with quicktime('loop'):             from .loop             import *
with quicktime('sim'):              from .sim              import *
with quicktime('run'):              from .run              import *
with quicktime('calibration'):      from .calibration      import *
with quicktime('calib_components'): from .calib_components import *
with quicktime('samples'):          from .samples          import *

# Assign the root folder
import sciris as sc
root = sc.thispath(__file__).parent

# Double-check key requirements -- should match pyproject.toml
reqs = ['sciris>=3.2.0', 'pandas>=2.0.0']
msg = f'The following dependencies for Starsim {__version__} were not met: <MISSING>.'
sc.require(reqs, message=msg)
del sc, reqs, msg # Don't keep this in the module
