# from .core.analyzers  import *
# from .core.connectors import *
# from .core.distributions import *
# from .core.interventions import *
# from .core.modules    import *
# from .core.parameters import *
# from .core.people     import *
# from .core.random     import *
# from .core.results    import *
# from .core.sim        import *
# from .core.demographics   import *

from .analyzers  import *
from .connectors import *
from .distributions import *
from .interventions import *
from .modules    import *
from .parameters import *
from .people     import *
from .random     import *
from .results    import *
from .sim        import *
from .demographics   import *
from .states.states   import *
from .states.dinamicview import *
from .states.fussedarray import *
from .diseases         import *
from .diseases.super.disease import *
from .diseases.super.sti import *
from .networks.networks         import *
from .networks.randnet          import *
from .networks.base_networks import *
from .utils.logger import *
from .utils.ndict  import *
from .utils.actions import *
from .version import __version__, __versiondate__, __license__
from .settings import options
from .models.SIR import *
from .models.NCD import *

import sciris as sc
root = sc.thispath(__file__).parent

# Import the version and print the license
if options.verbose:
  print(__license__)