"""
General module class -- base class for diseases, interventions, etc.
"""

import sciris as sc
import starsim as ss
from scipy.stats._distn_infrastructure import rv_frozen

__all__ = ['Module']


class Module(sc.quickobj):

    def __init__(self, pars=None, par_dists=None, name=None, label=None, requires=None, **kwargs):
        self.pars = ss.omerge(pars, kwargs)
        self.par_dists = ss.omerge(par_dists)
        self.name = name if (name is not None) else self.__class__.__name__.lower() # Default name is the class name
        self.label = label if (label is not None) else self.name
        self.requires = sc.mergelists(requires)
        self.results = ss.Results(self.name)
        self.initialized = False
        self.finalized = False
        return
    
    def disp(self, output=False):
        """ Display the full object """
        out = sc.prepr(self)
        if not output:
            print(out)
        else:
            return out

    def check_requires(self, sim):
        """ Check that the module's requirements (of other modules) are met """
        errs = sc.autolist()
        all_names = [m.__class__ for m in sim.modules] + [m.name for m in sim.modules if hasattr(m, 'name')]
        for req in self.requires:
            if req not in all_names:
                errs += req
        if len(errs):
            errormsg = f'{self.name} (label={self.label}) requires the following module(s), but the Sim does not contain them.'
            errormsg += sc.newlinejoin(errs)
            raise Exception(errormsg)
        return

    def initialize(self, sim):
        """
        Perform initialization steps

        This method is called once, as part of initializing a Sim
        """
        self.check_requires(sim)

        # First, convert any scalar pars to distributions if required
        for key in self.par_dists.keys():
            par = self.pars[key]
            if isinstance(par, ss.Dist):
                continue

            # Handle arguments
            args = ()
            kwargs = {}
            if isinstance(par, dict):
                kwargs = par
            elif isinstance(par, (tuple, list)):
                args = par
            else:
                args = [par]

            # Make the distribution
            par_dist = self.par_dists[key]
            if isinstance(par_dist, str):
                try:
                    par_dist = getattr(ss.distributions, par_dist)
                except Exception as E:
                    errormsg = f'"{par_dist}" is not a valid distribution name; valid distributions are: {sc.newlinejoin(ss.dists.dist_list)}'
                    raise ValueError(errormsg) from E
            
            if issubclass(par_dist, ss.Dist): # Main use case
                par_dist = par_dist(*args, sim=sim, module=self, **kwargs)
            
            if not isinstance(par_dist, ss.Dist):
                print('Warning, should probably be an ss.Dist already')
                par_dist = ss.Dist(dist=par_dist, *args, **kwargs)
            
            self.pars[key] = par_dist

        # Initialize distributions in pars # TODO: refactor
        for key, value in self.pars.items():
            if isinstance(value, rv_frozen):
                self.pars[key] = ss.Dist(dist=value)

        for key, value in self.__dict__.items():
            if isinstance(value, rv_frozen):
                setattr(self, key, ss.Dist(dist=value))
        
        # Initialize everything # TODO: shouldn't be needed, should be able to recurse more
        for key,val in list(self.pars.items()) + list(self.__dict__.items()):
            if isinstance(val, ss.Dist):
                if val.initialized is not True: # Catches False and 'partial'
                    val.initialize(module=self, sim=sim, force=True) # Actually a dist
                else:
                    raise RuntimeError(f'Trying to reinitialize {val}, this should not happen')

        # Connect the states to the sim
        # Will use random numbers, so do after distribution initialization
        for state in self.states:
            state.initialize(sim)

        self.initialized = True
        return

    def finalize(self, sim):
        self.finalize_results(sim)
        self.finalized = True
        return

    def finalize_results(self, sim):
        """
        Finalize results
        """
        # Scale results
        for reskey, res in self.results.items():
            if isinstance(res, ss.Result) and res.scale:
                self.results[reskey] = self.results[reskey]*sim.pars.pop_scale
        return
    
    def add_states(self, *args, check=True):
        """
        Add states to the module with the same attribute name as the state
        
        Args:
            args (states): list of states to add
            check (bool): whether to check that the object being added is a state
        """
        for arg in args:
            if isinstance(arg, (list, tuple)):
                state = ss.State(*arg)
            elif isinstance(arg, dict):
                state = ss.State(**arg)
            else:
                state = arg
                
            if check:
                assert isinstance(state, ss.State), f'Could not add {state}: not a State object'
                
            setattr(self, state.name, state)
        return

    @property
    def states(self):
        """
        Return a flat list of all states

        The base class returns all states that are contained in top-level attributes
        of the Module. If a Module stores states in a non-standard location (e.g.,
        within a list of states, or otherwise in some other nested structure - perhaps
        due to supporting features like multiple genotypes) then the Module should
        overload this attribute to ensure that all states appear in here.
        """
        return [x for x in self.__dict__.values() if isinstance(x, ss.State)]

    @property
    def statesdict(self):
        """
        Return a flat dictionary (objdict) of all states

        Note that name collisions may affect the output of this function
        """
        return sc.objdict({s.name:s for s in self.states})

    @classmethod
    def create(cls, name, *args, **kwargs):
        """
        Create a module instance by name
        Args:
            name (str): A string with the name of the module class in lower case, e.g. 'sir'
        """
        for subcls in ss.all_subclasses(cls):
            if subcls.__name__.lower() == name:
                return subcls(*args, **kwargs)
        else:
            raise KeyError(f'Module "{name}" did not match any known Starsim Modules')
