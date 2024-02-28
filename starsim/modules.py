"""
General module class -- base class for diseases, interventions, etc.
"""

import sciris as sc
import starsim as ss
from scipy.stats._distn_infrastructure import rv_frozen
from inspect import signature, _empty
import numpy as np

__all__ = ['Module']

class Module(sc.prettyobj):

    def __init__(self, pars=None, par_dists=None, name=None, label=None, requires=None, **kwargs):
        self.pars = ss.omerge(pars, kwargs)
        self.par_dists = ss.omerge(par_dists)
        self.name = name if name else self.__class__.__name__.lower() # Default name is the class name
        self.label = label if label else self.name
        self.requires = sc.mergelists(requires)
        self.results = ss.Results(self.name)
        self.initialized = False
        self.finalized = False
        return

    def check_requires(self, sim):
        """ Check that the module's requirements (of other modules) are met """
        errs = sc.autolist()
        all_names = [m.__class__ for m in sim.modules] + [m.name for m in sim.modules]
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
            if isinstance(par, rv_frozen):
                continue

            par_dist = self.par_dists[key]

            # If it's a lognormal distribution, initialize assuming the par is the desired mean
            if par_dist.name == 'lognorm':
                if sc.isiterable(par):
                    if isinstance(par, dict):
                        mu = par['mu']
                        stdev = par['stdev']
                    elif isinstance(par, list):
                        mu = par[0]
                        stdev = par[1]
                elif sc.isnumber(par):
                    mu = par
                    stdev = 1

                s, scale = ss.lognorm_params(mu, stdev)  # Assume stdev of 1
                self.pars[key] = self.par_dists[key](s=s, scale=scale)

            # Otherwise, figure out the required arguments and assume the user is trying to set them
            else:
                rqrd_args = [x for x, p in signature(par_dist._parse_args).parameters.items() if p.default == _empty]
                if len(rqrd_args) != 0:
                    par_dist_arg = rqrd_args[0]
                else:
                    par_dist_arg = 'loc'
                self.pars[key] = self.par_dists[key](**{par_dist_arg: par})

        # Initialize distributions in pars
        for key, value in self.pars.items():
            if isinstance(value, rv_frozen):
                self.pars[key] = ss.ScipyDistribution(value, f'{self.name}_{self.label}_{key}')
                self.pars[key].initialize(sim, self)

        for key, value in self.__dict__.items():
            if isinstance(value, rv_frozen):
                setattr(self, key, ss.ScipyDistribution(value, f'{self.name}_{self.label}_{key}'))
                getattr(self, key).initialize(sim, self)

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
