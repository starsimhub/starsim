"""
General module class -- base class for diseases, interventions, etc.
"""

import sciris as sc
import starsim as ss

__all__ = ['module_map', 'find_modules', 'Module', 'Connector']


def module_map(key=None):
    """ Define the mapping between module names and types """
    module_map = sc.objdict(
        networks      = ss.Network,
        demographics  = ss.Demographics,
        diseases      = ss.Disease,
        interventions = ss.Intervention,
        analyzers     = ss.Analyzer,
        connectors    = ss.Connector,
    )
    return module_map if key is None else module_map[key]


def find_modules(key=None):
    """ Find all subclasses of Module, divided by type """
    modules = sc.objdict()
    modmap = module_map()
    attrs = dir(ss)
    for modkey, modtype in modmap.items(): # Loop over each module type
        modules[modkey] = sc.objdict()
        for attr in attrs: # Loop over each attribute (inefficient, but doesn't need to be optimized)
            item = getattr(ss, attr)
            try:
                assert issubclass(item, modtype) # Check that it's a class, and instance of this module
                low_attr = attr.lower()
                modules[modkey][low_attr] = item # It passes, so assign it to the dict
                if modkey == 'networks' and low_attr.endswith('net'): # Also allow networks without 'net' suffix
                    modules[modkey][low_attr.removesuffix('net')] = item
            except:
                pass
    return modules if key is None else modules[key]
    

class Module(sc.quickobj):

    def __init__(self, name=None, label=None, requires=None):
        self.set_metadata(name, label, requires) # Usually reset as part of self.update_pars()
        self.pars = ss.Pars() # Usually populated via self.default_pars()
        self.results = ss.Results(self.name)
        self.initialized = False
        self.finalized = False
        return
    
    def __bool__(self):
        """ Ensure that zero-length modules (e.g. networks) are still truthy """
        return True
    
    def set_metadata(self, name, label, requires):
        """ Set metadata for the module """
        self.name = sc.ifelse(name, getattr(self, 'name', self.__class__.__name__.lower())) # Default name is the class name
        self.label = sc.ifelse(label, getattr(self, 'label', self.name))
        self.requires = sc.mergelists(requires)
        return
    
    def default_pars(self, inherit=True, **kwargs):
        """ Create or merge Pars objects """
        if inherit: # Merge with existing
            self.pars.update(**kwargs, create=True)
        else: # Or overwrite
            self.pars = ss.Pars(**kwargs)
        return self.pars
    
    def update_pars(self, pars, **kwargs):
        """ Pull out recognized parameters, returning the rest """
        pars = sc.mergedicts(pars, kwargs)
        
        # Update matching module parameters
        matches = {}
        for key in list(pars.keys()): # Need to cast to list to avoid "dict changed during iteration"
            if key in self.pars:
                matches[key] = pars.pop(key)
        self.pars.update(matches)
                
        # Update module attributes
        metadata = {key:pars.pop(key, None) for key in ['name', 'label', 'requires']}
        self.set_metadata(**metadata)
        
        # Should be no remaining pars
        if len(pars):
            errormsg = f'{len(pars)} unrecognized arguments for {self.name}: {sc.strjoin(pars.keys())}'
            raise ValueError(errormsg)
        return
    
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

    def init_pre(self, sim):
        """
        Perform initialization steps

        This method is called once, as part of initializing a Sim. Note: after
        initialization, initialized=False until init_post() is called (which is after
        distributions are initialized).
        
        Note: distributions cannot be used here because they aren't initialized 
        until after init_pre() is called. Use init_post() instead.
        """
        self.check_requires(sim)
        self.sim = sim # Link back to the sim object
        ss.link_dists(self, sim, skip=ss.Sim) # Link the distributions to sim and module
        sim.pars[self.name] = self.pars
        sim.results[self.name] = self.results
        sim.people.add_module(self) # Connect the states to the people
        return
    
    def init_post(self):
        """ Initialize the values of the states, including calling distributions; the last step of initialization """
        for state in self.states:
            if not state.initialized:
                state.init_vals()
        self.initialized = True
        return
    
    def disp(self, output=False):
        """ Display the full object """
        out = sc.prepr(self)
        if not output:
            print(out)
        else:
            return out

    def finalize(self):
        self.finalize_results()
        self.finalized = True
        return

    def finalize_results(self):
        """ Finalize results """
        # Scale results
        for reskey, res in self.results.items():
            if isinstance(res, ss.Result) and res.scale:
                self.results[reskey] = self.results[reskey]*self.sim.pars.pop_scale
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
                assert isinstance(state, ss.Arr), f'Could not add {state}: not an Arr object'
                
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
        return [x for x in self.__dict__.values() if isinstance(x, ss.Arr)] # TODO: use ndict

    @property
    def statesdict(self): # TODO: remove
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
            raise KeyError(f'Module "{name}" did not match any known Starsim modules')
            
    def to_json(self):
        """ Export to a JSON-compatible format """
        out = sc.objdict()
        out.type = self.__class__.__name__
        out.name = self.name
        out.label = self.label
        out.pars = self.pars.to_json()
        return out

    def plot(self):
        with sc.options.with_style('fancy'):
            flat = sc.flattendict(self.results, sep=': ')
            yearvec = self.sim.yearvec
            fig, axs = sc.getrowscols(len(flat), make=True)
            for ax, (k, v) in zip(axs.flatten(), flat.items()):
                ax.plot(yearvec, v)
                ax.set_title(k)
                ax.set_xlabel('Year')
        return fig


class Connector(Module):
    """
    Define a Connector, which mediates interactions between disease modules
    
    Because connectors can do anything, they have no specified structure: it is
    up to the user to define how they behave.    
    """
    pass
