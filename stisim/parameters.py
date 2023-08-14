"""
Set parameters
"""

import sciris as sc
from .settings import options as sso  # For setting global options

__all__ = ['make_pars']




class Parameters(sc.objdict):
    """
    Create the parameters for the simulation. Typically, this function is used
    internally rather than called by the user; e.g. typical use would be to do
    sim = ss.Sim() and then inspect sim.pars, rather than calling this function
    directly.

    Args:
        kwargs        (dict): any additional kwargs are interpreted as parameter names
    Returns:
        pars (dict): the parameters of the simulation
    """
    def __init__(self, **kwargs):

        # Population parameters
        self['n_agents']        = 10e3          # Number of agents
        self['total_pop']       = 10e3          # If defined, used for calculating the scale factor
        self['pop_scale']       = None          # How much to scale the population
        self['location']        = None          # What demographics to use - NOT CURRENTLY FUNCTIONAL
        self['birth_rates']     = None          # Birth rates, loaded below
        self['death_rates']     = None          # Death rates, loaded below
        self['rel_birth']       = 1.0           # Birth rate scale factor
        self['rel_death']       = 1.0           # Death rate scale factor
    
        # Simulation parameters
        self['start']           = 1995.         # Start of the simulation
        self['end']             = None          # End of the simulation
        self['n_years']         = 35            # Number of years to run, if end isn't specified. Note that this includes burn-in
        self['burnin']          = 25            # Number of years of burnin. NB, this is doesn't affect the start and end dates of the simulation, but it is possible remove these years from plots
        self['dt']              = 1.0           # Timestep (in years)
        self['dt_demog']        = 1.0           # Timestep for demographic updates (in years)
        self['rand_seed']       = 1             # Random seed, if None, don't reset
        self['verbose']         = sso.verbose   # Whether or not to display information during the run -- options are 0 (silent), 0.1 (some; default), 1 (default), 2 (everything)
        self['use_migration']   = True          # Whether to estimate migration rates to correct the total population size
    
        # Events and interventions
        self['connectors']      = sc.autolist()
        self['interventions']   = sc.autolist() # The interventions present in this simulation; populated by the user
        self['analyzers']       = sc.autolist() # The functions present in this simulation; populated by the user
        self['timelimit']       = None          # Time limit for the simulation (seconds)
        self['stopping_func']   = None          # A function to call to stop the sim partway through
    
        # Network parameters, generally initialized after the population has been constructed
        self['networks']        = sc.autolist()  # Network types and parameters
        self['debut']           = dict(f=dict(dist='normal', par1=15.0, par2=2.0),
                                       m=dict(dist='normal', par1=17.5, par2=2.0))

        # Update with any supplied parameter values and generate things that need to be generated
        self.update(kwargs)

        return


    def update_pars(self, pars=None, create=False):
        """
        Update internal dict with new pars.

        Args:
            pars (dict): the parameters to update (if None, do nothing)
            create (bool): if create is False, then raise a KeyNotFoundError if the key does not already exist
        """
        if pars is not None:
            if not isinstance(pars, dict):
                raise TypeError(f'The pars object must be a dict; you supplied a {type(pars)}')
            if not create:
                available_keys = list(self.keys())
                mismatches = [key for key in pars.keys() if key not in available_keys]
                if len(mismatches):
                    errormsg = f'Key(s) {mismatches} not found; available keys are {available_keys}'
                    raise sc.KeyNotFoundError(errormsg)
            self.update(pars)
        return

def make_pars(**kwargs):
    return Parameters(**kwargs)
    