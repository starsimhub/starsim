"""
Set parameters
"""

import sciris as sc
import starsim as ss

__all__ = ['Parameters', 'make_pars']

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
        self.n_agents        = 10e3  # Number of agents
        self.total_pop       = None  # If defined, used for calculating the scale factor
        self.pop_scale       = None  # How much to scale the population
        self.remove_dead     = 10    # How many timesteps to go between removing dead agents (0 to not remove)

        # Demographic parameters
        self.location    = None  #  NOT CURRENTLY FUNCTIONAL - what demographics to use
        self.birth_rate = None
        self.death_rate = None

        # Simulation parameters
        self.start           = 2000          # Start of the simulation
        self.end             = None          # End of the simulation
        self.n_years         = 49            # Number of years to run, if end isn't specified. Note that this includes burn-in
        self.burnin          = 0             # Number of years of burnin. NB, this is doesn't affect the start and end dates of the simulation, but it is possible remove these years from plots
        self.dt              = 1.0           # Timestep (in years)
        self.dt_demog        = 1.0           # Timestep for demographic updates (in years)
        self.rand_seed       = 1             # Random seed, if None, don't reset
        self.slot_scale      = 5             # Random slots will be assigned to newborn agents between min=n_agents and max=slot_scale*n_agents. Choosing a larger value here will reduce the probability of two agents using the same slot (and hence random draws), but increase the number of random numbers that are required.
        self.verbose         = ss.options.verbose # Whether or not to display information during the run -- options are 0 (silent), 0.1 (some; default), 1 (default), 2 (everything)

        # Plug-ins: demographics, diseases, connectors, networks, analyzers, and interventions
        self.demographics = ss.ndict()
        self.diseases = ss.ndict()
        self.networks = ss.ndict()
        self.connectors = ss.ndict()
        self.interventions = ss.ndict()
        self.analyzers = ss.ndict()

        # Update with any supplied parameter values and generate things that need to be generated
        self.update(kwargs)

        if self.slot_scale < 1:
            raise Exception('The value of the "slot_scale" parameter must be a number >= 1.0')

        return

    def update_pars(self, pars=None, create=False, **kwargs):
        """
        Update internal dict with new pars.
        Args:
            pars (dict): the parameters to update (if None, do nothing)
            create (bool): if create is False, then raise a KeyNotFoundError if the key does not already exist
        """
        if pars is not None:
            if not isinstance(pars, dict):
                raise TypeError(f'The pars object must be a dict; you supplied a {type(pars)}')

            pars = sc.mergedicts(pars, kwargs)
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
