"""
Set parameters
"""

import sciris as sc
import stisim as ss


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
        self.total_pop       = 10e3  # If defined, used for calculating the scale factor
        self.pop_scale       = None  # How much to scale the population
        self.remove_dead     = True          # Remove dead agents each timestep

        # Demographic parameters: NOT CURRENTLY FUNCTIONAL
        # TBC whether these parameters live here or in separate demographic modules
        self.location    = None  # What demographics to use
        self.birth_rates = None  # Birth rates
        self.death_rates = None  # Death rates
        self.rel_birth   = 1.0  # Birth rate scale factor
        self.rel_death   = 1.0  # Death rate scale factor

        # Simulation parameters
        self.start           = 1995.         # Start of the simulation
        self.end             = None          # End of the simulation
        self.n_years         = 35            # Number of years to run, if end isn't specified. Note that this includes burn-in
        self.burnin          = 25            # Number of years of burnin. NB, this is doesn't affect the start and end dates of the simulation, but it is possible remove these years from plots
        self.dt              = 1.0           # Timestep (in years)
        self.dt_demog        = 1.0           # Timestep for demographic updates (in years)
        self.rand_seed       = 1             # Random seed, if None, don't reset
        self.verbose         = ss.options.verbose # Whether or not to display information during the run -- options are 0 (silent), 0.1 (some; default), 1 (default), 2 (everything)
        self.remove_dead     = True          # Remove dead agents each timestep

        # Events and interventions
        self.connectors = sc.autolist()
        self.interventions = sc.autolist()  # The interventions present in this simulation; populated by the user
        self.analyzers = sc.autolist()  # The functions present in this simulation; populated by the user

        # Network parameters, generally initialized after the population has been constructed
        self.networks        = sc.autolist()  # Network types and parameters
        self.debut           = dict(f=ss.normal(mean=15.0, std=2.0),
                                    m=ss.normal(mean=17.5, std=2.0))

        # Update with any supplied parameter values and generate things that need to be generated
        self.update(kwargs)

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
