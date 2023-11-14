import numpy as np
import sciris as sc
import stisim as ss

__all__ = ['Disease', 'STI']

class Disease(ss.Module):
    """ Base module contains states/attributes that all modules have """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = ss.ndict(type=ss.Result)

    def initialize(self, sim):
        super().initialize(sim)
        self.validate_pars(sim)
        self.set_initial_states(sim)
        self.init_results(sim)
        return

    def finalize(self, sim):
        super().finalize(sim)
        self.finalize_results(sim)

    def validate_pars(self, sim):
        """
        Perform any parameter validation

        :return: None if parameters are all valid
        :raises: Exception if there are any invalid parameters (or if the initialization is otherwise invalid in some way)
        """
        pass

    def set_initial_states(self, sim):
        """
        Set initial values for states

        This could involve passing in a full set of initial conditions,
        or using init_prev, or other. Note that this is different to initialization of the State objects
        i.e., creating their dynamic array, linking them to a People instance. That should have already
        taken place by the time this method is called. This method is about supplying initial values
        for the states (e.g., seeding initial infections)
        """
        pass

    def init_results(self, sim):
        """
        Initialize results
        """
        pass

    def finalize_results(self, sim):
        """
        Finalize results
        """
        pass

    def update_pre(self, sim):
        """
        Carry out autonomous updates at the start of the timestep (prior to transmission)

        :param sim:
        :return:
        """
        pass

    def make_new_cases(self, sim):
        """
        Add new cases of the disease

        This method is agnostic as to the mechanism by which new cases occur. This
        could be through transmission (parametrized in different ways, which may or
        may not use the contact networks) or it may be based on risk factors/seeding,
        as may be the case for non-communicable diseases.
        """
        pass

    def set_prognoses(self, sim, uids):
        pass

    def update_results(self, sim):
        """
        Update results

        This function is executed after transmission in all modules has been resolved.
        This allows result updates at this point to capture outcomes dependent on multiple
        modules, where relevant.
        """
        pass



class STI(Disease):
    """ Base module contains states/attributes that all modules have """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rel_sus = ss.State('rel_sus', float, 1)
        self.rel_sev = ss.State('rel_sev', float, 1)
        self.rel_trans = ss.State('rel_trans', float, 1)
        self.susceptible = ss.State('susceptible', bool, True)
        self.infected = ss.State('infected', bool, False)
        self.ti_infected = ss.State('ti_infected', int, ss.INT_NAN)
        return

    def initialize(self, sim):
        super().initialize(sim)

        # Initialization steps
        self.validate_pars(sim)
        self.set_initial_states(sim)
        self.init_results(sim)
        return

    def validate_pars(self, sim):
        """
        Perform any parameter validation
        """
        if 'beta' not in self.pars:
            self.pars.beta = sc.objdict({k: [1, 1] for k in sim.people.networks})
        return

    def set_initial_states(self, sim):
        """
        Set initial values for states. This could involve passing in a full set of initial conditions,
        or using init_prev, or other. Note that this is different to initialization of the State objects
        i.e., creating their dynamic array, linking them to a People instance. That should have already
        taken place by the time this method is called.
        """
        n_init_cases = int(self.pars['init_prev'] * len(sim.people))
        initial_cases = np.random.choice(sim.people.uid, n_init_cases, replace=False)
        self.set_prognoses(sim, initial_cases)
        return

    def init_results(self, sim):
        """
        Initialize results
        """
        self.results += ss.Result(self.name, 'n_susceptible', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'n_infected', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'prevalence', sim.npts, dtype=float)
        self.results += ss.Result(self.name, 'new_infections', sim.npts, dtype=int)
        return

    def update_pre(self, sim):
        """
        Carry out autonomous updates at the start of the timestep (prior to transmission)

        :param sim:
        :return:
        """
        pass

    def make_new_cases(self, sim):
        """ Add new cases of module, through transmission, incidence, etc. """
        pars = self.pars
        for k, layer in sim.people.networks.items():
            if k in pars['beta']:
                contacts = layer.contacts
                rel_trans = (self.infected & sim.people.alive).astype(float) * self.rel_trans
                rel_sus = (self.susceptible & sim.people.alive).astype(float) * self.rel_sus
                for a, b, beta in [[contacts.p1, contacts.p2, pars.beta[k][0]],
                                   [contacts.p2, contacts.p1, pars.beta[k][1]]]:
                    # probability of a->b transmission
                    p_transmit = rel_trans[a] * rel_sus[b] * contacts.beta * beta
                    new_cases = np.random.random(len(a)) < p_transmit
                    if np.any(new_cases):
                        self.set_prognoses(sim, b[new_cases])

    def set_prognoses(self, sim, uids):
        pass

    def set_congenital(self, sim, uids):
        # Need to figure out whether we would have a methods like this here or make it
        # part of a pregnancy/STI connector
        pass

    def update_results(self, sim):
        self.results['n_susceptible'][sim.ti] = np.count_nonzero(self.susceptible)
        self.results['n_infected'][sim.ti] = np.count_nonzero(self.infected)
        self.results['prevalence'][sim.ti] = self.results.n_infected[sim.ti] / np.count_nonzero(sim.people.alive)
        self.results['new_infections'][sim.ti] = np.count_nonzero(self.ti_infected == sim.ti)
