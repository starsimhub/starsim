"""
Base classes for diseases
"""

import numpy as np
from stisim.utils.logger import InfectionLog
from stisim.utils.ndict import ndict as ssu
from stisim.core.modules import Module
import stisim as ss

class Disease(Module):
    """ Base module class for diseases """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = ss.ndict(type= ss.Result)
        self.log = InfectionLog()

    @property
    def _boolean_states(self):
        """
        Iterator over states with boolean type

        For diseases, these states typically represent attributes like 'susceptible',
        'infectious', 'diagnosed' etc. These variables are typically useful to

        :return:
        """
        for state in self.states:
            if state.dtype == bool:
                yield state

    def initialize(self, sim):
        super().initialize(sim)
        self.validate_pars(sim)
        self.init_results(sim)
        self.set_initial_states(sim)
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

        By default, diseases all report on counts for any boolean states e.g., if
        a disease contains a boolean state 'susceptible' it will automatically contain a
        Result for 'n_susceptible'
        """
        for state in self._boolean_states:
            self.results += ss.Result(self.name, f'n_{state.name}', sim.npts, dtype=int)
        return

    def finalize_results(self, sim):
        """
        Finalize results
        """
        # TODO - will probably need to account for rescaling outputs for the default results here
        pass

    def update_pre(self, sim):
        """
        Carry out autonomous updates at the start of the timestep (prior to transmission)

        :param sim:
        :return:
        """
        pass

    def update_death(self, sim, uids):
        """
        Carry out state changes upon death

        This function is triggered after deaths are resolved, and before analyzers are run.
        See the SIR example model for a typical use case - deaths are requested as an autonomous
        update, to take effect after transmission on the same timestep. State changes that occur
        upon death (e.g., clearing an `infected` flag) are executed in this function. That also
        allows an intervention to avert a death scheduled on the same timestep, without having
        to undo any state changes that have already been applied (because they only run via this
        function if the death actually occurs).

        Depending on the module and the results it produces, it may or may not be necessary
        to implement this.

        :param sim:
        :param uids:
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

        It is expected that this method will internally call Disease.set_prognoses()
        at some point.

        """
        pass


    def set_prognoses(self, sim, uids, from_uids=None):
        """
        Set prognoses upon infection/acquisition

        This function assigns state values upon infection or acquisition of
        the disease. It would normally be called somewhere towards the end of
        `Disease.make_new_cases()`. Infections will automatically be added to
        the log as part of this operation.

        The from_uids are relevant for infectious diseases, but would be left
        as `None` for NCDs.

        :param sim:
        :param uids: UIDs for agents to assign disease progoses to
        :param from_uids: Optionally specify the infecting agent
        :return:
        """
        if from_uids is None:
            for target in uids:
                self.log.append(np.nan, target, sim.year)
        else:
            for target, source in zip(uids, from_uids):
                self.log.append(source, target, sim.year)


    def update_results(self, sim):
        """
        Update results

        This function is executed after transmission in all modules has been resolved.
        This allows result updates at this point to capture outcomes dependent on multiple
        modules, where relevant.
        """
        for state in self._boolean_states:
            self.results[f'n_{state.name}'][sim.ti] = np.count_nonzero(state & sim.people.alive)
        return
