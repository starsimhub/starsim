"""
Defines the People class and functions associated with making people
"""

# %% Imports
import sciris as sc
from . import base as ssb
from . import misc as ssm
from .data import loaders as ssdata


__all__ = ['People']


# %% Define all properties of people


class People(ssb.BasePeople):
    """
    A class to perform all the operations on the people
    This class is usually created automatically by the sim. The only required input
    argument is the population size, but typically the full parameters dictionary
    will get passed instead since it will be needed before the People object is
    initialized.

    Note that this class handles the mechanics of updating the actual people, while
    ``ss.BasePeople`` takes care of housekeeping (saving, loading, exporting, etc.).
    Please see the BasePeople class for additional methods.

    Args:
        pars (dict): the sim parameters, e.g. sim.pars -- alternatively, if a number, interpreted as n_agents
        strict (bool): whether to only create keys that are already in self.meta.person; otherwise, let any key be set
        pop_trend (dataframe): a dataframe of years and population sizes, if available
        kwargs (dict): the actual data, e.g. from a popdict, being specified

    **Examples**::
        ppl = ss.People(2000)
    """

    # %% Basic methods

    def __init__(self, n, states=None, strict=True, **kwargs):
        """
        Initialize
        """
        super().__init__(n, states=states)
        if strict: self.lock()  # If strict is true, stop further keys from being set (does not affect attributes)
        self.kwargs = kwargs
        return

    def initialize(self, popdict=None):
        """ Perform initializations """
        super().initialize(popdict=popdict)  # Initialize states
        return

    def add_module(self, module, force=False):
        # Initialize all the states associated with a module
        # This is implemented as People.add_module rather than
        # Module.add_to_people(people) or similar because its primary
        # role is to modify the People object
        if hasattr(self, module.name) and not force:
            raise Exception(f'Module {module.name} already added')
        self.__setattr__(module.name, sc.objdict())

        for state_name, state in module.states.items():
            combined_name = module.name + '.' + state_name
            self._data[combined_name] = state.new(self._n)
            self._map_arrays(keys=combined_name)

        return

    def scale_flows(self, inds):
        """
        Return the scaled versions of the flows -- replacement for len(inds)
        followed by scale factor multiplication
        """
        return self.scale[inds].sum()

    def update_states(self, sim):
        """ Perform all state updates at the current timestep """

        self.age[~self.dead] += sim.dt
        self.dead[self.ti_dead <= sim.ti] = True

        for module in sim.modules.values():
            module.update_states(sim)

        # Perform network updates
        for lkey, layer in self.contacts.items():
            layer.update(self)

        return


def make_popdict(n=None, location=None, verbose=None, sex_ratio=0.5):
    """ Create a location-specific population dictionary """

    # Initialize total pop size
    total_pop = None

    # Load age data for this location if available
    if verbose:
        print(f'Loading location-specific data for "{location}"')
    try:
        age_data = ssdata.get_age_distribution(location, year=sim['start'])
        pop_trend = ssdata.get_total_pop(location)
        total_pop = sum(age_data[:, 2])  # Return the total population
        pop_age_trend = ssdata.get_age_distribution_over_time(location)
    except ValueError as E:
        warnmsg = f'Could not load age data for requested location "{location}" ({str(E)})'
        ssm.warn(warnmsg, die=True)

    uids, sexes, debuts = set_static(n_agents, sex_ratio=sex_ratio)

    # Set ages, rounding to nearest timestep if requested
    age_data_min = age_data[:, 0]
    age_data_max = age_data[:, 1]
    age_data_range = age_data_max - age_data_min
    age_data_prob = age_data[:, 2]
    age_data_prob /= age_data_prob.sum()  # Ensure it sums to 1
    age_bins = hpu.n_multinomial(age_data_prob, n_agents)  # Choose age bins
    if dt_round_age:
        ages = age_data_min[age_bins] + np.random.randint(
            age_data_range[age_bins] / dt) * dt  # Uniformly distribute within this age bin
    else:
        ages = age_data_min[age_bins] + age_data_range[age_bins] * np.random.random(
            n_agents)  # Uniformly distribute within this age bin

    # Store output
    popdict = {}
    popdict['uid'] = uids
    popdict['age'] = ages
    popdict['sex'] = sexes
    popdict['debut'] = debuts
    popdict['rel_sev'] = rel_sev
    popdict['partners'] = partners

    is_active = ages > debuts
    is_female = sexes == 0

    # Create the contacts
    lkeys = sim['partners'].keys()  # TODO: consider a more robust way to do this
    if microstructure in ['random', 'default']:
        contacts = dict()
        current_partners = np.zeros((len(lkeys), n_agents))
        lno = 0
        for lkey in lkeys:
            contacts[lkey], current_partners, _, _ = make_contacts(
                lno=lno, tind=0, partners=partners[lno, :], current_partners=current_partners,
                sexes=sexes, ages=ages, debuts=debuts, is_female=is_female, is_active=is_active,
                mixing=sim['mixing'][lkey], layer_probs=sim['layer_probs'][lkey], cross_layer=sim['cross_layer'],
                pref_weight=100, durations=sim['dur_pship'][lkey], acts=sim['acts'][lkey],
                age_act_pars=sim['age_act_pars'][lkey], **kwargs
            )
            lno += 1

    else:
        errormsg = f'Microstructure type "{microstructure}" not found; choices are random or TBC'
        raise NotImplementedError(errormsg)

    popdict['contacts'] = contacts
    popdict['current_partners'] = current_partners

    return popdict