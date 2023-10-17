'''
Networks that connect people within a population
'''

# %% Imports
import numpy as np
import sciris as sc
import stisim as ss
import scipy.optimize as spo
import scipy.spatial as sps


# Specify all externally visible functions this file defines
__all__ = ['Networks', 'Network', 'simple_sexual', 'simple_embedding', 'stable_monogamy', 'hpv_network', 'maternal']

class Network(sc.objdict):
    """
    A class holding a single network of contact edges (connections) between people
    as well as methods for updating these.

    The input is typically arrays including: person 1 of the connection, person 2 of
    the connection, the weight of the connection, the duration and start/end times of
    the connection.

    Args:
        p1 (array): an array of length N, the number of connections in the network, with the indices of people
                   on one side of the connection.
        p2 (array): an array of length N, the number of connections in the network, with the indices of people
                    on the other side of the connection.
        beta (array): an array representing relative transmissibility of each connection for this network - TODO, do we need this?
        label (str): the name of the network (optional)
        kwargs (dict): other keys copied directly into the network

    Note that all arguments (except for label) must be arrays of the same length,
    although not all have to be supplied at the time of creation (they must all
    be the same at the time of initialization, though, or else validation will fail).

    **Examples**::

        # Generate an average of 10 contacts for 1000 people
        n_contacts_pp = 10
        n_people = 1000
        n = n_contacts_pp * n_people
        p1 = np.random.randint(n_people, size=n)
        p2 = np.random.randint(n_people, size=n)
        beta = np.ones(n)
        network = ss.Network(p1=p1, p2=p2, beta=beta, label='rand')
        network = ss.Network(dict(p1=p1, p2=p2, beta=beta), label='rand') # Alternate method

        # Convert one network to another with extra columns
        index = np.arange(n)
        self_conn = p1 == p2
        network2 = ss.Network(**network, index=index, self_conn=self_conn, label=network.label)
    """

    def __init__(self, *args, pars=None, key_dict=None, vertical=False, label=None, **kwargs):
        default_keys = {
            'p1': ss.int_,
            'p2': ss.int_,
            'beta': ss.float_,
        }

        self.meta = sc.mergedicts(default_keys, key_dict)
        self.vertical = vertical  # Whether transmission is bidirectional
        self.basekey = 'p1'  # Assign a base key for calculating lengths and performing other operations
        self.label = label
        self.initialized = False

        # Handle args
        kwargs = sc.mergedicts(*args, kwargs)

        # Initialize the keys of the network
        for key, dtype in self.meta.items():
            self[key] = np.empty((0,), dtype=dtype)

        # Set data, if provided
        for key, value in kwargs.items():
            self[key] = np.array(value, dtype=self.meta.get(key))
            self.initialized = True

    @property
    def name(self):
        # The module name is a lower-case version of its class name
        return self.__class__.__name__.lower()

    def initialize(self, sim):
        pass
        # Auto initialization trick for streams does not work because objdict overwrites __dict__
        '''
        # Connect the streams to the sim
        for stream in self.streams.values():
            stream.initialize(sim.streams)
        return
        '''

    '''
    @property
    def streams(self):
       return ss.ndict({k:v for k,v in self.__dict__.items() if isinstance(v, ss.Stream)})
    '''

    def __len__(self):
        try:
            return len(self[self.basekey])
        except:  # pragma: no cover
            return 0

    def __repr__(self, **kwargs):
        """ Convert to a dataframe for printing """
        namestr = self.name
        labelstr = f'"{self.label}"' if self.label else '<no label>'
        keys_str = ', '.join(self.keys())
        output = f'{namestr}({labelstr}, {keys_str})\n'  # e.g. Network("r", f, m, beta)
        output += self.to_df().__repr__()
        return output

    def __contains__(self, item):
        """
        Check if a person is present in a network

        Args:
            item: Person index

        Returns: True if person index appears in any interactions
        """
        return (item in self['p1']) or (item in self['p2'])

    @property
    def members(self):
        """ Return sorted array of all members """
        return np.unique([self['p1'], self['p2']])

    def meta_keys(self):
        """ Return the keys for the network's meta information """
        return self.meta.keys()

    def validate(self, force=True):
        """
        Check the integrity of the network: right types, right lengths.

        If dtype is incorrect, try to convert automatically; if length is incorrect,
        do not.
        """
        n = len(self[self.basekey])
        for key, dtype in self.meta.items():
            if dtype:
                actual = self[key].dtype
                expected = dtype
                if actual != expected:
                    self[key] = np.array(self[key],
                                         dtype=expected)  # Probably harmless, so try to convert to correct type
            actual_n = len(self[key])
            if n != actual_n:
                errormsg = f'Expecting length {n} for network key "{key}"; got {actual_n}'  # We can't fix length mismatches
                raise TypeError(errormsg)
        return

    def get_inds(self, inds, remove=False):
        """
        Get the specified indices from the edgelist and return them as a dict.
        Args:
            inds (int, array, slice): the indices to find
            remove (bool): whether to remove the indices
        """
        output = {}
        for key in self.meta_keys():
            output[key] = self[key][inds]  # Copy to the output object
            if remove:
                self[key] = np.delete(self[key], inds)  # Remove from the original
        return output

    def pop_inds(self, inds, do_return=True):
        """
        "Pop" the specified indices from the edgelist and return them as a dict.
        Returns arguments in the right format to be used with network.append().

        Args:
            inds (int, array, slice): the indices to be removed
        """
        popped_inds = self.get_inds(inds, remove=True)
        if do_return: return popped_inds
        else: return

    def append(self, contacts):
        """
        Append contacts to the current network.

        Args:
            contacts (dict): a dictionary of arrays with keys f,m,beta, as returned from network.pop_inds()
        """
        for key in self.meta_keys():
            new_arr = contacts[key]
            n_curr = len(self[key])  # Current number of contacts
            n_new = len(new_arr)  # New contacts to add
            n_total = n_curr + n_new  # New size
            self[key] = np.resize(self[key], n_total)  # Resize to make room, preserving dtype
            self[key][n_curr:] = new_arr  # Copy contacts into the network
        return

    def to_dict(self):
        """ Convert to dictionary """
        d = {k:self[k] for k in self.meta_keys()}
        return d

    def to_df(self):
        """ Convert to dataframe """
        df = sc.dataframe.from_dict(self.to_dict())
        return df

    def from_df(self, df, keys=None):
        """ Convert from a dataframe """
        if keys is None:
            keys = self.meta_keys()
        for key in keys:
            self[key] = df[key].to_numpy()
        return self

    def find_contacts(self, inds, as_array=True):
        """
        Find all contacts of the specified people

        For some purposes (e.g. contact tracing) it's necessary to find all the contacts
        associated with a subset of the people in this network. Since contacts are bidirectional
        it's necessary to check both P1 and P2 for the target indices. The return type is a Set
        so that there is no duplication of indices (otherwise if the Network has explicit
        symmetric interactions, they could appear multiple times). This is also for performance so
        that the calling code doesn't need to perform its own unique() operation. Note that
        this cannot be used for cases where multiple connections count differently than a single
        infection, e.g. exposure risk.

        Args:
            inds (array): indices of people whose contacts to return
            as_array (bool): if true, return as sorted array (otherwise, return as unsorted set)

        Returns:
            contact_inds (array): a set of indices for pairing partners

        Example: If there were a network with
        - P1 = [1,2,3,4]
        - P2 = [2,3,1,4]
        Then find_contacts([1,3]) would return {1,2,3}
        """

        # Check types
        if not isinstance(inds, np.ndarray):
            inds = sc.promotetoarray(inds)
        if inds.dtype != np.int64:  # pragma: no cover # This is int64 since indices often come from hpv.true(), which returns int64
            inds = np.array(inds, dtype=np.int64)

        # Find the contacts
        contact_inds = ss.find_contacts(self['p1'], self['p2'], inds)
        if as_array:
            contact_inds = np.fromiter(contact_inds, dtype=ss.int_)
            contact_inds.sort()  # Sorting ensures that the results are reproducible for a given seed as well as being identical to previous versions of HPVsim

        return contact_inds

    def add_pairs(self):
        """ Define how pairs of people are formed """
        pass

    def update(self, people):
        """ Define how pairs/connections evolve (in time) """
        return

    def remove_uids(self, uids):
        """
        Remove interactions involving specified UIDs

        This method is typically called via `People.remove()` and
        is specifically used when removing agents from the simulation.

        """
        if len(uids) == 0:
            return

        keep = ~(np.isin(self.p1, uids) | np.isin(self.p2, uids))
        for k in self.meta_keys():
            self[k] = self[k][keep]
        return


class Networks(ss.ndict):
    def __init__(self, *args, type=Network, **kwargs):
        return super().__init__(self, *args, type=type, **kwargs)
    

class simple_sexual(Network):
    """
    A class holding a single network of contact edges (connections) between people.
    This network is built by **randomly pairing** males and female with variable relationship durations.
    """
    def __init__(self, mean_dur=5, **kwargs):
        key_dict = {
            'p1': ss.int_,
            'p2': ss.int_,
            'dur': ss.float_,
            'beta': ss.float_,
        }

        # Call init for the base class, which sets all the keys
        super().__init__(key_dict=key_dict, **kwargs)

        # Set other parameters
        self.mean_dur = mean_dur

        # Define random streams
        self.rng_pair_12 = ss.Stream('pair_12')
        self.rng_pair_21 = ss.Stream('pair_21')
        self.rng_mean_dur = ss.Stream('mean_dur')

        return

    def initialize(self, sim):
        super().initialize(sim)
        
        # Initialize random streams
        self.rng_pair_12.initialize(sim.streams, sim.people.slot)
        self.rng_pair_21.initialize(sim.streams, sim.people.slot)
        self.rng_mean_dur.initialize(sim.streams, sim.people.slot)

        self.add_pairs(sim.people, ti=0)
        return

    def add_pairs(self, people, ti=None):
        # Find unpartnered males and females - could in principle check other contact layers too
        # by having the People object passed in here

        available_m = np.setdiff1d(people.uid[~people.female], self.members)
        available_f = np.setdiff1d(people.uid[people.female], self.members)

        if len(available_m) <= len(available_f):
            p1 = available_m
            p2 = self.rng_pair_12.choice(a=available_f, size=len(p1), replace=False) # TODO: Stream-ify
        else:
            p2 = available_f
            p1 = self.rng_pair_21.choice(a=available_m, size=len(p2), replace=False) # TODO: Stream-ify

        beta = np.ones_like(p1)
        dur = self.rng_mean_dur.poisson(p1, self.mean_dur) # TODO: Stream-ify
        self['p1'] = np.concatenate([self['p1'], p1])
        self['p2'] = np.concatenate([self['p2'], p2])
        self['beta'] = np.concatenate([self['beta'], beta])
        self['dur'] = np.concatenate([self['dur'], dur])

    def update(self, people, dt=None):
        super().update(people)
        if dt is None: dt = people.dt
        # First remove any relationships due to end
        self['dur'] = self['dur'] - dt
        active = (self['dur'] > 0) & people.alive[self['p1']] & people.alive[self['p2']]
        self['p1'] = self['p1'][active]
        self['p2'] = self['p2'][active]
        self['beta'] = self['beta'][active]
        self['dur'] = self['dur'][active]

        # Then add new relationships for unpartnered people
        self.add_pairs(people)

class simple_embedding(simple_sexual):
    """
    A class holding a single network of contact edges (connections) between people.
    This network is built by **randomly pairing** males and female with variable relationship durations.
    """

    def add_pairs(self, people, ti=None):
        # Find unpartnered males and females - could in principle check other contact layers too
        # by having the People object passed in here

        available_m = np.setdiff1d(ss.true(~people.female & people.alive), self.members)
        available_f = np.setdiff1d(ss.true(people.female & people.alive), self.members)

        if not len(available_m) or not len(available_f):
            if ss.options.verbose > 1:
                print('No pairs to add')
            return 0

        loc_m = people.age[available_m].values - 5 + self.rng_pair_12.normal(size=available_m, std=3)
        loc_f = people.age[available_f].values     + self.rng_pair_21.normal(size=available_f, std=3)
        dist_mat = sps.distance_matrix(loc_m[:, np.newaxis], loc_f[:, np.newaxis])

        ind_m, ind_f = spo.linear_sum_assignment(dist_mat)
        # loc_f[ind_f[0]] is close to loc_m[ind_m[0]]

        n_pairs = len(ind_f)
        self['p1'] = np.concatenate([self['p1'], available_m[ind_m]])
        self['p2'] = np.concatenate([self['p2'], available_f[ind_f]])

        beta = np.ones(n_pairs)
        dur = self.rng_mean_dur.poisson(available_m[ind_m], self.mean_dur)
        self['beta'] = np.concatenate([self['beta'], beta])
        self['dur'] = np.concatenate([self['dur'], dur])

        return n_pairs


class stable_monogamy(simple_sexual):
    """
    Very simple network for debugging in which edges are:
    1-2, 3-4, 5-6, ...
    """
    def __init__(self, **kwargs):
        # Call init for the base class, which sets all the keys
        super().__init__(mean_dur=np.iinfo(int).max, **kwargs)
        return

    def initialize(self, sim):
        n = len(sim.people._uid_map)
        self['p1'] = np.arange(0,n,2) # EVEN
        self['p2'] = np.arange(1,n,2) # ODD
        self['beta'] = np.ones(len(self['p1']))
        self['dur'] = np.full(len(self['p1']), fill_value=np.iinfo(int).max, dtype=int)
        return
    
    def update(self, people, dt=None):
        pass


class hpv_network(Network):
    def __init__(self, pars=None, **kwargs):

        key_dict = {
            'p1': ss.int_,
            'p2': ss.int_,
            'dur': ss.float_,
            'acts': ss.float_,
            'start': ss.float_,
            'beta': ss.float_,
        }

        # Call init for the base class, which sets all the keys
        super().__init__(key_dict=key_dict, **kwargs)

        # Define default parameters
        self.pars = dict()
        self.pars['cross_layer']   = 0.05  # Proportion of agents who have concurrent cross-layer relationships
        self.pars['partners']      = ss.poisson(rate=0.01)  # The number of concurrent sexual partners
        self.pars['acts']          = ss.neg_binomial(mean=80, dispersion=40)  # The number of sexual acts per year
        self.pars['age_act_pars']  = dict(peak=30, retirement=100, debut_ratio=0.5, retirement_ratio=0.1) # Parameters describing changes in coital frequency over agent lifespans
        self.pars['condoms']       = 0.2  # The proportion of acts in which condoms are used
        self.pars['dur_pship']     = ss.normal_pos(mean=1, std=1)  # Duration of partnerships
        self.pars['participation'] = None  # Incidence of partnership formation by age
        self.pars['mixing']        = None  # Mixing matrices for storing age differences in partnerships

        # Define random number streams
        self.rng_partners  = ss.Stream('partners')
        self.rng_acts      = ss.Stream('acts')
        self.rng_dur_pship = ss.Stream('dur_pship')

        self.update_pars(pars)
        self.get_layer_probs()

    def initialize(self, sim):
        super().initialize(sim)

        # Initialize streams and connect to Distributions
        self.rng_partners.initialize(sim.streams)
        self.pars['partners'].set_stream(self.rng_partners)

        self.rng_acts.initialize(sim.streams)
        self.pars['acts'].set_stream(self.rng_acts)

        self.rng_dur_pship.initialize(sim.streams)
        self.pars['dur_pship'].set_stream(self.rng_dur_pship)

        self.add_pairs(sim.people, ti=0)
        return

    def update_pars(self, pars):
        if pars is not None:
            for k, v in pars.items():
                self.pars[k] = v
        return

    def get_layer_probs(self):

        defaults = {}
        mixing = np.array([
            #       0,  5,  10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75
            [ 0,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 5,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [10,    0,  0, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [15,    0,  0, .1, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [20,    0,  0, .1, .1, .1, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [25,    0,  0, .5, .1, .5, .1, .1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [30,    0,  0,  1, .5, .5, .5, .5, .1,  0,  0,  0,  0,  0,  0,  0,  0],
            [35,    0,  0, .5,  1,  1, .5,  1,  1, .5,  0,  0,  0,  0,  0,  0,  0],
            [40,    0,  0,  0, .5,  1,  1,  1,  1,  1, .5,  0,  0,  0,  0,  0,  0],
            [45,    0,  0,  0,  0, .1,  1,  1,  2,  1,  1, .5,  0,  0,  0,  0,  0],
            [50,    0,  0,  0,  0,  0, .1,  1,  1,  1,  1,  2, .5,  0,  0,  0,  0],
            [55,    0,  0,  0,  0,  0,  0, .1,  1,  1,  1,  1,  2, .5,  0,  0,  0],
            [60,    0,  0,  0,  0,  0,  0,  0, .1, .5,  1,  1,  1,  2, .5,  0,  0],
            [65,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  2, .5,  0],
            [70,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1, .5],
            [75,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1],
        ])

        participation = np.array([
                [ 0,  5,    10,    15,   20,   25,   30,   35,    40,    45,    50,    55,    60,    65,    70,    75],
                [ 0,  0,  0.10,   0.7,  0.8,  0.6,  0.6,  0.4,   0.1,  0.05, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001], # Share of females of each age newly having casual relationships
                [ 0,  0,  0.05,   0.7,  0.8,  0.6,  0.6,  0.4,   0.4,   0.3,   0.1,  0.05,  0.01,  0.01, 0.001, 0.001]], # Share of males of each age newly having casual relationships
            )

        defaults['mixing'] = mixing
        defaults['participation'] = participation

        for pkey, pval in defaults.items():
            if self.pars[pkey] is None:
                self.pars[pkey] = pval

        return

    def add_pairs(self, people, ti=0):

        female = people.female
        active = people.active
        f_active = female & active
        m_active = ~female & active

        # Compute number of partners
        f_partnered_inds, f_partnered_counts = np.unique(self['p1'], return_counts=True)
        m_partnered_inds, m_partnered_counts = np.unique(self['p2'], return_counts=True)
        current_partners = np.zeros((len(people)))
        current_partners[f_partnered_inds] = f_partnered_counts
        current_partners[m_partnered_inds] = m_partnered_counts
        partners = self.pars['partners'].sample(len(people)) + 1
        underpartnered = current_partners < partners  # Indices of underpartnered people
        f_eligible = f_active & underpartnered
        m_eligible = m_active & underpartnered

        # Bin the agents by age
        bins = self.pars['participation'][0, :]  # Extract age bins

        # Try randomly select females for pairing
        f_eligible_inds = ss.true(f_eligible)  # Inds of all eligible females
        age_bins_f = np.digitize(people.age[f_eligible_inds], bins=bins) - 1  # Age bins of selected females
        bin_range_f = np.unique(age_bins_f)  # Range of bins
        f = []  # Initialize the female partners
        for ab in bin_range_f:  # Loop over age bins
            these_f_contacts = ss.binomial_filter(self.pars['participation'][1][ab], f_eligible_inds[
                age_bins_f == ab])  # Select females according to their participation rate in this layer
            f += these_f_contacts.tolist()

        # Select males according to their participation rate in this layer
        m_eligible_inds = ss.true(m_eligible)
        age_bins_m = np.digitize(people.age[m_eligible_inds], bins=bins) - 1
        bin_range_m = np.unique(age_bins_m)  # Range of bins
        m = []  # Initialize the male partners
        for ab in bin_range_m:
            these_m_contacts = ss.binomial_filter(self.pars['participation'][2][ab], m_eligible_inds[
                age_bins_m == ab])  # Select males according to their participation rate in this layer
            m += these_m_contacts.tolist()

        # Create preference matrix between eligible females and males that combines age and geo mixing
        age_bins_f = np.digitize(people.age[f],
                                 bins=bins) - 1  # Age bins of females that are entering new relationships
        age_bins_m = np.digitize(people.age[m], bins=bins) - 1  # Age bins of active and participating males
        age_f, age_m = np.meshgrid(age_bins_f, age_bins_m)
        pair_probs = self.pars['mixing'][age_m, age_f + 1]

        f_to_remove = pair_probs.max(axis=0) == 0  # list of female inds to remove if no male partners are found for her
        f = [i for i, flag in zip(f, f_to_remove) if ~flag]  # remove the inds who don't get paired on this timestep
        selected_males = []
        if len(f):
            pair_probs = pair_probs[:, np.invert(f_to_remove)]
            choices = []
            fems = np.arange(len(f))
            f_paired_bools = np.full(len(fems), True, dtype=bool)
            np.random.shuffle(fems) # TODO: Stream-ify
            for fem in fems:
                m_col = pair_probs[:, fem]
                if m_col.sum() > 0:
                    m_col_norm = m_col / m_col.sum()
                    choice = np.random.choice(len(m_col_norm), 1, replace=False, p=m_col_norm) # TODO: Stream-ify
                    choices.append(choice)
                    pair_probs[choice, :] = 0  # Once male partner is assigned, remove from eligible pool
                else:
                    f_paired_bools[fem] = False
            selected_males = np.array(m)[np.array(choices).flatten()]
            f = np.array(f)[f_paired_bools]

        p1 = np.array(f)
        p2 = selected_males
        n_partnerships = len(p1)
        dur = self.pars['dur_pship'].sample(n_partnerships)
        acts = self.pars['acts'].sample(n_partnerships)
        age_p1 = people.age[p1]
        age_p2 = people.age[p2]

        age_debut_p1 = people.debut[p1]
        age_debut_p2 = people.debut[p2]

        # For each couple, get the average age they are now and the average age of debut
        avg_age = np.array([age_p1, age_p2]).mean(axis=0)
        avg_debut = np.array([age_debut_p1, age_debut_p2]).mean(axis=0)

        # Shorten parameter names
        dr = self.pars['age_act_pars']['debut_ratio']
        peak = self.pars['age_act_pars']['peak']
        rr = self.pars['age_act_pars']['retirement_ratio']
        retire = self.pars['age_act_pars']['retirement']

        # Get indices of people at different stages
        below_peak_inds = avg_age <= self.pars['age_act_pars']['peak']
        above_peak_inds = (avg_age > self.pars['age_act_pars']['peak']) & (
                    avg_age < self.pars['age_act_pars']['retirement'])
        retired_inds = avg_age > self.pars['age_act_pars']['retirement']

        # Set values by linearly scaling the number of acts for each partnership according to
        # the age of the couple at the commencement of the relationship
        below_peak_vals = acts[below_peak_inds] * (dr + (1 - dr) / (peak - avg_debut[below_peak_inds]) * (
                avg_age[below_peak_inds] - avg_debut[below_peak_inds]))
        above_peak_vals = acts[above_peak_inds] * (
                rr + (1 - rr) / (peak - retire) * (avg_age[above_peak_inds] - retire))
        retired_vals = 0

        # Set values and return
        scaled_acts = np.full(len(acts), np.nan, dtype=ss.float_)
        scaled_acts[below_peak_inds] = below_peak_vals
        scaled_acts[above_peak_inds] = above_peak_vals
        scaled_acts[retired_inds] = retired_vals
        start = np.array([ti] * n_partnerships, dtype=ss.float_)
        beta = np.ones(n_partnerships)

        new_contacts = dict(
            p1=p1,
            p2=p2,
            dur=dur,
            acts=scaled_acts,
            start=start,
            beta=beta
        )
        self.append(new_contacts)
        return

    def update(self, people, ti=None, dt=None):
        if ti is None: ti = people.ti
        if dt is None: dt = people.dt
        # First remove any relationships due to end
        self['dur'] = self['dur'] - dt
        active = self['dur'] > 0
        for key in self.meta.keys():
            self[key] = self[key][active]

        # Then add new relationships
        self.add_pairs(people, ti=ti)
        return


class maternal(Network):
    def __init__(self, key_dict=None, vertical=True, **kwargs):
        """
        Initialized empty and filled with pregnancies throughout the simulation
        """
        key_dict = sc.mergedicts({'dur': ss.float_}, key_dict)
        super().__init__(key_dict=key_dict, vertical=vertical, **kwargs)
        return

    def update(self, people, dt=None):
        if dt is None: dt = people.dt
        # Set beta to 0 for women who complete post-partum period
        # Keep connections for now, might want to consider removing
        self['dur'] = self['dur'] - dt
        inactive = self['dur'] <= 0
        self['beta'][inactive] = 0

    def initialize(self, people):
        """ No pairs added upon initialization """
        pass

    def add_pairs(self, mother_inds, unborn_inds, dur):
        """
        Add connections between pregnant women and their as-yet-unborn babies
        """
        beta = np.ones_like(mother_inds)
        self['p1'] = np.concatenate([self['p1'], mother_inds])
        self['p2'] = np.concatenate([self['p2'], unborn_inds])
        self['beta'] = np.concatenate([self['beta'], beta])
        self['dur'] = np.concatenate([self['dur'], dur])
        return