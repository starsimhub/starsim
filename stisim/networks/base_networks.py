'''
Networks that connect people within a population
'''

# %% Imports
import numpy as np
import sciris as sc
from stisim.utils.ndict import *
from stisim.utils.actions import *
from stisim.modules import *
from stisim.settings import *
from stisim.states.states import *


# Specify all externally visible functions this file defines
__all__ = ['Networks', 'Network', 'NetworkConnector']


class Network(Module):
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

    def __init__(self, pars=None, key_dict=None, vertical=False, *args, **kwargs):

        # Initialize as a module
        super().__init__(pars, *args, **kwargs)

        # Each relationship is characterized by these default set of keys, plus any user- or network-supplied ones
        default_keys = {
            'p1': int_,
            'p2': int_,
            'beta': float_,
        }
        self.meta = omerge(default_keys, key_dict)
        self.vertical = vertical  # Whether transmission is bidirectional

        # Initialize the keys of the network
        self.contacts = sc.objdict()
        for key, dtype in self.meta.items():
            self.contacts[key] = np.empty((0,), dtype=dtype)

        # Set data, if provided
        for key, value in kwargs.items():
            self.contacts[key] = np.array(value, dtype=self.meta.get(key))
            self.initialized = True

        # Define states using placeholder values
        self.participant = State('participant', bool, fill_value=False)
        self.debut = State('debut', float, fill_value=0)
        return


    def initialize(self, sim):
        super().initialize(sim)
        return

    def __len__(self):
        try:
            return len(self.contacts.p1)
        except:  # pragma: no cover
            return 0

    def __repr__(self, **kwargs):
        """ Convert to a dataframe for printing """
        namestr = self.name
        labelstr = f'"{self.label}"' if self.label else '<no label>'
        keys_str = ', '.join(self.contacts.keys())
        output = f'{namestr}({labelstr}, {keys_str})\n'  # e.g. Network("r", p1, p2, beta)
        output += self.to_df().__repr__()
        return output

    def __contains__(self, item):
        """
        Check if a person is present in a network

        Args:
            item: Person index

        Returns: True if person index appears in any interactions
        """
        return (item in self.contacts.p1) or (item in self.contacts.p2)

    @property
    def members(self):
        """ Return sorted array of all members """
        return np.unique([self.contacts.p1, self.contacts.p2])

    def meta_keys(self):
        """ Return the keys for the network's meta information """
        return self.meta.keys()

    def set_network_states(self, people):
        """
        Many network states depend on properties of people -- e.g. MSM depends on being male,
        age of debut varies by sex and over time, and participation rates vary by age.
        Each time states are dynamically grown, this function should be called to set the network
        states that depend on other states.
        """
        pass

    def validate(self, force=True):
        """
        Check the integrity of the network: right types, right lengths.

        If dtype is incorrect, try to convert automatically; if length is incorrect,
        do not.
        """
        n = len(self.contacts.p1)
        for key, dtype in self.meta.items():
            if dtype:
                actual = self.contacts[key].dtype
                expected = dtype
                if actual != expected:
                    self.contacts[key] = np.array(self.contacts[key], dtype=expected)  # Try to convert to correct type
            actual_n = len(self.contacts[key])
            if n != actual_n:
                errormsg = f'Expecting length {n} for network key "{key}"; got {actual_n}'  # Report length mismatches
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
            output[key] = self.contacts[key][inds]  # Copy to the output object
            if remove:
                self.contacts[key] = np.delete(self.contacts[key], inds)  # Remove from the original
        return output

    def pop_inds(self, inds):
        """
        "Pop" the specified indices from the edgelist and return them as a dict.
        Returns arguments in the right format to be used with network.append().

        Args:
            inds (int, array, slice): the indices to be removed
        """
        popped_inds = self.get_inds(inds, remove=True)
        return popped_inds

    def append(self, contacts):
        """
        Append contacts to the current network.

        Args:
            contacts (dict): a dictionary of arrays with keys p1,p2,beta, as returned from network.pop_inds()
        """
        for key in self.meta_keys():
            new_arr = contacts[key]
            n_curr = len(self.contacts[key])  # Current number of contacts
            n_new = len(new_arr)  # New contacts to add
            n_total = n_curr + n_new  # New size
            self.contacts[key] = np.resize(self.contacts[key], n_total)  # Resize to make room, preserving dtype
            self.contacts[key][n_curr:] = new_arr  # Copy contacts into the network
        return

    def to_dict(self):
        """ Convert to dictionary """
        d = {k: self.contacts[k] for k in self.meta_keys()}
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
            self.contacts[key] = df[key].to_numpy()
        return self

    def find_contacts(self, inds, as_array=True):
        """
        Find all contacts of the specified people

        For some purposes (e.g. contact tracing) it's necessary to find all the contacts
        associated with a subset of the people in this network. Since contacts are bidirectional
        it's necessary to check both p1 and p2 for the target indices. The return type is a Set
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
        - p1 = [1,2,3,4]
        - p2 = [2,3,1,4]
        Then find_contacts([1,3]) would return {1,2,3}
        """

        # Check types
        if not isinstance(inds, np.ndarray):
            inds = sc.promotetoarray(inds)
        if inds.dtype != np.int64:  # pragma: no cover # This is int64 since indices often come from utils.true(), which returns int64
            inds = np.array(inds, dtype=np.int64)

        # Find the contacts
        contact_inds = find_contacts(self.contacts.p1, self.contacts.p2, inds)
        if as_array:
            contact_inds = np.fromiter(contact_inds, dtype=int_)
            contact_inds.sort()

        return contact_inds

    def add_pairs(self):
        """ Define how pairs of people are formed """
        pass

    def update(self, people):
        """ Define how pairs/connections evolve (in time) """
        pass

    def remove_uids(self, uids):
        """
        Remove interactions involving specified UIDs
        This method is typically called via `People.remove()` and
        is specifically used when removing agents from the simulation.
        """
        keep = ~(np.isin(self.contacts.p1, uids) | np.isin(self.contacts.p2, uids))
        for k in self.meta_keys():
            self.contacts[k] = self.contacts[k][keep]

class Networks(ndict):
    def __init__(self, *args, type=Network, connectors=None, **kwargs):
        self.setattribute('_connectors', ndict(connectors))
        super().__init__(*args, type=type, **kwargs)
        return

    def initialize(self, sim):
        for nw in self.values():
            nw.initialize(sim)
        for cn in self._connectors.values():
            cn.initialize(sim)
        return

    def update(self, people):
        for nw in self.values():
            nw.update(people)
        for cn in self._connectors.values():
            cn.update(people)
        return

class NetworkConnector(Module):
    """
    Template for a connector between networks.
    """
    def __init__(self, *args, networks=None, pars=None, **kwargs):
        super().__init__(pars, requires=networks, *args, **kwargs)
        return

    def set_participation(self, people, upper_age=None):
        pass

    def update(self, people):
        pass
