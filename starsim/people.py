"""
Defines the People class and functions associated with making people
"""
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
from pathlib import Path
import matplotlib.pyplot as plt

__all__ = ['People', 'Person']


class People:
    """
    A class to perform all the operations on the people
    This class is usually created automatically by the sim. The only required input
    argument is the population size, but typically the full parameters dictionary
    will get passed instead since it will be needed before the People object is
    initialized.

    Note that this class handles the mechanics of updating the actual people,
    as well as the additional housekeeping methods (saving, loading, exporting, etc.).

    Args:
        pars (dict): the sim parameters, e.g. sim.pars -- alternatively, if a number, interpreted as n_agents
        age_data (dataframe): a dataframe of years and population sizes, if available
        extra_states (list): non-default states to initialize
        mock (bool): if True, initialize the People object with a mock Sim object (for debugging only)

    **Examples**:
        ppl = ss.People(2000)
    """

    def __init__(self, n_agents, age_data=None, extra_states=None, mock=False):
        """ Initialize """

        # We internally store states in a dict keyed by the memory ID of the state, so that we can have colliding names
        # e.g., across modules, but we will never change the size of a state multiple times in the same iteration over
        # _states. This is a hidden variable because it is internally used to synchronize the size of all states contained
        # within the sim, regardless of where they are. In contrast, `People.states` offers a more user-friendly way to access
        # a selection of the states e.g., module states could be added in there, while intervention states might not.
        self._states = {}
        self.version = ss.__version__  # Store version info
        self.initialized = False
        self.n_agents_init = n_agents

        # Handle the three fundamental arrays: UIDs for tracking agents, slots for
        # tracking random numbers, and AUIDs for tracking alive agents
        n = int(n_agents)
        uids = ss.uids(np.arange(n))
        self.auids = uids.copy() # This tracks all active UIDs (in practice, agents who are alive)
        self.uid    = ss.IndexArr('uid')  # This variable tracks all UIDs
        self.slot   = ss.IndexArr('slot') # A slot is a special state managed internally to generate random numbers
        self.parent = ss.IndexArr('parent', label='UID of parent')  # UID of parent, if any (used with pregnancy)
        self.uid.grow(new_vals=uids)
        self.slot.grow(new_vals=uids)
        self.parent.grow(new_uids=uids, new_vals=np.full(len(uids), self.parent.nan))
        for state in [self.uid, self.slot, self.parent]:
            state.people = self # Manually link to people since we don't want to link to states

        # Handle additional states
        extra_states = sc.promotetolist(extra_states)
        states = [
            ss.BoolState('alive', default=True),  # Time index for death
            ss.BoolState('female', default=ss.bernoulli(name='female', p=0.5)),
            ss.FloatArr('age', default=self.get_age_dist(age_data)), # NaN until conceived
            ss.FloatArr('ti_dead'),  # Time index for death
            ss.FloatArr('scale', default=1.0), # The scale factor for the agents (multiplied for making results)
        ]
        states.extend(extra_states)
        self.states = ss.ndict(type=ss.Arr)
        for state in states:
            self.states.append(state, overwrite=False)
            setattr(self, state.name, state)
            state.link_people(self)
        self._linked_modules = []

        if mock:
            self.init_mock()

        return

    def brief(self, output=False):
        n = self.n_agents_init
        alive = len(self)
        alive_str = f'{alive=:n}; ' if alive != n else ''
        age_mean = self.age.mean()
        age_std = self.age.std()
        out = f'People({n=:n}; {alive_str}age={age_mean:0.1f}±{age_std:0.1f})'
        return out if output else print(out)

    def __repr__(self):
        return self.brief(output=True)

    def __str__(self):
        out = self.brief(output=True)[:-1] # Remove trailing parenthesis
        out += f'; states=[{sc.strjoin(self.states.keys())}])'
        return out

    def disp(self):
        """ Full object information """
        return sc.pr(self)

    @staticmethod
    def get_age_dist(age_data):
        """
        Return an age distribution based on provided data

        The data should be provided in the form of either an Nx2 array, a pandas series
        with age as the index and counts/probability as the value, or a pandas DataFrame
        with "age" and "value" as columns. Each of these should look like e.g.:

            age      value
            0      220.548
            1      206.188
            2      195.792
            3      187.442

        The ages will be interpreted as lower bin edges. An upper bin edge will
        automatically be added based on the final age plus the difference of the
        last two bins. To explicitly control the width of the upper age bin, add
        an extra entry to the `age_data` with a value of 0 and an age value
        corresponding to the desired upper age bound.

        Args:
            age_data: An array/series/dataframe with an index corresponding to age values, and a value corresponding to histogram counts
                         or relative proportions. A distribution will be estimated based on the histogram. The histogram will be
                         assumed to correspond to probability densitiy if the sum of the histogram values is equal to 1, otherwise
                         it will be assumed to correspond to counts.

        Note: `age_data` can also be provided as a string (interpreted as a filename).

        If no value is provided, uniform ages from 0-60 are created (to match the
        global mean age of ~30 years).

        Returns:
            An [`ss.Dist`](`starsim.distributions.Dist`) instance that returns an age for newly created agents
        """
        if age_data is None:
            dist = ss.uniform(low=0, high=60, name='Age distribution')
        else:
            # Try loading from file
            if isinstance(age_data, str) or isinstance(age_data, Path):
                age_data = pd.read_csv(age_data)

            # Process
            if isinstance(age_data, np.ndarray): # TODO: accept output of np.histogram()
                age_bins = age_data[:,0]
                age_props = age_data[:,1]
            elif isinstance(age_data, pd.Series):
                age_bins = age_data.index
                age_props = age_data.values
            elif isinstance(age_data, pd.DataFrame):
                age_bins = age_data['age'].values
                age_props = age_data['value'].values

            # Convert to a histogram
            dist = ss.histogram(values=age_props, bins=age_bins, name='Age distribution')

        return dist

    def link_sim(self, sim, init=False):
        """ Initialization """
        if self.initialized:
            errormsg = 'Cannot re-initialize a People object directly; use sim.init(reset=True)'
            raise RuntimeError(errormsg)
        self.sim = sim # Store the sim
        ss.link_dists(obj=self.states, sim=sim, module=self, skip=[ss.Sim, ss.Module], init=init)
        return

    def add_module(self, module, force=False):
        """
        Add a Module to the People instance

        This method is used to add a module to the People. It will register any module states with this
        people instance for dynamic resizing, and expose the states contained in the module to the user
        via `People.states.<module_name>.<state_name>`

        The entries created below make it possible to do `sim.people.hiv.susceptible` or
        `sim.people.states['hiv.susceptible']` and have both of them work
        """
        # Map the module's states into the People state ndict
        if hasattr(self, module.name) and not force:
            raise Exception(f'Module {module.name} already added')

        if len(module.state_list):
            module_states = sc.objdict()
            setattr(self, module.name, module_states)
            self._linked_modules.append(module.name)
            for state in module.state_list:
                state.link_people(self)
                combined_name = module.name + '.' + state.name  # We will have to resolve how this works with multiple instances of the same module (e.g., for strains). The underlying machinery should be fine though, with People._states being flat and keyed by ID
                self.states[combined_name] = state # Register the state on the user-facing side using the combined name. Within the original module, it can still be referenced by its original name
                module_states[state.name] = state
        return

    def init_vals(self):
        """ Populate states with initial values, the final step of initialization """
        for state in self.states():
            if not state.initialized:
                state.init_vals()
        self.initialized = True
        return

    def init_mock(self):
        """ Initialize with a mock simulation (for debugging purposes only) """
        sim = ss.mock_sim(n_agents=self.n_agents)
        self.link_sim(sim, init=True)
        self.init_vals()
        return

    def __bool__(self):
        """ Ensure that zero-length people are still truthy """
        return True

    def __len__(self):
        """ Length of people """
        return len(self.auids)

    @property
    def n_agents(self):
        """ Alias for len(people) """
        return len(self)

    @property
    def n_uids(self):
        return self.uid.len_used

    def _link_state(self, state, die=True):
        """
        Link a state with the People instance for dynamic resizing; usually called by
        `state.link_people()`

        All states should be registered by this function for the purpose of connecting them to the
        People's UIDs and to have them be automatically resized when the number of agents changes.
        This operation is normally triggered as part of initializing the state (via `Arr.init()`)
        """
        if id(state) not in self._states:
            self._states[id(state)] = state
        elif die:
            errormsg = f'Cannot add state {state} since already added'
            raise ValueError(errormsg)
        return

    def grow(self, n=None, new_slots=None):
        """
        Increase the number of agents

        Args:
            n: Integer number of agents to add
            new_slots: Optionally specify the slots to assign for the new agents. Otherwise, it will default to the new UIDs
        """
        if n is None:
            if new_slots is None:
                errormsg = 'Must supply either n or new_slots'
                raise ValueError(errormsg)
            else:
                n = len(new_slots)

        if n == 0:
            return np.array([], dtype=ss.dtypes.int)

        start_uid = self.uid.len_used
        stop_uid = start_uid + n
        new_uids = ss.uids(np.arange(start_uid, stop_uid))
        self.uid.grow(new_uids, new_vals=new_uids)

        # We need to grow the slots as well
        new_slots = new_slots if new_slots is not None else new_uids
        self.slot.grow(new_uids, new_vals=new_slots)

        self.parent.grow(new_uids, new_vals=self.parent.nan) # Grow parent array

        # Grow the states
        for state in self._states.values():
            state.grow(new_uids)

        # Finally, update the alive indices
        self.auids = self.auids.concat(new_uids)
        return new_uids

    def __getitem__(self, key):
        """
        Allow people['attr'] instead of getattr(people, 'attr')
        If the key is an integer, alias `people.person()` to return a `Person` instance
        """
        if isinstance(key, int):
            return self.person(key)
        else:
            return getattr(self, key)

    def __setitem__(self, key, value):
        """ Ditto """
        return setattr(self, key, value)

    def __iter__(self):
        """ Iterate over people """
        for i in range(len(self)):
            yield self[i]

    def __setstate__(self, state):
        """
        Set the state upon unpickling/deepcopying

        If a People instance is copied (by any mechanism) then the keys in the `_states`
        registry will no longer match the memory addresses of the new copied states. Therefore,
        after copying, we need to re-create the states registry with the new object IDs
        """
        state['_states'] =  {id(v):v for v in state['_states'].values()}
        self.__dict__ = state
        return

    def filter(self, criteria=None, uids=None, split=False):
        """
        Store indices to allow for easy filtering of the People object.

        Args:
            criteria (bool array): a boolean array for the filtering critria
            uids (array): alternatively, explicitly filter by these indices
            split (bool): if True, return separate People objects matching both True and False

        Returns:
            A filtered People object, which works just like a normal People object
            except only operates on a subset of indices.
        """
        filt = Filter(self)
        return filt.filter(criteria=criteria, uids=uids, split=split)

    def scale_flows(self, inds):
        """
        Return the scaled versions of the flows -- replacement for len(inds)
        followed by scale factor multiplication
        """
        return self.scale[inds].sum()

    def update_post(self):
        """ Final updates at the very end of the timestep """
        sim = self.sim
        if sim.pars.use_aging:
            self.age[self.alive.uids] += sim.t.dt_year
        return

    def request_death(self, uids):
        """
        External-facing function to request an agent die at the current timestep

        In general, users should not directly interact with `People.ti_dead` to minimize
        interactions between modules (e.g., if a module requesting a future death, overwrites
        death due to a different module taking place at the current timestep).

        Modules that have a future time of death (e.g., due to disease duration) should keep
        track of that internally. When the module is ready to cause the agent to die, it should
        call this method, and can update its own results for the cause of death. This way, if
        multiple modules request death on the same day, they can each record a death due to their
        own cause.

        The actual deaths are resolved after modules have all run, but before analyzers. That way,
        regardless of whether removing dead agents is enabled or not, analyzers will be able to
        see and record outcomes for agents that died this timestep.

        **WARNING** - this function allows multiple modules to each independently carry out and
        record state changes associated with death. It is therefore important that they can
        guarantee that after requesting death, the death is guaranteed to occur.

        Args:
            uids: Agent IDs to request deaths for

        Returns:
            UIDs of agents that have been scheduled to die on this timestep
        """
        self.ti_dead[uids] = self.sim.ti
        return

    def step_die(self):
        """ Carry out any deaths that took place this timestep """
        death_uids = (self.ti_dead <= self.sim.ti).uids
        self.alive[death_uids] = False

        # Execute deaths that took place this timestep (i.e., changing the `alive` state of the agents). This is executed
        # before analyzers have run so that analyzers are able to inspect and record outcomes for agents that died this timestep
        for disease in self.sim.diseases():
            if isinstance(disease, ss.Disease):
                disease.step_die(death_uids)

        return death_uids

    def remove_dead(self):
        """
        Remove dead agents
        """
        uids = self.dead.uids
        if len(uids):

            # Remove the UIDs from the networks too
            for network in self.sim.networks.values():
                network.remove_uids(uids) # TODO: only run once every nth timestep

            # Calculate the indices to keep
            self.auids = self.auids[np.isin(self.auids, np.unique(uids), assume_unique=True, invert=True, kind='sort')]

        return

    def update_results(self):
        ti = self.sim.ti
        res = self.sim.results
        res.n_alive[ti] = np.count_nonzero(self.alive)
        res.new_deaths[ti] = np.count_nonzero(self.ti_dead == ti)
        res.cum_deaths[ti] = np.sum(res.new_deaths[:ti]) # TODO: inefficient to compute the cumulative sum on every timestep!
        return

    def finish_step(self):
        self.remove_dead()
        self.update_post()
        return

    @property
    def dead(self):
        """ Dead boolean """
        return ~self.alive

    @property
    def male(self):
        """ Male boolean """
        return ~self.female

    def to_df(self):
        """ Export to dataframe """
        df = sc.dataframe(uid=self.uid, slot=self.slot, **self.states)
        return df

    def plot(self, key=None, alive=True, bins=50, hist_kw=None, max_plots=20, figsize=(16,12), **kwargs):
        """
        Plot all the properties of the People object

        Args:
            key (str/list): the state(s) to plot
            alive (bool): if false, use all agents in the simulation instead of active (alive) ones
            bins (int): the number of bins to use in `plt.hist()`
            hist_kw (dict): passed to `plt.hist()`
            max_plots (int): the maximum number of plots to create
            figsize (tuple): passed to `plt.figure()`; by default large since the plots are crowded
            **kwargs (dict): passed to `sc.getrowscols()`, including `fig_kw`
        """
        # Handle figsize
        if 'figsize' not in kwargs:
            kwargs['figsize'] = figsize

        # Decide what to plot
        if key is None:
            key = self.states.keys()
        key = sc.tolist(key)
        if max_plots:
            key = key[:max_plots]
        n_keys = len(key)

        # Create figure and plot
        fig,axs = sc.getrowscols(n_keys, make=True, **kwargs)
        for i,k in enumerate(key):
            ax = fig.axes[i]
            state = self.states[k]
            if alive:
                data = state.values
            else:
                data = state.raw

            # It's a float array, plot a histogram
            if isinstance(state, ss.FloatArr):
                data = sc.rmnans(data)
                if not len(data):
                    print(f'Not plotting "{k}" since all values are NaN')
                ax.hist(data, bins=bins, **sc.mergedicts(hist_kw))

            # It's a boolean array, plot the proportions
            elif isinstance(state, ss.BoolArr):
                true,false = data.sum(), (~data).sum()
                labels = ['False', 'True']
                ax.bar(labels, [false,true], color=['#E0D3DE', '#3C88A3'])

            # Give up
            else:
                warnmsg = f'Cannot understand state of {type(state)} for plotting; skipping'
                ss.warn(warnmsg)

            # Tidy up
            ax.set_title(f'{state.name} (mean={data.mean():n})')
            ax.set_ylabel('Count')
            sc.commaticks(ax)
            sc.boxoff(ax)

        plt.tight_layout()
        return ss.return_fig(fig)

    def plot_ages(self, bins=None, absolute=False, fig_kw=None):
        """
        Plot the age pyramid for males and females.

        Args:
            bins (list/int): optional list or int for binning age groups (default: 5-year bins up to max age)
            absolute (bool): whether to show absolute numbers or percentage of the population
            fig_kw (dict): passed to `plt.subplots()`

        **Example**:

            sim = ss.demo(plot=False)
            sim.people.plot_age()
        """
        # Preliminaries
        age = self.age
        female = self.female
        f_age = age[female]
        m_age = age[~female]
        max_age = age.max()

        # Define age bins
        if np.iterable(bins):
            width = None # Already defined, don't need
        if np.isscalar(bins):
            width = bins
        else:
            width = 5
        if width is not None:
            max_bin = (np.ceil(max_age / width) + 1) * width
            bins = np.arange(0, max_bin, width)
        y_pos = np.arange(len(bins)-1)

        # Histogram counts
        f_vals, _ = np.histogram(f_age, bins=bins)
        m_vals, _ = np.histogram(m_age, bins=bins)

        # Optionally convert to percentages
        if absolute:
            xlabel = 'Number of people'
        else:
            total = f_vals.sum() + f_vals.sum()
            f_vals = f_vals / total * 100
            m_vals = m_vals / total * 100
            xlabel = 'Percentage of population'

        # Plot
        fig,ax = plt.subplots(**sc.mergedicts(fig_kw))
        ax.barh(y_pos, -f_vals, color='salmon', label='Female')  # Negative = left
        ax.barh(y_pos,  m_vals, color='skyblue', label='Male')

        # Annotate
        y_labels = [f'{bins[i]:n}–{bins[i+1]-1:n}' if i < len(bins)-1 else f'{bins[i]:n}+' for i in range(len(bins)-1)]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Age group')
        ax.legend(loc='lower right')
        max_val = max(f_vals.max(), m_vals.max()) * 1.1
        ax.set_xlim([-max_val, max_val])
        ax.axvline(0, color='black', linewidth=0.8)

        # Tidy up
        sc.boxoff()
        plt.tight_layout()
        return ss.return_fig(fig)

    def person(self, ind):
        """
        Get all the properties for a single person.

        **Example**:

            sim = ss.Sim(diseases='sir', networks='random', n_agents=100).run()
            print(sim.people.person(5)) # The 5th agent in the simulation
        """
        person = Person()
        for key in ['uid', 'slot']:
            person[key] = self[key][ind]
        for key in self.states.keys():
            person[key] = self.states[key][ind]
        return person


class Person(sc.objdict):
    """
    A simple class to hold all attributes of a person

    **Example**:

        sim = ss.Sim(diseases='sir', networks='random', n_agents=100).run()
        print(sim.people.person(5)) # The 5th agent in the simulation
    """
    def to_df(self):
        """ Convert to a dataframe """
        df = sc.dataframe.from_dict(self, orient='index', columns=['value'])
        df.index.name = 'key'
        return df


class Filter(sc.prettyobj):
    """
    A filter on states; see `ss.People.filter()` for details.
    """
    def __init__(self, people, uids=None):
        self.people = people
        self._uids = uids if uids is not None else people.auids
        self.orig_uids = people.auids
        self.states = ss.ndict(type=ss.Arr)
        self.is_filtered = False
        self._key = None
        self._stale = False
        self._linked_modules = people._linked_modules

        # Copy states
        for key,state in self.people.states.items():
            new_state = object.__new__(state.__class__)
            new_state.__dict__ = state.__dict__.copy() # Shallow copy
            new_state.people = self
            self.states[key] = new_state

        # Set up links to modules
        for mod in self._linked_modules:
            setattr(self, mod, sc.dictobj())
            for key in getattr(self.people, mod).keys():
                getattr(self, mod)[key] = self.states[f'{mod}.{key}']
        return

    def __getitem__(self, key):
        return self.states[key]

    def __getattr__(self, key):
        return self.states[key]

    def __setitem__(self, key, value):
        self.states[key] = value
        return

    def __call__(self, key):
        self._key = key
        self._stale = True
        return self

    @property
    def stale(self):
        return self._key is not None

    @property
    def auids(self):
        """ Rename for use as a "people" object """
        return self._uids

    @property
    def uids(self):
        """ Get the UIDs, computing if necessary """
        if self._stale:
            return self.filter(self._key, new=False)._uids
        else:
            return self._uids

    def _func(self, obj, op):
        if not self.stale:
            errormsg = "To use logical operations on a Filter object, call first, e.g. filt('age') > 5"
            raise RuntimeError(errormsg)
        else:
            arr = self.states[self._key]
            self._key = None
            self._stale = False

        match op:
            case '==': criteria = arr == obj
            case '!=': criteria = arr != obj
            case '<':  criteria = arr <  obj
            case '<=': criteria = arr <= obj
            case '>':  criteria = arr >  obj
            case '>=': criteria = arr >= obj
            case '~': criteria = ~arr
            case _: raise ValueError(f'Unsupported operator: {op}')

        return self.filter(criteria=criteria)

    def __eq__(self, obj): return self._func(obj, '==')
    def __ne__(self, obj): return self._func(obj, '!=')
    def __lt__(self, obj): return self._func(obj, '<')
    def __le__(self, obj): return self._func(obj, '<=')
    def __gt__(self, obj): return self._func(obj, '>')
    def __ge__(self, obj): return self._func(obj, '>=')
    def __invert__(self): return self._func(None, '~')

    def filter(self, criteria=None, uids=None, split=False, new=None):
        """
        Store indices to allow for easy filtering of the People object.

        Args:
            criteria (bool array): a boolean array for the filtering critria
            uids (array): alternatively, explicitly filter by these indices
            split (bool): if True, return separate filter objects matching both True and False

        **Example**:

            sim = ss.Sim(n_agents=100e3, dur=10, networks='random', diseases='sir', verbose=0)
            sim.run()
            ppl = sim.people

            # Traditional filtering
            f1 = ppl.female == True
            f2 = f1 * (ppl.age>5)
            f3 = f2 * (~ppl.sir.infected)

            # Equivalent using filter
            f1 = ppl.filter('female')
            f2 = f1('age')>5
            f3 = ~f2('sir.infected')
        """
        if new is True:
            filtered = Filter(self)
        elif new is False:
            filtered = self
        else:
            filtered = Filter(self) if (self.is_filtered and not split) else self

        # Perform the filtering
        if uids is not None: # Unless indices are supplied directly, in which case use them
            new_uids = np.intersect1d(self._uids, uids)
            if split:
                inv_uids = np.setdiff1d(self._uids, uids)
                f1 = Filter(self, uids=new_uids)
                f2 = Filter(self, uids=inv_uids)
                return f1, f2
            else:
                filtered._uids = new_uids

        if criteria is not None: # Main use case: perform filtering
            if isinstance(criteria, str): # Allow e.g. filter('female')
                if criteria[0] == '~': # Allow e.g. filter('~female') # TODO: may not be working
                    key = criteria[1:]
                    criteria = ~self.states[key]
                else:
                    self._key = criteria
                    criteria = self.states[criteria]
            len_criteria = len(criteria)
            if len_criteria == len(filtered._uids): # Main use case: a new filter applied on an already filtered object, e.g. filtered.filter(filtered.age > 5)
                new_uids = filtered._uids[criteria] # Criteria is already filtered, just get the indices
                if split:
                    inv_uids = filtered._uids[~criteria]
                    f1 = Filter(self, uids=new_uids)
                    f2 = Filter(self, uids=inv_uids)
                    return f1, f2
            else:
                errormsg = f'"criteria" must be boolean array matching either current filter length ({len(self)}) or else the total number of people ({self.n_uids}), not {len(criteria)}'
                raise ValueError(errormsg)
            filtered._uids = new_uids

        filtered.is_filtered = True

        return filtered