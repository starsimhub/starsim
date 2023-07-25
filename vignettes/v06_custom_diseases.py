"""
Typical disease dynamics:
Case A.1:
A custom disease module based on a SEIRS model: A->B->C->D->A, where the duration of each state is predefined (attribute).
Interventions and individual behaviours can affect the duration parameters.

Case A.2:
A custom disease module based on a SEIRS model: A->B->C->D->A, where the duration of each state is an output, because the
transition between two states is given by transition probabilities (attribute).
Interventions and individual behaviours can affect the transition probability parameters.

Case A.3: (default structure of all modules).
A custom disease module based on a SEIRS model: A->B->C->D->A, where some transition between two states are predefined,
and other transitions are given by transition probabilities. For instance the transition from A to B is governed by
by a transition probability that depends on network interactions.

Case B:
A custom disease module based on a A.3 model but it's different for females and males:
female:  A->B->C->D->A
male:    A->B->C->F->A

Case C:
A custom disease module based on a nested state machine model, that is there are superstates and substates.
Superstate A:
    Substate B1: B1.1 -> B1.2 -> B1.3
    Substate B2: B2.1 -> B2.2 -> B2.3
Superstate C:
    Substate D1: D1.1 -> D1.2 -> D1.3
    Substate D2: D2.1 -> D2.2 -> D2.3
Where the transitions between substates are governed by predefined durations,
and transitions between superstates are determined by transition probabilities.

"""

# Import STIsimimport stisim as ss
import numpy as np
import stisim as ss
import sciris as sc



class CustomDisease(ss.Module):
    """
    Based on Case A.3:  A->B->C->D->A, the transition from A to B is governed by by a transition probability
    that depends on network interactions.

    Uses custom states to handle duration and transition prob distributions
    """
    def __init__(self, pars=None):
        super().__init__(pars)
        # Define disease states or stages of disease progression, and state trackers
        self.states = ss.omerge(ss.named_dict(
            ss.CustomState('susceptible', bool, True, next_state='infected'),
             self.states)   # Recovered, before becoming susceptible again

        # Initialise additional states to track events
        self.init_event_tracker(['infected'])

    def init_event_trackers(self, states):
        """
        Initialise EventTracker instances
        """
        event_trackers = sc.objdict()
        for state_name in states:
            event_tracker_name = f"ti_{self.states[state_name]}"
            event_trackers[event_tracker_name] = EventTracker(event_tracker_name, fill_value=np.nan)
        self.states = ss.omerge(event_trackers, self.states)

    def next_state_start(self, sim, n, source_state=None):
        """
        Draw len(uids) samples with durations of "source state, to determine the start of "next state".
        """
        ns_start = sim.ti + self.states[source_state].samples(n) / sim.pars.dt
        return ns_start

    def next_state_prob(self, sim, n, source_state=None):
        """
        Draw len(uids) samples with durations of "source state, to determine the start of "next state".
        """
        ns_prob = sim.ti + self.states[source_state].samples(n)
        return ns_prob


class DurationBasedDisease(CustomDisease):
    """
    Case A.1-ish:
    A custom disease module based on a SEIRS model: A->B->C->D->A, where the duration of each state is predefined (attribute).
    Transition from susceptibility to infected is governed by a transition probability that depends on network interaction.
    """
    def __init__(self, pars=None):
        super().__init__(pars)
        # Define disease states or stages of disease progression, and state trackers
        self.states = ss.omerge(ss.named_dict(
            ss.CustomState('susceptible', bool, True, next_state='infected'),
            ss.CStochState('infected',   bool, False, duration=dict(dist='lognormal', par1=0.5, par2=0.5), next_state='infectious'),     # infected
            ss.CStochState('infectious', bool, False, duration=dict(dist='lognormal', par1=0.8, par2=1.2), next_state='recovered'),      # infectious
            ss.CStochState('recovered',  bool, False, duration=dict(dist='lognormal', par1=0.05, par2=0.2), next_state='susceptible')),  # recovered
            self.states)   # Recovered, before becoming susceptible again

        # Initialise additional states to track events
        self.init_event_tracker(['infectious', 'recovered'])

    def set_prognoses(self, sim, uids):
        # Time at which another event will happen (ie, transition from source state to a destination state)
        #infectious_start = sim.ti + np.random.poisson(self.state_durations['infected'] / sim.pars.dt,
        #                                              len(uids[is_infectious]))

        # The assumption here is that the infected state is not infectious, but latent. Infected state has a duration
        # before becoming infectious, and assumes everyone who is infected will become infectious eventually.
        infectious_start = self.next_state_start(sim, len(uids), source_state='infected')

        # Keep track of when events happen
        sim.people[self.name].ti_infectious[uids] = infectious_start




class ProbabilityBasedDisease(ss.Module):
    """
    Case A.2-ish: A custom disease module, where we specify transition probability distributions.
    Transition from susceptibility to infected is governed by a transition probability that depends on network interaction.
    """

    def __init__(self, pars=None):
        super().__init__(pars)
        # Define disease states or stages of disease progression, and state trackers
        self.states = ss.omerge(ss.named_dict(
            ss.CustomState('susceptible', bool, True, next_state='infected'),
            ss.CStochState('infected', bool, False,   transition_prob=dict(dist='uniform', par1=0.0, par2=1.0), next_state='infectious'),  # transition_prob to infectious
            ss.CStochState('infectious', bool, False, transition_prob=dict(dist='normal', par1=0.8, par2=0.1), next_state='recovered'),    # transition_prob to recovered
            ss.CStochState('recovered', bool, False,  transition_prob=1.0, next_state='susceptible')),                                     # transition_prob to susceptible
            self.states)

        # Initialise additional states to track events
        self.init_event_tracker(['infectious', 'recovered'])

    def set_prognoses(self, sim, uids):
        """
        This function updates the disease states (and outcomes) for individuals
        who have newly contracted the disease.
        """
        # Transition from susceptible to infected
        sim.people[self.name].susceptible[uids] = False
        sim.people[self.name].infected[uids] = True
        sim.people[self.name].ti_infected[uids] = sim.ti

        # Proportion of those who are infected decide who will become infectious and who will become latent
        # If transition probabilty is specified as a single number in self.pars
        #is_infectious = np.random.random(len(uids)) < self.get_state_transition_prob('infected', 'infectious')

        # OR if transition probability is specified with custom states
        is_infectious = np.random.random(len(uids)) < self.next_state_prob(sim, len(uids), source_state='infected')

        # Keep track of time events
        sim.people[self.name].ti_infectious[uids[is_infectious]] = sim.ti

        # More prognoses logic here .... check who recovers, etc



# Custom states
class EventTracker(ss.State):
    """
    Convinience class for ti_state States
    """
    def __init__(self, name, dtype=ss.default_float, fill_value=np.nan, **kwargs):
        super().__init__(name, dtype, fill_value, kwargs)
        return


class CustomState(ss.State):
    """
    Custom sate with a hint to the "next state", not meaningfully used yet, just a label
    """
    def __init__(self, name, dtype, fill_value=0, next_state=None, **kwargs):
        super().__init__(name, dtype, fill_value, kwargs)
        self.next_state = next_state
        return

    @property
    def ndim(self):
        return len(sc.tolist(self.shape)) + 1

    def new(self, n):
        shape = sc.tolist(self.shape)
        shape.append(n)
        return np.full(shape, dtype=self.dtype, fill_value=self.fill_value)


class CStochState(ss.State):
    """
    A custom stoachastic state, that returns samples of duration or probabilities
    """
    def __init__(self, name, dtype, duration=None, transition_prob=None, **kwargs):
        super().__init__(name, dtype, **kwargs)
        self.distdict = duration or transition_prob
        return

    def samples(self, n):
        """
        Sample durations or transition probabilities from a distribution
        """
        shape = sc.tolist(self.shape)
        shape.append(n)
        if isinstance(self.distdict, (int, float)):
            return np.full(shape, dtype=self.dtype, fill_value=self.distdict)
        return ss.sample(**self.distdict, size=tuple(shape))
