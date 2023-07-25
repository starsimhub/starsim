"""
Anatomy of a custom disease module. Based on Chlamydia, that has different states for male and females.
Transition probabilities are made up. State durations are based on values found on the literature.

Notes:
    - Easy to make normal gonorrhea protective/exclusive of custom gonorrhea, but not the other way round?
"""

# Import STIsim
import stisim as ss
import numpy as np


# Step 0: A new disease is defined as a derived from STIsim's Module class.
#         A new disease has multiple attributes. Two two essential attributes are:
#         - disease states: possible progression states of this disease, including other conditions (ie, cancer, PID).
#                   Typical disease-specific states are susceptible, infected, recovered, though there could be more
#                   to represent severe stages of disease. The 'dead' state is not included in a disease module, as this
#                   is considered a state that is intrinsic to an individual, and can be reached independently of a disease.
#        - disease state event trackers: instants at which a state starts or event occurs
class Disease(ss.Module):
    """
    Durations of each disease state
    | State        | Duration (in years) | Description                                                                                                                             | References                                                                 |
    |--------------|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
    | Susceptible  | NA                  | Not applicable because duration depends on individual behaviors, physiology, and exposure risks                                         | Hatt & Stefan (2022)                                                                       |
    | Infected     | NA                  | Individuals who have acquired chlamydia infection., its duration depends on substates.   | (Brunham & Rey-Ladino, 2005; , Beale et al., 2019; , Redmond et al., 2015; , Geisler, 2010)                                                     |
    | Infectious   | NA                  | Individuals who can transmit chlamydia infection., its duration depends on substates.   | (Brunham & Rey-Ladino, 2005; , Beale et al., 2019; , Redmond et al., 2015; , Geisler, 2010)                                                     |
    | Latent       | 0.5                 | Individuals who have but cannot transmit infection.   | (Brunham & Rey-Ladino, 2005; , Beale et al., 2019; , Redmond et al., 2015; , Geisler, 2010)                                                     |
    | Asymptomatic | 0.15 to 0.82        | Infected individuals who do not exhibit any symptoms but can still transmit the infection. | (Brunham & Rey-Ladino, 2005; , Redmond et al., 2015; , Geisler, 2010; , Templeton et al., 2008)                                                     |
    | Symptomatic  | 0.15 to 0.82        | Infected individuals who develop symptoms such as genital discharge, pain, or discomfort.  | (Brunham & Rey-Ladino, 2005; , Redmond et al., 2015; , Geisler, 2010; , Templeton et al., 2008)                                                     |
    | PID Mild     | 0.15 to 2.19        | Individuals with mild pelvic inflammatory disease (PID) as a complication of chlamydia infection.  | (Brunham & Rey-Ladino, 2005; , Redmond et al., 2015; , Geisler, 2010)                                                             |
    | Epididymitis | 0.08 to 0.15        | Infected males who develop epididymitis as a complication of chlamydia infection.  | (Brunham & Rey-Ladino, 2005; , Redmond et al., 2015; , Geisler (2010)                                                             |
    | Infertile    | 0.49 to inf         | Individuals who experience infertility as a consequence of chlamydia infection. The duration may be permanent in some cases | (Brunham & Rey-Ladino, 2005; , Redmond et al., 2015; , Geisler, 2010; , Chaban et al., 2018)                                                      |

    Add a state transition matrix

    """

    def __init__(self, pars=None):
        super().__init__(pars)
        # Define disease states or stages of disease progression, and state trackers
        self.states = ss.omerge(ss.named_dict(
            ss.State('susceptible', bool, True),
            ss.State('infected', bool, False),  # state infected
            ss.State('ti_infected', float, 0.0),  # state tracker
            ss.State('latent', bool, False),  # noninfectious state
            ss.State('ti_latent', float, 0.0),  # event tracker
            ss.State('infectious', bool, False),  # infectious
            ss.State('ti_infectious', float, 0.0),  # tracker
            ss.State('asymptomatic', bool, False),  # infectious state
            ss.State('ti_asymptomatic', float, 0.0),  # tracker
            ss.State('symptomatic', bool, False),  # infectious state
            ss.State('ti_symptomatic', float, 0.0),  # tracker
            # Female progression states
            ss.State('pid', bool, False),  # infectious, pelvic inflammatory disease
            ss.State('ti_pid', float, 0.0),  # tracker
            # Male progression states
            ss.State('epididymitis', bool, False),  # infectious,
            ss.State('ti_epididymitis', bool, False),  # tracker
            # Both sexes can progress to infertility
            ss.State('infertile', bool, False),  # noninfectious, state
            ss.State('ti_infertile', bool, False),  # noninfectious, tracker

        ), self.states)
        # Step 1: We need to specify additional parameters to make the new disease module work
        #        - parameters: disease-specific parameters that govern progression and propagation dynamics
        #                       - time-related parameters:
        #                                 - duration of each state
        #                                 - intervals (defined as two instants, whose difference defines a duration)
        #                                 - Some states do not have a duration specified upfront, because their duration
        #                                   is the sum of the durations of their substates (ie, infectious), or their
        #                                   duration depends on human behaviour and other risk factors
        #                                   (ie, the susceptible state).
        self.pars = ss.omerge({
            'duration': {'susceptible': None, 'infected': None, 'infectious': None,
                         'latent': 0.15, 'asymptotmatic': 0.82, 'symptomatic': 0.82,
                         'pid': 0.3, 'epididymitis': 0.15,
                         'infertile': 10.0},
            # This line is a placeholder that could be used if we hadn't network interactions
            'from_susceptible': {'susceptible': 0.0, 'infected': 0.0, 'infectious': 0.0,
                                 'latent': 0.0, 'asymptotmatic': 0.0, 'symptomatic': 0.0,
                                 'pid': 0.0, 'epididymitis': 0.0,
                                 'infertile': 0.0, 'dead': 0.0},
            # Probability of transitioning immediately from infected to infectious
            'from_infected': {'susceptible': 0.0, 'infected': 0.0, 'infectious': 0.7,
                              'latent': 0.3, 'asymptotmatic': 0.0, 'symptomatic': 0.0,
                              'pid': 0.0, 'epididymitis': 0.0,
                              'infertile': 0.0, 'dead': 0.0},
            # Probability of becoming either asymptomatic or symptomatic (infectious)
            'from_infectious': {'susceptible': 0.0, 'infected': 0.0, 'infectious': 0.0,
                                'latent': 0.0, 'asymptotmatic': 0.15, 'symptomatic': 0.85,
                                'pid': 0.0, 'epididymitis': 0.0,
                                'infertile': 0.0, 'dead': 0.0},
            # Probability of becoming infectious from latent state is 100%, but duration of latency varies
            'from_latent': {'susceptible': 0.0, 'infected': 0.0, 'infectious': 1.0,
                            'latent': 0.0, 'asymptotmatic': 0.0, 'symptomatic': 0.0,
                            'pid': 0.0, 'epididymitis': 0.0,
                            'infertile': 0.0, 'dead': 0.0},
            # Probability of recovering (back to susceptible) or progressing onto a chronic and severe state
            'from_asymptomatic': {'susceptible': 0.8, 'infected': 0.0, 'infectious': 0.0,
                                  'latent': 0.0, 'asymptotmatic': 0.0, 'symptomatic': 0.0,
                                  'pid': 0.2, 'epididymitis': 0.0,
                                  'infertile': 0.0, 'dead': 0.0},
            # These probabilities should be different for male and females
            'from_symptomatic': {'susceptible': 0.2, 'infected': 0.0, 'infectious': 0.0,
                                 'latent': 0.0, 'asymptotmatic': 0.0, 'symptomatic': 0.0,
                                 'pid': 0.8, 'epididymitis': 0.8,
                                 'infertile': 0.0, 'dead': 0.0},
            # Only applicable to females
            'from_pid': {'susceptible': 0.6, 'infected': 0.0, 'infectious': 0.0,
                         'latent': 0.0, 'asymptotmatic': 0.0, 'symptomatic': 0.0,
                         'pid': 0.0, 'epididymitis': 0.0,
                         'infertile': 0.3, 'dead': 0.1},
            # Only applicable to males
            'from_epididymitis': {'susceptible': 0.7, 'infected': 0.0, 'infectious': 0.0,
                                  'latent': 0.0, 'asymptotmatic': 0.0, 'symptomatic': 0.0,
                                  'pid': 0.0, 'epididymitis': 0.0,
                                  'infertile': 0.2, 'dead': 0.1},
        }, self.pars)
        # NB: I'm adding a new attribute just for convienience in case state durations become more complex, and need validation
        self.state_durations = self.set_state_durations()
        self.state_transition_prob_matrix = self.set_state_transition_matrix()
        return

    def state_coexistance_matrix(self):
        """
         This is essentially a transition matrix, except that two states can coexist because
         a disease can be a nested state machine (ie, there are superstates like infectious that have substates)
         """
        # NB: this method would be useful for validation, so there are not agents that have two mutually-exclusive
        # states
        pass

    def set_state_duration(self):
        """
        Define the average durations of different disease states.
        """
        state_durations = {}
        for state_name in self.states:
            if state_name in self.pars['duration'].items():
                state_durations[state_name] = self.pars['duration'][state_name]
            else:
                raise ValueError(f"State duration not provided for state: {state_name}")
            return state_durations

    def set_state_transition_matrix(self):
        """
        Set the state transition probabilities for source states
        """
        state_transition_matrix = {}
        for state_name in self.states:
            from_source_state = f"from_{state_name}"
            try:
                state_transition_matrix[state_name] = self.pars[from_source_state]
            except:
                raise ValueError(f"Transition probabilities from state {state_name} to other states are not defined.")
            return state_transition_matrix

    def get_state_transition_prob(self, source_state, dest_state):
        """
        Get the transition probability from a source state to a destination state.

        Args:
            source_state (str): The name of the source state.
            dest_state (str): The name of the destination state.

        Returns:
           float: The transition probability from the source state to the destination state. If no transition
               probability is defined between the source and destination states, the method returns 0.0.
        """
        transition_probs = self.state_transition_prob_matrix[source_state]
        return transition_probs.get(dest_state, 0.0)

    def update(self, sim):
        """
        Disease dynamics
        """
        self.progress(sim)
        self.propagate(sim)
        self.update_results(sim)

    def progress(self, sim):
        """
        Define the rules of evolution of disease progression.
        """


        # People who are latent will all eventually become infectious
        self.latent_to_infectious(sim)
        # Those who become infectious, can develop symptoms
        newly_infectious = sim.people[self.name].ti_infectious == sim.ti
        self.develop_symptoms(sim, newly_infectious)

        self.set_sequela(sim)
        return

    def init_results(self, sim):
        """
        Initialise Results objects to keep track of how this disease evolves
        """
        # intialise basic results
        ss.Module.init_results(self, sim)
        # intialise results specific to Chlamydia
        sim.results[self.name]['n_latent'] = ss.Result('n_latent', self.name, sim.npts, dtype=int)
        sim.results[self.name]['n_asymptomatic'] = ss.Result('n_asymptomatic', self.name, sim.npts, dtype=int)
        sim.results[self.name]['n_symptomatic'] = ss.Result('n_symptomatic', self.name, sim.npts, dtype=int)

    def update_results(self, sim):
        """
        Define what we measures/outputs we are are tracking as results
        """
        # The essential ones like number of infected, etc
        super(Disease, self).update_results(sim)
        # More specific results
        sim.results[self.name]['n_latent'] = np.count_nonzero(sim.people.alive & sim.people[self.name].latent)
        sim.results[self.name]['n_asymptomatic'] = np.count_nonzero(
            sim.people.alive & sim.people[self.name].asymptomatic)
        sim.results[self.name]['n_symptomatic'] = np.count_nonzero(sim.people.alive & sim.people[self.name].symptomatic)

    def propagate(self, sim):
        """
        Propagate this disease. This method implements the rules of through transmission,
        incidence which result in new cases.
        """
        pars = sim.pars[self.name]
        for k, network in sim.people.networks.items():
            if k in pars['beta']:
                rel_trans = (sim.people[self.name].infected & ~sim.people.dead).astype(float)
                rel_sus = (sim.people[self.name].susceptible & ~sim.people.dead).astype(float)
                for a, b, beta in [[network['p1'], network['p2'], pars['beta'][k][0]],
                                   [network['p2'], network['p1'], pars['beta'][k][1]]]:
                    # probability of a->b transmission
                    p_transmit = rel_trans[a] * rel_sus[b] * network['beta'] * \
                                 beta * self.get_state_transition_prob('susceptible', 'infected')
                    newly_infected = np.random.random(len(a)) < p_transmit
                    if newly_infected.any():
                        self.set_prognoses(sim, b[newly_infected])

    def set_deaths(self, uids):
        """
        Determine the individuals that should die based on a fixed state transition probability
        from a source state to dead state.
        """
        dead = np.zeros(len(uids), dtype=bool)
        # NB: The order is a user-specific choice and may bias the results.
        states_to_dead = ['symptomatic', 'pid_mild', 'epididymitis']
        already_dead = np.zeros(len(uids), dtype=bool)  # Mask to keep track of individuals already marked as dead

        for source_state in states_to_dead:
            # Get the transition probability to the 'dead' state from the current source state
            transition_prob = self.get_state_transition_prob(source_state, 'dead')

            # Generate a random array of the same length as nondead uids and compare it with the transition probability
            # Only consider individuals who are not already marked as dead.
            not_dead_indices = np.where(~already_dead)[0]
            dead_from_state = np.zeros(len(not_dead_indices), dtype=bool)
            dead_from_state |= np.random.random(len(not_dead_indices)) < transition_prob[not_dead_indices]

            # Update the 'dead' array and the 'already_dead' mask
            dead[dead_from_state] = True
            already_dead[dead_from_state] = True
        return dead

    def set_prognoses(self, sim, uids):
        """
        This function updates the disease states (and outcomes) for individuals
        who have newly contracted the disease. It also sets the times of at which a new state
        starts (`ti_recovered` or `ti_dead`) based on disease-specific parameters.

        Args:
            sim (stisim.Sim): The STIsim simulation instance.
            uids (numpy.ndarray): An array of unique identifiers of individuals who have newly contracted the disease.
        """
        # Transition from susceptible to infected
        sim.people[self.name].susceptible[uids] = False
        sim.people[self.name].infected[uids] = True
        sim.people[self.name].ti_infected[uids] = sim.ti

        # Proportion of those who are infected decide who will become infectious and who will become latent
        to_infectious = np.random.random(len(uids)) < self.get_state_transition_prob('infected', 'infectious')
        # Infectious
        sim.people[self.name].infectious[uids[to_infectious]] = True
        sim.people[self.name].ti_infectious[uids[to_infectious]] = sim.ti
        # Noninfectious, Latent
        sim.people[self.name].latent[uids[~to_infectious]] = True
        sim.people[self.name].ti_latent[uids[~to_infectious]] = sim.ti

        # Those who are infectious can be either asymptomatic or symptomatic
        self.develop_symptoms(sim, uids[to_infectious])
        self.develop_severe_disease()
        self.death()
        self.

        # A proportion of people will get pid or epididymitis, decide who and when
        ti_pid = self.transition_to_next_state(sim, 'symptomatic', to_pid)
        sim.people[self.name].ti_pid[] = ti_pid

        ti_pid = self.transition_to_next_state(sim, 'asymptomatic', to_pid)
        sim.people[self.name].ti_pid[] = ti_pid

        ti_pid = self.transition_to_next_state(sim, 'symptomatic', to_epi)
        sim.people[self.name].ti_pid[] = ti_pid

        ti_pid = self.transition_to_next_state(sim, 'asymptomatic', to_epi)
        sim.people[self.name].ti_pid[] = ti_pid




    def transition_to_next_state(sim, source_state, uids):
        next_event = sim.ti + np.random.poisson(self.state_durations[source_state] / sim.pars.dt, len(uids))
        return next_event


        # Determine which individuals will die after a period of time
        dead = self.set_deaths(uids)

        # Time at which another event will happen (ie, transition from source state to a destination state)
        next_event = sim.ti + np.random.poisson(self.state_durations['infected'] / sim.pars.dt, len(uids))

        # Keep track of when events happen
        sim.people[self.name].ti_recovered[uids[~dead]] = next_event[~dead]
        sim.people[self.name].ti_dead[uids[dead]] = next_event[dead]

    def develop_symptoms(self, sim, uids):

        # Those who are infectious can be either asymptomatic or symptomatic
        to_symptomatic = np.random.random(len(uids)) < self.get_state_transition_prob('infectious', 'symptomatic')

        # Infectious, symptomatic
        sim.people[self.name].symptomatic[uids[to_symptomatic]] = True
        sim.people[self.name].ti_symptomatic[uids[to_symptomatic]] = sim.ti
        # Infectious, asymptomatic
        sim.people[self.name].asymptomatic[uids[~to_symptomatic]] = True
        sim.people[self.name].ti_symptomatic[uids[~to_symptomatic]] = sim.ti


    def latent_to_infectious(self, sim):
        # People who are latent will all eventually become infectious
        uids   = sim.people.alive & sim.people[self.name].latent & ((sim.ti - sim.people[self.name].ti_latent) > self.state_durations['latent'])  # years
        sim.people[self.name].infectious[uids] = True
        sim.people[self.name].ti_infectious[uids] = sim.ti


    def set_sequela(self, sim):
        """
        Modify people's attributes that have been permanently altered by the disease
        """
        male_uids   = sim.people.alive  & sim.people[self.name].epididymitis & ((sim.ti - sim.people[self.name].ti_epididymitis) > self.state_durations['epididymitis'])  # years
        female_uids = sim.people.alive  & sim.people[self.name].pid & ((sim.ti - sim.people[self.name].ti_pid) > self.state_durations['pid'])  # years
        sim.people.infertile[male_uids] = True
        sim.people.infertile[female_uids] = True
