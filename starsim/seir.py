import numpy as np
import starsim as ss

__all__ = ['SEIR_AMS']

class SEIR_AMS(ss.Infection):
    """
    SEIR model with Asymtomatic/Mild/Severe infections

    This class implements a basic SEIR model with states for susceptible, exposed,
    infected/infectious, and recovered. Within infectious, a person may have asymtomatic,
    mild or severe disease, with a different probability of death based on severity of disease.

    Args:
        beta (float/`ss.prob`): the infectiousness
        init_prev (float/s`s.bernoulli`): the fraction of people to start of being infected
        dur_exp (float/`ss.dur`/`ss.Dist`): how long (in years) people are exposed/incubating for
        dur_inf (float/`ss.dur`/`ss.Dist`): how long (in years) people are infected for
        p_symp (`ss.choice`): probability of a person developing symptoms of a particular severity
        p_death_mild (float/`ss.bernoulli`): the probability of death from mild disease
        p_death_severe (float/`ss.bernoulli`): the probability of death from severe disease

    """
    def __init__(self, pars=None, beta=None, init_prev=None, dur_inf=None, p_death=None, **kwargs):
        super().__init__()
        self.define_pars(
            init_prev = ss.bernoulli(p=0.01),
            beta = ss.perday(0.0907*24),
            dur_exp = ss.lognorm_ex(mean=ss.days(10/24), std=ss.days(0.2)),
            dur_inf = ss.lognorm_ex(mean=ss.days(77/24), std=ss.days(0.5)),
            p_symp = ss.choice(a=3, p=[0.30, 0.42, 0.28]),  # 0=asymptomatic, 1=mild, 2=severe
            p_death_mild = ss.bernoulli(p=0.25),
            p_death_severe = ss.bernoulli(p=0.70),
        )
        self.update_pars(pars, **kwargs)

        # Example of defining all states, redefining those from ss.Infection, using overwrite=True
        self.define_states(
            ss.BoolState('susceptible', default=True, label='Susceptible'),
            ss.BoolState('exposed', label='Exposed'),
            ss.BoolState('infected', label='Infectious'),
            ss.FloatArr('symptom_cat', label='Symptom category'),
            ss.BoolState('recovered', label='Recovered'),
            ss.FloatArr('ti_exposed', label='TIme of exposure'),
            ss.FloatArr('ti_infected', label='Time of infection'),
            ss.FloatArr('ti_recovered', label='Time of recovery'),
            ss.FloatArr('ti_dead', label='Time of death'),
            ss.FloatArr('rel_sus', default=1.0, label='Relative susceptibility'),
            ss.FloatArr('rel_trans', default=1.0, label='Relative transmission'),
            reset = True, # Remove any existing states (from super().define_states())
        )
        return

    def step_state(self):
        # Progress exposed -> infected
        sim = self.sim
        infected = self.exposed & (self.ti_infected <= sim.ti)
        self.exposed[infected] = False
        self.infected[infected] = True
        # Progress infectious -> recovered
        recovered = (self.infected & (self.ti_recovered <= sim.ti)).uids
        self.infected[recovered] = False
        self.recovered[recovered] = True
        # Trigger deaths
        deaths = (self.ti_dead <= sim.ti).uids
        if len(deaths):
            sim.people.request_death(deaths)
        return

    def set_prognoses(self, uids, sources=None):
        """ Set prognoses """
        super().set_prognoses(uids, sources)
        ti = self.t.ti
        self.susceptible[uids] = False
        self.exposed[uids] = True
        self.infected[uids] = False
        self.ti_exposed[uids] = ti
        p = self.pars
        # Sample duration of exposed/incubation
        dur_exp = p.dur_exp.rvs(uids)
        self.ti_infected[uids] = ti + dur_exp

        # Sample duration of infection and severity of symptoms
        dur_inf = p.dur_inf.rvs(uids)
        symp = p.p_symp.rvs(uids)
        self.symptom_cat[uids] = symp

        asym_uids = uids[symp == 0]
        mild_uids = uids[symp == 1]
        severe_uids = uids[symp == 2]

        # Determine who dies and who recovers and when
        mild_die   = p.p_death_mild.rvs(mild_uids)
        severe_die = p.p_death_severe.rvs(severe_uids)

        will_die = np.zeros(len(uids), dtype=bool)
        mild_mask = symp == 1
        severe_mask = symp == 2
        will_die[mild_mask]   = p.p_death_mild.rvs(uids[mild_mask])
        will_die[severe_mask] = p.p_death_severe.rvs(uids[severe_mask])

        self.ti_dead[uids[will_die]] = ti + dur_exp[will_die] + dur_inf[will_die] # Consider rand round, but not CRN safe
        self.ti_recovered[uids[~will_die]] = ti + dur_exp[~will_die] + dur_inf[~will_die]
        return

    def step_die(self, uids):
        """ Reset infected/recovered flags for dead agents """
        self.susceptible[uids] = False
        self.exposed[uids] = False
        self.infected[uids] = False
        self.recovered[uids] = False
        return
