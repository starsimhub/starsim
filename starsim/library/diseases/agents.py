"""
Diseases written as code: Starsim 3's SIR, SEIR, and SIS

These are the Starsim 3 library models, written as `ss.Infection` subclasses with
per-agent states and methods. They are examples of writing a disease as code, and
base classes for the library diseases that add per-agent behavior. The core `ss.SIR`,
`ss.SIS`, and `ss.SEIR` are declared from states and transitions instead. Written as
code, these run only as agents, hence their names (`ssl.AgentSIR` and the others).
"""
import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

__all__ = ['AgentSIR', 'AgentSEIR', 'AgentSIS']

_ = None # For function signatures


class AgentSIR(ss.Infection):
    """
    An SIR model written as code: Starsim 3's `ss.SIR`

    This class implements a basic SIR model with states for susceptible,
    infected/infectious, and recovered, as per-agent arrays updated in code (see
    `ss.Infection`). It also includes deaths, and basic results. It is an example of
    writing a disease as code, and a base class for diseases with per-agent behavior
    (e.g. `ssl.Ebola`); for an SIR model, use the declared `ss.SIR`, which runs with
    every solver. A Starsim 3 subclass of `ss.SIR` that overrides its methods can
    subclass this class instead, unchanged.

    Its defaults are Starsim 3's: a lognormal `dur_inf` (a mean of 6 years, std 1 year)
    and `p_death=0.01`, where the declared `ss.SIR` is the textbook SIR, with an
    exponential `dur_inf` and no deaths. Use it to keep a Starsim 3 model as it was.

    Args:
        beta (float/`ss.prob`): the infectiousness
        init_prev (float/s`s.bernoulli`): the fraction of people to start of being infected
        dur_inf (float/`ss.dur`/`ss.Dist`): how long (in years) people are infected for
        p_death (float/`ss.bernoulli`): the probability of death from infection
    """
    plot_states = ['n_susceptible', 'n_infected', 'n_recovered'] # Which results to show in plot()

    def __init__(self, pars=None, beta=_, init_prev=_, dur_inf=_, p_death=_, **kwargs):
        super().__init__()
        self.define_pars(
            beta = ss.peryear(0.1),
            init_prev = ss.bernoulli(p=0.01),
            dur_inf = ss.lognorm_ex(mean=ss.years(6)),
            p_death = ss.bernoulli(p=0.01),
        )
        self.update_pars(pars, **kwargs)

        # Example of defining all states, redefining those from ss.Infection, using overwrite=True
        self.define_states(
            ss.BoolState('susceptible', default=True, label='Susceptible'),
            ss.BoolState('infected', label='Infected'),
            ss.BoolState('recovered', label='Recovered'),
            ss.FloatArr('ti_infected', label='Time of infection'),
            ss.FloatArr('ti_recovered', label='Time of recovery'),
            ss.FloatArr('ti_dead', label='Time of death'),
            ss.FloatArr('rel_sus', default=1.0, label='Relative susceptibility'),
            ss.FloatArr('rel_trans', default=1.0, label='Relative transmission'),
            reset = True, # Remove any existing states and aliases (from super().define_states())
            infectious = 'infected', # Everyone infected is infectious; AgentSEIR replaces this with a state of its own
        )
        return

    def step_state(self):
        # Progress infectious -> recovered
        sim = self.sim
        recovered = (self.infected & (self.ti_recovered <= self.ti)).uids
        self.clear_infection(recovered)
        self.recovered[recovered] = True

        # Trigger deaths
        deaths = (self.ti_dead <= self.ti).uids
        if len(deaths):
            sim.people.request_death(deaths, cause=self)
        return

    def set_prognoses(self, uids, sources=None):
        """ Set prognoses: infect the agents, then schedule what happens to them """
        super().set_prognoses(uids, sources)
        self.set_infection(uids)
        self.set_progression(uids)
        return

    def set_infection(self, uids):
        """ Make the agents infectious, immediately (`AgentSEIR` overrides this to add a latent period) """
        self.infected[uids] = True
        self.ti_infected[uids] = self.ti_entry # At the end of the step, as a declared disease dates it
        return

    def clear_infection(self, uids):
        """ Clear the infection states, on recovery or death (`AgentSEIR` overrides this, since its `infected` is derived) """
        self.infected[uids] = False
        return

    def set_progression(self, uids):
        """ Schedule recovery or death, relative to when each agent becomes infectious """
        p = self.pars

        # Sample duration of infection, being careful to only sample from the
        # distribution once per timestep.
        dur_inf = p.dur_inf.rvs(uids)

        # Determine who dies and who recovers and when. In AgentSIR agents are infectious as
        # soon as they are infected, so the infectious period runs from ti_infected;
        # AgentSEIR overrides this to run it from ti_infectious instead.
        will_die = p.p_death.rvs(uids)
        dead_uids = uids[will_die]
        rec_uids = uids[~will_die]
        self.ti_dead[dead_uids] = self.ti_infected[dead_uids] + dur_inf[will_die] # Consider rand round, but not CRN safe
        self.ti_recovered[rec_uids] = self.ti_infected[rec_uids] + dur_inf[~will_die]
        return

    def step_die(self, uids):
        """ Reset infected/recovered flags for dead agents """
        self.susceptible[uids] = False
        self.clear_infection(uids)
        self.recovered[uids] = False
        return

    def plot(self, **kwargs):
        """ Default plot for SIR model """
        fig = plt.figure()
        kw = sc.mergedicts(dict(lw=2, alpha=0.8), kwargs)
        res = self.results
        for rkey in self.plot_states:
            plt.plot(res.timevec, res[rkey], label=res[rkey].label, **kw)
        plt.legend(frameon=False)
        plt.xlabel('Time')
        plt.ylabel('Number of people')
        plt.ylim(bottom=0)
        sc.boxoff()
        sc.commaticks()
        return ss.return_fig(fig)


class AgentSEIR(AgentSIR):
    """
    An SEIR model written as code: Starsim 3's `ss.SEIR` (see `ssl.AgentSIR`), with its lognormal `dur_exp` and `dur_inf` and `p_death=0.01`; for an SEIR model, use `ss.SEIR`

    This class extends `AgentSIR` with an exposed state: agents who are infected but not
    yet infectious. `exposed` and `infectious` are the literal E and I compartments, and
    `infected` is derived from them: an agent is infected from the moment it acquires the
    infection, whether or not it is yet transmitting. So `n_infected` and `prevalence`
    count E plus I, while transmission depends on `infectious` alone.

    Correspondingly, `ti_exposed` is the time of acquisition and `ti_infectious` is the
    time of becoming infectious. `ti_infected` is deliberately not defined, since it is
    too easily confused with `ti_infectious`.

    Args:
        beta (float/`ss.prob`): the infectiousness
        init_prev (float/`ss.bernoulli`): the fraction of people to start off being infected
        dur_exp (float/`ss.dur`/`ss.Dist`): how long people are exposed (latent) for
        dur_inf (float/`ss.dur`/`ss.Dist`): how long people are infectious for
        p_death (float/`ss.bernoulli`): the probability of death from infection
    """
    plot_states = ['n_susceptible', 'n_exposed', 'n_infectious', 'n_recovered']

    def __init__(self, pars=None, dur_exp=_, **kwargs):
        super().__init__()
        self.define_pars(
            dur_exp = ss.lognorm_ex(mean=ss.years(0.5)),
        )
        self.update_pars(pars, **kwargs)

        # SIR states are added automatically; here we split I off from "infected", which
        # becomes E plus I, and drop ti_infected in favor of ti_exposed and ti_infectious
        self.define_states(
            ss.BoolState('exposed', label='Exposed'),
            ss.BoolState('infectious', label='Infectious'),
            ss.FloatArr('ti_exposed', label='Time of exposure'),
            ss.FloatArr('ti_infectious', label='Time of becoming infectious'),
            reset = ['infected', 'infectious', 'ti_infected'],
            infected = lambda self: self.exposed | self.infectious, # Derived: E and I are both infected
        )
        return

    def set_infection(self, uids):
        """ Agents are exposed first, and become infectious after the latent period """
        self.exposed[uids] = True
        self.ti_exposed[uids] = self.ti_entry # At the end of the step, as a declared disease dates it
        self.ti_infectious[uids] = self.ti_exposed[uids] + self.pars.dur_exp.rvs(uids)
        return

    def set_progression(self, uids):
        """ Schedule recovery or death, relative to the end of the latent period """
        p = self.pars
        dur_inf = p.dur_inf.rvs(uids)

        # As AgentSIR, except that the infectious period runs from ti_infectious rather than
        # from the time of acquisition, so the latent period delays it rather than eating into it
        will_die = p.p_death.rvs(uids)
        dead_uids = uids[will_die]
        rec_uids = uids[~will_die]
        self.ti_dead[dead_uids] = self.ti_infectious[dead_uids] + dur_inf[will_die]
        self.ti_recovered[rec_uids] = self.ti_infectious[rec_uids] + dur_inf[~will_die]
        return

    def clear_infection(self, uids):
        """ `infected` is derived, so clear the states it is derived from """
        self.exposed[uids] = False
        self.infectious[uids] = False
        return

    def step_state(self):
        """ Progress exposed -> infectious, then the usual SIR transitions """
        infectious = (self.exposed & (self.ti_infectious <= self.ti)).uids
        self.exposed[infectious] = False
        self.infectious[infectious] = True
        super().step_state()
        return


class AgentSIS(ss.Infection):
    """
    An SIS model with waning immunity, written as code: Starsim 3's `ss.SIS` (see `ssl.AgentSIR`), with its lognormal `dur_inf`

    This class implements a basic SIS model with states for susceptible,
    infected/infectious, and back to susceptible, with a per-agent immunity level
    that is boosted by each infection and wanes exponentially, which sets each agent's
    `rel_sus`. There is no death in this case. For an SIS model without immunity,
    use the declared `ss.SIS`, which runs with every solver.

    Args:
        beta (float/`ss.prob`): the infectiousness
        init_prev (float/`ss.bernoulli`): the fraction of people to start of being infected
        dur_inf (float/`ss.dur`/`ss.Dist`): how long (in years) people are infected for
        waning (float/`ss.rate`): how quickly immunity wanes
        imm_boost (float): how much an infection boosts immunity
    """
    def __init__(self, pars=None, beta=_, init_prev=_, dur_inf=_, waning=_, imm_boost=_, **kwargs):
        super().__init__()
        self.define_pars(
            beta = ss.peryear(0.05),
            init_prev = ss.bernoulli(p=0.01),
            dur_inf = ss.lognorm_ex(mean=ss.years(10)),
            waning = ss.peryear(0.05),
            imm_boost = 1.0,
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.FloatArr('ti_recovered'),
            ss.FloatArr('immunity', default=0.0),
        )
        return

    def step_state(self):
        """ Progress infectious -> recovered """
        recovered = (self.infected & (self.ti_recovered <= self.ti)).uids
        self.infected[recovered] = False
        self.susceptible[recovered] = True
        self.update_immunity()
        return

    def make_naive(self, uids, skip_states=None):
        """ As `ss.Disease.make_naive()`, but also resets `rel_sus` by default, since SIS sets it from its own immunity """
        if skip_states is None:
            skip_states = ['rel_trans']
        return super().make_naive(uids, skip_states=skip_states)

    def update_immunity(self):
        """ Apply exponential waning to immunity and update relative susceptibility. """
        waning = self.pars.waning.to_prob() # Exponential waning (NB: the exponential conversion is calculated automatically by the timepar)
        has_imm = (self.immunity > 0).uids
        self.immunity[has_imm] *= (1-waning)
        self.rel_sus[has_imm] = np.maximum(0, 1 - self.immunity[has_imm])
        return

    def set_prognoses(self, uids, sources=None):
        """ Set prognoses """
        super().set_prognoses(uids, sources) # Also makes the agents non-susceptible
        self.infected[uids] = True
        self.ti_infected[uids] = self.ti_entry # At the end of the step, as a declared disease dates it
        self.immunity[uids] += self.pars.imm_boost

        # Sample duration of infection
        dur_inf = self.pars.dur_inf.rvs(uids)

        # Determine when people recover
        self.ti_recovered[uids] = self.ti_infected[uids] + dur_inf

        return

    def init_results(self):
        """ Initialize results """
        super().init_results()
        self.define_results(
            ss.Result('rel_sus', dtype=float, scale=False, label='Relative susceptibility')
        )
        return

    @ss.required()
    def update_results(self):
        """ Store the population immunity (susceptibility) """
        super().update_results()
        self.results['rel_sus'][self.ti] = self.rel_sus.mean()
        return

    def plot(self, **kwargs):
        """ Default plot for SIS model """
        fig = plt.figure()
        kw = sc.mergedicts(dict(lw=2, alpha=0.8), kwargs)
        res = self.results
        for rkey in ['n_susceptible', 'n_infected']:
            plt.plot(res.timevec, res[rkey], label=res[rkey].label, **kw)
        plt.legend(frameon=False)
        plt.xlabel('Time')
        plt.ylabel('Number of people')
        plt.ylim(bottom=0)
        sc.boxoff()
        sc.commaticks()
        return ss.return_fig(fig)
