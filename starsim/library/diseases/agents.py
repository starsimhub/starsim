"""
Diseases written as code: Starsim 3's SIR, SEIR, and SIS

These are the Starsim 3 library models, written as `ss.AgentDisease` subclasses with
per-agent states and methods. They are teaching examples: they spell out, step by step
in plain Python, what the declared `ss.SIR`, `ss.SIS`, and `ss.SEIR` do under the hood
when run as agents, and they are starting points for a disease of your own written as
code. In most cases, it would be preferable to use the declared form, but these are 
included as examples of what explicit agent-based code looks like. Needless to say,
they run only as agents.

They schedule each move in whole steps as the agent engine for a declared disease does (see `ss.AgentDisease`): an
infection in a step happens at its end, so `dur` steps after it is due at `self.ti + 1 + dur`, and is
taken when `ss.TimeArr.due()`, at the start of a step, before transmission, rounding a time between steps up or
down at random in proportion. So each case, seeds included, is infectious for `dur` steps on average, and with the
same parameters these agree with `ss.SIR` and `ss.SEIR` run as agents at any step.
"""
# Authorship: human drafted, AI edited, human reviewed and edited

import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

__all__ = ['AgentSIR', 'AgentSEIR', 'AgentSIS']

_ = None # For function signatures


class AgentSIR(ss.AgentDisease):
    """
    An SIR model written as code: Starsim 3's `ss.SIR`

    This class implements a basic SIR model with states for susceptible,
    infected/infectious, and recovered, as per-agent arrays updated in code (see
    `ss.AgentDisease`). It also
    includes deaths, and basic results. It is a teaching
    example, showing in code what the declared `ss.SIR` does as agents, and a starting
    point for a disease written as code (e.g. `ssl.HIV`).

    Its defaults are Starsim 3's: a lognormal `dur_inf` (a mean of 6 years, std 1 year)
    and `p_death=0.01`, where the declared `ss.SIR` is the textbook SIR, with an
    exponential `dur_inf` and no deaths. Use it to keep a Starsim 3 model's parameters;
    its timing is the agent engine's (see above), so each case is infectious for half a
    step longer on average than in Starsim 3.

    Note: CRN safe.

    Args:
        beta (float/`ss.prob`): the infectiousness
        init_prev (float/`ss.bernoulli`): the fraction of people to start off being infected
        dur_inf (float/`ss.Duration`/`ss.Dist`): how long (in years) people are infected for
        p_death (float/`ss.bernoulli`): the probability of death from infection
    """
    plot_states = ['susceptible', 'infected', 'recovered'] # Which results to show in plot()

    def __init__(self, pars=None, beta=_, init_prev=_, dur_inf=_, p_death=_, **kwargs):
        super().__init__()
        self.define_pars(
            beta = ss.peryear(0.1),
            init_prev = ss.bernoulli(p=0.01),
            dur_inf = ss.lognormal(mean=ss.years(6)),
            p_death = ss.bernoulli(p=0.01),
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.BoolState('recovered', label='Recovered'),
            ss.TimeArr('ti_recovered', label='Time of recovery'),
            ss.TimeArr('ti_dead', label='Time of death'),
        )
        return

    def step_state(self):
        # Progress infectious -> recovered
        sim = self.sim
        recovered = (self.infected & self.ti_recovered.due()).uids
        self.clear_infection(recovered)
        self.recovered[recovered] = True

        # Trigger deaths: at the start of the step, as recovery is, so they do not transmit in it
        deaths = self.ti_dead.due().uids
        self.clear_infection(deaths)
        if len(deaths):
            sim.people.request_death(deaths, cause=self)
        return

    def set_prognoses(self, uids, sources=None):
        """ Infect the agents, then schedule what happens to them """
        self.susceptible[uids] = False
        self.set_infection(uids)
        self.set_progression(uids)
        return

    def set_infection(self, uids):
        """ Make the agents infectious, immediately (`AgentSEIR` overrides this to add a latent period) """
        self.infected[uids] = True
        self.ti_infected[uids] = self.ti
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
        # soon as they are infected, at the end of the step, so the infectious period runs
        # from the next step (ti_infected + 1); AgentSEIR overrides this to run it from
        # ti_infectious instead. due() rounds the time to a step (see ss.TimeArr).
        will_die = p.p_death.rvs(uids)
        dead_uids = uids[will_die]
        rec_uids = uids[~will_die]
        self.ti_dead[dead_uids] = self.ti_infected[dead_uids] + 1 + dur_inf[will_die]
        self.ti_recovered[rec_uids] = self.ti_infected[rec_uids] + 1 + dur_inf[~will_die]
        return

    def on_death(self, uids):
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
    An SEIR model written as agent-based code rather than declared: Starsim 3's `ss.SEIR` 
    (see `ssl.AgentSIR`), with its lognormal `dur_exp` and `dur_inf` and `p_death=0.01`.

    This class extends `AgentSIR` with an exposed state: agents who are infected but not
    yet infectious. `exposed` and `infectious` are the literal E and I compartments, and
    `infected` is derived from them: an agent is infected from the moment it acquires the
    infection, whether or not it is yet transmitting. So `infected` and `prevalence`
    count E plus I, while transmission depends on `infectious` alone.

    Correspondingly, `ti_exposed` is the time of acquisition and `ti_infectious` is the
    time of becoming infectious. `ti_infected` is deliberately not defined, since it is
    too easily confused with `ti_infectious`.

    Note: CRN safe.

    Args:
        beta (float/`ss.prob`): the infectiousness
        init_prev (float/`ss.bernoulli`): the fraction of people to start off being infected
        dur_exp (float/`ss.Duration`/`ss.Dist`): how long people are exposed (latent) for
        dur_inf (float/`ss.Duration`/`ss.Dist`): how long people are infectious for
        p_death (float/`ss.bernoulli`): the probability of death from infection
    """
    plot_states = ['susceptible', 'exposed', 'infectious', 'recovered']

    def __init__(self, pars=None, dur_exp=_, **kwargs):
        super().__init__()
        self.define_pars(
            dur_exp = ss.lognormal(mean=ss.years(0.5)),
        )
        self.update_pars(pars, **kwargs)

        # SIR states are added automatically; here we split I off from "infected", which
        # becomes E plus I, and drop ti_infected in favor of ti_exposed and ti_infectious
        self.define_states(
            ss.BoolState('exposed', label='Exposed'),
            ss.BoolState('infectious', label='Infectious'),
            ss.FloatArr('ti_exposed', label='Time of exposure'),
            ss.TimeArr('ti_infectious', label='Time of becoming infectious'),
            reset = ['infected', 'infectious', 'ti_infected'],
            infected = lambda self: self.exposed | self.infectious, # Derived: E and I are both infected
        )
        return

    def set_infection(self, uids):
        """ Agents are exposed first, and become infectious after the latent period """
        self.exposed[uids] = True
        self.ti_exposed[uids] = self.ti
        self.ti_infectious[uids] = self.ti_exposed[uids] + 1 + self.pars.dur_exp.rvs(uids) # Exposed at the end of this step, so the latent period runs from the next
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

    def not_naive(self):
        """ The agents who have ever been infected, for dynamic rescaling """
        return self.ti_exposed.notnan.uids

    def clear_infection(self, uids):
        """ `infected` is derived, so clear the states it is derived from """
        self.exposed[uids] = False
        self.infectious[uids] = False
        return

    def step_state(self):
        """ Progress exposed -> infectious, then the usual SIR transitions """
        infectious = (self.exposed & self.ti_infectious.due()).uids
        self.exposed[infectious] = False
        self.infectious[infectious] = True
        super().step_state()
        return


class AgentSIS(ss.AgentDisease):
    """
    An SIS model with waning immunity, written as code: Starsim 3's `ss.SIS` (see `ssl.AgentSIR`), with its lognormal `dur_inf`

    This class implements a basic SIS model with states for susceptible,
    infected/infectious, and back to susceptible, with a per-agent immunity level
    that is boosted by each infection and wanes exponentially, which sets each agent's
    `rel_sus`. There is no death in this case. For an SIS model without immunity,
    use the declared `ss.SIS`, which runs with every solver.

    Note: CRN safe.

    Args:
        beta (float/`ss.prob`): the infectiousness
        init_prev (float/`ss.bernoulli`): the fraction of people to start off being infected
        dur_inf (float/`ss.Duration`/`ss.Dist`): how long (in years) people are infected for
        waning (float/`ss.per`): how quickly immunity wanes
        imm_boost (float): how much an infection boosts immunity
    """
    def __init__(self, pars=None, beta=_, init_prev=_, dur_inf=_, waning=_, imm_boost=_, **kwargs):
        super().__init__()
        self.define_pars(
            beta = ss.peryear(0.05),
            init_prev = ss.bernoulli(p=0.01),
            dur_inf = ss.lognormal(mean=ss.years(10)),
            waning = ss.peryear(0.05),
            imm_boost = 1.0,
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            ss.TimeArr('ti_recovered'),
            ss.FloatArr('immunity', default=0.0),
        )
        return

    def init_results(self):
        """ As for `ss.AgentDisease`, and the mean susceptibility """
        super().init_results()
        self.define_results(ss.Result('rel_sus', dtype=float, label='Relative susceptibility', kind='ratio'))
        return

    def step_state(self):
        """ Progress infectious -> recovered """
        recovered = (self.infected & self.ti_recovered.due()).uids
        self.clear_infection(recovered)
        self.update_immunity()
        return

    def clear_infection(self, uids):
        """ Clear the infection, on recovery or cure (e.g. by `ss.treat()`): the agents are susceptible again """
        self.infected[uids] = False
        self.susceptible[uids] = True
        return

    def make_naive(self, uids):
        """ Reset agents to never having had the disease, for dynamic rescaling: every state to its default (`rel_sus` too, since SIS sets it from its own immunity), except `rel_trans` """
        for state in self.state_list:
            if state.name != 'rel_trans':
                state.set(uids)
        return

    def update_immunity(self):
        """ Apply exponential waning to immunity and update relative susceptibility. """
        waning = self.pars.waning.to_prob() # Exponential waning (NB: the exponential conversion is calculated automatically by the timepar)
        has_imm = (self.immunity > 0).uids
        self.immunity[has_imm] *= (1-waning)
        self.rel_sus[has_imm] = np.maximum(0, 1 - self.immunity[has_imm])
        return

    def set_prognoses(self, uids, sources=None):
        """ Infect the agents, at the end of this step, and schedule their recovery """
        super().set_prognoses(uids, sources)
        self.immunity[uids] += self.pars.imm_boost

        # Sample duration of infection
        dur_inf = self.pars.dur_inf.rvs(uids)

        # Determine when people recover: infected at the end of this step, so infectious from the next
        self.ti_recovered[uids] = self.ti_infected[uids] + 1 + dur_inf
        return

    def record_states(self):
        """ As for `ss.AgentDisease`, and the population immunity (susceptibility), at the start of the step """
        super().record_states()
        res = self.results
        res.rel_sus[self.ti] = self.rel_sus.mean()
        return

    def plot(self, **kwargs):
        """ Default plot for SIS model """
        fig = plt.figure()
        kw = sc.mergedicts(dict(lw=2, alpha=0.8), kwargs)
        res = self.results
        for rkey in ['susceptible', 'infected']:
            plt.plot(res.timevec, res[rkey], label=res[rkey].label, **kw)
        plt.legend(frameon=False)
        plt.xlabel('Time')
        plt.ylabel('Number of people')
        plt.ylim(bottom=0)
        sc.boxoff()
        sc.commaticks()
        return ss.return_fig(fig)
