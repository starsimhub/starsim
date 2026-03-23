import starsim as ss

__all__ = ['SEIR']

class SEIR(ss.SIR):
    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.define_pars(
            dur_exp = ss.lognorm_ex(0.5),
        )
        self.update_pars(pars, **kwargs)

        # Additional states beyond the SIR ones 
        self.define_states(
            ss.BoolState('exposed', label='Exposed'),
            ss.FloatArr('ti_exposed', label='TIme of exposure'),
        )
        return

    @property
    def infectious(self):
        return self.infected | self.exposed

    def step_state(self):
        """ Make all the updates from the SIR model """
        # Perform SIR updates
        super().step_state()

        # Additional updates: progress exposed -> infected
        infected = self.exposed & (self.ti_infected <= self.ti)
        self.exposed[infected] = False
        self.infected[infected] = True
        return

    def step_die(self, uids):
        super().step_die(uids)
        self.exposed[uids] = False
        return

    def set_prognoses(self, uids, sources=None):
        """ Carry out state changes associated with infection """
        super().set_prognoses(uids, sources)
        ti = self.ti
        self.susceptible[uids] = False
        self.exposed[uids] = True
        self.ti_exposed[uids] = ti

        # Calculate and schedule future outcomes
        p = self.pars # Shorten for convenience
        dur_exp = p.dur_exp.rvs(uids)
        self.ti_infected[uids] = ti + dur_exp
        dur_inf = p.dur_inf.rvs(uids)
        will_die = p.p_death.rvs(uids)        
        self.ti_recovered[uids[~will_die]] = ti + dur_inf[~will_die]
        self.ti_dead[uids[will_die]] = ti + dur_inf[will_die]
        return