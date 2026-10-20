"""
Sexual networks beyond the core ones (`ss.MFNet`, `ss.PartnerNet`): partnerships between men
"""
import numpy as np
import sciris as sc
import starsim as ss

_ = None # Sentinel for "not provided" default arguments


class MSMNet(ss.SexualNetwork):
    """
    A network that randomly pairs males

    Args:
        dur (`ss.Dist`): the duration of each partnership; can vary by age, year, and individual pair. Set scale=exp(mu) and s=sigma where mu,sigma are of the underlying normal distribution.
        debut (`ss.Dist`): Age of debut can vary by using callable parameter values
        acts (`ss.Dist`): Number of acts per year
        participation (`ss.Dist`): Probability of participating in this network - can vary by individual properties (age, sex, ...) using callable parameter values
    """
    def __init__(self, pars=None, dur=_, debut=_, acts=_, participation=_, **kwargs):
        super().__init__()
        ss.networks.check_duration(self, pars, kwargs)
        self.define_pars(
            dur = ss.lognorm_ex(mean=ss.years(2), std=ss.years(1)),
            debut = ss.normal(loc=16, scale=2),
            acts = ss.lognorm_ex(mean=ss.time.freq(80, 'year'), std=ss.time.freq(20, 'year')),
            participation = ss.bernoulli(p=0.1),
        )
        self.update_pars(pars, **kwargs)
        self.dist = ss.random(name='MSMNet') # For shuffling available males before pairing
        self.dist.hash_dtype = np.float64 # Avoid ties, as for ssl.RandomSafeNet
        self.debut = ss.FloatArr('debut') # Drawn for each man on his first step in the sim, as for ss.MFNet
        self.decided = ss.BoolArr('decided') # Whether his participation has been drawn, which happens at his debut
        return

    def init_post(self):
        self.set_network_states()
        super().init_post()
        return

    def set_network_states(self):
        """ Draw the age of debut of any man who has none (newborns and arrivals too), and whether he takes part when he reaches it, as for ss.MFNet """
        people = self.sim.people
        new = (people.alive & people.male & self.debut.isnan).uids
        self.debut[new] = self.pars.debut.rvs(new)
        due = (people.alive & people.male & ~self.decided & (people.age >= self.debut)).uids
        self.participant[due] = self.pars.participation.rvs(due)
        self.decided[due] = True
        return

    def add_pairs(self):
        """ Pair all unpartnered MSM """
        available_m = self.available(self.sim.people, 'male')
        available_m = available_m[np.argsort(self.dist.rvs(available_m), kind='stable')] # Shuffle so pairing is random; use a stable sort since the default order of ties depends on the CPU
        n_pairs = int(len(available_m)/2)
        p1 = available_m[:n_pairs]
        p2 = available_m[n_pairs:n_pairs*2]

        # Figure out durations
        if (len(p1) == len(np.unique(p1))):
            # No duplicates, so use slotting based on p1
            dur = self.pars.dur.rvs(p1)
            act_vals = self.pars.acts.rvs(p1)
        else:
            dur = self.pars.dur.rvs(len(p1)) # Just use len(p1) to say how many draws are needed
            act_vals = self.pars.acts.rvs(len(p1))

        self.append(p1=p1, p2=p2, beta=np.ones_like(p1), dur=dur, acts=act_vals)

        return len(p1)

    def step(self):
        self.end_pairs()
        self.set_network_states()
        self.add_pairs()
        return
