"""
Define default HIV disease module and related interventions
"""

import numpy as np
import sciris as sc
import starsim as ss


__all__ = ['HIV', 'ART', 'CD4_analyzer']

class HIV(ss.Infection):

    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.default_pars(
            beta = 1.0, # Placeholder value
            art_efficacy = 0.96,
            VMMC_efficacy = 0.6,
            init_prev = ss.bernoulli(p=0.05),
            survival_without_art = ss.weibull(c=2, scale=13),
            #survival_without_art = ss.weibull(c=2, scale=lambda self, sim, uids: 21.182-0.2717*sim.people.age[uids]), # Adult survival from EMOD
        )
        self.update_pars(pars=pars, **kwargs)

        # States
        self.add_states(
            ss.BoolArr('on_art', label='On ART'),
            ss.BoolArr('circumcised', label='Circumcised'),
            ss.FloatArr('ti_art', label='Time of ART initiation'),
            ss.FloatArr('ti_dead', label='Time of death'), # Time of HIV-caused death
        )
        return

    def update_pre(self):
        """ Update CD4 """
        people = self.sim.people

        self.rel_trans[self.infected & self.on_art] = 1 - self.pars['art_efficacy']
        self.rel_sus[people.male & self.circumcised] = 1 - self.pars['VMMC_efficacy']

        hiv_deaths = (self.ti_dead == self.sim.ti).uids
        people.request_death(hiv_deaths)
        return

    def init_results(self):
        """ Initialize results """
        super().init_results()
        self.results += ss.Result(self.name, 'new_deaths', self.sim.npts, dtype=int, label='Deaths')
        self.results += ss.Result(self.name, 'art_coverage', self.sim.npts, dtype=float, label='ART Coverage')
        self.results += ss.Result(self.name, 'vmmc_coverage', self.sim.npts, dtype=float, label='VMMC Coverage')
        self.results += ss.Result(self.name, 'vmmc_coverage_15_49', self.sim.npts, dtype=float, label='VMMC Coverage 15-49')
        self.results += ss.Result(self.name, 'prevalence_15_49', self.sim.npts, dtype=float, label='Prevalence 15-49')
        return

    def update_results(self):
        super().update_results()
        ti = self.sim.ti
        self.results['new_deaths'][ti] = np.count_nonzero(self.ti_dead == ti)
        n_inf = np.count_nonzero(self.infected)
        self.results['art_coverage'][ti] = np.count_nonzero(self.on_art) / n_inf if n_inf > 0 else 0
        self.results['vmmc_coverage'][ti] = np.count_nonzero(self.circumcised) / np.count_nonzero(self.sim.people.male)
        inds = (self.sim.people.age >= 15) & (self.sim.people.age < 50)
        self.results['vmmc_coverage_15_49'][ti] = np.count_nonzero(self.circumcised[inds]) / np.count_nonzero(self.sim.people.male[inds])
        self.results['prevalence_15_49'][ti] = np.count_nonzero(self.infected[inds]) / np.count_nonzero(self.sim.people.alive[inds])
        return 

    def set_prognoses(self, uids, source_uids=None):
        super().set_prognoses(uids, source_uids)
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = self.sim.ti

        prog = self.pars.survival_without_art.rvs(uids)
        self.ti_dead[uids] = self.sim.ti + np.round(prog/self.sim.dt).astype(int) # Survival without treatment
        return

    def set_congenital(self, uids, source_uids):
        return self.set_prognoses(uids, source_uids)


# %% HIV-related interventions

class ART(ss.Intervention):

    def __init__(self, year, coverage, pars=None, **kwargs):
        self.requires = HIV
        self.year = sc.toarray(year)
        self.coverage = sc.toarray(coverage)
        super().__init__()
        self.default_pars(
        #    art_delay = ss.lognorm_im(mean=1) # Value in years
        )
        self.update_pars(pars=pars, **kwargs)

        prob_art_at_init = lambda self, sim, uids: np.interp(sim.year, self.year, self.coverage)
        self.prob_art_at_infection = ss.bernoulli(p=prob_art_at_init)
        self.prob_art_post_infection = ss.bernoulli(p=0) # Set in init_pre
        return

    def init_pre(self, sim):
        super().init_pre(sim)

        # Importance sampling
        cvec = np.interp(sim.yearvec, self.year, self.coverage)
        pvec = np.zeros_like(cvec)
        np.divide(np.diff(cvec), 1-cvec[:-1], where=cvec[:-1]<1, out=pvec[1:])
        pvec = np.clip(pvec, a_min=0, a_max=1)

        prob_art_post_init = lambda self, sim, uids, pvec=pvec: pvec[sim.ti]
        self.prob_art_post_infection.set(p=prob_art_post_init)

        self.results += ss.Result(self.name, 'n_art', sim.npts, dtype=int)
        self.initialized = True
        return

    def apply(self, sim):
        if sim.year < self.year[0]:
            return

        hiv = sim.people.hiv
        infected = hiv.infected.uids
        ti_delay = 1 + np.round(2/sim.dt).astype(int) # About 2 years to ART initiation, 1+ due to order of operations, we're on the next time step
        recently_infected = infected[hiv.ti_infected[infected] == sim.ti - ti_delay]
        notrecent_noart = infected[(hiv.ti_infected[infected] < sim.ti - ti_delay) & (~hiv.on_art[infected])]

        n_added = 0
        if len(recently_infected):
            inds = self.prob_art_at_infection.filter(recently_infected)
            hiv.on_art[inds] = True
            hiv.ti_art[inds] = sim.ti
            hiv.ti_dead[inds] = np.nan
            n_added += len(inds)

        if len(notrecent_noart):
            inds = self.prob_art_post_infection.filter(notrecent_noart)
            hiv.on_art[inds] = True
            hiv.ti_art[inds] = sim.ti
            hiv.ti_dead[inds] = np.nan
            n_added += len(inds)

        # Add results
        self.results['n_art'][sim.ti] = np.count_nonzero(hiv.on_art)

        return n_added


class VMMC(ss.Intervention):

    def __init__(self, year, coverage, pars=None, **kwargs):
        self.requires = HIV
        self.year = sc.toarray(year)
        self.coverage = sc.toarray(coverage)
        super().__init__()
        self.default_pars()
        self.update_pars(pars=pars, **kwargs)

        prob_VMMC_at_debut = lambda self, sim, uids: np.interp(sim.year, self.year, self.coverage)
        self.prob_VMMC_at_debut = ss.bernoulli(p=prob_VMMC_at_debut)
        self.prob_VMMC_post_debut = ss.bernoulli(p=0) # Set in init_pre
        return

    def init_pre(self, sim):
        super().init_pre(sim)

        # Importance sampling
        cvec = np.interp(sim.yearvec, self.year, self.coverage)
        pvec = np.zeros_like(cvec)
        np.divide(np.diff(cvec), 1-cvec[:-1], where=cvec[:-1]<1, out=pvec[1:])
        pvec = np.clip(pvec, a_min=0, a_max=1)

        p_vmmc = lambda self, sim, uids, pvec=pvec: pvec[sim.ti]
        self.prob_VMMC_post_debut.set(p=p_vmmc)

        self.results += ss.Result(self.name, 'n_vmmc', sim.npts, dtype=int)
        self.initialized = True
        return

    def apply(self, sim):
        if sim.year < self.year[0]:
            return

        hiv = sim.people.hiv
        debut = None
        for net in sim.networks.values():
            if hasattr(net, 'debut'):
                debut = net.debut
                break # Take first
        assert debut != None
        recent_debut = (self.sim.people.age >= debut) & (self.sim.people.age < debut + sim.dt) & (self.sim.people.male) & (~hiv.circumcised)
        male_novmmc = (~recent_debut) & (self.sim.people.male) & (~hiv.circumcised)

        n_added = 0

        if recent_debut.any():
            inds = self.prob_VMMC_at_debut.filter(recent_debut)
            hiv.circumcised[inds] = True
            #hiv.ti_circumcised[inds] = sim.ti
            n_added += len(inds)

        if male_novmmc.any():
            inds = self.prob_VMMC_post_debut.filter(male_novmmc)
            hiv.circumcised[inds] = True
            #hiv.ti_circumcised[inds] = sim.ti
            n_added += len(inds)

        # Add results
        self.results['n_vmmc'][sim.ti] = n_added

        return n_added


#%% Analyzers

class CD4_analyzer(ss.Analyzer):

    def __init__(self):
        self.requires = HIV
        self.cd4 = None
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.cd4 = np.zeros((sim.npts, sim.people.n), dtype=int)
        return

    def apply(self, sim):
        self.cd4[sim.t] = sim.people.hiv.cd4
        return
