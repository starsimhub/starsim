"""
Test demographic consistency
"""
import starsim as ss
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sciris as sc
import pytest

sc.options(interactive=False) # Assume not running interactively
datadir = ss.root / 'tests/test_data'


@sc.timer()
def test_nigeria(which='births', dt=1, start=1995, dur=15, do_plot=False):
    """
    Make a Nigeria sim with demographic modules
    Switch between which='births' or 'pregnancy' to determine which demographic module to use
    """
    sc.heading('Testing Nigeria demographics')

    # Make demographic modules
    demographics = sc.autolist()

    if which == 'births':
        birth_rates = pd.read_csv(datadir/'nigeria_births.csv')
        births = ss.Births(birth_rate=birth_rates)
        demographics += births

    elif which == 'pregnancy':
        fertility_rates = pd.read_csv(datadir/'nigeria_asfr.csv')
        pregnancy = ss.Pregnancy(fertility_rate=fertility_rates, rel_fertility=1, burnin=False)  # 4/3
        demographics += pregnancy

    death_rates = pd.read_csv(datadir/'nigeria_deaths.csv')
    death = ss.Deaths(death_rate=death_rates, rate_units=1)
    demographics += death

    # Make people
    n_agents = 5_000
    nga_pop_1995 = 106819805
    age_data = pd.read_csv(datadir/'nigeria_age.csv')
    ppl = ss.People(n_agents, age_data=age_data)

    sim = ss.Sim(
        dt=dt,
        total_pop=nga_pop_1995,
        start=start,
        dur=dur,
        people=ppl,
        demographics=demographics,
    )

    if do_plot:
        sim.init()
        # Plot histograms of the age distributions - simulated vs data
        bins = np.arange(0, 101, 1)
        init_scale = nga_pop_1995 / n_agents
        counts, bins = np.histogram(sim.people.age, bins)
        plt.bar(bins[:-1], counts * init_scale, alpha=0.5, label='Simulated')
        plt.bar(bins, age_data.value.values * 1000, alpha=0.5, color='r', label='Data')
        plt.legend(loc='upper right')

    sim.run()

    stop = start + dur
    nigeria_popsize = pd.read_csv(datadir/'nigeria_popsize.csv')
    data = nigeria_popsize[(nigeria_popsize.year >= start) & (nigeria_popsize.year <= stop)]

    nigeria_cbr = pd.read_csv(datadir/'nigeria_births.csv')
    cbr_data = nigeria_cbr[(nigeria_cbr.Year >= start) & (nigeria_cbr.Year <= stop)]

    nigeria_cmr = pd.read_csv(datadir/'nigeria_cmr.csv')
    cmr_data = nigeria_cmr[(nigeria_cmr.Year >= start) & (nigeria_cmr.Year <= stop)]

    # Tests
    if which == 'pregnancy':

        print("Check we don't have more births than pregnancies")
        assert sum(sim.results.pregnancy.births) <= sum(sim.results.pregnancy.pregnancies)
        print('✓ (births <= pregnancies)')

        if dt == 1:
            print("Checking that births equal pregnancies with dt=1")
            assert np.array_equal(sim.results.pregnancy.pregnancies, sim.results.pregnancy.births)
            print('✓ (births == pregnancies)')

    rtol = 0.05
    print(f'Check final pop size within {rtol*100:n}% of data')
    assert np.isclose(data.n_alive.values[-1], sim.results.n_alive[-1], rtol=rtol), f'Final population size not within {rtol*100:n}% of data'
    print(f'✓ (simulated/data={sim.results.n_alive[-1] / data.n_alive.values[-1]:.2f})')

    # Plots
    if do_plot:
        tvec = sim.timevec
        fig, ax = plt.subplots(2, 2)
        ax = ax.ravel()
        ax[0].scatter(data.year, data.n_alive, alpha=0.5)
        ax[0].plot(tvec, sim.results.n_alive, color='k')
        ax[0].set_title('Population')

        ax[1].plot(tvec, 1000 * sim.results.deaths.cmr, label='Simulated CMR')
        ax[1].scatter(cmr_data.Year, cmr_data.CMR, label='Data CMR')
        ax[1].set_title('CMR')
        ax[1].legend()

        if which == 'births':
            ax[2].plot(tvec, sim.results.births.cbr, label='Simulated CBR')
        elif which == 'pregnancy':
            ax[2].plot(tvec, sim.results.pregnancy.cbr, label='Simulated CBR')
        ax[2].scatter(cbr_data.Year, cbr_data.CBR, label='Data CBR')
        ax[2].set_title('CBR')
        ax[2].legend()

        if which == 'pregnancy':
            ax[3].plot(tvec, sim.results.pregnancy.pregnancies, label='Pregnancies')
            ax[3].plot(tvec, sim.results.pregnancy.births, label='Births')
            ax[3].set_title('Pregnancies and births')
            ax[3].legend()

        fig.tight_layout()

    return sim


@sc.timer()
def test_constant_pop(do_plot=False):
    """ Test pars for constant pop size """
    sc.heading('Testing constant population size')
    sim = ss.Sim(n_agents=10e3, birth_rate=ss.freqperyear(10), death_rate=ss.freqperyear(10/1010*1000), dur=ss.years(200), rand_seed=1).run()
    print("Check final pop size within 5% of starting pop")
    assert np.isclose(sim.results.n_alive[0], sim.results.n_alive[-1], rtol=0.05)
    print(f'✓ (final pop / starting pop={sim.results.n_alive[-1] / sim.results.n_alive[0]:.2f})')

    # Plots
    if do_plot:
        sim.plot()
    return sim


@sc.timer()
def test_module_adding():
    """ Test that modules can't be added twice """
    sc.heading('Testing module duplication')
    births = ss.Births(birth_rate=ss.freqperyear(10))
    deaths = ss.Deaths(death_rate=ss.freqperyear(10))
    demographics = [births, deaths]
    with pytest.raises(ValueError):
        ss.Sim(n_agents=1e3, demographics=demographics, birth_rate=ss.freqperyear(10), death_rate=ss.freqperyear(10)).run()
    return demographics


@sc.timer()
def test_aging():
    """ Test that aging is configured properly """
    sc.heading('Testing aging')
    n_agents = int(1e3)

    # With no demograhpics, people shouldn't age
    s1 = ss.Sim(n_agents=n_agents).init()
    orig_ages = s1.people.age.raw.copy()
    orig_age = orig_ages.mean()
    s1.run()
    end_age = s1.people.age.mean()
    assert orig_age == end_age, f'By default there should be no aging, but {orig_age} != {end_age}'

    # We should be able to manually turn on aging
    s2 = ss.Sim(n_agents=n_agents, use_aging=True).run()
    age2 = s2.people.age.mean()
    assert orig_age < age2, f'With aging, start age {orig_age} should be less than end age {age2}'

    # Aging should turn on automatically if we add demographics
    s3 = ss.Sim(n_agents=n_agents, demographics=True).run()
    agent = s3.people.auids[0] # Find first alive agent
    orig_agent_age = orig_ages[agent]
    age3 = s3.people.age[ss.uids(agent)]
    assert orig_agent_age < age3, f'With demographics, original agent age {orig_agent_age} should be less than end age {age3}'

    # ...but can be turned off manually
    s4 = ss.Sim(n_agents=n_agents, demographics=True, use_aging=False).run()
    agent = s4.people.auids[0] # Find first alive agent
    orig_agent_age = orig_ages[agent]
    age4 = s4.people.age[ss.uids(agent)]
    assert orig_agent_age == age4, f'With aging turned off, original agent age {orig_agent_age} should match end age {age4}'

    return s3 # Most interesting sim


def test_pregnancy():
    """ Test pregnancy module"""
    sc.heading('Testing pregnancy module')

    sim = ss.Sim(
        n_agents=10e3,
        demographics=[
            ss.Pregnancy(fertility_rate=ss.freqperyear(10), burnin=True),
            ss.Deaths(death_rate=ss.freqperyear(10/1010*1000)),
        ],
        dur=ss.years(10),
        rand_seed=1,
        dt=ss.years(1/12),
        networks=ss.PrenatalNet(),
    )
    sim.run()

    # Tests
    # Check that everyone in their 2nd trimester conceived 3-6 months ago
    tri2 = sim.demographics.pregnancy.tri2_uids
    conception_time = sim.people.pregnancy.ti_pregnant[tri2]
    time_since_conception = ss.years((sim.ti - conception_time) * sim.t.dt_year).weeks
    assert np.all((time_since_conception >= 13) & (time_since_conception <= 26)), 'Some people in 2nd trimester did not conceive 3-6 months ago'

    # Check that gestational clock is present for all pregnant women, and that they all have a defined delivery time
    pregnant = sim.people.pregnancy.pregnant
    assert np.all(~np.isnan(sim.people.pregnancy.gestation[pregnant])), 'Some pregnant people are missing gestational clock'
    ti_delivery = sim.people.pregnancy.ti_delivery[pregnant]
    assert np.all(~np.isnan(ti_delivery)), 'Some people with ti_pregnant are missing ti_delivery'

    # Check that gestational clock is NaN for all non-pregnant women
    assert np.all(np.isnan(sim.people.pregnancy.gestation[~sim.people.pregnancy.pregnant])), 'Some non-pregnant people have gestational clock'
    sc.printgreen('✓ Pregnancy duration tests passed')

    # Check that everyone born during the sim has gestational age at birth recorded
    born_during_sim = (sim.people.age < sim.pars.dur.years) & (sim.people.age > sim.t.dt.years)
    aa = (sim.people.pregnancy.gestation_at_birth.isnan & born_during_sim).uids
    assert len(aa) == 0, f'Some babies born during the sim are missing gestational age at birth: {len(aa)} found'

    # Check that the mean gestational age at birth is approximately 40 weeks
    mean_gestation = np.nanmean(sim.people.pregnancy.gestation_at_birth[born_during_sim])
    assert np.isclose(mean_gestation, 40, atol=2), f'Mean gestational age at birth is {mean_gestation:.1f} weeks, expected ~40 weeks'

    # Check that the mean gestational age of babies born to women <18 or >35 is shorter than otherwise
    mothers = sim.people.parent[born_during_sim]
    age_of_mother = sim.people.age[mothers]
    age_of_mother_at_birth = age_of_mother - sim.people.age[born_during_sim]
    key_cohorts = {
        '<18': age_of_mother_at_birth < 18,
        '>35': age_of_mother_at_birth > 35,
        '18-35': (age_of_mother_at_birth >= 18) & (age_of_mother_at_birth <= 35),
    }
    mean_results = {k:None for k in key_cohorts.keys()}
    ptb_results = {k:None for k in key_cohorts.keys()}
    for k, v in key_cohorts.items():
        these_mothers_idx = v.nonzero()[-1]
        babies_of_these_mothers = born_during_sim.uids[these_mothers_idx]
        mean_gestation = sim.people.pregnancy.gestation_at_birth[babies_of_these_mothers].mean()
        print(f'Mean gestational age at birth for mothers aged {k} at birth: {mean_gestation:.1f} weeks')
        mean_results[k] = mean_gestation
        n = len(babies_of_these_mothers)
        print(f'  {n} babies born to mothers aged {k} at birth')
        # Share of babies born preterm (<37 weeks)
        n_preterm = np.sum(sim.people.pregnancy.gestation_at_birth[babies_of_these_mothers] < 37)
        preterm_share = n_preterm / n if n > 0 else np.nan
        print(f'  Preterm births (<37 weeks): {n_preterm} ({preterm_share:.1%})')
        ptb_results[k] = preterm_share

    # Assertions
    assert mean_results['<18'] < mean_results['18-35'], '<18 mothers do not have lower gestational age at birth than 18-35 mothers'
    assert mean_results['>35'] < mean_results['18-35'], '>35 mothers do not have lower gestational age at birth than 18-35 mothers'
    assert ptb_results['<18'] > ptb_results['18-35'], '<18 mothers do not have higher preterm birth share than 18-35 mothers'
    assert ptb_results['>35'] > ptb_results['18-35'], '>35 mothers do not have higher preterm birth share than 18-35 mothers'

    sc.printgreen('✓ Gestational age at birth by maternal age tests passed')

    return sim


def test_pregnancy_short_sim():
    """ Test that Pregnancy.finalize() works for sims shorter than 1 year """
    sc.heading('Testing pregnancy with short sim duration')

    sim = ss.Sim(
        n_agents=1e3,
        demographics=[
            ss.Pregnancy(fertility_rate=ss.freqperyear(10), burnin=False),
            ss.Deaths(death_rate=ss.freqperyear(10/1010*1000)),
        ],
        dur=ss.days(250),
        rand_seed=1,
        dt=ss.days(10),
        networks=ss.PrenatalNet(),
    )
    sim.run()  # Should not raise ValueError in finalize()

    sc.printgreen('✓ Short sim pregnancy finalize test passed')
    return sim


def test_fetal_health(do_plot=False):
    """
    Test the FetalHealth module across three scenarios:
      1. Baseline: no disease — birth weights ~3400g at term
      2. Disease: SIR infection worsens fetal outcomes via a connector
      3. Treatment: treating infected pregnant women partially reverses damage
    """
    sc.heading('Testing fetal health module')

    # -- Helper classes --

    class tx(ss.Intervention):
        """ Treat infected pregnant women each timestep """
        def __init__(self, disease='sir', **kwargs):
            super().__init__(**kwargs)
            self.disease_name = disease
            self.define_pars(
                p_treat=ss.bernoulli(p=0.9),
            )
            self.define_states(
                ss.FloatArr('ti_treated', label='Time of treatment'),
            )
            return

        def step(self):
            preg = self.sim.people.pregnancy
            disease = self.sim.diseases[self.disease_name]
            eligible = preg.pregnant & disease.infected
            treated = self.pars.p_treat.filter(eligible)
            if len(treated):
                disease.infected[treated] = False
                disease.recovered[treated] = True
                self.ti_treated[treated] = self.ti
            return

    class fetal_tx(ss.Connector):
        """ Connector: infection damages fetal health, treatment partially reverses it """
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.define_pars(
                timing_shift=ss.lognorm_ex(mean=3.0, std=1.0),
                growth_penalty=0.15,
                tx_growth_reversal=0.7,
                tx_timing_reversal=0.7,
            )
            return

        def init_pre(self, sim):
            super().init_pre(sim)
            fh = sim.custom['fetal_health']
            fh.add_conception_callback(self._on_conception)
            return

        def _on_conception(self, uids):
            infected = self.sim.diseases.sir.infected[uids]
            infected_uids = uids[infected]
            if len(infected_uids):
                self._apply_damage(infected_uids)
            return

        def _apply_damage(self, uids):
            fh = self.sim.custom['fetal_health']
            shifts = self.pars.timing_shift.rvs(uids)
            fh.apply_timing_shift(uids, shifts)
            fh.apply_growth_restriction(uids, self.pars.growth_penalty)
            return

        def _apply_treatment_reversal(self, uids):
            fh = self.sim.custom['fetal_health']
            reversible = self.pars.growth_penalty * self.pars.tx_growth_reversal
            fh.reverse_growth_restriction(uids, reversible)
            fh.reverse_timing_shift(uids, self.pars.tx_timing_reversal)
            return

        def step(self):
            sim = self.sim
            preg = sim.people.pregnancy
            if not preg.pregnant.any():
                return

            pregnant_uids = preg.pregnant.uids

            # New infections in pregnant women
            newly_infected = sim.diseases.sir.ti_infected == self.ti
            affected = pregnant_uids[newly_infected[pregnant_uids]]
            if len(affected):
                self._apply_damage(affected)

            # Newly treated pregnant women
            if 'tx' in sim.interventions:
                intv = sim.interventions['tx']
                just_treated = intv.ti_treated == self.ti
                treated_pregnant = pregnant_uids[just_treated[pregnant_uids]]
                if len(treated_pregnant):
                    self._apply_treatment_reversal(treated_pregnant)
            return

    # -- Shared sim parameters --

    sim_kw = dict(
        n_agents=5_000,
        dur=ss.years(5),
        dt=ss.years(1/12),
        rand_seed=1,
    )
    demo = lambda: [
        ss.Pregnancy(fertility_rate=ss.freqperyear(30), burnin=True),
        ss.Deaths(death_rate=ss.freqperyear(10/1010*1000)),
    ]
    sir_kw = dict(dur_inf=ss.lognorm_ex(mean=ss.dur(6), std=ss.dur(1)), beta=0.1, init_prev=0.1)

    def mean_bw(sim):
        bw = sim.results.fetal_health.mean_birth_weight
        return bw[bw > 0].mean()

    # -- 1. Baseline (no disease) --
    sim_baseline = ss.Sim(
        demographics=demo(),
        modules=ss.FetalHealth(),
        networks=ss.PrenatalNet(),
        **sim_kw,
    )
    sim_baseline.run()

    # Basic sanity checks
    total_deliveries = sim_baseline.results.fetal_health.n_deliveries.sum()
    assert total_deliveries > 0, 'No deliveries recorded'
    print(f'Total deliveries: {total_deliveries}')

    bw_baseline = mean_bw(sim_baseline)
    print(f'Mean birth weight (baseline): {bw_baseline:.0f}g')
    assert 2500 < bw_baseline < 4000, f'Mean birth weight {bw_baseline:.0f}g outside expected range'

    ga = sim_baseline.results.fetal_health.mean_ga_at_birth
    mean_ga = ga[ga > 0].mean()
    print(f'Mean GA at birth: {mean_ga:.1f} weeks')
    assert 36 < mean_ga < 44, f'Mean GA {mean_ga:.1f} weeks outside expected range'

    # -- 2. Disease only (no treatment) --
    sim_disease = ss.Sim(
        demographics=demo(),
        diseases=ss.SIR(**sir_kw),
        connectors=fetal_tx(),
        modules=ss.FetalHealth(),
        networks=[ss.PrenatalNet(), ss.RandomNet()],
        **sim_kw,
    )
    sim_disease.run()

    bw_disease = mean_bw(sim_disease)
    print(f'Mean birth weight (disease): {bw_disease:.0f}g')
    assert bw_disease < bw_baseline, f'Disease ({bw_disease:.0f}g) should be lower than baseline ({bw_baseline:.0f}g)'

    ptb_disease = sim_disease.results.fetal_health.n_preterm.sum()
    ptb_baseline = sim_baseline.results.fetal_health.n_preterm.sum()
    print(f'Preterm births — baseline: {ptb_baseline}, disease: {ptb_disease}')
    assert ptb_disease > ptb_baseline, 'Disease scenario should have more preterm births'

    # -- 3. Disease + treatment --
    sim_treated = ss.Sim(
        demographics=demo(),
        diseases=ss.SIR(**sir_kw),
        connectors=fetal_tx(),
        interventions=tx(name='tx', disease='sir'),
        modules=ss.FetalHealth(),
        networks=[ss.PrenatalNet(), ss.RandomNet()],
        **sim_kw,
    )
    sim_treated.run()

    bw_treated = mean_bw(sim_treated)
    print(f'Mean birth weight (treated): {bw_treated:.0f}g')
    assert bw_treated > bw_disease, f'Treated ({bw_treated:.0f}g) should be higher than disease-only ({bw_disease:.0f}g)'

    ptb_treated = sim_treated.results.fetal_health.n_preterm.sum()
    print(f'Preterm births — disease: {ptb_disease}, treated: {ptb_treated}')
    assert ptb_treated <= ptb_disease, 'Treatment should not increase preterm births'

    print(f'Summary — baseline: {bw_baseline:.0f}g, disease: {bw_disease:.0f}g, treated: {bw_treated:.0f}g')

    if do_plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for sim, label in [(sim_baseline, 'Baseline'), (sim_disease, 'Disease'), (sim_treated, 'Treated')]:
            fh = sim.custom['fetal_health']
            born = fh.birth_weight > 0
            axes[0].hist(np.asarray(fh.birth_weight[born]), bins=30, alpha=0.4, label=label, edgecolor='k')
        axes[0].axvline(2500, color='r', ls='--', label='LBW threshold')
        axes[0].set_xlabel('Birth weight (g)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Birth weight: baseline vs disease vs treated')
        axes[0].legend()

        tvec = sim_baseline.timevec
        axes[1].plot(tvec, sim_baseline.results.fetal_health.preterm_rate, label='Baseline')
        axes[1].plot(tvec, sim_disease.results.fetal_health.preterm_rate, label='Disease')
        axes[1].plot(tvec, sim_treated.results.fetal_health.preterm_rate, label='Treated')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Preterm rate')
        axes[1].set_title('Preterm birth rate')
        axes[1].legend()

        fig.tight_layout()

    sc.printgreen('✓ Fetal health tests passed')
    return sim_baseline, sim_disease, sim_treated


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    s1 = test_nigeria(do_plot=do_plot)
    s2 = test_nigeria(do_plot=do_plot, dt=1/12, which='pregnancy')
    s3 = test_constant_pop(do_plot=do_plot)
    s4 = test_module_adding()
    s5 = test_aging()
    s6 = test_pregnancy()
    s7 = test_fetal_health(do_plot=do_plot)
    plt.show()
