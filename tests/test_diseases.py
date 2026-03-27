"""
Run tests of disease models
"""

# %% Imports and settings
import sciris as sc
import matplotlib.pyplot as plt
import numpy as np
import starsim as ss
import starsim_examples as sse

test_run = True
n_agents = [10_000, 2_000][test_run]
do_plot = True
sc.options(interactive=do_plot) # Assume not running interactively


@sc.timer()
def test_sir():
    sc.heading('Testing SIR dynamics')

    ppl = ss.People(n_agents)
    network_pars = {
        'n_contacts': ss.poisson(4), # Contacts Poisson distributed with a mean of 4
    }
    networks = ss.RandomNet(**network_pars)

    sir_pars = {
        'dur_inf': ss.normal(loc=10),  # Override the default distribution
    }
    sir = ss.SIR(**sir_pars)

    # Change pars after creating the SIR instance
    sir.pars.beta = {'random': ss.freqperyear(0.1)}

    # You can also change the parameters of the default lognormal distribution directly
    sir.pars.dur_inf.set(loc=5)

    # Or use a function, here a lambda that makes the mean for each agent equal to their age divided by 10
    sir.pars.dur_inf.set(loc = lambda self, sim, uids: sim.people.age[uids] / 10)

    sim = ss.Sim(people=ppl, diseases=sir, networks=networks)
    sim.run()

    plt.figure()
    res = sim.results
    plt.stackplot(
        sim.timevec,
        res.sir.n_susceptible,
        res.sir.n_infected,
        res.sir.n_recovered,
        res.new_deaths.cumsum(),
    )
    plt.legend(['Susceptible', 'Infected', 'Recovered', 'Dead'])
    plt.xlabel('Year')
    plt.title('SIR')
    return sim


@sc.timer()
def test_sir_epi():
    sc.heading('Test basic epi dynamics')

    base_pars = dict(n_agents=n_agents, networks=dict(type='random'), diseases=dict(type='sir'))

    # Define the parameters to vary
    par_effects = dict(
        beta = [0.01, 0.99],
        n_contacts = [1, 20],
        init_prev = [0.1, 0.9],
        dur_inf = [1, 8],
        p_death = [.01, .1],
    )

    # Loop over each of the above parameters and make sure they affect the epi dynamics in the expected ways
    for par, par_val in par_effects.items():
        lo = par_val[0]
        hi = par_val[1]

        # Make baseline pars
        pars0 = sc.dcp(base_pars)
        pars1 = sc.dcp(base_pars)

        if par != 'n_contacts':
            pars0['diseases'] = sc.mergedicts(pars0['diseases'], {par: lo})
            pars1['diseases'] = sc.mergedicts(pars1['diseases'], {par: hi})
        else:
            pars0['networks'] = sc.mergedicts(pars0['networks'], {par: lo})
            pars1['networks'] = sc.mergedicts(pars1['networks'], {par: hi})

        # Run the simulations and pull out the results
        s0 = ss.Sim(pars0, label=f'{par} {par_val[0]}').run()
        s1 = ss.Sim(pars1, label=f'{par} {par_val[1]}').run()

        # Check results
        if par == 'p_death':
            v0 = s0.results.cum_deaths[-1]
            v1 = s1.results.cum_deaths[-1]
        else:
            ind = 1 if par == 'init_prev' else -1
            v0 = s0.results.sir.prevalence[ind]
            v1 = s1.results.sir.prevalence[ind]

        print(f'Checking with varying {par:10s} ... ', end='')
        assert v0 <= v1, f'Expected prevalence to be lower with {par}={lo} than with {par}={hi}, but {v0} > {v1})'
        print(f'✓ ({v0} <= {v1})')

    return s0, s1


@sc.timer()
def test_sis(do_plot=do_plot):
    sc.heading('Testing SIS dynamics')

    pars = dict(
        n_agents = n_agents,
        diseases = 'sis',
        networks = 'random'
    )
    sim = ss.Sim(pars)
    sim.run()
    if do_plot:
        sim.plot()
        sim.diseases.sis.plot()
    return sim


@sc.timer()
def test_ncd():
    sc.heading('Testing NCDs')

    ppl = ss.People(n_agents)
    ncd = ss.NCD()
    sim = ss.Sim(people=ppl, diseases=ncd, copy_inputs=False, dt=ss.years(1), analyzers='infection_log') # Since using ncd directly below
    sim.run()
    log = sim.analyzers[0].logs[0]

    assert len(log.out_edges) == log.number_of_edges()
    df = log.to_df()  # Check generation of line-list
    assert df.source.isna().all()

    plt.figure()
    plt.stackplot(
        sim.timevec,
        ncd.results.n_not_at_risk,
        ncd.results.n_at_risk - ncd.results.n_affected,
        ncd.results.n_affected,
        sim.results.new_deaths.cumsum(),
    )
    assert ncd.results.n_not_at_risk.label == 'Not at risk'
    assert ncd.results.n_affected.label == 'Affected'
    plt.legend(['Not at risk', 'At risk', 'Affected', 'Dead'])
    plt.xlabel('Year')
    plt.title('NCD')
    return sim


@sc.timer()
def test_gavi():
    sc.heading('Testing GAVI diseases')
    ss.register_modules(sse) # So we can use strings for these

    sims = sc.autolist()
    for disease in ['cholera', 'measles', 'ebola']:
        pars = dict(
            diseases = disease,
            n_agents = n_agents,
            networks = 'random',
        )
        sim = ss.Sim(pars)
        sim.run()
        sim.plot()
        sims += sim
    return sims


@sc.timer()
def test_multidisease():
    sc.heading('Testing simulating multiple diseases')

    ppl = ss.People(n_agents)
    sir1 = ss.SIR(name='sir1')
    sir2 = ss.SIR(name='sir2')

    sir1.pars.beta = {'randomnet': 0.1}
    sir2.pars.beta = {'randomnet': 0.2}
    networks = ss.RandomNet(n_contacts=ss.poisson(4))

    sim = ss.Sim(people=ppl, diseases=[sir1, sir2], networks=networks)
    sim.run()
    return sim


@sc.timer()
def test_mtct():
    sc.heading('Test mother-to-child transmission routes')
    ppl = ss.People(n_agents)
    sis = ss.SIS(
        beta = dict(
            random = [ss.permonth(0.005)]*2,
            prenatal = [ss.permonth(0.1), 0],
            breastfeeding = [ss.permonth(0.1), 0]
        )
    )
    networks = [ss.RandomNet(), ss.PrenatalNet(), ss.BreastfeedingNet()]
    demographics = ss.Pregnancy(fertility_rate=ss.freqperyear(20))
    sim = ss.Sim(dt=ss.month, people=ppl, diseases=sis, networks=networks, demographics=demographics)
    sim.run()
    sim.plot()
    return sim


@sc.timer()
def test_fetal_health(do_plot=do_plot):
    """
    Test the FetalHealth module across three scenarios:
      1. Baseline: no disease — birth weights ~3400g at term
      2. Disease: SIR infection worsens fetal outcomes via a connector
      3. Treatment: treating infected pregnant women partially reverses damage
    """
    sc.heading('Testing fetal health module')

    sim_kw = dict(
        n_agents=5_000,
        dur=ss.years(5),
        dt=ss.years(1/12),
        rand_seed=1,
    )
    def demog():
        return [
            ss.Pregnancy(fertility_rate=ss.freqperyear(30), burnin=True),
            ss.Deaths(death_rate=ss.freqperyear(10/1010*1000)),
        ]
    sir_kw = dict(dur_inf=ss.lognorm_ex(mean=ss.dur(6), std=ss.dur(1)), beta=0.1, init_prev=0.1)

    def mean_bw(sim):
        bw = sim.results.fetal_health.mean_birth_weight
        return bw[bw > 0].mean()

    # -- 1. Baseline (no disease) --
    sim_baseline = ss.Sim(
        demographics=demog(),
        custom=ss.FetalHealth(),
        networks=ss.PrenatalNet(),
        **sim_kw,
    )
    sim_baseline.run()

    # Basic sanity checks
    res_fh = sim_baseline.results.fetal_health
    total_births = res_fh.n_births.sum()
    assert total_births > 0, 'No births recorded'
    print(f'Total births: {total_births}')

    bw_baseline = mean_bw(sim_baseline)
    print(f'Mean birth weight (baseline): {bw_baseline:.0f}g')
    assert 2500 < bw_baseline < 4000, f'Mean birth weight {bw_baseline:.0f}g outside expected range'

    ga = res_fh.mean_ga_at_birth
    mean_ga = ga[ga > 0].mean()
    print(f'Mean GA at birth: {mean_ga:.1f} weeks')
    assert 36 < mean_ga < 44, f'Mean GA {mean_ga:.1f} weeks outside expected range'

    # -- 2. Disease only (no treatment) --
    sim_disease = ss.Sim(
        demographics=demog(),
        diseases=ss.SIR(**sir_kw),
        connectors=sse.fetal_infection(),
        custom=ss.FetalHealth(),
        networks=[ss.PrenatalNet(), ss.RandomNet()],
        **sim_kw,
    )
    sim_disease.run()

    bw_disease = mean_bw(sim_disease)
    print(f'Mean birth weight (disease): {bw_disease:.0f}g')
    assert bw_disease < bw_baseline, f'Disease ({bw_disease:.0f}g) should be lower than baseline ({bw_baseline:.0f}g)'

    ptb_disease = sim_disease.results.pregnancy.n_preterm.sum()
    ptb_baseline = sim_baseline.results.pregnancy.n_preterm.sum()
    print(f'Preterm births — baseline: {ptb_baseline}, disease: {ptb_disease}')
    assert ptb_disease > ptb_baseline, 'Disease scenario should have more preterm births'

    # -- 3. Disease + treatment --
    sim_treated = ss.Sim(
        demographics=demog(),
        diseases=ss.SIR(**sir_kw),
        connectors=sse.fetal_infection(),
        interventions=sse.treat_pregnant(disease='sir'),
        custom=ss.FetalHealth(),
        networks=[ss.PrenatalNet(), ss.RandomNet()],
        **sim_kw,
    )
    sim_treated.run()

    bw_treated = mean_bw(sim_treated)
    print(f'Mean birth weight (treated): {bw_treated:.0f}g')
    rtol = 0.05 # Generous tolerance for stochastic comparison
    assert bw_treated > bw_disease * (1 - rtol), f'Treated ({bw_treated:.0f}g) should be at least as high as disease-only ({bw_disease:.0f}g)'

    ptb_treated = sim_treated.results.pregnancy.n_preterm.sum()
    print(f'Preterm births — disease: {ptb_disease}, treated: {ptb_treated}')

    print(f'Summary — baseline: {bw_baseline:.0f}g, disease: {bw_disease:.0f}g, treated: {bw_treated:.0f}g')

    if do_plot:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))

        for sim, label in [(sim_baseline, 'Baseline'), (sim_disease, 'Disease'), (sim_treated, 'Treated')]:
            fh = sim.custom['fetal_health']
            born = fh.birth_weight > 0
            ax0.hist(np.asarray(fh.birth_weight[born]), bins=30, alpha=0.4, label=label, edgecolor='k')
        ax0.axvline(2500, color='r', ls='--', label='LBW threshold')
        ax0.set_xlabel('Birth weight (g)')
        ax0.set_ylabel('Count')
        ax0.set_title('Birth weight: baseline vs disease vs treated')
        ax0.legend()

        tvec = sim_baseline.timevec
        ax1.plot(tvec, sim_baseline.results.pregnancy.preterm_rate, label='Baseline')
        ax1.plot(tvec, sim_disease.results.pregnancy.preterm_rate, label='Disease')
        ax1.plot(tvec, sim_treated.results.pregnancy.preterm_rate, label='Treated')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Preterm rate')
        ax1.set_title('Preterm birth rate')
        ax1.legend()

        fig.tight_layout()

    sc.printgreen('✓ Fetal health tests passed')
    return sim_baseline, sim_disease, sim_treated


@sc.timer()
def test_congenital():
    """
    Test the generic congenital outcome framework in the base Infection class.
    Uses CongenitalDisease example and verifies outcomes are sampled and fired.
    """
    sc.heading('Testing congenital framework...')

    sim = ss.Sim(
        n_agents=5_000,
        demographics=[
            ss.Pregnancy(fertility_rate=ss.freqperyear(30), burnin=True),
            ss.Deaths(death_rate=ss.freqperyear(10/1010*1000)),
        ],
        diseases=sse.CongenitalDisease(beta=0.2, init_prev=0.2),
        dur=ss.years(5),
        dt=ss.years(1/12),
        rand_seed=1,
        networks=[ss.PrenatalNet(), ss.RandomNet()],
    )
    sim.run()

    preg = sim.demographics.pregnancy
    sb           = preg.results['stillbirths'].sum()
    nnds         = preg.results['nnds'].sum()
    births       = preg.results['births'].sum()
    n_congenital = sim.diseases[0].congenital.sum()
    print(f'  Stillbirths: {sb}, NNDs: {nnds}, Births: {births}, Congenital: {n_congenital}')

    assert sb > 0, 'Expected stillbirths from congenital disease outcomes'
    assert n_congenital > 0, 'Expected some congenital infections'
    assert births > 0, 'Expected some live births'

    sc.printgreen('✓ Congenital framework tests passed')
    return sim


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    sir   = test_sir()
    s1,s2 = test_sir_epi()
    sis   = test_sis()
    ncd   = test_ncd()
    gavi  = test_gavi()
    multi = test_multidisease()
    mtct  = test_mtct()
    fetal = test_fetal_health(do_plot=do_plot)
    cong  = test_congenital()
