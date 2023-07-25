"""
Vignette 05: Scenarios
The implementations below have some features in common, but each focuses on a different aspect
    - RA focus: iterating over par values
    - PSL focus: branching a sim -- running to a certain point, then brancing into multiple scenarios
    - RS focus: replication of covasim behavior

NB, all of us agreed that we don't use scenarios much. Possibly this should be deprioritized?
"""
import stisim as ss

##################
# @RomeshA
##################
# This is presumably not the user-friendly way of doing it, but is how I would envisage writing my own scripts

def scen_sim(scen_par=value, seed=0):
    ... # apply config e.g., in modules, or in people
    sim = ss.Sim()
    sim.run()
    return sim

sim1 = scen_sim(beta=0.1)
sim2 = scen_sim(beta=0.2)
sim3 = scen_sim(beta=0.3)

# Something more standard?
scen = ss.Scenario(sim, 'hpv.eff_condoms'=[0.1,0.2,0.3])
sims = scen.run()
# But in this setup I'm not sure when the parameter value is consumed. It would have to be after
# the module has been instantiated, but what if the module's `__init__` method did something that depended
# on the parameter value? Would it be too late to apply the scenario? Does the Scenario instead need a sim generation
# function? (that would be too complex for the simple UI)



##################
# @pausz
##################


##################################### Case 1: Branching using multisims ################################################
# Step 0: Run multiple simulations with different seeds
pars = {'start': 1950,
        'end': 2022}
sim = ss.Sim(pars=pars, modules=ss.hiv())
msim = ss.MultiSim(sim)
msim.run(n_runs=100)

# Step 1: estimate average behaviour
median_sim = msim.median()
# Save last 10 years of data to disk, doesn't modify median_sim
median_sim.save(horizon=10)

# Step 2a: use the median sim as history for the next simulation, with a horizon of 10 years (ie, use the last 10 years of
# the "median simulation" instead of using 'burnin' years to avoid trnasient dyanamics).
# This new sim will have to set the "transplanted state": ie, initialise the new simulation's state using the saved
# state from the source simulation. This involves (re)setting the population's attributes, disease states, and any other
# relevant parameters to match the saved state.
pars = {'end': 2100.0,
        'horizon': 10.0}
s1 = ss.Sim(history=median_sim, pars=pars,  modules=ss.syphilis(), label='Syphilis outbreak')

# OR we could do
# Discard everything older than the last 10 years of simulation
median_sim.keep(horizon=10.0)
# Step 2b: transplant history
s1 = ss.Sim(history=median_sim, modules=ss.syphilis(), label='Syphilis outbreak')

# Step 3: Progress forward in time from the end of median_sim onward,
# introduce changes to explore alternative scenarios (branching).
rel_betas = [0.9, 1.0]
msims = []
for beta in rel_betas:
    s1.initialize()
    s1.pars.syphilis.rel_beta = beta
    msim = ss.MultiSim(s1)
    msim.run(n_runs=20)
    msim.mean()
    msims.append(msim)

merged = ss.MultiSim.merge(msims)
merged.plot(color_by_sim=True)

##################################### Case 2: Branching using scenarios ################################################
# Step 0: Define interventions
campaign_a = ss.ther_vaccine(eligibility={'female': {'min_age': 30, 'max_age': 55}}, prob=0.8, start_year=2020)
campaign_b = ss.prev_vaccine(eligibility={'female': {'min_age': 15, 'max_age': 25},
                                          'male': {'min_age': 10, 'max_age': 20}}, prob=0.8, start_year=2010)

screening_campaign = ss.screening(prob=0.2, start_year=2010)

# Step 1: Define parameters that are common to all simulations
basepars = {'beta': [0.9],
            'interventions': [screening_campaign]}

# Step 2: Define specific configurations for scenarios
scenarios = {'campaign_a': {
    'name': 'Therapeutic Campaign',
    'pars': {
        'interventions': [campaign_a]
    }
},
    'campaign_b': {
        'name': 'Preventative Campaign',
        'pars': {
            'interventions': [campaign_b],
        }
    },
}

# Step 3: Each scenario branches from median_sim
scens = ss.Scenarios(history_sim=median_sim, basepars=basepars, scenarios=scenarios)
scens.run()


##################
# @robynstuart
##################
import stisim as ss
import pandas as pd

# Case 1: multisim with stochastic variation
sim = ss.Sim(modules=ss.hiv())
msim = ss.MultiSim(sim)
msim.run(n_runs=5)
msim.plot()

# Case 2: average over stochastic variation
msim.mean()
msim.plot_result('hiv.new_infections')

# Case 3: run simple scenarios
s1 = ss.Sim(modules=ss.syphilis(), label='Without HIV')
s2 = ss.Sim(modules=[ss.hiv(), ss.syphilis()], label='With HIV')
ss.parallel(s1, s2).plot(['syphilis.cum_deaths', 'syphilis.cum_infections'])


# Case 4: run combination msims & scenarios
module_dict = {'Without HIV': ss.syphilis(), 'With HIV': [ss.hiv(), ss.syphilis()]}
syph_rel_betas = [0.9, 1, 1.1]

msims = []
for label, modules in module_dict.items():
    sims = []
    for beta in syph_rel_betas:
        sim = ss.Sim(modules=modules, label=label)
        sim.initialize()
        sim.pars.syphilis.rel_beta = beta  # Could also allow this to be supplied directly
        sims.append(sim)
    msim = ss.MultiSim(sims)
    msim.run()
    msim.mean()
    msims.append(msim)

merged = ss.MultiSim.merge(msims)
merged.plot(color_by_sim=True)


# Case 5: scenarios
basepars = {'modules': [ss.hiv(), ss.syphilis]}

products_dx = pd.read_csv('products_dx.csv')
products_tx = pd.read_csv('products_tx.csv')
syph_dx = ss.Product(efficacy=products_dx[products_dx.name == 'syph_dx'])
syph_tx = ss.Product(efficacy=products_tx[products_tx.name == 'syph_tx'])
syph_screening = ss.screening(products=syph_dx, prob=0.3, start_year=2015)
syph_treatment = ss.treatment(eligibility='syph.diagnosed', products=syph_tx, prob=0.8, start_year=2015)

# Settings for each scenario
scenarios = {'baseline': {
              'name': 'Baseline',
              'pars': {}
              },
            'with_treatment': {
              'name': 'With treatment',
              'pars': {
                  'interventions': [syph_screening, syph_treatment],
                  }
              },
             }

# Run and plot the scenarios
scens = ss.Scenarios(basepars=basepars, scenarios=scenarios)
scens.run()
scens.plot()


