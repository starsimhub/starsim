"""
Vignette 03: "Hello world" results output
DECISIONS:
    - sim.plot() should do something sensible based on which modules are present
    - sim.save_csv() likewise
    - sim.plot() is equivalent to sim.plot('default')
    - sim.hiv.plot() is equivalent to sim.plot('hiv')
    - sim.plot_result('hiv.prevalence') is equivalent to plt.plot(sim.t, sim.results.hiv['prevalence'])
    - make a result_splitter as in RS.3.2 to get coinfection results

TO RESOLVE:
    - how can individual disease results be accessed?
    - what are the default plots for the sim? (i.e. from ss.Sim(modules='hpv').run().plot() )
    - what are the default plots for each module?
    - what style should we use for naming results?
    - what should come from sim.summary()?
"""

import stisim as ss

##################
# @RomeshA
##################
sim = ss.Sim(modules=ss.HPV())
sim.run()
sim.plot()  # Do something sensible based on which modules are present
plt.plot(sim.t, sim.results.hpv['new_infections'])
sim.save_csv()


##################
# @cliffckerr
##################
# CK.3.1 Plots for a single module
ss.Sim(modules='hpv').run().plot()

# CK.3.2 Plot for multiple modules
sim = ss.Sim(modules=['hiv', 'gonorrhea'])
sim.run()
sim.plot('default', 'hiv', 'gonorrhea')  # Plot default as well as disease-specific plots
print(sim.results.hiv) # All HIV results
print(sim.results['hiv']['incidence']) # A single result, can also be accessed as dict keys
print(sim.results.hiv_gonorrhea_coinfections) # Result populated by the HIV-gonorrhea connector
sim.plot() # Equivalent to sim.plot('default')
sim.hiv.plot() # Equivalent to sim.plot('hiv')
sim.gonorrhea.plot() # Equivalent to sim.plot('gonorrhea')


##################
# @robynstuart
##################
# RS.3.1 Plots for a single module
sim = ss.Sim(modules=ss.hiv())
sim.run()
ss.plot()  # Default plots: prevalence, deaths, new infections
sim.plot_result('hiv.prevalence')  # Alternative method

# RS.3.2 Plot for multiple modules
# By default, I don't think we will create results of the numbers of people with dual infection.
# However, we can make a result splitter to divide standard results by boolean people attributes.
results_by_hiv = ss.result_splitter(
    results=['syphilis.prev'],  # If blank, throw error
    by=['hiv.infected', 'hiv.susceptible', 'hiv.treated'],  # If blank, throw error?
    years=2020  # If blank, do all years
)
results_by_sex = ss.result_splitter(
    results=['syphilis.prev'],
    by=['female', 'male'],
)
results_by_pregnancy = ss.result_splitter(
    results=['syphilis.prev'],
    by=['pregnant'],
)
# Could maybe also allow people to pass in a function that returns a boolean array for people, e.g.
# NB I don't really like this. Another option could be:
#   class ppl(ss.People()):
#       @property
#       def young(self):
#           return self.age < 21
#   sim = cv.Sim(people=ppl())

youth = lambda sim: ss.true(sim.people.age < 21)
results_by_age = ss.result_splitter(
    results=['syphilis.new_infections'],
    by=[youth],  # or similar
)
sim.run(analyzers=[results_by_hiv, results_by_sex, results_by_pregnancy, results_by_age])

res = sim.get_analyzer().results
res.syphylis.prev.hiv.infected
res.syphylis.prev.hiv.susceptible
res.syphylis.prev.hiv.treated
res.syphylis.prev.youth  # TBC
