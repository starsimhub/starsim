"""
Vignette 01: "Hello world" single STI
DECISIONS:
    - sim cannot be called empty, needs modules
    - module parameters can be set via the sim's par dict or via the module

TO RESOLVE:
    - what are the default network(s)?
    - what are the default demographics?
    - can the modules be added by string? (see CK.1.1)
    - how should the people parameters be changed? via the par dict (RS.1.2.3, CK.1.2.3) or via a people constructor (RA.1.2.3)?
"""
import stisim as ss

##################
# @RomeshA
##################
# RA.1.1 Simplest
sim = ss.Sim(modules=ss.HPV())
sim = ss.Sim(modules='hpv')

# RA.1.2 Par modification
# RA.1.2.1 Changing module pars via the module
sim = ss.Sim(modules=ss.HPV(eff_condoms=0.5))
sim = ss.Sim(modules=ss.HPV(pars={'eff_condoms':0.5})) # Depends what other arguments a module might need? Not a big fan of this otherwise

# RA.1.2.2 Changing module pars via the sim par dict
sim = ss.Sim(modules=ss.HPV(), pars={'hpv':{'eff_condoms':0.5}})

# RA.1.2.3 Changing people
# I am thinking here that we could have a standard people generation function and then
# if `people` is not a `People` instance, then it is instead some kwargs that are passed
# to this function. Structurally then this would be the same mechanism people could use to implement
# some of the workflows below. It also makes it very clear which parameters are module parameters and
# which are people parameters
sim = ss.Sim(modules=ss.HPV(), people={'n':10000})

# "Hello world" single STI (HPV) with a modified number of contacts
sim = ss.Sim(modules=ss.HPV(), people = {'n': 10000, 'n_partners': {'casual': 2}})

# NB. this is specific to STIsim's people generation function i.e.
def make_people(n, n_partners, ...):
    people = ss.People(n=n)
    ...
    return people

##################
# @pausz
##################
# PSL.1.1 Simplest
sim = ss.Sim(modules=ss.hiv())

##################
# @cliffckerr
##################
# CK.1.1 Simplest
sim = ss.Sim(modules='hpv')

# CK.1.2 Par modification
# CK.1.2.1 Changing module pars via the module

# Create some genotype pars
genotype_pars = {
    16: {
        'sev_fn': dict(form='logf2', k=0.25, x_infl=0, ttc=30)
    }
}
hpv = ss.HPV(genotype_pars=genotype_pars)
sim = ss.Sim(modules=hpv)

# CK.1.2.2 Changing module pars via the sim par dict
# See CK.2.2.2

# CK.1.2.3 Changing people
pars = dict(n_agents=5e3)
sim = ss.Sim(pars=pars)


##################
# @robynstuart
##################
# RS.1.1 Simplest
sim = ss.Sim(modules=ss.syphylis())

# RS.1.2 Par modification
# RS.1.2.1: via the sim's par dict
pars = dict(n_agents=100, syphilis=dict(init_prev=0.02))
sim = ss.Sim(pars=pars, modules=ss.syphilis())  # Ensure that any module keys in the parameters are added correctly

# RS.1.2.2: via the module par dict
syph_pars = dict(init_prev=0.02)
syph = ss.syphilis(pars=syph_pars)
sim = ss.Sim(modules=syph)

# RS.1.2.3 Changing people
sim = ss.Sim(people=ss.People(100), modules=ss.syphilis())
sim = ss.Sim(pars=dict(n_agents=100), modules=ss.syphilis())

