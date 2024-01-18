"""
Vignette 02: "Hello world" multiple STI
DECISIONS:
    - all examples follow similar syntax for specifying more than one disease
TO RESOLVE:
    - what are the default connectors?
"""

import stisim as ss

##################
# @RomeshA
##################
# RA.2.1 Simplest
sim = ss.Sim(modules=[ss.HPV(), ss.Gonorrhea()])

# RA.2.2.1 Changing module pars via the module
hpv_pars = {'eff_condoms':0.5}
hpv = ss.HPV(**hpv_pars)
gonorrhea_pars = {'eff_condoms':0.5}
gonorrhea = ss.Gonorrhea(**gonorrhea_pars)
sim = ss.Sim(modules=[hpv, gonorrhea])

# RA.2.2.2 Changing module pars via the sim par dict
# See RA.1.2.2 in v01_single.py


##################
# @cliffckerr
##################
# CK.2.1 Simplest
sim = ss.Sim(modules=['hiv', 'gonorrhea'])

# CK.2.2.1 Changing module pars via the module
sti_pars = dict(
    n_agents = 10e3,
    start = 1970,
    end = 2020,
)

hiv_pars = dict(
    cd4_pars = dict(
        initial = 500,
        rate = 50,
    ),
)

# Create modules first
hiv = ss.HIV(hiv_pars)
gon = ss.Gonorrhea(transm2f=4.0)

# Create and run the sim
sim = ss.Sim(sti_pars, modules=[hiv, gon])
sim.initialize() # Happens automatically as part of sim.run(), but can call explicitly
assert id(sim.pars.hiv) == id(sim.hiv.pars) # These are the same object after initialization

# CK.2.2.2 Changing module pars via the sim par dict
pars = dict(
    n_agents = 10e3,
    start = 1970,
    end = 2020,
    hiv = dict(
        cd4_pars = dict(
            initial = 500,
            rate = 50,
        ),
    ),
    gonorrhea = dict(
        transm2f = 4.0,
    ),
)

sim = ss.Sim(pars, modules=['hiv', 'gonorrhea'])

##################
# @robynstuart
##################
# RS.2.1 Simplest
sim = ss.Sim(modules=[ss.syphylis(), ss.hiv()])

# RS.2.2.1 Changing module pars via the module
syph_pars = dict(init_prev=0.02)
hiv_pars = dict(init_prev=0.02)
syph = ss.syphilis(pars=syph_pars)
hiv = ss.hiv(pars=hiv_pars)
sim = ss.Sim(modules=[syph, hiv])
sim.initialize()  # Adds modules and parameters
# Parameters can be read like a conditional probability, i.e. prob(transmitting hiv | syphilis infection) = 2
sim['hiv']['syphilis']['rel_trans'] = 4


# RS.2.2.2 Changing module pars via the sim par dict
# See RS.1.2.2