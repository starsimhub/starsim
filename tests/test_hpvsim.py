'''
Simple tests
'''

#%% Imports and settings
import sciris as sc
import numpy as np
import stisim as ss
import pylab as pl

#%% Define the tests


def test_hpv():
    sc.heading('Testing out people and pop updating with hpvsim')

    # Create a simulation with demographics from a given location
    geostruct = 1
    pars = dict(
        start=1980,
        end=2020,
        location='nigeria',
        geostructure=geostruct
    )


    sim = ss.Sim(pars=pars, modules=[ss.HPV])
    sim.run()

    return



#%% Run as a script
if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    sim = test_hpv()

    sc.toc(T)
    print('Done.')