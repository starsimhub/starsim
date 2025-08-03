"""
Simple Starsim demo script
"""
import starsim as ss

sim = ss.Sim(diseases='sir', networks='random')
sim.run()
sim.plot()