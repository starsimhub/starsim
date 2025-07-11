"""
Profile LASER vs Starsim simulations
"""
import sciris as sc
import starsim as ss
import laser_example_orig as laser
import laser_starsim_full as starsim

to_run = [
    'cprofile',
    'profile',
    'time',
]

if 'cprofile' in to_run:

    with sc.cprofile(sort='selfpct') as lcprof:
        laser.run_sim(plot=False)

    with sc.cprofile(sort='selfpct') as scprof:
        starsim.run_sim(plot=False)

if 'profile' in to_run:
    kwargs = dict(plot=False)
    sprof = sc.profile(run=starsim.run_sim, follow=[ss.Sim.run], **kwargs)

if 'time' in to_run:
    with sc.timer('LASER') as ltime:
        laser.run_sim(plot=False)

    with sc.timer('Starsim') as stime:
        starsim.run_sim(plot=False)

    print(f'Ratio: {stime.total/ltime.total:n}')
