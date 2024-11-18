"""
Test calibration
"""

#%% Imports and settings
import sciris as sc
import starsim as ss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import init_notebook_plotting, render

do_plot = 1
do_save = 0
n_agents = 2e3

#%% Helper functions

def make_sim():
    sir = ss.SIR(
        beta = ss.beta(0.9),
        dur_inf = ss.lognorm_ex(mean=ss.dur(6)),
        init_prev = ss.bernoulli(0.01),
    )

    #deaths = ss.Deaths(death_rate=15)
    #births = ss.Births(birth_rate=15)

    random = ss.RandomNet(n_contacts=ss.poisson(4))

    sim = ss.Sim(
        dt = 1,
        unit = 'day',
        n_agents = n_agents,
        #total_pop = 9980999,
        start = sc.date('2024-01-01'),
        stop = sc.date('2024-01-31'),
        diseases = sir,
        networks = random,
        #demographics = [deaths, births],
    )

    return sim


def build_sim(sim, calib_pars, **kwargs):
    """ Modify the base simulation by applying calib_pars """

    for k, v in calib_pars.items():
        if k == 'beta':
            sim.diseases.sir.pars['beta'] = ss.beta(v)
        elif k == 'dur_inf':
            sim.diseases.sir.pars['dur_inf'] = ss.lognorm_ex(mean=ss.dur(v)), #ss.dur(v)
        elif k == 'n_contacts':
            sim.networks.randomnet.pars.n_contacts = v # Typically a Poisson distribution, but this should set the distribution parameter value appropriately
        else:
            sim.pars[k] = v # Assume sim pars

    return sim

def eval_sim(pars):
    sim = make_sim()
    sim.init()
    sim = build_sim(sim, pars)
    sim.run()
    #print('pars:', pars, ' --> Final prevalence:', sim.results.sir.prevalence[-1])
    fig = sim.plot()
    fig.suptitle(pars)
    fig.subplots_adjust(top=0.9)
    plt.show()

    return dict(
        prevalence_error = ((sim.results.sir.prevalence[-1] - 0.10)**2, None),
        prevalence = (sim.results.sir.prevalence[-1], None),
    )


#%% Define the tests
def test_calibration(do_plot=False):
    sc.heading('Testing calibration')

    # Define the calibration parameters
    calib_pars = [
        dict(name='beta', type='range', bounds=[0.01, 1.0], value_type='float', log_scale=True),
        dict(name='dur_inf', type='range', bounds=[1, 60], value_type='float', log_scale=False),
        #dict(name='init_prev', type='range', bounds=[0.01, 0.30], value_type='float', log_scale=False),
        dict(name='n_contacts', type='range', bounds=[2, 10], value_type='int', log_scale=False),
    ]

    best_pars, values, exp, model = optimize(
        experiment_name = 'starsim',
        parameters = calib_pars,
        evaluation_function = eval_sim,
        objective_name = 'prevalence_error',
        minimize = True,
        parameter_constraints = None,
        outcome_constraints = None,
        total_trials = 10,
        arms_per_trial = 3,
    )

    return best_pars, values, exp, model


#%% Run as a script
if __name__ == '__main__':

    T = sc.timer()
    do_plot = True

    best_pars, values, exp, model = test_calibration(do_plot=do_plot)

    print('best_pars:', best_pars)
    print('values:', values)
    print('exp:', exp)
    print('model:', model)

    render(plot_contour(model=model, param_x='beta', param_y='init_prev', metric_name='prevalence'))

    # `plot_single_method` expects a 2-d array of means, because it expects to average means from multiple
    # optimization runs, so we wrap out best objectives array in another array.

    for trial in exp.trials.values():
        print(trial)
        print(dir(trial))
        print(f"Trial {trial.index} with parameters {trial.arm.parameters} "
              f"has objective {trial.objective_mean}.") 

    best_objectives = np.array(
        [[trial.objective_mean for trial in exp.trials.values()]]
    )
    best_objective_plot = optimization_trace_single_method(
        y = np.minimum.accumulate(best_objectives, axis=1),
        optimum = 0.10, #hartmann6.fmin,
        title = "Model performance vs. # of iterations",
        ylabel = "Prevalence",
    )
    render(best_objective_plot)

    plt.show()

    T.toc()
