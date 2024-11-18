"""
Test calibration
"""

#%% Imports and settings
import sciris as sc
import starsim as ss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from ax.plot.contour import plot_contour
#from ax.plot.trace import optimization_trace_single_method
#from ax.service.managed_loop import optimize
#from ax.utils.notebook.plotting import init_notebook_plotting, render

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.notebook.plotting import init_notebook_plotting, render

from ax.modelbridge.cross_validation import cross_validate
from ax.plot.contour import interact_contour
from ax.plot.diagnostic import interact_cross_validation
from ax.plot.scatter import interact_fitted, plot_objective_vs_constraints, tile_fitted
from ax.plot.slice import plot_slice
from ax.service.utils.report_utils import exp_to_df

do_plot = 1
do_save = 0
n_agents = [2e3, 25_000][1]

ax_client = AxClient(enforce_sequential_optimization=False)

#%% Helper functions

def make_sim(calib_pars):
    sir = ss.SIR(
        beta = ss.beta( calib_pars.get('beta', 0.9) ),
        dur_inf = ss.lognorm_ex(mean=ss.dur( calib_pars.get('dur_inf', 6))),
        init_prev = ss.bernoulli(0.01),
    )

    #deaths = ss.Deaths(death_rate=15)
    #births = ss.Births(birth_rate=15)

    random = ss.RandomNet(n_contacts=ss.poisson(calib_pars.get('n_contacts', 4)))

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
        rand_seed = np.random.randint(1e6),
    )

    return sim


def eval_sim(pars):
    sim = make_sim(pars)
    sim.run()

    if False:
        fig = sim.plot()
        fig.suptitle(pars)
        fig.subplots_adjust(top=0.9)
        plt.show()

    return dict(
        prevalence_error = (np.abs(sim.results.sir.prevalence[-1] - 0.20), None),
        #prevalence = (sim.results.sir.prevalence[-1], None),
    )

#%% Define the tests
def test_calibration(do_plot=False):
    sc.heading('Testing calibration')

    # Define the calibration parameters
    calib_pars = [
        dict(name='beta', type='range', bounds=[0.005, 0.1], value_type='float', log_scale=True),
        #dict(name='dur_inf', type='range', bounds=[1, 120], value_type='float', log_scale=False),
        dict(name='dur_inf', type='fixed', value=60, value_type='float'),
        #dict(name='init_prev', type='range', bounds=[0.01, 0.30], value_type='float', log_scale=False),
        dict(name='n_contacts', type='range', bounds=[1, 10], value_type='int', log_scale=False),
    ]

    ax_client.create_experiment(
        name = 'starsim test',
        parameters = calib_pars,
        objectives={'prevalence_error': ObjectiveProperties(minimize=True)},
        parameter_constraints = None,
        outcome_constraints = None,
        choose_generation_strategy_kwargs={"max_parallelism_override": 25},
    )

    print('Max parallelism:', ax_client.get_max_parallelism()) # Seems to require manual specification of generation_strategy

    for i in range(5):
        print('THINKING...')
        trial_index_to_param, idk = ax_client.get_next_trials(max_trials=1_000)

        print('STEP', i, len(trial_index_to_param))

        # Does NOT work to complete_trial in the parallel loop
        results = sc.parallelize(eval_sim, iterkwargs=dict(pars=list(trial_index_to_param.values())), serial=False)
        for trial_index, result in zip(trial_index_to_param.keys(), results):
            ax_client.complete_trial(trial_index=trial_index, raw_data=result)

        print(exp_to_df(ax_client.experiment))


    best_pars, values = ax_client.get_best_parameters()

    return best_pars, values#, exp, model


#%% Run as a script
if __name__ == '__main__':


    T = sc.timer()
    do_plot = True

    best_pars, values = test_calibration(do_plot=do_plot)

    sim = make_sim(best_pars)
    sim.run()
    sim.plot()

    print('best_pars:', best_pars)
    print('values:', values)

    #render(ax_client.get_contour_plot(param_x='beta', param_y='dur_inf', metric_name='prevalence_error'))
    render(ax_client.get_optimization_trace(objective_optimum=0))

    model = ax_client.generation_strategy.model
    render(interact_contour(model=model, metric_name='prevalence_error'))

    cv_results = cross_validate(model)
    render(interact_cross_validation(cv_results))

    render(plot_slice(model, 'beta', 'prevalence_error'))
    render(plot_slice(model, 'n_contacts', 'prevalence_error'))

    render(interact_fitted(model, rel=False))

    plt.show()

    T.toc()
