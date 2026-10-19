"""Check what the v3 and v4 forms of each family can and cannot do."""
import sciris as sc
import starsim as ss

def attempt(label, fn):
    try:
        out = fn()
        print(f'[ok]  {label}: {out}')
    except Exception as E:
        print(f'[ERR] {label}: {type(E).__name__}: {str(E)[:300]}')

# 1. v3 SIR class under the new solvers
attempt('v3 ss.SIR as agents', lambda: ss.Sim(diseases=ss.SIR(), networks='random', n_agents=500, dur=2, verbose=0).run().summarize()['sir_cum_infections'])
attempt('v3 ss.SIR with solver=ode', lambda: ss.Sim(diseases=ss.SIR(), networks='random', n_agents=500, dur=2, solver='ode', verbose=0).run())
# v4 declared SIR under all solvers
def declared():
    sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.3), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir')
    return sir
attempt('v4 Disease.solvers()', lambda: declared().solvers())
attempt('v4 Disease with v3 RandomNet', lambda: ss.Sim(declared(), networks=ss.RandomNet(n_contacts=5), n_agents=500, stop=20, verbose=0).run().results.sir.n_I[-1])
attempt('v4 Disease with v3 MixingPool, ode', lambda: ss.Sim(declared(), networks=ss.MixingPool(n_contacts=10), solver='ode', total_pop=1000, stop=20, verbose=0).run().results.sir.n_I[-1])
attempt('v4 Disease with v3 Births/Deaths, ode', lambda: ss.Sim(declared(), demographics=[ss.Births(birth_rate=20), ss.Deaths(death_rate=10)], solver='ode', total_pop=1000, stop=20, verbose=0).run().results.n_alive[-1])
attempt('v4 Disease with v3 seasonality connector, ode', lambda: ss.Sim(declared(), connectors=ss.seasonality(diseases='sir'), solver='ode', total_pop=1000, stop=20, verbose=0).run().results.sir.n_I[-1])
attempt('v4 Disease with v3 routine_vx, ode', lambda: ss.Sim(declared(), interventions=ss.routine_vx(product=ss.simple_vx(efficacy=0.9), prob=0.5, start_year=5), solver='ode', total_pop=1000, stop=20, verbose=0).run().results.sir.n_I[-1])
attempt('v4 Disease with v3 routine_vx, abm', lambda: ss.Sim(declared(), interventions=ss.routine_vx(product=ss.simple_vx(efficacy=0.9), prob=0.5, start_year=5), solver='abm', n_agents=500, stop=20, verbose=0).run().results.sir.n_I[-1])
attempt('v3 SIR with v4 vaccinate', lambda: ss.Sim(diseases=ss.SIR(), networks='random', interventions=ss.vaccinate(ss.Vaccine(efficacy=0.9), coverage=0.5), n_agents=500, dur=2, verbose=0).run().results.sir.cum_infections[-1])
attempt('v3 SIR with v4 modify', lambda: ss.Sim(diseases=ss.SIR(), networks='random', connectors=ss.modify('sir.pars.beta', multiply=0.5), n_agents=500, dur=2, verbose=0).run().results.sir.cum_infections[-1])
attempt('ss.Vx() is ss.Vaccine', lambda: type(ss.Vx(efficacy=0.5)).__name__)
attempt('sim.calibrate on v3 SIR (no bounds)', lambda: ss.Sim(diseases=ss.SIR(), networks='random', n_agents=200, dur=1, verbose=0).calibrate())
print('solver names:', ss.solver_names)
