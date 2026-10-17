"""SEIR with an Erlang dwell time, each way."""
import starsim as ss, numpy as np, sciris as sc

print('=== string form, source forgotten')
seir_str = ss.Disease(
    infection = ss.transmission('S -> E', beta=0.3),
    latency   = ss.progression('E -> I', dur=ss.erlang(k=3, mean=ss.days(5))),
    recovery  = ss.progression('I -> R', dur=ss.days(10)),
    init_prev = 0.01, name='seir',
)
seir_str.describe()
sim = ss.Sim(seir_str, stop=60, verbose=0).run()
print('peak I with source=E (wrong guess):', sim.results.seir.n_I.max())

print('\n=== string form, source given')
seir_str2 = ss.Disease(
    infection = ss.transmission('S -> E', beta=0.3, source='I'),
    latency   = ss.progression('E -> I', dur=ss.erlang(k=3, mean=ss.days(5))),
    recovery  = ss.progression('I -> R', dur=ss.days(10)),
    init_prev = 0.01, name='seir',
)
sim2 = ss.Sim(seir_str2, stop=60, verbose=0).run()
print('peak I with source=I:', sim2.results.seir.n_I.max())

print('\n=== operator form with erlang and stages=')
S, E, I, R = ss.states(4)
infection = S >> E @ 0.3 * I
latency   = E >> I @ ss.erlang(k=3, mean=ss.days(5))
recovery  = I >> R @ ss.days(10)
seir_op = ss.Disease(infection, latency, recovery, init_prev=0.01, name='seir')
seir_op.describe()
seir_op.equations()
print(seir_op.to_json()['transitions'][1])
sim3 = ss.Sim(seir_op, stop=60, verbose=0).run()
print('peak I operator form:', sim3.results.seir.n_I.max(), ' same as string+source?', np.isclose(sim3.results.seir.n_I.max(), sim2.results.seir.n_I.max()))

print('\n=== stages=3 instead of erlang')
S, E, I, R = ss.states(4)
seir_st = ss.Disease(infection=S >> E @ 0.3 * I, latency=ss.Transition(E, I, dur=ss.days(5), stages=3), recovery=I >> R @ ss.days(10), init_prev=0.01, name='seir')
sim4 = ss.Sim(seir_st, stop=60, verbose=0).run()
print('peak I stages=3:', sim4.results.seir.n_I.max(), ' same as erlang?', np.isclose(sim4.results.seir.n_I.max(), sim3.results.seir.n_I.max()))
print('to_json latency:', seir_st.to_json()['transitions'][1])

print('\n=== where does init_prev put people? (spec: E)')
sim3b = ss.Sim(sc.dcp(seir_op), stop=10, verbose=0); sim3b.init()
r = sim3b.diseases.seir
print('n_E[0], n_I[0]:', sim3b.results.seir.n_E[0] if len(sim3b.results.seir.n_E) else None)
sim3b.run()
print('after run n_E[0], n_I[0]:', sim3b.results.seir.n_E[0], sim3b.results.seir.n_I[0])

print('\n=== lognormal dwell under ode: Erlang approximation note')
S, E, I, R = ss.states(4)
seir_ln = ss.Disease(infection=S >> E @ 0.3 * I, latency=E >> I @ ss.lognorm_ex(mean=5, std=1.5, unit='day'), recovery=I >> R @ ss.days(10), init_prev=0.01, name='seir')
sim5 = ss.Sim(seir_ln, stop=60, verbose=1).run()
print('notes:', sim5.notes)
print('\n=== as agents (n_agents=2000)')
sim6 = ss.Sim(sc.dcp(seir_op), stop=60, n_agents=2000, verbose=0).run()
print('solver:', sim6.pars.solver, 'peak I:', sim6.results.seir.n_I.max())
sim6.diseases.seir.describe()
