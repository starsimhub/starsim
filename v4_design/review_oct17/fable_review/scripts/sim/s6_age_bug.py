import starsim as ss, sciris as sc, traceback
sir = ss.Disease(infection=ss.transmission('S -> I', beta=0.05), recovery=ss.progression('I -> R', dur=10), init_prev=0.01, name='sir')
ppl = ss.People(1)
try:
    ss.Sim(sir, stratify=dict(age=[0,15,65]), connectors=ss.modify(sir.pars.beta, multiply=0.5, where=ppl.age < 10), stop=10, verbose=0).run()
except Exception:
    traceback.print_exc()
sc.heading('n_agents=None directly')
try:
    ss.Sim(sir, n_agents=None, stop=10, verbose=0).run()
except Exception as E:
    print('ERROR', type(E).__name__, E)
