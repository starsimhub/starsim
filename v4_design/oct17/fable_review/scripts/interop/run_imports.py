import warnings, sys, traceback
import sciris as sc
import starsim as ss
ex = sc.path('/home/cliffk/idm/lingua_franca_design/examples')
out = sc.path('/tmp/claude-1000/-home-cliffk-idm-ssv4/b3102841-11e6-46bf-942c-f372bd4682c4/scratchpad/interop')
cases = {
  'emulsion': lambda: ss.load(ex/'emulsion/sir/original/compart_SIR.yaml', verbose=True),
  'odin':     lambda: ss.load(ex/'odin/seir_array/original/seir_array_odin.R', format='odin', stop=365, verbose=True),
  'flepimop': lambda: ss.load(ex/'flepimop/seir_2pop/original/config_sample_2pop.yml', verbose=True),
  'atomica':  lambda: ss.load(ex/'atomica/sir/original/sir_framework.xlsx', format='atomica', databook=ex/'atomica/sir/original/sir_databook.xlsx', dt=1, verbose=True),
  'siminf':   lambda: ss.load(ex/'siminf/sir_herds/original/events_SIR.R', format='siminf', u0=ex/'siminf/sir_herds/reference/u0.csv', events=ex/'siminf/sir_herds/reference/events.csv.gz', stop=20, verbose=True),
}
for name, fn in cases.items():
    print(f'\n################ {name}')
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            sim = fn()
        for x in w: print('WARNING:', str(x.message)[:1500])
        code = sim.to_code(out/f'code_{name}.py')
        print(f'--- to_code: {len(code.splitlines())} lines')
        print(code)
        T = sc.timer()
        sim.run(verbose=0)
        print(f'--- ran in {T.toc(output=True):.2f}s; n_states={len(sim.diseases[0].graph.states)}')
    except Exception:
        traceback.print_exc()
