import sciris as sc, starsim as ss
ex = sc.path('/home/cliffk/idm/lingua_franca_design/examples')
sim = ss.load(ex/'emulsion/sir/original/compart_SIR.yaml', verbose=0)
sim.init()
sim.describe()
print('---- sim.imported keys:', list(sim.imported.keys()))
print(sim.imported.choices)
print('---- odin describe')
sim2 = ss.load(ex/'odin/seir_array/original/seir_array_odin.R', format='odin', stop=365, verbose=0); sim2.init(); sim2.describe()
print('---- ss.diff import vs hand')
import importlib.util, sys
spec = importlib.util.spec_from_file_location('m', '/home/cliffk/idm/lingua_franca_design/vignettes/emulsion/sir/model.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ss.diff(sim, m.compartment)
