import time, sys
t0=time.time(); import starsim as ss; t1=time.time()
print(f'import starsim: {t1-t0:.2f}s')
mods=[m for m in sys.modules if m.startswith('starsim.interop')]
print('interop modules loaded at import:', len(mods))
print('scipy.optimize loaded:', 'scipy.optimize' in sys.modules, ' pandas:', 'pandas' in sys.modules, ' openpyxl:', 'openpyxl' in sys.modules)
print(ss.interop.formats.keys())
print(ss.interop.exporters.keys())
