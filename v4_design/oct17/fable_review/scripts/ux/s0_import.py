import time; t=time.time()
import starsim as ss
print('import', round(time.time()-t,2), ss.__version__)
print([n for n in dir(ss) if not n.startswith('_')])
