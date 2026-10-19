import time, ast, pathlib, types
t0 = time.perf_counter()
import starsim as ss
t1 = time.perf_counter()
print(f'import starsim: {t1-t0:.2f} s')
pub = [n for n in dir(ss) if not n.startswith('_')]
mods = [n for n in pub if isinstance(getattr(ss, n), types.ModuleType)]
classes = [n for n in pub if isinstance(getattr(ss, n), type)]
funcs = [n for n in pub if callable(getattr(ss, n)) and not isinstance(getattr(ss, n), type)]
other = [n for n in pub if n not in mods and n not in classes and n not in funcs]
print(f'public names: {len(pub)}; submodules {len(mods)}; classes {len(classes)}; functions {len(funcs)}; other {len(other)}')
print('submodules:', mods)
print('other:', other)
# where do they come from
import collections
src = collections.Counter()
bymod = collections.defaultdict(list)
for n in classes + funcs:
    m = getattr(getattr(ss, n), '__module__', '?')
    src[m] += 1
    bymod[m].append(n)
for m, c in sorted(src.items(), key=lambda x: -x[1]):
    print(f'{c:4d} {m}: {", ".join(sorted(bymod[m]))}')

# method lists for big classes
R = pathlib.Path('/home/cliffk/idm/ssv4/starsim')
for fname, cls in [('sim.py','Sim'), ('solvers.py','Engine'), ('solvers.py','Compartmental'), ('solvers.py','Agents'), ('transitions.py','Graph'), ('solvers.py','Mixing')]:
    tree = ast.parse((R/fname).read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            print(f'\n=== {fname}:{cls} ===')
            rows = [(m.end_lineno-m.lineno+1, m.name, m.lineno) for m in node.body if isinstance(m, ast.FunctionDef)]
            for n, name, ln in rows:
                print(f'{n:4d} L{ln:<5d} {name}')
