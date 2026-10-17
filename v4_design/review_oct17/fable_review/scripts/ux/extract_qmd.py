"""Extract the python cells of a .qmd into a script; skip IPython magics and hide-cell pip installs."""
import sys, re, pathlib
src = pathlib.Path(sys.argv[1]); dst = pathlib.Path(sys.argv[2])
text = src.read_text()
cells = re.findall(r'```\{python\}\n(.*?)```', text, flags=re.S)
out = ['import matplotlib; matplotlib.use("agg")', 'import pylab as _pl', '_pl.show = lambda *a, **k: None']
for i, c in enumerate(cells):
    lines = [l for l in c.splitlines() if not l.strip().startswith(('%', '!', '#|'))]
    out.append(f'\n# ---- cell {i+1} ----')
    out.append('\n'.join(lines))
    out.append('_pl.close("all")')
dst.write_text('\n'.join(out) + '\n')
print(src.name, len(cells), 'cells')
