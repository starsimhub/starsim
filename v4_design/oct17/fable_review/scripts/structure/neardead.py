"""Top-level public defs in starsim/*.py: where are they referenced? Flag those used only in their own file + tests/docs (not by other core modules, library, examples, or vignettes)."""
import ast, re, pathlib
R = pathlib.Path('/home/cliffk/idm/ssv4')
V = pathlib.Path('/home/cliffk/idm/lingua_franca_design/vignettes')
core = sorted((R/'starsim').glob('*.py')) + sorted((R/'starsim/interop').glob('*.py'))
areas = {
    'core': core,
    'lib': list((R/'starsim/library').rglob('*.py')),
    'tests': list((R/'tests').glob('*.py')),
    'docs': list((R/'docs').rglob('*.qmd')) + list((R/'docs').rglob('*.py')),
    'ex': list((R/'v4_design/examples').rglob('*.py')),
    'vig': list(V.rglob('*.py')),
}
texts = {a: {p: p.read_text(errors='ignore') for p in ps} for a, ps in areas.items()}
rows = []
for f in core:
    src = texts['core'][f]
    tree = ast.parse(src)
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef)): continue
        if node.name.startswith('_'): continue
        n = node.end_lineno - node.lineno + 1
        pat = re.compile(r'(?<![A-Za-z0-9_])' + re.escape(node.name) + r'(?![A-Za-z0-9_])')
        cnt = {}
        for a, d in texts.items():
            c = 0
            for p, t in d.items():
                if a == 'core' and p == f: continue
                c += len(pat.findall(t))
            cnt[a] = c
        own = len(pat.findall(src)) - len(pat.findall(ast.get_source_segment(src, node) or ''))
        rows.append((f.name, node.lineno, node.name, n, own, cnt))
print('Public top-level defs NOT used by other core modules, library, examples, or vignettes (only own file / tests / docs):')
print(f'{"lines":>5} {"file:line":28s} {"name":28s} {"own":>4} {"core":>5} {"lib":>4} {"tests":>5} {"docs":>5} {"ex":>3} {"vig":>4}')
tot = 0
for f, ln, name, n, own, c in sorted(rows, key=lambda r: -r[3]):
    if c['core'] == 0 and c['lib'] == 0 and c['ex'] == 0 and c['vig'] == 0:
        tot += n
        print(f'{n:5d} {f+":"+str(ln):28s} {name:28s} {own:4d} {c["core"]:5d} {c["lib"]:4d} {c["tests"]:5d} {c["docs"]:5d} {c["ex"]:3d} {c["vig"]:4d}')
print('total lines:', tot)
