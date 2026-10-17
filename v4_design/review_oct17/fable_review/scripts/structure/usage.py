"""Count uses of names across tests/, library/, docs/, vignettes/ and starsim core (excluding the defining module)."""
import re, sys, pathlib, collections
R = pathlib.Path('/home/cliffk/idm/ssv4')
V = pathlib.Path('/home/cliffk/idm/lingua_franca_design/vignettes')
areas = {
    'core': [p for p in (R/'starsim').glob('*.py')] + list((R/'starsim/interop').glob('*.py')),
    'library': list((R/'starsim/library').rglob('*.py')),
    'tests': list((R/'tests').glob('test_*.py')),
    'docs': list((R/'docs').rglob('*.qmd')) + list((R/'docs').rglob('*.py')),
    'examples': list((R/'v4_design/examples').rglob('*.py')),
    'vignettes': list(V.rglob('*.py')) if V.exists() else [],
}
texts = {a: [(p, p.read_text(errors='ignore')) for p in ps] for a, ps in areas.items()}
names = sys.argv[1:]
print(f'{"name":28s} ' + ' '.join(f'{a:>9s}' for a in areas) + '   files(tests/lib/docs/vign)')
for name in names:
    pat = re.compile(r'(?<![A-Za-z0-9_])' + re.escape(name) + r'(?![A-Za-z0-9_])')
    counts = {}
    files = collections.defaultdict(set)
    for a, items in texts.items():
        c = 0
        for p, t in items:
            # skip the file that defines it (class/def) for core
            if a == 'core' and re.search(r'^(class|def) ' + re.escape(name) + r'\b', t, re.M):
                continue
            k = len(pat.findall(t))
            if k:
                c += k
                files[a].add(p.name if a != 'vignettes' else str(p.relative_to(V)))
        counts[a] = c
    fl = '; '.join(f'{a}: {",".join(sorted(files[a]))[:160]}' for a in ('tests','library','docs','vignettes') if files[a])
    print(f'{name:28s} ' + ' '.join(f'{counts[a]:9d}' for a in areas) + '   ' + fl)
